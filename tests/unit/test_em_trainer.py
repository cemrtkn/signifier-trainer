"""Unit tests for EMTrainer.

A real transformers.Trainer cannot be instantiated in the plain test env
(TrainingArguments device setup needs a newer torch than the Intel-Mac wheel,
and accelerate is absent), so these tests exercise the EMTrainer logic at the
method level via __new__ + targeted patches. The full run_sft integration is
covered by the cluster smoke (issue #2)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer

from finetune.em_model import EMModel
from finetune.em_trainer import EMTrainer, _PhaseCallback, build_phase_step_map
from finetune.sft_types import EMConfig, TrainingConfig

ORIG_VOCAB = 64
N_NEW = 4


def tiny_em(tie: bool = False, n_new: int = N_NEW) -> EMModel:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=ORIG_VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
    )
    model = EMModel(Qwen2ForCausalLM(cfg), n_new, pad_to_multiple_of=None)
    model.resize_token_embeddings(ORIG_VOCAB + n_new)
    return model


def base_cfg(**over):
    cfg = dict(
        model="m",
        train_args={},
        train_dataset_config=dict(
            data_path="x",
            parser_config={"fields": {}},
            signifier_config={
                "mode": "token_signifier",
                "new_special_tokens": ["<|A|>"],
            },
        ),
    )
    cfg.update(over)
    return cfg


class TestEMConfigValidation:
    def test_em_with_peft_rejected(self):
        with pytest.raises(ValueError, match="LoRA|peft"):
            TrainingConfig(
                **base_cfg(
                    em_config={"status": True},
                    peft_config={
                        "peft_type": "LORA",
                        "task_type": "CAUSAL_LM",
                        "lora_alpha": 16,
                        "lora_dropout": 0.05,
                        "r": 8,
                    },
                )
            )

    def test_em_with_quantization_rejected(self):
        with pytest.raises(ValueError, match="quantization|LoRA"):
            TrainingConfig(
                **base_cfg(
                    em_config={"status": True},
                    quantization={
                        "do_quantization": True,
                        "load_in_4bit": True,
                        "load_in_8bit": False,
                        "double_quant": True,
                        "quant_type_4bit": "nf4",
                    },
                )
            )

    def test_em_with_partial_ft_rejected(self):
        with pytest.raises(ValueError, match="partial_fine_tuning|LoRA"):
            TrainingConfig(
                **base_cfg(
                    em_config={"status": True},
                    partial_fine_tuning={"unfrozen_layers": 2},
                )
            )

    def test_em_requires_token_signifier(self):
        with pytest.raises(ValueError, match="token_signifier"):
            TrainingConfig(
                **base_cfg(
                    em_config={"status": True},
                    train_dataset_config=dict(
                        data_path="x",
                        parser_config={"fields": {}},
                        signifier_config={"mode": "nl_signifier"},
                    ),
                )
            )

    def test_lr_fields_without_status_rejected(self):
        with pytest.raises(ValueError, match="status"):
            TrainingConfig(
                **base_cfg(em_config={"status": False, "e_learning_rate": 1e-4})
            )

    def test_stray_training_sequence_when_off_rejected(self):
        with pytest.raises(ValueError, match="status"):
            TrainingConfig(
                **base_cfg(em_config={"status": False, "training_sequence": "meme"})
            )

    @pytest.mark.parametrize("seq", ["em", "emem", "meme", "eem"])
    def test_training_sequence_valid(self, seq):
        EMConfig(status=True, training_sequence=seq)

    @pytest.mark.parametrize("seq", ["emx", "", "EM", "e m"])
    def test_training_sequence_invalid(self, seq):
        with pytest.raises(ValueError, match="training_sequence"):
            EMConfig(status=True, training_sequence=seq)

    def test_valid_em_config_accepted(self):
        cfg = TrainingConfig(
            **base_cfg(
                em_config={
                    "status": True,
                    "training_sequence": "emem",
                    "e_learning_rate": 1e-4,
                    "m_learning_rate": 2e-6,
                }
            )
        )
        assert cfg.em_config.status
        assert cfg.em_config.training_sequence == "emem"

    def test_absent_em_config_is_none(self):
        assert TrainingConfig(**base_cfg()).em_config is None


class TestPhaseSchedule:
    @pytest.mark.parametrize(
        "seq,expected",
        [
            ("em", ["E", "M"]),
            ("emem", ["E", "M", "E", "M"]),
            ("meme", ["M", "E", "M", "E"]),
            ("eem", ["E", "E", "M"]),
        ],
    )
    def test_phase_for_epoch(self, seq, expected):
        t = EMTrainer.__new__(EMTrainer)
        t.training_sequence = seq
        assert [t.phase_for_epoch(k) for k in range(len(seq))] == expected


class TestInit:
    """__init__ up to (patched) super: sequence -> num_train_epochs, lr resolution."""

    def _make(self, em_config, num_train_epochs=99, learning_rate=3e-6):
        targs = SimpleNamespace(
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=0.01,
        )
        with (
            patch.object(
                Trainer,
                "__init__",
                lambda self, *a, **k: setattr(self, "args", k["args"]),
            ),
            patch.object(Trainer, "add_callback", lambda self, cb: None),
        ):
            t = EMTrainer(model=object(), args=targs, em_config=em_config)
        return t, targs

    def test_derives_num_train_epochs_and_resolves_lrs(self):
        em = EMConfig(
            status=True,
            training_sequence="emem",
            e_learning_rate=1e-4,
            m_learning_rate=2e-6,
        )
        t, targs = self._make(em)
        assert targs.num_train_epochs == 4
        assert t.training_sequence == "emem"
        assert (t.e_lr, t.m_lr) == (1e-4, 2e-6)
        assert t._current_phase == "E"

    def test_default_sequence_is_em(self):
        t, targs = self._make(EMConfig(status=True))
        assert t.training_sequence == "em"
        assert targs.num_train_epochs == 2

    def test_lr_fallback_to_learning_rate(self):
        t, _ = self._make(EMConfig(status=True), learning_rate=3e-6)
        assert (t.e_lr, t.m_lr) == (3e-6, 3e-6)


class TestCreateOptimizer:
    def _make(self, model, e_lr=1e-4, m_lr=2e-6):
        t = EMTrainer.__new__(EMTrainer)
        t.optimizer = None
        t.model = model
        t.args = SimpleNamespace(weight_decay=0.01)
        # Present so create_optimizer skips get_optimizer_cls_and_kwargs; the
        # stray lr / weight_decay must be popped, not applied to the groups.
        t.optimizer_cls_and_kwargs = (
            torch.optim.AdamW,
            {"lr": 9.9, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.5},
        )
        t.e_lr = e_lr
        t.m_lr = m_lr
        return t

    @pytest.mark.parametrize("tie", [False, True])
    def test_param_groups(self, tie):
        model = tiny_em(tie)
        t = self._make(model, e_lr=1e-4, m_lr=2e-6)
        opt = t.create_optimizer()
        groups = opt.param_groups
        assert len(groups) == 3

        table_group, m_decay, m_nodecay = groups
        assert table_group["lr"] == 1e-4 and table_group["weight_decay"] == 0.01
        assert m_decay["lr"] == 2e-6 and m_decay["weight_decay"] == 0.01
        assert m_nodecay["lr"] == 2e-6 and m_nodecay["weight_decay"] == 0.0

        tables = (
            (model.new_shared,) if model.tied else (model.new_embed, model.new_lm_head)
        )
        table_ids = {id(p) for mod in tables for p in mod.parameters()}
        group0_ids = {id(p) for p in table_group["params"]}
        assert group0_ids == table_ids

        # every param lands in exactly one group (a partition)
        all_group_ids = [id(p) for g in groups for p in g["params"]]
        assert sorted(all_group_ids) == sorted(id(p) for p in model.parameters())
        assert len(set(all_group_ids)) == len(all_group_ids)

    def test_stray_lr_and_wd_popped(self):
        t = self._make(tiny_em(False))
        opt = t.create_optimizer()
        assert all(g["lr"] in (1e-4, 2e-6) for g in opt.param_groups)
        assert all(g["weight_decay"] in (0.01, 0.0) for g in opt.param_groups)


class TestLogLr:
    def _make(self):
        t = EMTrainer.__new__(EMTrainer)
        p_e = torch.nn.Parameter(torch.zeros(2))
        p_m = torch.nn.Parameter(torch.zeros(2))
        t.optimizer = torch.optim.AdamW(
            [{"params": [p_e], "lr": 1e-4}, {"params": [p_m], "lr": 2e-6}]
        )
        return t

    @pytest.mark.parametrize("phase,expected", [("E", 1e-4), ("M", 2e-6)])
    def test_active_phase_learning_rate(self, phase, expected):
        t = self._make()
        t._current_phase = phase
        logs = {"learning_rate": 0.123, "loss": 1.0}
        with patch.object(Trainer, "log", lambda self, logs, *a, **k: None):
            t.log(logs)
        assert logs["lr_e"] == 1e-4
        assert logs["lr_m"] == 2e-6
        assert logs["learning_rate"] == expected

    def test_no_learning_rate_key_is_untouched(self):
        t = self._make()
        t._current_phase = "M"
        logs = {"eval_loss": 2.0}
        with patch.object(Trainer, "log", lambda self, logs, *a, **k: None):
            t.log(logs)
        assert logs["lr_e"] == 1e-4 and logs["lr_m"] == 2e-6
        assert "learning_rate" not in logs


class TestPhaseStepMap:
    """The pure step->phase-timeline map that gives each phase its memory."""

    def test_emem_own_timeline_pauses_and_continues(self):
        # emem over 40 steps: epochs [0,10) E, [10,20) M, [20,30) E, [30,40) M.
        m = build_phase_step_map("emem", 40)
        assert m.totals == {"E": 20, "M": 20}
        # E advances 0..9 in its first epoch, holds flat across the M epoch,
        # then *continues* 10..19 in the second E (not restart, not the M tail).
        assert m.own_elapsed("E", 9) == 9
        assert m.own_elapsed("E", 10) == m.own_elapsed("E", 19) == 10  # held
        assert m.own_elapsed("E", 20) == 10 and m.own_elapsed("E", 29) == 19
        # M is symmetric: flat until its first epoch, then continues across E.
        assert m.own_elapsed("M", 9) == 0
        assert m.own_elapsed("M", 19) == 9
        assert m.own_elapsed("M", 20) == m.own_elapsed("M", 29) == 10  # held
        assert m.own_elapsed("M", 39) == 19
        assert [m.phase_of(s) for s in (0, 10, 20, 30)] == ["E", "M", "E", "M"]

    def test_uneven_split_totals_sum_to_steps(self):
        m = build_phase_step_map("emem", 37)
        assert m.totals["E"] + m.totals["M"] == 37


class TestCreateScheduler:
    def _make(self, seq="em", warmup=2):
        t = EMTrainer.__new__(EMTrainer)
        t.lr_scheduler = None
        t.training_sequence = seq
        # Three groups mirror create_optimizer (E table, M decay, M no-decay);
        # LambdaLR needs one base group per phase lambda. base lr 1.0 so each
        # sched.lr_lambdas[i] is the bare factor.
        params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
        t.optimizer = torch.optim.SGD([{"params": [p], "lr": 1.0} for p in params])
        t.args = SimpleNamespace(
            lr_scheduler_type="linear",
            lr_scheduler_kwargs={},
            get_warmup_steps=lambda n: warmup,
        )
        return t

    def test_preset_scheduler_delegates_to_stock(self):
        t = self._make()
        t.lr_scheduler = "already-set"
        sentinel = object()
        with patch.object(
            Trainer, "create_scheduler", lambda self, n, o=None: sentinel
        ):
            assert t.create_scheduler(20) is sentinel

    def test_em_each_phase_warms_and_decays_over_its_own_epoch(self):
        # Single cycle, 20 steps: E owns [0,10), M owns [10,20). Each phase
        # warms 2 steps then decays over its own epoch — the sensible default
        # that used to require an explicit reset at the E->M boundary.
        t = self._make(seq="em", warmup=2)
        sched = t.create_scheduler(20)
        fn_e, fn_m = sched.lr_lambdas[0], sched.lr_lambdas[1]
        # E: warmup peak at step 2, decaying (still > 0) by the end of its epoch.
        assert fn_e(0) == pytest.approx(0.0)
        assert fn_e(2) == pytest.approx(1.0)
        assert 0.0 < fn_e(9) < 1.0
        # M does NOT inherit E's decay: it holds at its warmup start through the
        # E epoch, then warms fresh over its own epoch (peak at step 12).
        assert fn_m(9) == pytest.approx(0.0)
        assert fn_m(10) == pytest.approx(0.0)
        assert fn_m(12) == pytest.approx(1.0)
        assert 0.0 < fn_m(19) < 1.0

    def test_emem_phase_schedules_have_memory(self):
        # emem over 40 steps. E owns [0,10)+[20,30); M owns [10,20)+[30,40).
        t = self._make(seq="emem", warmup=2)
        sched = t.create_scheduler(40)
        fn_e, fn_m = sched.lr_lambdas[0], sched.lr_lambdas[1]
        # E group lambda == M group lambda for the two M param groups.
        assert sched.lr_lambdas[1] is not sched.lr_lambdas[2]
        # E holds flat across the intervening M epoch...
        assert fn_e(10) == pytest.approx(fn_e(15)) == pytest.approx(fn_e(19))
        # ...and the 2nd E *continues* the 1st E's decay: it resumes exactly at
        # the paused value, below where the 1st E ended, and is not a re-warmup
        # to the peak nor the fully-decayed tail.
        assert fn_e(20) == pytest.approx(fn_e(19))
        assert fn_e(20) < fn_e(9)
        assert 0.0 < fn_e(20) < fn_e(2)
        # M is symmetric: flat across the 2nd E epoch, 2nd M continues the decay.
        assert fn_m(20) == pytest.approx(fn_m(25)) == pytest.approx(fn_m(29))
        assert fn_m(30) == pytest.approx(fn_m(29))
        assert 0.0 < fn_m(30) < fn_m(12)

    def test_lr_lambdas_are_pure_functions_of_step(self):
        # Resume safety: same global step -> same factor, no hidden state.
        t = self._make(seq="emem", warmup=2)
        sched = t.create_scheduler(40)
        for fn in sched.lr_lambdas:
            assert [fn(s) for s in range(40)] == [fn(s) for s in range(40)]

    def test_unsupported_scheduler_type_rejected(self):
        t = self._make()
        t.args.lr_scheduler_type = "reduce_lr_on_plateau"
        with pytest.raises(ValueError, match="LambdaLR-family"):
            t.create_scheduler(20)


class TestPhaseCallback:
    def _trainer(self, seq="em"):
        t = EMTrainer.__new__(EMTrainer)
        t.training_sequence = seq
        return t

    def _layer0_trainable(self, model):
        return any(
            p.requires_grad for n, p in model.base.named_parameters() if "layers.0" in n
        )

    def test_epoch0_sets_e_phase(self):
        t = self._trainer()
        model = tiny_em(False)
        cb = _PhaseCallback(t)
        state = SimpleNamespace(epoch=0.0, is_world_process_zero=False)
        cb.on_epoch_begin(None, state, None, model=model)
        assert t._current_phase == "E"
        assert model.new_embed.weight.requires_grad
        assert not model.base.get_input_embeddings().weight.requires_grad
        assert not self._layer0_trainable(model)

    def test_epoch1_sets_m_phase(self):
        t = self._trainer()
        model = tiny_em(False)
        cb = _PhaseCallback(t)
        cb.on_epoch_begin(
            None,
            SimpleNamespace(epoch=1.0, is_world_process_zero=False),
            None,
            model=model,
        )
        assert t._current_phase == "M"
        assert not model.new_embed.weight.requires_grad
        assert self._layer0_trainable(model)

    def test_float_epoch_rounds(self):
        # state.epoch drifts slightly off the integer at epoch begin.
        t = self._trainer()
        model = tiny_em(False)
        cb = _PhaseCallback(t)
        cb.on_epoch_begin(
            None,
            SimpleNamespace(epoch=0.999, is_world_process_zero=False),
            None,
            model=model,
        )
        assert t._current_phase == "M"
