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
from finetune.em_trainer import EMTrainer, _PhaseCallback
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
        with patch.object(
            Trainer, "__init__", lambda self, *a, **k: setattr(self, "args", k["args"])
        ), patch.object(Trainer, "add_callback", lambda self, cb: None):
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

        tables = (model.new_shared,) if model.tied else (model.new_embed, model.new_lm_head)
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


class TestPhaseCallback:
    def _trainer(self, seq="em"):
        t = EMTrainer.__new__(EMTrainer)
        t.training_sequence = seq
        return t

    def _layer0_trainable(self, model):
        return any(
            p.requires_grad
            for n, p in model.base.named_parameters()
            if "layers.0" in n
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
        cb.on_epoch_begin(None, SimpleNamespace(epoch=1.0, is_world_process_zero=False), None, model=model)
        assert t._current_phase == "M"
        assert not model.new_embed.weight.requires_grad
        assert self._layer0_trainable(model)

    def test_float_epoch_rounds(self):
        # state.epoch drifts slightly off the integer at epoch begin.
        t = self._trainer()
        model = tiny_em(False)
        cb = _PhaseCallback(t)
        cb.on_epoch_begin(None, SimpleNamespace(epoch=0.999, is_world_process_zero=False), None, model=model)
        assert t._current_phase == "M"
