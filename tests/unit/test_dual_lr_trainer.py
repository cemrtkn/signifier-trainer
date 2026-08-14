"""Unit tests for DualLRTrainer (embedding_lr / model_lr dual-rate SFT).

Same constraint as test_em_trainer.py: a real transformers.Trainer cannot be
instantiated in the plain test env, so the trainer logic is exercised at the
method level via __new__ + targeted patches. run_sft dispatch is covered by
patching the three trainer classes at the module boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer

import finetune.sft as sft
from finetune.dual_lr_trainer import DualLRTrainer
from finetune.sft_types import TrainingConfig

EMB_LR, MODEL_LR = 1e-3, 2e-5

PEFT = {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "r": 8,
}
QUANT = {
    "do_quantization": True,
    "load_in_4bit": True,
    "load_in_8bit": False,
    "double_quant": True,
    "quant_type_4bit": "nf4",
}


def tiny(tie: bool = False) -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
    )
    return Qwen2ForCausalLM(cfg)


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


class TestDualLRConfigValidation:
    def test_lone_embedding_lr_rejected(self):
        with pytest.raises(ValueError, match="both embedding_lr and model_lr"):
            TrainingConfig(**base_cfg(embedding_lr=EMB_LR))

    def test_lone_model_lr_rejected(self):
        with pytest.raises(ValueError, match="both embedding_lr and model_lr"):
            TrainingConfig(**base_cfg(model_lr=MODEL_LR))

    def test_dual_lr_with_em_on_rejected(self):
        with pytest.raises(ValueError, match="excludes EM training"):
            TrainingConfig(
                **base_cfg(
                    embedding_lr=EMB_LR,
                    model_lr=MODEL_LR,
                    em_config={"status": True},
                )
            )

    @pytest.mark.parametrize(
        "extra",
        [
            {"peft_config": PEFT},
            {"quantization": QUANT},
            {"partial_fine_tuning": {"unfrozen_layers": 2}},
        ],
        ids=["peft", "quantization", "partial_fine_tuning"],
    )
    def test_dual_lr_with_non_full_ft_rejected(self, extra):
        with pytest.raises(ValueError, match="full-FT only"):
            TrainingConfig(**base_cfg(embedding_lr=EMB_LR, model_lr=MODEL_LR, **extra))

    def test_both_rates_accepted(self):
        cfg = TrainingConfig(**base_cfg(embedding_lr=EMB_LR, model_lr=MODEL_LR))
        assert (cfg.embedding_lr, cfg.model_lr) == (EMB_LR, MODEL_LR)

    def test_neither_rate_accepted(self):
        cfg = TrainingConfig(**base_cfg())
        assert cfg.embedding_lr is None and cfg.model_lr is None

    def test_both_rates_with_em_off_accepted(self):
        cfg = TrainingConfig(
            **base_cfg(
                embedding_lr=EMB_LR,
                model_lr=MODEL_LR,
                em_config={"status": False},
            )
        )
        assert (cfg.embedding_lr, cfg.model_lr) == (EMB_LR, MODEL_LR)
        assert not cfg.em_config.status


def _make_trainer(model, embedding_lr=EMB_LR, model_lr=MODEL_LR):
    t = DualLRTrainer.__new__(DualLRTrainer)
    t.optimizer = None
    t.model = model
    t.args = SimpleNamespace(weight_decay=0.01)
    # Present so create_optimizer skips get_optimizer_cls_and_kwargs; the stray
    # lr / weight_decay must be popped, not applied to the groups.
    t.optimizer_cls_and_kwargs = (
        torch.optim.AdamW,
        {"lr": 9.9, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.5},
    )
    t.embedding_lr = embedding_lr
    t.model_lr = model_lr
    return t


class TestCreateOptimizer:
    @staticmethod
    def _embed_ids(model):
        modules = [model.get_input_embeddings(), model.get_output_embeddings()]
        return {id(p) for m in modules if m is not None for p in m.parameters()}

    @pytest.mark.parametrize("tie", [False, True])
    def test_param_groups(self, tie):
        model = tiny(tie)
        t = _make_trainer(model)
        groups = t.create_optimizer().param_groups
        assert len(groups) == 3

        g_embed, g_decay, g_nodecay = groups
        assert g_embed["lr"] == EMB_LR and g_embed["weight_decay"] == 0.01
        assert g_decay["lr"] == MODEL_LR and g_decay["weight_decay"] == 0.01
        assert g_nodecay["lr"] == MODEL_LR and g_nodecay["weight_decay"] == 0.0

        embed_ids = self._embed_ids(model)
        assert {id(p) for p in g_embed["params"]} == embed_ids
        # tied -> the one shared tensor; untied -> input + output matrices
        assert len(g_embed["params"]) == (1 if tie else 2)

        decay = set(DualLRTrainer.get_decay_parameter_names(t, model))
        expected_decay = {
            id(p)
            for n, p in model.named_parameters()
            if id(p) not in embed_ids and n in decay
        }
        expected_nodecay = {
            id(p)
            for n, p in model.named_parameters()
            if id(p) not in embed_ids and n not in decay
        }
        assert expected_decay and expected_nodecay
        assert {id(p) for p in g_decay["params"]} == expected_decay
        assert {id(p) for p in g_nodecay["params"]} == expected_nodecay

        # every param lands in exactly one group (a partition)
        all_ids = [id(p) for g in groups for p in g["params"]]
        assert len(set(all_ids)) == len(all_ids)
        assert sorted(all_ids) == sorted(id(p) for p in model.parameters())

    def test_tied_model_shares_the_embedding_tensor(self):
        # Guards the premise of the tied case above: id()-based membership is
        # only meaningful if the two matrices really are one tensor.
        model = tiny(True)
        assert (
            model.get_input_embeddings().weight is model.get_output_embeddings().weight
        )

    def test_optimizer_cls_and_kwargs_respected(self):
        t = _make_trainer(tiny(False))
        opt = t.create_optimizer()
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.defaults["betas"] == (0.9, 0.999)
        assert opt.defaults["eps"] == 1e-8
        # top-level lr / weight_decay popped, not leaked into the groups
        assert all(g["lr"] in (EMB_LR, MODEL_LR) for g in opt.param_groups)
        assert all(g["weight_decay"] in (0.01, 0.0) for g in opt.param_groups)

    def test_existing_optimizer_reused(self):
        t = _make_trainer(tiny(False))
        first = t.create_optimizer()
        assert t.create_optimizer() is first


class TestLogLr:
    def test_injects_both_rates_and_repoints_learning_rate(self):
        t = _make_trainer(tiny(False))
        t.create_optimizer()
        logs = {"learning_rate": 0.123, "loss": 1.0}
        with patch.object(Trainer, "log", lambda self, logs, *a, **k: None):
            t.log(logs)
        assert logs["lr_embed"] == EMB_LR
        assert logs["lr_model"] == MODEL_LR
        assert logs["learning_rate"] == MODEL_LR
        assert logs["loss"] == 1.0

    def test_reads_live_group_lrs(self):
        # The scheduler mutates group lrs in place; the log must follow them
        # rather than echo the configured constants.
        t = _make_trainer(tiny(False))
        t.create_optimizer()
        t.optimizer.param_groups[0]["lr"] = 5e-4
        t.optimizer.param_groups[1]["lr"] = 1e-5
        logs = {"learning_rate": 0.123}
        with patch.object(Trainer, "log", lambda self, logs, *a, **k: None):
            t.log(logs)
        assert (logs["lr_embed"], logs["lr_model"]) == (5e-4, 1e-5)
        assert logs["learning_rate"] == 1e-5

    def test_no_learning_rate_key_is_untouched(self):
        t = _make_trainer(tiny(False))
        t.create_optimizer()
        logs = {"eval_loss": 2.0}
        with patch.object(Trainer, "log", lambda self, logs, *a, **k: None):
            t.log(logs)
        assert logs["lr_embed"] == EMB_LR and logs["lr_model"] == MODEL_LR
        assert "learning_rate" not in logs

    def test_no_optimizer_leaves_logs_untouched(self):
        t = _make_trainer(tiny(False))
        assert t.optimizer is None
        logs = {"learning_rate": 0.123, "loss": 1.0}
        with patch.object(Trainer, "log", lambda self, logs, *a, **k: None):
            t.log(logs)
        assert logs == {"learning_rate": 0.123, "loss": 1.0}

    def test_single_param_group_leaves_logs_untouched(self):
        t = _make_trainer(tiny(False))
        t.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-4)
        logs = {"learning_rate": 0.123}
        with patch.object(Trainer, "log", lambda self, logs, *a, **k: None):
            t.log(logs)
        assert logs == {"learning_rate": 0.123}

    def test_forwards_args_and_kwargs_to_super(self):
        t = _make_trainer(tiny(False))
        t.create_optimizer()
        seen = {}
        with patch.object(
            Trainer,
            "log",
            lambda self, logs, *a, **k: seen.update(logs=logs, args=a, kwargs=k),
        ):
            t.log({"loss": 1.0}, 7, start_time=3)
        assert seen["args"] == (7,)
        assert seen["kwargs"] == {"start_time": 3}
        assert seen["logs"]["lr_model"] == MODEL_LR


class TestRunSFTBranch:
    """Which trainer class run_sft picks. Everything heavy is patched out."""

    @staticmethod
    def _config(**over):
        train_args = SimpleNamespace(
            seed=0,
            output_dir="out",
            eval_strategy="no",
            resume_from_checkpoint=None,
            logging_first_step=False,
            learning_rate=1e-5,
        )
        cfg = SimpleNamespace(
            model="fake-model",
            train_args=train_args,
            ptmp_dir=None,
            em_config=None,
            embedding_lr=None,
            model_lr=None,
            train_dataset_config=SimpleNamespace(
                resolve_signifier_config=lambda: SimpleNamespace(
                    mode="natural_language_sys", new_special_tokens=[]
                ),
                test_fold=0,
            ),
            run_profiler=False,
        )
        for key, value in over.items():
            setattr(cfg, key, value)
        return cfg

    @staticmethod
    def _run(config):
        """run_sft with every heavy piece stubbed; returns the trainer mocks."""
        with (
            patch.object(sft, "set_seed"),
            patch.object(sft, "get_model", return_value=MagicMock()),
            patch.object(
                sft.AutoTokenizer, "from_pretrained", return_value=MagicMock()
            ),
            patch.object(
                sft,
                "load_dataset_and_collator",
                return_value=({"train": MagicMock()}, MagicMock()),
            ),
            patch.object(sft.os, "makedirs"),
            patch.object(sft.dist, "is_initialized", return_value=False),
            # sft.py prints dist.get_rank() unguarded in the save tail
            patch.object(sft.dist, "get_rank", return_value=0),
            patch.object(sft, "Trainer") as stock,
            patch.object(sft, "DualLRTrainer") as dual,
            patch.object(sft, "EMTrainer") as em,
        ):
            for mock in (stock, dual, em):
                mock.return_value.is_fsdp_enabled = False
            sft.run_sft(config)
            return stock, dual, em

    def test_both_rates_select_dual_lr_trainer(self):
        stock, dual, em = self._run(
            self._config(embedding_lr=EMB_LR, model_lr=MODEL_LR)
        )
        assert dual.call_count == 1
        assert stock.call_count == 0 and em.call_count == 0
        kwargs = dual.call_args.kwargs
        assert kwargs["embedding_lr"] == EMB_LR
        assert kwargs["model_lr"] == MODEL_LR

    def test_no_rates_select_stock_trainer(self):
        stock, dual, em = self._run(self._config())
        assert stock.call_count == 1
        assert dual.call_count == 0 and em.call_count == 0
        assert "embedding_lr" not in stock.call_args.kwargs

    def test_em_on_selects_em_trainer(self):
        em_config = SimpleNamespace(status=True)
        stock, dual, em = self._run(self._config(em_config=em_config))
        assert em.call_count == 1
        assert stock.call_count == 0 and dual.call_count == 0
        assert em.call_args.kwargs["em_config"] is em_config
