"""Unit tests for DPO mode (#13): config gating, the preference dataset
loader, and the pair-preserving sampler.

Same constraint as test_em_trainer.py: no real trainer is instantiated;
sampler/trainer logic is exercised at the method level via __new__."""

from types import SimpleNamespace

import pytest
import yaml
from datasets import Dataset, DatasetDict

from finetune.dpo import PairedDPOTrainer, PairPreservingSampler, check_signifier_tokens
from finetune.dpo_config import SignifierDPOConfig
from finetune.sft_types import TrainingConfig
from finetune.utils.config import load_config
from finetune.utils.dataset import load_dpo_dataset

PARSER_FIELDS = {
    "system_prompt": {
        "text": "<|im_start|>system\n{signifiers}<|im_end|>\n<|im_start|>user",
        "baseline": "<|im_start|>system\nbase<|im_end|>\n<|im_start|>user",
    },
    "question": {"text": "\n{question}<|im_end|>\n<|im_start|>assistant"},
    "answer": {"text": "\n{answer}<|im_end|>"},
}


def base_config(**overrides) -> dict:
    cfg = {
        "model": "m",
        "train_args": {"output_dir": "x", "bf16": False},
        "train_dataset_config": {
            "data_path": "x",
            "signifier_config": {
                "mode": "token_signifier",
                "new_special_tokens": ["<|party:A|>", "<|party:B|>"],
            },
            "parser_config": {"fields": PARSER_FIELDS},
        },
    }
    cfg.update(overrides)
    return cfg


def load_yaml_config(tmp_path, cfg: dict) -> TrainingConfig:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return load_config(str(path))


class TestConfigGate:
    def test_dpo_roundtrips_signifier_dpo_config(self, tmp_path):
        cfg = base_config(mode="dpo")
        cfg["train_args"].update({"beta": 0.05, "batch_bidirectionals": True})
        loaded = load_yaml_config(tmp_path, cfg)
        assert isinstance(loaded.train_args, SignifierDPOConfig)
        assert loaded.train_args.beta == 0.05
        assert loaded.train_args.batch_bidirectionals is True

    def test_sft_rejects_dpo_knobs(self, tmp_path):
        cfg = base_config()
        cfg["train_args"]["beta"] = 0.05
        with pytest.raises(TypeError, match="beta"):
            load_yaml_config(tmp_path, cfg)

    def test_sft_rejects_batch_bidirectionals(self, tmp_path):
        cfg = base_config()
        cfg["train_args"]["batch_bidirectionals"] = True
        with pytest.raises(TypeError, match="batch_bidirectionals"):
            load_yaml_config(tmp_path, cfg)

    def test_bidirectionals_defaults_false(self, tmp_path):
        loaded = load_yaml_config(tmp_path, base_config(mode="dpo"))
        assert loaded.train_args.batch_bidirectionals is False

    def test_mode_defaults_sft(self, tmp_path):
        loaded = load_yaml_config(tmp_path, base_config())
        assert loaded.mode == "sft"
        assert not isinstance(loaded.train_args, SignifierDPOConfig)

    def test_dpo_excludes_em(self, tmp_path):
        cfg = base_config(mode="dpo", em_config={"status": True})
        with pytest.raises(ValueError, match="excludes EM"):
            load_yaml_config(tmp_path, cfg)

    def test_dpo_excludes_dual_lr(self, tmp_path):
        cfg = base_config(mode="dpo", embedding_lr=1e-3, model_lr=2e-5)
        with pytest.raises(ValueError, match="uniform-lr"):
            load_yaml_config(tmp_path, cfg)

    def test_dpo_excludes_peft(self, tmp_path):
        peft = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "r": 8,
        }
        cfg = base_config(mode="dpo", peft_config=peft)
        with pytest.raises(ValueError, match="full-FT only"):
            load_yaml_config(tmp_path, cfg)

    def test_dpo_rejects_plain_training_arguments(self):
        from transformers import TrainingArguments

        cfg = base_config(mode="dpo")
        cfg["train_args"] = TrainingArguments(output_dir="x", bf16=False)
        with pytest.raises(ValueError, match="DPOConfig"):
            TrainingConfig(**cfg)


class TestPairPreservingSampler:
    def test_pairs_adjacent_and_complete(self):
        sampler = PairPreservingSampler(20, seed=0)
        order = list(sampler)
        assert sorted(order) == list(range(20))
        assert all(
            order[i] % 2 == 0 and order[i + 1] == order[i] + 1 for i in range(0, 20, 2)
        )

    def test_epochs_reshuffle_deterministically(self):
        e1, e2 = list(PairPreservingSampler(40, seed=0)), None
        sampler = PairPreservingSampler(40, seed=0)
        assert list(sampler) == e1  # same seed, first epoch identical
        e2 = list(sampler)  # second __iter__ advances the generator
        assert e2 != e1 and sorted(e2) == sorted(e1)

    def test_odd_row_count_rejected(self):
        with pytest.raises(ValueError, match="even row count"):
            PairPreservingSampler(21, seed=0)

    def test_paired_trainer_uses_sampler(self):
        trainer = PairedDPOTrainer.__new__(PairedDPOTrainer)
        trainer.args = SimpleNamespace(seed=0)
        trainer.train_dataset = list(range(8))
        sampler = trainer._get_train_sampler()
        assert isinstance(sampler, PairPreservingSampler)
        assert len(sampler) == 8


class TestCheckSignifierTokens:
    class _Tok:
        def __init__(self, single: set):
            self.single = single

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] if text in self.single else [0, 1]}

    def test_passes_on_merged_checkpoint(self):
        sig = SimpleNamespace(mode="token_signifier", new_special_tokens=["<|a|>"])
        check_signifier_tokens(self._Tok({"<|a|>"}), sig)

    def test_fails_on_unmerged_tokens(self):
        sig = SimpleNamespace(mode="token_signifier", new_special_tokens=["<|a|>"])
        with pytest.raises(ValueError, match="merged"):
            check_signifier_tokens(self._Tok(set()), sig)

    def test_skips_nl_mode(self):
        sig = SimpleNamespace(mode="nl_signifier", new_special_tokens=None)
        check_signifier_tokens(self._Tok(set()), sig)


class TestLoadDpoDataset:
    def _rows(self, n_pairs: int, party=("<|party:A|>", "<|party:B|>")):
        rows = []
        for i in range(n_pairs):
            a, b = f"ans-a{i}", f"ans-b{i}"
            for sig, chosen, rejected in ((party[0], a, b), (party[1], b, a)):
                rows.append(
                    {
                        "signifiers": sig,
                        "question": f"q{i}",
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )
        return rows

    def _config(self, data_path) -> TrainingConfig:
        cfg = base_config(mode="dpo")
        cfg["train_args"] = SignifierDPOConfig(output_dir="x", bf16=False)
        cfg["train_dataset_config"]["data_path"] = str(data_path)
        return TrainingConfig(**cfg)

    def _save(self, tmp_path, folds: dict) -> str:
        path = tmp_path / "processed"
        DatasetDict({k: Dataset.from_list(v) for k, v in folds.items()}).save_to_disk(
            str(path)
        )
        return str(path)

    def test_renders_prompt_and_strips_eos(self, tmp_path):
        path = self._save(tmp_path, {"0": self._rows(1), "1": self._rows(2)})
        tok = SimpleNamespace(eos_token="<|im_end|>")
        dd = load_dpo_dataset(self._config(path), tok, test_fold=0)
        assert set(dd.keys()) == {"train", "test"}
        assert len(dd["train"]) == 4 and len(dd["test"]) == 2
        row = dd["test"][0]
        assert set(row.keys()) == {"prompt", "chosen", "rejected"}
        assert row["prompt"] == (
            "<|im_start|>system\n<|party:A|><|im_end|>\n<|im_start|>user"
            "\n\nq0<|im_end|>\n<|im_start|>assistant"
        )
        assert row["chosen"] == "\n\nans-a0"  # template eos stripped
        assert row["rejected"] == "\n\nans-b0"

    def test_adjacent_mirror_survives_fold_merge(self, tmp_path):
        path = self._save(
            tmp_path, {"0": self._rows(1), "1": self._rows(2), "2": self._rows(3)}
        )
        tok = SimpleNamespace(eos_token="<|im_end|>")
        dd = load_dpo_dataset(self._config(path), tok, test_fold=0)
        train = dd["train"]
        for i in range(0, len(train), 2):
            assert train[i]["chosen"] == train[i + 1]["rejected"]
            assert train[i]["rejected"] == train[i + 1]["chosen"]

    def test_answer_field_must_be_last(self, tmp_path):
        path = self._save(tmp_path, {"0": self._rows(1), "1": self._rows(1)})
        config = self._config(path)
        fields = config.train_dataset_config.parser_config.fields
        reordered = {"answer": fields["answer"], **fields}
        config.train_dataset_config.parser_config.fields = reordered
        with pytest.raises(ValueError, match="'answer' field last"):
            load_dpo_dataset(
                config, SimpleNamespace(eos_token="<|im_end|>"), test_fold=0
            )

    def test_stray_signifier_fails_fast(self, tmp_path):
        rows = self._rows(1, party=("<|party:A|>", "<|party:UNDECLARED|>"))
        path = self._save(tmp_path, {"0": rows, "1": rows})
        with pytest.raises(ValueError, match="not.*covered by new_special_tokens"):
            load_dpo_dataset(
                self._config(path), SimpleNamespace(eos_token="e"), test_fold=0
            )
