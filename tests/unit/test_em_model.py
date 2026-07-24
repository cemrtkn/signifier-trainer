"""Unit tests for the EMModel wrapper (tiny untied and tied checkpoints)."""

import json
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from finetune.em_model import EMModel, get_em_auto_wrap_policy

ORIG_VOCAB = 64
NEW_VOCAB = 68


def tiny_model(tie: bool):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=ORIG_VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
    )
    return Qwen2ForCausalLM(config)


def resized_em_model(tie: bool) -> EMModel:
    model = EMModel(tiny_model(tie))
    model.resize_token_embeddings(NEW_VOCAB)
    return model.eval()


def batch_with_new_ids():
    torch.manual_seed(1)
    ids = torch.randint(0, ORIG_VOCAB, (2, 10))
    ids[:, 3] = ORIG_VOCAB + 1
    ids[:, 7] = NEW_VOCAB - 1
    return ids


def tables_of(model: EMModel):
    if model.tied:
        return (model.new_shared,)
    return (model.new_embed, model.new_lm_head)


class TestConstructionGuards:
    def test_rejects_quantized(self):
        base = tiny_model(False)
        base.is_quantized = True
        with pytest.raises(ValueError, match="quantized"):
            EMModel(base)

    def test_rejects_peft(self):
        from peft import PeftModel

        with pytest.raises(ValueError, match="LoRA"):
            EMModel(PeftModel.__new__(PeftModel))

    def test_methods_require_resize_first(self):
        model = EMModel(tiny_model(False))
        with pytest.raises(RuntimeError):
            model(input_ids=torch.tensor([[1, 2]]))
        with pytest.raises(RuntimeError):
            model.set_phase("E")
        with pytest.raises(RuntimeError):
            model.save_merged("/dev/null")
        with pytest.raises(RuntimeError):
            get_em_auto_wrap_policy(model)


@pytest.mark.parametrize("tie", [False, True])
class TestResizeDispatch:
    def test_builds_strategy_tables_and_freezes_originals(self, tie):
        model = resized_em_model(tie)
        n_new = NEW_VOCAB - ORIG_VOCAB
        embed_weight = model.base.get_input_embeddings().weight
        assert embed_weight.shape[0] == NEW_VOCAB
        assert not embed_weight.requires_grad
        assert not model.base.get_output_embeddings().weight.requires_grad
        for table in tables_of(model):
            assert table.weight.shape[0] == n_new
            assert table.weight.requires_grad
        assert torch.equal(tables_of(model)[0].weight, embed_weight[ORIG_VOCAB:])

    def test_guards(self, tie):
        model = resized_em_model(tie)
        with pytest.raises(RuntimeError, match="once"):
            model.resize_token_embeddings(NEW_VOCAB + 4)
        with pytest.raises(ValueError, match="new tokens"):
            EMModel(tiny_model(tie)).resize_token_embeddings(ORIG_VOCAB)

    def test_plain_model_resize_is_stock(self, tie):
        model = tiny_model(tie)
        model.resize_token_embeddings(NEW_VOCAB)
        assert model.get_input_embeddings().weight.shape[0] == NEW_VOCAB
        assert model.get_input_embeddings().weight.requires_grad


@pytest.mark.parametrize("tie", [False, True])
class TestForward:
    def test_logit_and_loss_equivalence_with_resized_base(self, tie):
        model = resized_em_model(tie)
        old_ids = torch.randint(0, ORIG_VOCAB, (2, 10))
        for ids in (old_ids, batch_with_new_ids()):
            with torch.no_grad():
                out = model(input_ids=ids, labels=ids.clone())
                ref = model.base(input_ids=ids, labels=ids.clone())
            assert out.logits.shape == (2, 10, NEW_VOCAB)
            assert torch.allclose(out.logits, ref.logits, atol=1e-6)
            assert torch.allclose(out.loss, ref.loss, atol=1e-6)

    def test_gradients_reach_only_the_tables(self, tie):
        model = resized_em_model(tie).train()
        ids = batch_with_new_ids()
        model(input_ids=ids, labels=ids.clone()).loss.backward()
        for table in tables_of(model):
            assert table.weight.grad is not None
            assert table.weight.grad.abs().sum() > 0
        assert model.base.get_input_embeddings().weight.grad is None


@pytest.mark.parametrize("tie", [False, True])
class TestSetPhase:
    def test_trainable_numel_per_phase(self, tie):
        model = resized_em_model(tie)
        table_numel = sum(t.weight.numel() for t in tables_of(model))
        total_numel = sum(p.numel() for p in model.parameters())

        def trainable():
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.set_phase("E")
        assert trainable() == table_numel
        model.set_phase("M")
        assert trainable() == total_numel - table_numel
        assert model.base.get_input_embeddings().weight.requires_grad
        model.set_phase("E")
        assert trainable() == table_numel

    def test_rejects_unknown_phase(self, tie):
        with pytest.raises(ValueError, match="phase"):
            resized_em_model(tie).set_phase("X")


@pytest.mark.parametrize("tie", [False, True])
class TestSaveMerged:
    def test_round_trip(self, tie, tmp_path):
        model = resized_em_model(tie)
        with torch.no_grad():
            for table in tables_of(model):
                table.weight.add_(torch.randn_like(table.weight) * 0.1)
        model.save_merged(str(tmp_path))

        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["vocab_size"] == NEW_VOCAB
        assert config["tie_word_embeddings"] == tie
        assert os.path.exists(tmp_path / "model.safetensors")

        ids = batch_with_new_ids()
        with torch.no_grad():
            assert torch.allclose(
                model(input_ids=ids).logits,
                reloaded(input_ids=ids).logits,
                atol=1e-5,
            )
        if tie:
            assert (
                reloaded.get_input_embeddings().weight
                is reloaded.get_output_embeddings().weight
            )


class TestTiedStrategy:
    def test_builds_shared_table_and_warns(self, capsys):
        model = resized_em_model(True)
        assert hasattr(model, "new_shared")
        assert "new_embed" not in model._modules
        assert "new_lm_head" not in model._modules
        assert "tie_word_embeddings" in capsys.readouterr().out

    def test_untied_builds_pair_without_warning(self, capsys):
        model = resized_em_model(False)
        assert hasattr(model, "new_embed") and hasattr(model, "new_lm_head")
        assert "new_shared" not in model._modules
        assert "tie_word_embeddings" not in capsys.readouterr().out


@pytest.mark.parametrize("tie", [False, True])
class TestAutoWrapPolicy:
    def test_policy_picks_tables_and_decoder_layers(self, tie):
        model = resized_em_model(tie)
        policy = get_em_auto_wrap_policy(model)
        picked = [
            name
            for name, module in model.named_modules()
            if policy(module=module, recurse=False, nonwrapped_numel=0)
        ]
        expected_tables = ["new_shared"] if tie else ["new_embed", "new_lm_head"]
        for table_name in expected_tables:
            assert table_name in picked
        assert sum(".layers." in name for name in picked) == 2

    def test_units_uniform_in_requires_grad(self, tie):
        model = resized_em_model(tie)
        policy = get_em_auto_wrap_policy(model)
        units = {
            name: module
            for name, module in model.named_modules()
            if policy(module=module, recurse=False, nonwrapped_numel=0)
        }
        for phase in ("E", "M"):
            model.set_phase(phase)
            unit_param_ids = set()
            for name, module in units.items():
                grads = {p.requires_grad for p in module.parameters()}
                assert len(grads) == 1, f"mixed unit {name} in phase {phase}"
                unit_param_ids |= {id(p) for p in module.parameters()}
            root_rest = {
                p.requires_grad
                for p in model.parameters()
                if id(p) not in unit_param_ids
            }
            assert len(root_rest) == 1, f"mixed root remainder in phase {phase}"
