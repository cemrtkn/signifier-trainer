"""Unit tests for the EMModel wrapper (tiny untied and tied checkpoints)."""

import json
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from finetune.em_model import EMModel, get_em_auto_wrap_policy

ORIG_VOCAB = 64
NEW_VOCAB = 68
N_NEW = NEW_VOCAB - ORIG_VOCAB


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
    # pad_to_multiple_of=None: the production default of 128 would round the
    # tiny 68-row matrix up to 128; the padded path is covered in TestPaddedVocab.
    model = EMModel(tiny_model(tie), N_NEW, pad_to_multiple_of=None)
    model.resize_token_embeddings(NEW_VOCAB)
    return model.eval()


def assert_healthy_init(table_weight, used_weight):
    """Mean-resizing init: rows near the used-row mean vector, not zero and
    not blown up."""
    mean_norm = used_weight.mean(dim=0).norm()
    row_norms = table_weight.norm(dim=-1)
    assert (row_norms > 0.5 * mean_norm).all()
    assert (row_norms < 2.0 * mean_norm).all()


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
            EMModel(base, N_NEW)

    def test_rejects_peft(self):
        from peft import PeftModel

        with pytest.raises(ValueError, match="LoRA"):
            EMModel(PeftModel.__new__(PeftModel), N_NEW)

    def test_rejects_zero_new_tokens(self):
        with pytest.raises(ValueError, match="n_new_tokens"):
            EMModel(tiny_model(False), 0)

    def test_methods_require_resize_first(self):
        model = EMModel(tiny_model(False), N_NEW)
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
            assert_healthy_init(table.weight, embed_weight[:ORIG_VOCAB])

    def test_guards(self, tie):
        model = resized_em_model(tie)
        with pytest.raises(RuntimeError, match="once"):
            model.resize_token_embeddings(NEW_VOCAB + 4)
        with pytest.raises(ValueError, match="original vocab"):
            EMModel(tiny_model(tie), N_NEW).resize_token_embeddings(N_NEW)

    def test_plain_model_resize_is_stock(self, tie):
        model = tiny_model(tie)
        model.resize_token_embeddings(NEW_VOCAB)
        assert model.get_input_embeddings().weight.shape[0] == NEW_VOCAB
        assert model.get_input_embeddings().weight.requires_grad


@pytest.mark.parametrize("tie", [False, True])
class TestForward:
    def test_original_vocab_logits_match_resized_base(self, tie):
        # The tables are freshly initialised, so full-tensor equivalence with
        # the unwrapped base no longer holds (that invariant lives in the
        # merged-save round trip) — but on batches without new tokens the
        # original-vocab columns must be identical.
        model = resized_em_model(tie)
        torch.manual_seed(2)
        old_ids = torch.randint(0, ORIG_VOCAB, (2, 10))
        with torch.no_grad():
            out = model(input_ids=old_ids, labels=old_ids.clone())
            ref = model.base(input_ids=old_ids, labels=old_ids.clone())
        assert out.logits.shape == (2, 10, NEW_VOCAB)
        assert torch.allclose(
            out.logits[..., :ORIG_VOCAB], ref.logits[..., :ORIG_VOCAB], atol=1e-6
        )
        assert torch.isfinite(out.loss)
        with torch.no_grad():
            out_new = model(input_ids=batch_with_new_ids())
        assert torch.isfinite(out_new.logits).all()

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


class TestPaddedVocab:
    """Qwen-style checkpoints pad the embedding matrix beyond the tokenizer
    (152064 rows vs 151665 entries): new token ids land inside the padding,
    the padding rows ship zeroed, and pad_to_multiple_of keeps the matrix
    at its aligned shape instead of shrinking to the tokenizer length."""

    def qwen_mimic(self):
        base = tiny_model(False)   # matrix 64, tokenizer had 56
        with torch.no_grad():
            base.get_input_embeddings().weight[56:] = 0.0
        model = EMModel(base, 6, pad_to_multiple_of=64)
        model.resize_token_embeddings(62)
        return model.eval()

    def test_matrix_shape_and_init(self):
        model = self.qwen_mimic()
        embed_weight = model.base.get_input_embeddings().weight
        assert model.orig_vocab_size == 56
        assert embed_weight.shape[0] == 64          # kept aligned, no shrink
        assert model.base.config.vocab_size == 64
        assert model.new_embed.weight.shape == (6, 16)
        # zeroed padding never becomes the init
        assert_healthy_init(model.new_embed.weight, embed_weight[:56])
        assert_healthy_init(model.new_lm_head.weight, model.base.get_output_embeddings().weight[:56])
        # untouched alignment rows stay as shipped
        assert (embed_weight[62:] == 0).all()

    def test_forward_and_merged_round_trip(self, tmp_path):
        model = self.qwen_mimic()
        torch.manual_seed(1)
        ids = torch.randint(0, 56, (2, 10))
        ids[:, 3] = 56
        ids[:, 7] = 61
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.shape == (2, 10, 62)      # boundary + n_new, not matrix width
        assert torch.isfinite(out.loss)

        model.save_merged(str(tmp_path))
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
        assert reloaded.config.vocab_size == 64     # shape-identical to the base release
        with torch.no_grad():
            full = reloaded(input_ids=ids).logits
        assert torch.allclose(out.logits, full[..., :62], atol=1e-5)
        merged_embed = reloaded.get_input_embeddings().weight
        assert torch.equal(merged_embed[56:62], model.new_embed.weight)
        assert (merged_embed[62:] == 0).all()

    def test_growth_past_headroom_stays_aligned(self):
        model = EMModel(tiny_model(False), 12, pad_to_multiple_of=64)
        model.resize_token_embeddings(68)           # 56 used + 12 new > 64 rows
        assert model.base.get_input_embeddings().weight.shape[0] == 128
        assert model.orig_vocab_size == 56
        assert model.new_embed.weight.shape[0] == 12


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
