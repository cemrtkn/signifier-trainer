"""Unit tests for sft.resize_for_new_tokens (padded-vocab preservation)."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from finetune.em_model import EMModel
from finetune.sft import resize_for_new_tokens

MATRIX_ROWS = 80  # padded matrix, only USED_VOCAB rows are real tokens
USED_VOCAB = 64


def tiny_padded_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=MATRIX_ROWS,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config)


class TestPlainSft:
    def test_new_tokens_inside_padding_keep_shape(self):
        model = tiny_padded_model()
        before_embed = model.get_input_embeddings().weight.detach().clone()
        before_head = model.get_output_embeddings().weight.detach().clone()

        resize_for_new_tokens(model, USED_VOCAB + 6, em_on=False)

        assert model.get_input_embeddings().weight.shape[0] == MATRIX_ROWS
        assert model.get_output_embeddings().weight.shape[0] == MATRIX_ROWS
        assert model.config.vocab_size == MATRIX_ROWS
        assert torch.equal(model.get_input_embeddings().weight, before_embed)
        assert torch.equal(model.get_output_embeddings().weight, before_head)

    def test_new_tokens_beyond_rows_grow(self):
        model = tiny_padded_model()
        resize_for_new_tokens(model, MATRIX_ROWS + 6, em_on=False)
        assert model.get_input_embeddings().weight.shape[0] == MATRIX_ROWS + 6
        assert model.get_output_embeddings().weight.shape[0] == MATRIX_ROWS + 6


class TestEmDelegation:
    def test_em_model_always_resizes(self):
        # EMModel requires the resize call (it builds the E-phase tables);
        # its own pad_to_multiple_of restores the padded shape.
        model = EMModel(tiny_padded_model(), 6, pad_to_multiple_of=MATRIX_ROWS)
        resize_for_new_tokens(model, USED_VOCAB + 6, em_on=True)
        assert model.tied is not None  # resize_token_embeddings ran
        assert model.new_embed.weight.shape == (6, 16)
        assert model.base.get_input_embeddings().weight.shape[0] == MATRIX_ROWS
