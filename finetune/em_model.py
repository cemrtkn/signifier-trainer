from functools import partial
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class EMModel(nn.Module):
    """Wraps a causal LM for EM training: new-token embeddings live in small
    separate tables (built by resize_token_embeddings) so the E and M phases
    can be gated per-module via set_phase."""

    def __init__(
        self,
        base: PreTrainedModel,
        n_new_tokens: int,
        pad_to_multiple_of: int = 512,
    ):
        super().__init__()
        if isinstance(base, PeftModel):
            raise ValueError("EM training is incompatible with LoRA/QLoRA models.")
        if getattr(base, "is_quantized", False):
            raise ValueError("EM training is incompatible with quantized models.")
        if n_new_tokens < 1:
            raise ValueError(f"EM training requires new tokens, got n_new_tokens={n_new_tokens}.")
        self.base = base
        self.n_new_tokens = n_new_tokens
        # Megatron pads the vocab to make_vocab_size_divisible_by (128) × TP:
        # 512 reproduces Qwen's exact 152064 (= 512 × 297), so the merged
        # checkpoint keeps the base release's shape and every TP-4/8 shard
        # stays 128-aligned.
        self.pad_to_multiple_of = pad_to_multiple_of
        # Set at resize as new_num_tokens - n_new_tokens. Padded vocabs (Qwen:
        # 152064 matrix rows vs 151665 tokenizer entries) place new ids below
        # the matrix row count, so the boundary cannot come from the matrix shape.
        self.orig_vocab_size = None
        self.tied = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)

    def resize_token_embeddings(self, new_num_tokens: int):
        if self.tied is not None:
            raise RuntimeError("EMModel.resize_token_embeddings can only be called once.")
        if new_num_tokens <= self.n_new_tokens:
            raise ValueError(
                f"new_num_tokens={new_num_tokens} leaves no original vocab below "
                f"the {self.n_new_tokens} new tokens."
            )

        self.base.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of=self.pad_to_multiple_of
        )
        self.orig_vocab_size = new_num_tokens - self.n_new_tokens
        n_new = self.n_new_tokens
        embed_weight = self.base.get_input_embeddings().weight
        dim = embed_weight.shape[1]
        factory = dict(dtype=embed_weight.dtype, device=embed_weight.device)
        self.tied = self.base.config.tie_word_embeddings

        if self.tied:
            self.new_shared = nn.Embedding(n_new, dim, **factory)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    "WARNING: tie_word_embeddings=True — EM uses a single shared "
                    "new-token table whose rows are pulled both as input conditioners "
                    "and as output logits. EM training is untested on tied models."
                )
        else:
            self.new_embed = nn.Embedding(n_new, dim, **factory)
            self.new_lm_head = nn.Linear(dim, n_new, bias=False, **factory)
        self._init_new_token_tables(dim)

        for module in (self.base.get_input_embeddings(), self.base.get_output_embeddings()):
            for param in module.parameters():
                param.requires_grad = False

        return self.base.get_input_embeddings()

    def _init_new_token_tables(self, dim: int) -> None:
        """Initialise the table rows with transformers' own mean_resizing math
        (Hewitt 2021: multivariate normal with the old rows' mean+covariance).
        The stock resize only applies it when the matrix grows — padded vocabs
        (Qwen) shrink instead and would leave the tables on zeroed padding rows
        (issue #4). The helpers write into the last n rows of the module they
        are handed, which here is the whole table."""
        n_new = self.n_new_tokens
        if self.tied:
            self.base._init_added_embeddings_weights_with_mean(
                self.base.get_input_embeddings(),
                self.new_shared,
                dim,
                self.orig_vocab_size,
                n_new,
            )
        else:
            self.base._init_added_embeddings_weights_with_mean(
                self.base.get_input_embeddings(),
                self.new_embed,
                dim,
                self.orig_vocab_size,
                n_new,
            )
            self.base._init_added_lm_head_weights_with_mean(
                self.base.get_output_embeddings(),
                self.new_lm_head,
                dim,
                self.orig_vocab_size,
                n_new,
            )

    def set_phase(self, phase: Literal["E", "M"]) -> None:
        if self.tied is None:
            raise RuntimeError("EMModel.set_phase called before resize_token_embeddings.")
        if phase not in ("E", "M"):
            raise ValueError(f"Unknown phase {phase!r}; expected 'E' or 'M'.")
        tables = (self.new_shared,) if self.tied else (self.new_embed, self.new_lm_head)
        self.base.requires_grad_(phase == "M")
        for module in tables:
            module.requires_grad_(phase == "E")

    def save_merged(self, output_dir: str) -> None:
        if self.tied is None:
            raise RuntimeError("EMModel.save_merged called before resize_token_embeddings.")
        with torch.no_grad():
            rows = slice(self.orig_vocab_size, self.orig_vocab_size + self.n_new_tokens)
            embed_weight = self.base.get_input_embeddings().weight
            if self.tied:
                embed_weight[rows].copy_(self.new_shared.weight)
            else:
                embed_weight[rows].copy_(self.new_embed.weight)
                lm_head_weight = self.base.get_output_embeddings().weight
                lm_head_weight[rows].copy_(self.new_lm_head.weight)
        self.base.save_pretrained(output_dir)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if self.tied is None:
            raise RuntimeError("EMModel.forward called before resize_token_embeddings.")
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)

        table = self.new_shared if self.tied else self.new_embed
        new_mask = input_ids >= self.orig_vocab_size
        embeds = self.base.get_input_embeddings()(
            input_ids.clamp(max=self.orig_vocab_size - 1)
        )
        embeds[new_mask] = table(input_ids[new_mask] - self.orig_vocab_size)

        outputs = self.base.model(
            inputs_embeds=embeds, attention_mask=attention_mask, **kwargs
        )
        hidden = outputs.last_hidden_state

        orig_logits = self.base.get_output_embeddings()(hidden)[
            ..., : self.orig_vocab_size
        ]
        if self.tied:
            new_logits = nn.functional.linear(hidden, self.new_shared.weight)
        else:
            new_logits = self.new_lm_head(hidden)
        logits = torch.cat([orig_logits, new_logits], dim=-1)

        loss = None
        if labels is not None:
            loss = self.base.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=logits.shape[-1],
                num_items_in_batch=num_items_in_batch,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
        )


def get_em_auto_wrap_policy(model: EMModel):
    """Auto-wrap policy making each new-token table its own FSDP unit next to
    the transformer layers, so every unit stays uniform in requires_grad
    across phase flips."""
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    if model.tied is None:
        raise RuntimeError(
            "get_em_auto_wrap_policy called before resize_token_embeddings."
        )
    tables = (
        {model.new_shared} if model.tied else {model.new_embed, model.new_lm_head}
    )
    layer_names = set(model.base._no_split_modules or [])

    def lambda_fn(module):
        return module in tables or module.__class__.__name__ in layer_names

    return partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
