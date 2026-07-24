import torch.distributed as dist
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel


class EMModel(nn.Module):
    """Wraps a causal LM for EM training: new-token embeddings live in small
    separate tables (built by resize_token_embeddings) so the E and M phases
    can be gated per-module via set_phase."""

    def __init__(self, base: PreTrainedModel):
        super().__init__()
        if isinstance(base, PeftModel):
            raise ValueError("EM training is incompatible with LoRA/QLoRA models.")
        if getattr(base, "is_quantized", False):
            raise ValueError("EM training is incompatible with quantized models.")
        self.base = base
        self.orig_vocab_size = base.get_input_embeddings().weight.shape[0]
        self.tied = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)

    def resize_token_embeddings(self, new_num_tokens: int):
        if self.tied is not None:
            raise RuntimeError("EMModel.resize_token_embeddings can only be called once.")
        if new_num_tokens <= self.orig_vocab_size:
            raise ValueError(
                f"EM training requires new tokens: new_num_tokens={new_num_tokens} "
                f"<= original vocab size {self.orig_vocab_size}."
            )

        self.base.resize_token_embeddings(new_num_tokens)
        n_new = new_num_tokens - self.orig_vocab_size
        embed_weight = self.base.get_input_embeddings().weight
        dim = embed_weight.shape[1]
        factory = dict(dtype=embed_weight.dtype, device=embed_weight.device)
        self.tied = self.base.config.tie_word_embeddings

        if self.tied:
            self.new_shared = nn.Embedding(n_new, dim, **factory)
            self.new_shared.weight.data.copy_(embed_weight.data[self.orig_vocab_size:])
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    "WARNING: tie_word_embeddings=True — EM uses a single shared "
                    "new-token table whose rows are pulled both as input conditioners "
                    "and as output logits. EM training is untested on tied models."
                )
        else:
            lm_head_weight = self.base.get_output_embeddings().weight
            self.new_embed = nn.Embedding(n_new, dim, **factory)
            self.new_embed.weight.data.copy_(embed_weight.data[self.orig_vocab_size:])
            self.new_lm_head = nn.Linear(dim, n_new, bias=False, **factory)
            self.new_lm_head.weight.data.copy_(lm_head_weight.data[self.orig_vocab_size:])

        for module in (self.base.get_input_embeddings(), self.base.get_output_embeddings()):
            for param in module.parameters():
                param.requires_grad = False

        return self.base.get_input_embeddings()
