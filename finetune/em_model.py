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

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)
