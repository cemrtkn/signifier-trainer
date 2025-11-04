from collections import defaultdict

import torch
from transformers import (
    PreTrainedModel,
    TrainerCallback,
)
from transformers.generation.utils import GenerateOutput

def generate_different_sequences(
    model: PreTrainedModel,
    context: torch.Tensor,
    sequence_bias_add: int,
    sequence_bias_decay: float,
    generation_args: dict,
    num_generations: int,
) -> list[torch.Tensor]:
    """Generates different completions using an LLM

    Uses sequence bias to ensure generation of different completions.
    All tokens that were generated in previous completions get a negative bias.

    Args:
        model: The LLM model to use for generation
        context: The context to generate completions for
        sequence_bias_add: The amount to add to the sequence bias for each token generated
        sequence_bias_decay: The amount to decay the sequence bias for each generation
        generation_args: The arguments to pass to the model.generate method
        num_generations: The number of completions to generate
    Returns:
        list[torch.Tensor]: A list of generated completions
    """
    gen_sequences = []
    sequence_bias = defaultdict(float)
    for _ in range(num_generations):
        gen = model.generate(context, sequence_bias=sequence_bias or None, **generation_args)
        if isinstance(gen, GenerateOutput):
            gen = gen.sequences
        gen = gen[0][context.shape[-1] :]
        gen_sequences.append(gen)
        for key in sequence_bias:
            sequence_bias[key] *= sequence_bias_decay
        if sequence_bias_add != 0:
            for tok in gen:
                sequence_bias[(tok.item(),)] += sequence_bias_add
    return gen_sequences



class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True
