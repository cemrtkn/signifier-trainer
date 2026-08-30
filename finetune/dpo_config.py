"""DPOConfig extension carrying the trainer's own DPO knobs."""

from dataclasses import dataclass

from trl import DPOConfig


@dataclass
class SignifierDPOConfig(DPOConfig):
    # Pack both directions of each preference pair (dataset rows 2i / 2i+1)
    # into the same optimizer step via a pair-preserving sampler.
    batch_bidirectionals: bool = False
