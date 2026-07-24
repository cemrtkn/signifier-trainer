from typing import Optional

from transformers import Trainer, TrainerCallback

from finetune.sft_types import EMConfig
from finetune.utils.setup_model import print_trainable_parameters


class _PhaseCallback(TrainerCallback):
    """Sets the EMModel phase at each epoch boundary from the trainer's
    training_sequence. state.epoch is a float sitting on the integer at epoch
    begin, so round() is the phase index and stays correct across resume."""

    def __init__(self, trainer: "EMTrainer"):
        self._trainer = trainer

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        phase = self._trainer.phase_for_epoch(epoch)
        model.set_phase(phase)
        if state.is_world_process_zero:
            print(f"[EM] epoch {epoch} -> {phase} phase")
            print_trainable_parameters(model)


class EMTrainer(Trainer):
    """Trainer for EM training: adopts the stock loop and only adds per-epoch
    phase switching driven by em_config.training_sequence. Optimizer param
    groups and the merged save land in later steps of issue #2."""

    def __init__(self, *args, em_config: Optional[EMConfig] = None, **kwargs):
        seq = (em_config.training_sequence if em_config else None) or "em"
        self.training_sequence = seq
        # The sequence length is the epoch count; override whatever
        # num_train_epochs carried (its default or an explicit value) so the
        # schedule and the phase plan can never disagree.
        train_args = kwargs.get("args") if "args" in kwargs else args[1]
        train_args.num_train_epochs = len(seq)

        super().__init__(*args, **kwargs)

        lr = self.args.learning_rate
        self.e_lr = (
            em_config.e_learning_rate
            if em_config is not None and em_config.e_learning_rate is not None
            else lr
        )
        self.m_lr = (
            em_config.m_learning_rate
            if em_config is not None and em_config.m_learning_rate is not None
            else lr
        )
        self.add_callback(_PhaseCallback(self))

    def phase_for_epoch(self, epoch: int) -> str:
        """Phase for a 0-based epoch: the epoch-th character of
        training_sequence ('e'/'m' -> 'E'/'M')."""
        return self.training_sequence[epoch].upper()
