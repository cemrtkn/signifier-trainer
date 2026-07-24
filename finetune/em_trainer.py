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

    def create_optimizer(self):
        """One optimizer over *all* params (no requires_grad filter) so a
        param's group survives phase flips: the new-token table(s) at e_lr,
        everything else at m_lr split into decay / no-decay like the stock
        optimizer. Membership is by id(p) against the table modules, the same
        no-name-matching rule as set_phase. Which group actually steps is
        gated at runtime by requires_grad (frozen params get no grad, and the
        optimizer skips params whose grad is None)."""
        if self.optimizer is None:
            model = self.model
            tables = (
                (model.new_shared,)
                if model.tied
                else (model.new_embed, model.new_lm_head)
            )
            table_ids = {id(p) for m in tables for p in m.parameters()}
            decay = set(self.get_decay_parameter_names(model))
            wd = self.args.weight_decay
            groups = [
                {
                    "params": [p for p in model.parameters() if id(p) in table_ids],
                    "lr": self.e_lr,
                    "weight_decay": wd,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if id(p) not in table_ids and n in decay
                    ],
                    "lr": self.m_lr,
                    "weight_decay": wd,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if id(p) not in table_ids and n not in decay
                    ],
                    "lr": self.m_lr,
                    "weight_decay": 0.0,
                },
            ]
            if self.optimizer_cls_and_kwargs is not None:
                opt_cls, opt_kwargs = self.optimizer_cls_and_kwargs
            else:
                opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(self.args, model)
            opt_kwargs = {
                k: v
                for k, v in opt_kwargs.items()
                if k not in ("lr", "weight_decay", "params")
            }
            self.optimizer = opt_cls(groups, **opt_kwargs)
        return self.optimizer
