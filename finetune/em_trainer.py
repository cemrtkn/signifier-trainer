from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer, TrainerCallback, get_scheduler

from finetune.sft_types import EMConfig
from finetune.utils.setup_model import print_trainable_parameters


@dataclass(frozen=True)
class PhaseStepMap:
    """Maps the global optimizer-step axis onto each phase's own timeline.

    Built once from training_sequence + num_training_steps; a pure function of
    the global step thereafter (no mutable state), so resume-from-checkpoint
    reproduces every LR for free. Epoch i spans global steps
    [bounds[i], bounds[i + 1]) and runs phase sequence[i]. A phase's own
    timeline advances only on its own steps and holds flat while the other
    phase runs — this is what lets a phase's schedule skip the intervening
    opposite phase and resume where it left off.
    """

    sequence: str  # phases per epoch, e.g. "EMEM"
    bounds: tuple  # len = num_epochs + 1; bounds[0] = 0, bounds[-1] = total
    totals: dict  # own step count per phase, {"E": int, "M": int}

    def phase_of(self, step: int) -> str:
        """Phase of the epoch that global `step` falls in (last phase for any
        step at/beyond the final boundary)."""
        for i in range(len(self.sequence)):
            if step < self.bounds[i + 1]:
                return self.sequence[i]
        return self.sequence[-1]

    def own_elapsed(self, phase: str, step: int) -> int:
        """Number of `phase` steps strictly before global `step` — the 0-based
        position along `phase`'s own timeline. Increments only within `phase`'s
        epochs and stays flat across the other phase's steps."""
        n = 0
        for i, ph in enumerate(self.sequence):
            if ph != phase:
                continue
            lo, hi = self.bounds[i], self.bounds[i + 1]
            n += max(0, min(hi, step) - lo)
        return n


def build_phase_step_map(
    training_sequence: str, num_training_steps: int
) -> PhaseStepMap:
    """Precompute the epoch->global-step boundaries and per-phase totals for a
    training_sequence. Boundaries match create_optimizer/scheduler's existing
    even split: epoch i spans [round(i·N/E), round((i+1)·N/E))."""
    seq = training_sequence.upper()
    num_epochs = len(seq)
    bounds = tuple(
        round(i * num_training_steps / num_epochs) for i in range(num_epochs + 1)
    )
    totals = {
        ph: sum(bounds[i + 1] - bounds[i] for i, p in enumerate(seq) if p == ph)
        for ph in ("E", "M")
    }
    return PhaseStepMap(sequence=seq, bounds=bounds, totals=totals)


class _PhaseCallback(TrainerCallback):
    """Sets the EMModel phase at each epoch boundary from the trainer's
    training_sequence. state.epoch is a float sitting on the integer at epoch
    begin, so round() is the phase index and stays correct across resume."""

    def __init__(self, trainer: "EMTrainer"):
        self._trainer = trainer

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        phase = self._trainer.phase_for_epoch(epoch)
        self._trainer._current_phase = phase
        model.set_phase(phase)
        if state.is_world_process_zero:
            print(f"[EM] epoch {epoch} -> {phase} phase")
            print_trainable_parameters(model)


class _StepPhaseCallback(TrainerCallback):
    """Sets the EMModel phase from step windows over the whole run, for
    em_config.phase_unit == "steps". The map is built at train begin, the first
    point where state.max_steps holds the Trainer's resolved total step count
    (from num_train_epochs or max_steps); the phase is a pure function of
    state.global_step thereafter, so resume needs no extra state."""

    def __init__(self, trainer: "EMTrainer"):
        self._trainer = trainer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        step_map = build_phase_step_map(
            self._trainer.training_sequence, state.max_steps
        )
        self._trainer._step_map = step_map
        phase = step_map.phase_of(state.global_step)
        self._trainer._current_phase = phase
        model.set_phase(phase)
        if state.is_world_process_zero:
            print(f"[EM] step-wise phases over {state.max_steps} optimizer steps:")
            for i, ph in enumerate(step_map.sequence):
                lo, hi = step_map.bounds[i], step_map.bounds[i + 1]
                print(f"[EM]   phase {i}: {ph}, steps [{lo}, {hi}), {hi - lo} steps")
            totals = ", ".join(f"{ph}={step_map.totals[ph]}" for ph in ("E", "M"))
            print(f"[EM] total steps per phase: {totals}")
            print(f"[EM] step {state.global_step} -> {phase} phase")
            print_trainable_parameters(model)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        phase = self._trainer._step_map.phase_of(state.global_step)
        if phase == self._trainer._current_phase:
            return
        self._trainer._current_phase = phase
        model.set_phase(phase)
        if state.is_world_process_zero:
            print(f"[EM] step {state.global_step} -> {phase} phase")
            print_trainable_parameters(model)


class EMTrainer(Trainer):
    """Stock Trainer plus E/M phase switching (from em_config.training_sequence,
    one phase per epoch or per step window depending on em_config.phase_unit),
    phase-scaled optimizer param groups, and a merged final save."""

    def __init__(self, *args, em_config: Optional[EMConfig] = None, **kwargs):
        seq = (em_config.training_sequence if em_config else None) or "em"
        self.training_sequence = seq
        self.phase_unit = (em_config.phase_unit if em_config else None) or "epoch"
        self._current_phase = seq[0].upper()
        self._step_map = None
        if self.phase_unit == "epoch":
            # One phase per epoch: sequence length is the epoch count and
            # overrides num_train_epochs. In steps mode the general config owns
            # the run length and the sequence only splits it.
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
        self.add_callback(
            _PhaseCallback(self)
            if self.phase_unit == "epoch"
            else _StepPhaseCallback(self)
        )

    def phase_for_epoch(self, epoch: int) -> str:
        """Phase for a 0-based epoch: the epoch-th character of
        training_sequence ('e'/'m' -> 'E'/'M')."""
        return self.training_sequence[epoch].upper()

    def log(self, logs, *args, **kwargs):
        """Trainer logs only param-group 0's lr as 'learning_rate', and group 0
        is the e_lr table group — so the stock field reports e_lr in both
        phases (and wandb, reading the same dict, would too). Expose both group
        scales as lr_e / lr_m and repoint 'learning_rate' at the active phase."""
        if self.optimizer is not None and len(self.optimizer.param_groups) >= 2:
            logs["lr_e"] = self.optimizer.param_groups[0]["lr"]
            logs["lr_m"] = self.optimizer.param_groups[1]["lr"]
            if "learning_rate" in logs:
                logs["learning_rate"] = (
                    logs["lr_e"] if self._current_phase == "E" else logs["lr_m"]
                )
        return super().log(logs, *args, **kwargs)

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
                opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, model
                )
            opt_kwargs = {
                k: v
                for k, v in opt_kwargs.items()
                if k not in ("lr", "weight_decay", "params")
            }
            self.optimizer = opt_cls(groups, **opt_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Per-phase LR schedules with memory. Each phase (E, M) gets its own
        warmup->decay over its *own* total steps, evaluated through
        PhaseStepMap so the factor advances only on that phase's steps and
        holds flat across the other — a later same-phase epoch continues the
        schedule rather than re-warming. One lambda per param group in
        create_optimizer's order (group 0 = table(s) at e_lr, groups 1-2 = M
        surface at m_lr). Pure in the global step, so resume reproduces the
        LR for free."""
        if self.lr_scheduler is not None:
            return super().create_scheduler(num_training_steps, optimizer)
        optimizer = optimizer if optimizer is not None else self.optimizer
        step_map = build_phase_step_map(self.training_sequence, num_training_steps)

        # One warmup->decay factor per phase over that phase's own step count,
        # built on a throwaway optimizer so construction never touches the real
        # param-group LRs; we keep only the step->factor lambda (base-LR-free).
        phase_lambdas = {}
        for ph in ("E", "M"):
            phase_len = max(1, step_map.totals[ph])
            dummy = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
            sub = get_scheduler(
                self.args.lr_scheduler_type,
                dummy,
                num_warmup_steps=self.args.get_warmup_steps(phase_len),
                num_training_steps=phase_len,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            if not hasattr(sub, "lr_lambdas"):
                raise ValueError(
                    f"EM per-phase scheduling needs a LambdaLR-family "
                    f"lr_scheduler_type; '{self.args.lr_scheduler_type}' is not one."
                )
            phase_lambdas[ph] = sub.lr_lambdas[0]

        def phase_factor(phase):
            fn = phase_lambdas[phase]
            return lambda step: fn(step_map.own_elapsed(phase, step))

        # Group order mirrors create_optimizer: [E table, M decay, M no-decay].
        group_lambdas = [phase_factor("E"), phase_factor("M"), phase_factor("M")]
        self.lr_scheduler = LambdaLR(optimizer, group_lambdas)
        self._created_lr_scheduler = True
        return self.lr_scheduler

    def save_model(self, output_dir=None, _internal_call=False):
        """Write the final checkpoint as a vanilla merged HF model rather than
        the wrapper's state_dict. Under FSDP the base matrices are sharded, so
        summon the full params (collective on all ranks; materialised on rank 0
        only, the one rank that scatters the tables in and writes). Mid-training
        save_steps checkpoints (_internal_call) keep the stock wrapper-shaped
        path so resume still works."""
        if _internal_call:
            return super().save_model(output_dir, _internal_call=_internal_call)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.is_fsdp_enabled:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(
                self.model_wrapped,
                writeback=False,
                rank0_only=True,
                offload_to_cpu=True,
            ):
                if self.args.should_save:
                    self.model.save_merged(output_dir)
        elif self.args.should_save:
            self.model.save_merged(output_dir)
