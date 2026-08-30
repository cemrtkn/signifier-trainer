import os

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer, set_seed
from trl import DPOTrainer

from finetune.sft_types import TrainingConfig
from finetune.utils.dataset import load_dpo_dataset
from finetune.utils.setup_model import get_model
from finetune.utils.value_util import EvaluateFirstStepCallback


class PairPreservingSampler(Sampler):
    """Shuffles at pair granularity and emits 2i then 2i+1, so both directions
    of a preference pair stay adjacent in the sample stream."""

    def __init__(self, num_rows: int, seed: int):
        if num_rows % 2:
            raise ValueError(
                f"batch_bidirectionals needs an even row count (pairs at rows "
                f"2i/2i+1); the train dataset has {num_rows} rows."
            )
        self.num_rows = num_rows
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        for pair in torch.randperm(self.num_rows // 2, generator=self.generator):
            yield 2 * int(pair)
            yield 2 * int(pair) + 1

    def __len__(self):
        return self.num_rows


class PairedDPOTrainer(DPOTrainer):
    """DPOTrainer whose train sampler keeps the two directions of each pair
    adjacent, landing them in the same optimizer step."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return PairPreservingSampler(len(dataset), self.args.seed)


def check_signifier_tokens(tokenizer, signifier_config) -> None:
    """Fail fast when the checkpoint's tokenizer splits any signifier token —
    DPO expects a merged token checkpoint that already carries them."""
    if signifier_config.mode != "token_signifier":
        return
    missing = [
        tok
        for tok in signifier_config.new_special_tokens
        if len(tokenizer(tok, add_special_tokens=False)["input_ids"]) != 1
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} signifier token(s) are not single tokens in the "
            f"model's tokenizer, e.g. {missing[:3]}. mode: dpo trains a merged "
            "token checkpoint; point `model` at one."
        )


def run_dpo(config: TrainingConfig):
    """DPO-train a merged checkpoint using the given configuration (mode: dpo)."""
    set_seed(config.train_args.seed)

    saving_dir = config.train_args.output_dir
    config.train_args.output_dir = (
        os.path.join(config.ptmp_dir, saving_dir) if config.ptmp_dir else saving_dir
    )
    os.makedirs(config.train_args.output_dir, exist_ok=True)

    print("=" * 8, "Load Policy Model.", "=" * 8)
    model = get_model(config)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        print("Setting pad token to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    check_signifier_tokens(
        tokenizer, config.train_dataset_config.resolve_signifier_config()
    )

    print("=" * 8, "Prepare Dataset.", "=" * 8)
    datasetdict = load_dpo_dataset(
        config, tokenizer, test_fold=config.train_dataset_config.test_fold or 0
    )

    if config.batch_bidirectionals:
        effective_batch = (
            config.train_args.per_device_train_batch_size
            * config.train_args.world_size
            * config.train_args.gradient_accumulation_steps
        )
        if effective_batch % 2:
            raise ValueError(
                "batch_bidirectionals needs an even effective batch; got "
                f"per_device x world_size x grad_accum = {effective_batch}."
            )

    print("=" * 8, "Start Training.", "=" * 8)

    config.train_args.eval_strategy = (
        "no" if "test" not in datasetdict else config.train_args.eval_strategy
    )

    trainer_cls = PairedDPOTrainer if config.batch_bidirectionals else DPOTrainer
    trainer = trainer_cls(
        model=model,
        ref_model=None,
        args=config.train_args,
        train_dataset=datasetdict["train"],
        eval_dataset=datasetdict["test"] if "test" in datasetdict else None,
        processing_class=tokenizer,
    )

    checkpoint = None
    if config.train_args.resume_from_checkpoint is not None:
        checkpoint = config.train_args.resume_from_checkpoint
        print(f"Resuming from checkpoint: {checkpoint}")

    if config.train_args.logging_first_step and "test" in datasetdict:
        trainer.add_callback(EvaluateFirstStepCallback())

    (
        trainer.train(resume_from_checkpoint=checkpoint)
        if checkpoint is not None
        else trainer.train()
    )

    if dist.is_initialized():
        print(f"[Rank {dist.get_rank()} Reaching pre-save barrier...")
        dist.barrier()

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Saving tokenizer to {config.train_args.output_dir}")
        tokenizer.save_pretrained(config.train_args.output_dir)

    if trainer.is_fsdp_enabled:
        print(
            f"[Rank {dist.get_rank()}] Setting FSDP state_dict_type to FULL_STATE_DICT..."
        )
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if dist.is_initialized():
        print(f"[Rank {dist.get_rank()} Reaching save barrier...")
        dist.barrier()

    trainer.save_model()
    print("save_model() call completed.")

    # Keep non-zero ranks alive until rank 0 finishes the state-dict
    # gather — early exit tears down NCCL and deadlocks rank 0 at scale.
    if dist.is_initialized():
        dist.barrier()
        print(f"[Rank {dist.get_rank()}] Passed post-save barrier.")
