import re
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    PydanticUndefinedAnnotation,
    field_validator,
    model_validator,
)

from finetune.dataset import DatasetConfig
from finetune.training_mode import FreezeLayerConfig, PeftConfig, QuantizationConfig


class EMConfig(BaseModel):
    status: bool = False
    # phase per epoch (e first); None -> "em". Length overrides num_train_epochs.
    training_sequence: Optional[str] = None
    e_learning_rate: Optional[float] = None
    m_learning_rate: Optional[float] = None
    # boundary the phase flips on: one phase per epoch, or per step window.
    phase_unit: Literal["epoch", "steps"] = "epoch"

    @field_validator("training_sequence")
    @classmethod
    def _check_training_sequence(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.fullmatch(r"[em]+", v):
            raise ValueError(
                "training_sequence must be a non-empty string of 'e'/'m' "
                "characters, e.g. 'em', 'emem', 'meme'."
            )
        return v


class TrainingConfig(BaseModel):
    model: str = Field(..., description="The name of the model to use")
    ptmp_dir: Optional[str] = Field(
        None,
        description="The path to the PTMP directory to save the model weights under ptmp_dir + train_args.output_dir.",
    )
    mode: Literal["sft", "dpo"] = "sft"
    train_args: Any
    train_dataset_config: DatasetConfig
    partial_fine_tuning: Optional[FreezeLayerConfig] = None
    peft_config: Optional[PeftConfig] = None
    quantization: Optional[QuantizationConfig] = None
    em_config: Optional[EMConfig] = None
    embedding_lr: Optional[float] = None
    model_lr: Optional[float] = None
    output_dir_root: Optional[str] = None
    run_profiler: bool = False
    use_flash_attention: Optional[bool] = True
    resume_from_checkpoint: Optional[str] = None

    @model_validator(mode="after")
    def _check_em_config(self) -> "TrainingConfig":
        em = self.em_config
        if em is None or not em.status:
            if em is not None and (
                em.e_learning_rate is not None
                or em.m_learning_rate is not None
                or em.training_sequence is not None
                or em.phase_unit != "epoch"
            ):
                raise ValueError(
                    "em_config.e_learning_rate / m_learning_rate / "
                    "training_sequence / phase_unit are set but em_config.status "
                    "is false; enable status or drop them (they would otherwise "
                    "be silently ignored)."
                )
            return self
        if (
            self.peft_config is not None
            or self.quantization is not None
            or self.partial_fine_tuning is not None
        ):
            raise ValueError(
                "EM training (em_config.status: true) excludes LoRA/QLoRA "
                "(peft_config), quantization, and partial_fine_tuning — unset "
                "them or disable EM."
            )
        signifier = self.train_dataset_config.resolve_signifier_config()
        if signifier.mode != "token_signifier":
            raise ValueError(
                "EM training requires train_dataset_config signifier mode "
                f"'token_signifier' (got '{signifier.mode}'): there must be new "
                "tokens to E-step on."
            )
        return self

    @model_validator(mode="after")
    def _check_dpo(self) -> "TrainingConfig":
        if self.mode != "dpo":
            return self
        if self.em_config is not None and self.em_config.status:
            raise ValueError(
                "mode: dpo excludes EM training for now (tracked in issue #14) — "
                "DPO runs on a merged token checkpoint whose signifiers are "
                "ordinary vocab rows; drop em_config or use mode: sft."
            )
        if self.embedding_lr is not None or self.model_lr is not None:
            raise ValueError(
                "mode: dpo is uniform-lr full-FT — drop embedding_lr / model_lr."
            )
        if (
            self.peft_config is not None
            or self.quantization is not None
            or self.partial_fine_tuning is not None
        ):
            raise ValueError(
                "mode: dpo is plain full-FT only — unset peft_config, "
                "quantization, and partial_fine_tuning."
            )
        from transformers import TrainingArguments

        from trl import DPOConfig

        if isinstance(self.train_args, TrainingArguments) and not isinstance(
            self.train_args, DPOConfig
        ):
            raise ValueError(
                "mode: dpo needs train_args validated as trl's DPOConfig "
                "(utils.config.load_config does this from YAML); got plain "
                "TrainingArguments."
            )
        return self

    @model_validator(mode="after")
    def _check_dual_lr(self) -> "TrainingConfig":
        if self.embedding_lr is None and self.model_lr is None:
            return self
        if self.embedding_lr is None or self.model_lr is None:
            raise ValueError(
                "dual-lr SFT needs both embedding_lr and model_lr — set both, or "
                "neither (a lone rate would leave the other half of the model on "
                "train_args.learning_rate, which is not what you meant)."
            )
        if self.em_config is not None and self.em_config.status:
            raise ValueError(
                "embedding_lr / model_lr excludes EM training "
                "(em_config.status: true), which carries its own per-phase rates "
                "in em_config.e_learning_rate / m_learning_rate — drop one or the "
                "other."
            )
        if (
            self.peft_config is not None
            or self.quantization is not None
            or self.partial_fine_tuning is not None
        ):
            raise ValueError(
                "dual-lr SFT (embedding_lr / model_lr) is full-FT only and "
                "excludes LoRA/QLoRA (peft_config), quantization, and "
                "partial_fine_tuning — unset them or drop the two rates."
            )
        return self


try:
    TrainingConfig.model_rebuild()
except PydanticUndefinedAnnotation as exc_info:
    assert exc_info.code == "undefined-annotation"
