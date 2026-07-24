from typing import Any, Optional, List

from pydantic import BaseModel, Field, PydanticUndefinedAnnotation, model_validator

from finetune.dataset import DatasetConfig
from finetune.training_mode import FreezeLayerConfig, PeftConfig, QuantizationConfig


class EMConfig(BaseModel):
    status: bool = False
    e_learning_rate: Optional[float] = None
    m_learning_rate: Optional[float] = None


class TrainingConfig(BaseModel):
    model: str = Field(..., description="The name of the model to use")
    ptmp_dir: Optional[str] = Field(
        None,
        description="The path to the PTMP directory to save the model weights under ptmp_dir + train_args.output_dir.",
    )
    train_args: Any
    train_dataset_config: DatasetConfig
    partial_fine_tuning: Optional[FreezeLayerConfig] = None
    peft_config: Optional[PeftConfig] = None
    quantization: Optional[QuantizationConfig] = None
    em_config: Optional[EMConfig] = None
    output_dir_root: Optional[str] = None
    run_profiler: bool = False
    use_flash_attention: Optional[bool] = True
    resume_from_checkpoint: Optional[str] = None

    @model_validator(mode="after")
    def _check_em_config(self) -> "TrainingConfig":
        em = self.em_config
        if em is None or not em.status:
            if em is not None and (
                em.e_learning_rate is not None or em.m_learning_rate is not None
            ):
                raise ValueError(
                    "em_config.e_learning_rate / m_learning_rate are set but "
                    "em_config.status is false; enable status or drop the rates "
                    "(they would otherwise be silently ignored)."
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



try:
    TrainingConfig.model_rebuild()
except PydanticUndefinedAnnotation as exc_info:
    assert exc_info.code == "undefined-annotation"
