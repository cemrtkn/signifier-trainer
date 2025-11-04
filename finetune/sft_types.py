from typing import Any, Optional, List

from pydantic import BaseModel, Field, PydanticUndefinedAnnotation

from finetune.dataset import DatasetConfig
from finetune.training_mode import FreezeLayerConfig, PeftConfig, QuantizationConfig

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
    output_dir_root: Optional[str] = None
    run_profiler: bool = False
    use_flash_attention: Optional[bool] = True
    resume_from_checkpoint: Optional[str] = None



try:
    TrainingConfig.model_rebuild()
except PydanticUndefinedAnnotation as exc_info:
    assert exc_info.code == "undefined-annotation"
