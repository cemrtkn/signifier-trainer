from typing import List, Optional
from pydantic import BaseModel

class QuantizationConfig(BaseModel):
    do_quantization: bool
    load_in_4bit: bool
    load_in_8bit: bool
    double_quant: bool
    quant_type_4bit: str

class PeftConfig(BaseModel):
    peft_type: str
    task_type: str
    lora_alpha: float
    lora_dropout: float
    r: int
    target_modules: Optional[List[str]] = None 

class FreezeLayerConfig(BaseModel):
    unfrozen_layers: int