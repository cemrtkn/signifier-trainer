from itertools import islice

import bitsandbytes as bnb
import torch
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from finetune.sft_types import TrainingConfig

def get_model(config: TrainingConfig):
    if config.peft_config is not None:
        if config.quantization is not None:
            model = get_qlora_model(config)
        else:
            model = get_lora_model(config)
    elif config.partial_fine_tuning is not None:
        model = get_partial_froozen_model(config)
    else:
        model = get_full_trainable_model(config)

    return model


def _get_model(config: TrainingConfig):
    if config.use_flash_attention:
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
    return model


def find_all_linear_names(bits, model):
    # Source https://github.com/artidoro/qlora/blob/main/qlora.py#L248
    cls = (
        bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def get_lora_model(config):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    """
    model = _get_model(config)

    if config.peft_config.target_modules is None:
        config.peft_config.target_modules = find_all_linear_names(bits=32, model=model)

    lora_config = LoraConfig(
        r=config.peft_config.r,
        lora_alpha=config.peft_config.lora_alpha,
        target_modules=config.peft_config.target_modules,
        lora_dropout=config.peft_config.lora_dropout,
        bias="none",
        task_type=config.peft_config.task_type,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    print("low-rank approximation technique is activated")
    print_trainable_parameters(model)

    return model


def get_qlora_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        load_in_4bit=config.quantization.load_in_4bit,
        load_in_8bit=config.quantization.load_in_8bit,
        max_memory={},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=config.quantization.load_in_4bit,
            load_in_8bit=config.quantization.load_in_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_quant_type=config.quantization.quant_type_4bit,
        ),
    )

    if config.peft_config.target_modules is None:
        config.peft_config.target_modules = find_all_linear_names(
            bits=(
                4
                if config.quantization.load_in_4bit
                else (8 if config.quantization.load_in_8bit else 32)
            ),
            model=model,
        )

    lora_config = LoraConfig(
        r=config.peft_config.r,
        lora_alpha=config.peft_config.lora_alpha,
        target_modules=config.peft_config.target_modules,
        lora_dropout=config.peft_config.lora_dropout,
        bias="none",
        task_type=config.peft_config.task_type,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.train_args.gradient_checkpointing
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    print("low-rank approximation with quantization technique is activated")
    print_trainable_parameters(model)

    return model


def get_partial_froozen_model(config):
    model = AutoModelForCausalLM.from_pretrained(config.model)
    num_params = len(list(model.named_parameters()))

    if num_params < config.partial_fine_tuning.unfrozen_layers:
        raise TypeError(
            "The given unfrozen_layers are more than model layer. Please decrease the number of unfrozen_layers"
        )

    for name, param in islice(
        list(model.named_parameters()), num_params - config.partial_fine_tuning.unfrozen_layers
    ):
        param.requires_grad = False
        if torch.distributed.get_rank() == 0:
            print(f"The layer {name} has been frozen.")

    return model


def get_full_trainable_model(config):
    model = _get_model(config)

    print("Full Fine-tuning is activated")
    print_trainable_parameters(model)

    return model
