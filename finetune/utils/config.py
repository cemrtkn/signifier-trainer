import yaml
from transformers import TrainingArguments

from finetune.sft_types import TrainingConfig


def load_config(config: TrainingConfig | str) -> TrainingConfig:
    """Load the configuration from a file if necessary.

    Args:
        config: A TrainingConfig object or a path to a yaml file with TrainingConfig contents.

    Returns:
        TrainingConfig: The TrainingConfig object.
    """
    # Guard clause for when we pass in a TrainingConfig object.
    if isinstance(config, TrainingConfig):
        return config

    # Load the configuration from the given file.
    with open(config) as f:
        config_dict = yaml.safe_load(f)

        # The line below is only needed because we use SkipValidation
        # for the TrainingArguments field.
        config_dict["train_args"] = TrainingArguments(**config_dict["train_args"])

        # Typecast the config_dict to a TrainingConfig object.
        training_config: TrainingConfig = TrainingConfig(**config_dict)

        return training_config

