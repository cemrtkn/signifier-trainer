from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from finetune.dataset import DataCollatorWithPadding, CustomDatasetDict
from finetune.sft_types import TrainingConfig

IGNORE_INDEX = LabelSmoother.ignore_index


def load_dataset_and_collator(
    config: TrainingConfig, tokenizer: PreTrainedTokenizer, test_fold: int = 0
):
    return CustomDatasetDict(
        config.train_dataset_config, tokenizer, test_fold
    ), DataCollatorWithPadding(
        feature_name_to_padding_value={
            "input_ids": tokenizer.pad_token_id,
            "labels": IGNORE_INDEX,
        }
    )
