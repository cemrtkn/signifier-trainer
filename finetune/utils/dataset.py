from datasets import DatasetDict
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from finetune.dataset import (
    CrossvalDatasetDict,
    CustomDatasetDict,
    DataCollatorWithPadding,
    validate_signifiers,
)
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


def load_dpo_dataset(
    config: TrainingConfig, tokenizer: PreTrainedTokenizer, test_fold: int = 0
) -> DatasetDict:
    """DPO twin of load_dataset_and_collator: same fold merge and signifier
    validation, but rows stay text. The prompt is the joined non-answer parser
    fields; chosen/rejected come from the answer field with the trailing eos
    stripped, because DPOTrainer.tokenize_row appends eos_token_id itself."""
    dataset_config = config.train_dataset_config
    raw = CrossvalDatasetDict.load_from_disk(dataset_config.data_path, test_fold)

    signifier_values = set()
    for split in raw.values():
        signifier_values.update(split.unique("signifiers"))
    validate_signifiers(
        signifier_values,
        dataset_config.resolve_signifier_config(),
        dataset_config.parser_config,
    )

    fields = dataset_config.parser_config.fields
    keys = list(fields)
    if keys[-1] != "answer":
        raise ValueError(
            "dpo mode expects the parser's 'answer' field last so the prompt "
            f"is the joined prefix; got field order {keys}."
        )
    eos = tokenizer.eos_token or ""

    def to_completion(answer: str) -> str:
        text = "\n" + fields["answer"]["text"].format(answer=answer)
        return text[: -len(eos)] if eos and text.endswith(eos) else text

    def render_batch(batch):
        prompts, chosens, rejecteds = [], [], []
        for signifiers, question, chosen, rejected in zip(
            batch["signifiers"], batch["question"], batch["chosen"], batch["rejected"]
        ):
            parts = []
            for key in keys[:-1]:
                if key == "system_prompt" and signifiers == "":
                    template = fields[key]["baseline"]
                else:
                    template = fields[key]["text"]
                parts.append(template.format(signifiers=signifiers, question=question))
            prompts.append("\n".join(parts))
            chosens.append(to_completion(chosen))
            rejecteds.append(to_completion(rejected))
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    result = DatasetDict()
    for name, ds in raw.items():
        result[name] = ds.map(
            render_batch,
            batched=True,
            remove_columns=ds.column_names,
            desc=f"Rendering {name} preferences",
        )
        print(f"Loaded {len(result[name])} preference rows from {name} dataset.")
    return result
