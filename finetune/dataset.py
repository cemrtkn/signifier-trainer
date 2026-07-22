from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from pydantic import BaseModel, model_validator
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import LabelSmoother

import json
import yaml
from transformers import AutoTokenizer
import os
import sys

from finetune.utils.parser import Parser, ParserConfig, QA

IGNORE_INDEX = LabelSmoother.ignore_index


@dataclass
# source: https://github.com/center-for-humans-and-machines/transformer-heads/blob/main/transformer_heads/util/helpers.py#L27-L62
class DataCollatorWithPadding:
    """
    A data collator that pads sequences to the same length.

    Attributes:
        feature_name_to_padding_value (dict[str, int]): A dictionary mapping feature names to their padding values.

    Methods:
        __call__(features: List[Dict[str, Any]]) -> Dict[str, Any]: Pad the sequences in the features to the same length.
    """

    feature_name_to_padding_value: dict[str, int | float]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pad the sequences in the features to the same length.

        Args:
            features (List[Dict[str, Any]]): A list of features, where each feature is a dictionary mapping feature names to sequences.

        Returns:
            Dict[str, Any]: A dictionary mapping feature names to padded sequences.
        """
        batch = dict()
        for key, value in self.feature_name_to_padding_value.items():
            batch[key] = pad_sequence(
                [feature[key].clone().detach() for feature in features],
                batch_first=True,
                padding_value=value,
            )
        for key in features[0].keys():
            if key not in self.feature_name_to_padding_value:
                batch[key] = torch.stack([feature[key].clone().detach() for feature in features])
        return batch



def find_sequence(input_list, start_sequence):
    """Find the last start sequence in a list."""
    last_start_index = -1
    start_sequence_len = len(start_sequence)
    for i in range(len(input_list) - start_sequence_len + 1):
        if input_list[i : i + start_sequence_len] == start_sequence:
            last_start_index = i

    return last_start_index + start_sequence_len


def texts_to_training_tensors_instruct(
    data: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    mask_untrainable_tokens=True,
    start_target_text="<|start_header_id|>assistant<|end_header_id|>",
) -> dict[str, Any]:
    """Turns a list of texts into tokenized training tensors.
    If mask_untrainable_tokens is set, the labels of all text
    before start_target_text are set to the ignore_token.

    Note: only works for single target at the end of each data point. we don't expect to train on multiple targets in a single data point when training an Instruct model.
    Note: No padding is done here. Padding is done in the collator."""

    # create a copy of data
    result = data.copy()
    input_ids_list = []
    labels_list = []
    
    start_target_sequence = tokenizer(start_target_text, add_special_tokens=False)["input_ids"]

    tokenized_games = tokenizer(data["text"], add_special_tokens=False)["input_ids"]

    # Cloning tokenized_games to labels using a deep copy
    labels = [list(game) for game in tokenized_games]
    if mask_untrainable_tokens:
        for idx, input_list in tqdm(enumerate(tokenized_games), total=len(tokenized_games)):
            target_start_index = find_sequence(input_list, start_target_sequence)

            if target_start_index != len(start_target_sequence) - 1:
                # Set labels before the start index to IGNORE_INDEX
                labels[idx][:target_start_index] = [IGNORE_INDEX] * target_start_index
                input_ids_list.append(input_list)
                labels_list.append(labels[idx])
            else:
                print("Instruction not found in input list.")

    result["input_ids"] = input_ids_list
    result["labels"] = labels_list

    return result

class SignifierConfig(BaseModel):
    mode: Literal["token_signifier", "nl_signifier"]
    new_special_tokens: Optional[List[str]] = None
    signifier_prompts_path: Optional[str] = None
    nl_prompts: Optional[List[str]] = None  # derived from signifier_prompts_path, do not set

    @model_validator(mode="after")
    def _resolve_and_check(self) -> "SignifierConfig":
        if self.new_special_tokens and self.signifier_prompts_path:
            raise ValueError("Set either new_special_tokens or signifier_prompts_path, not both")

        if self.signifier_prompts_path:
            if not os.path.isfile(self.signifier_prompts_path):
                raise ValueError(f"signifier_prompts_path not found: {self.signifier_prompts_path}")
            with open(self.signifier_prompts_path, "r") as f:
                prompts = json.load(f)
            if not isinstance(prompts, dict) or not prompts:
                raise ValueError(f"{self.signifier_prompts_path} must be a non-empty {{author: prompt}} json")
            if self.mode == "token_signifier":
                # the file's keys define the personas; their tokens are derived
                self.new_special_tokens = [f"<|{author}|>" for author in prompts]
            else:
                # the prompt values are what the signifier column must contain
                self.nl_prompts = list(prompts.values())

        if self.mode == "token_signifier" and not self.new_special_tokens:
            raise ValueError(
                "token_signifier mode requires new_special_tokens (or a signifier_prompts_path "
                "whose keys the tokens are derived from)"
            )
        if self.mode == "nl_signifier" and self.new_special_tokens:
            raise ValueError("nl_signifier mode must not set new_special_tokens")
        return self


class DatasetConfig(BaseModel):
    mask_untrainable_tokens: bool = True
    signifier_config: Optional[SignifierConfig] = None
    new_special_tokens: Optional[List[str]] = None  # deprecated: set signifier_config instead
    data_path: str
    parser_config: ParserConfig
    test_fold: Optional[int] = 0
    start_target_text: str = "<|start_header_id|>assistant<|end_header_id|>"
    max_length: Optional[int] = None

    def resolve_signifier_config(self) -> SignifierConfig:
        if self.signifier_config is not None:
            if self.new_special_tokens:
                raise ValueError(
                    "Set new_special_tokens inside signifier_config, not at the top level."
                )
            return self.signifier_config
        if self.new_special_tokens:
            print(
                "Deprecation: top-level new_special_tokens is legacy; declare "
                "signifier_config with mode 'token_signifier' instead."
            )
            return SignifierConfig(mode="token_signifier", new_special_tokens=self.new_special_tokens)
        raise ValueError(
            "No signifier_config set. Declare train_dataset_config.signifier_config with "
            "mode 'token_signifier' (plus its new_special_tokens) or 'nl_signifier'."
        )


def is_token_shaped(value: str) -> bool:
    parts = value.split()
    return bool(parts) and all(p.startswith("<|") and p.endswith("|>") for p in parts)


def validate_signifiers(signifier_values, signifier_config, parser_config, revert_special_tokens=False):
    """Fail fast on config/data mismatches before any training starts."""
    system_prompt = parser_config.fields.get("system_prompt", {})
    if "{signifiers}" not in system_prompt.get("text", ""):
        raise ValueError("The system prompt must contain a {signifiers} placeholder.")

    if "" in signifier_values and "baseline" not in system_prompt:
        raise ValueError(
            "The dataset contains rows with an empty signifier but the system prompt "
            "has no 'baseline' template to route them to."
        )

    if revert_special_tokens:
        return  # the ablation rewrites tokens to plain text on purpose

    non_empty = sorted(v for v in signifier_values if v != "")
    if signifier_config.mode == "token_signifier":
        allowed = set(signifier_config.new_special_tokens)
        strays = [v for v in non_empty if not (is_token_shaped(v) and set(v.split()) <= allowed)]
        if strays:
            raise ValueError(
                f"token_signifier mode: {len(strays)} signifier value(s) in the dataset are not "
                f"covered by new_special_tokens, e.g. {strays[:3]}. Fix the token list or rebuild "
                "the dataset."
            )
    elif signifier_config.nl_prompts is not None:
        allowed = set(signifier_config.nl_prompts)
        strays = [v for v in non_empty if v not in allowed]
        if strays:
            raise ValueError(
                f"nl_signifier mode: {len(strays)} signifier value(s) in the dataset are not "
                f"among the prompts in signifier_prompts_path, e.g. {[s[:60] for s in strays[:3]]}. "
                "The dataset and the prompts file are out of sync."
            )
    else:
        token_like = [v for v in non_empty if is_token_shaped(v)]
        if token_like:
            raise ValueError(
                f"nl_signifier mode: {len(token_like)} signifier value(s) look like special "
                f"tokens, e.g. {token_like[:3]}. This dataset was built for token signifiers — "
                "use mode token_signifier or rebuild it with signifier prompts."
            )

    

class QADataset(Dataset):
    @classmethod
    def from_qas(
        cls,
        qas: List[QA],
        tokenizer: PreTrainedTokenizerBase,
        parser: Parser,
        mask_untrainable_tokens: bool,
    ):
        texts = parser.parse(qas)
        data = (
            texts_to_training_tensors_instruct(texts, tokenizer, mask_untrainable_tokens)
        )

        result = cls.from_dict(data)
        result.set_format(type="torch", columns=["input_ids", "labels"])
        return result


class CrossvalDatasetDict(DatasetDict):
    """DatasetDict with multiple datasets for each fold in crossvalidation."""

    @classmethod
    def load_from_disk(cls, path: str, test_fold: int = 0) -> DatasetDict:
        """Loads a dataset dictionary from a given path and returns train and test sets.

        Args:
            path (str): The path to load the dataset from.

        Returns:
            CrossvalDatasetDict: The loaded datawiset.
        """
        dataset_dict = super().load_from_disk(path)

        # Merge the folds into train and test sets
        return cls._merge_datasets(dataset_dict=dataset_dict, test_fold=test_fold)

    @classmethod
    def _merge_datasets(cls, dataset_dict: DatasetDict, test_fold: int) -> DatasetDict:
        """Merges all datasets except the test fold into one.

        Args:
            dataset_dict (DatasetDict): The datasets to merge.

        Returns:
            DatasetDict: The merged dataset.
        """
        result_dic = {}

        # Merge all datasets except the test fold
        # if the dataset_dict contains only one fold then set the test dataset to None
        if len(dataset_dict) == 1:
            result_dic["train"] = dataset_dict[str(test_fold)]
        else:
            train_datasets = [ds for i, ds in dataset_dict.items() if i != str(test_fold)]

            # Concatenate training datasets into a single dataset
            if len(train_datasets) > 0:
                train_dataset = concatenate_datasets(train_datasets)
                result_dic["train"] = train_dataset

            # Get the test fold as a dataset
            test_dataset = dataset_dict[str(test_fold)]
            result_dic["test"] = test_dataset

        return DatasetDict(result_dic)


class CustomDatasetDict(CrossvalDatasetDict):
    def __init__(
        self, config: DatasetConfig, tokenizer: PreTrainedTokenizerBase, test_fold: int = 0, revert_special_tokens: bool = False
    ):
        """Loads the dataset from the config and splits it into train and test.

        Args:
            config (DatasetConfig): The config to load the datasets from.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
            test_fold (int, optional): The fold to use as test. Defaults to 0.
        """
        super().__init__()

        # Instance attributes
        self.config = config
        self.test_fold = test_fold
        self.revert_special_tokens = revert_special_tokens

        self._load_datasets(tokenizer)
    
    def _revert_new_tokens(self, signifiers):
        signifier_tokens = signifiers.split()
        result = ""

        for token in signifier_tokens:
            if token == "":
                return ""
            stripped_token = token[2:-2]  # remove '<|' and '|>'
            normal_text_tokens = " ".join(stripped_token.split("_"))
            result += normal_text_tokens

        return result

    def _load_datasets(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = 1000):
        """Tokenise streamingly via `ds.map(batched=True)` so a full-corpus
        DatasetDict never materialises into Python lists. Memory stays
        bounded to one batch of ~`batch_size` rows at a time; results
        stream to an Arrow table on disk."""
        raw_datasets: DatasetDict = self.load_from_disk(self.config.data_path, self.test_fold)

        signifier_values = set()
        for split in raw_datasets.values():
            signifier_values.update(split.unique("signifiers"))
        validate_signifiers(
            signifier_values,
            self.config.resolve_signifier_config(),
            self.config.parser_config,
            revert_special_tokens=self.revert_special_tokens,
        )

        parser = Parser(self.config.parser_config)
        start_target_sequence = tokenizer(
            self.config.start_target_text, add_special_tokens=False
        )["input_ids"]
        mask = self.config.mask_untrainable_tokens
        revert = self.revert_special_tokens
        revert_fn = self._revert_new_tokens

        def process_batch(batch):
            keys = list(batch.keys())
            rows = [dict(zip(keys, vals)) for vals in zip(*batch.values())]
            if revert:
                for r in rows:
                    r["signifiers"] = revert_fn(r["signifiers"])
            qas = [QA(**r) for r in rows]
            texts = parser.parse(qas)["text"]
            tok_kwargs: dict = {"add_special_tokens": False}
            if self.config.max_length is not None:
                tok_kwargs["max_length"] = self.config.max_length
                tok_kwargs["truncation"] = True
            tokenized = tokenizer(texts, **tok_kwargs)["input_ids"]
            input_ids_out: list[list[int]] = []
            labels_out: list[list[int]] = []
            for input_list in tokenized:
                target_start_index = find_sequence(input_list, start_target_sequence)
                if target_start_index == len(start_target_sequence) - 1:
                    # No assistant header found — drop the row (mirrors prior
                    # behaviour). Different output/input batch size is fine
                    # for ds.map(batched=True).
                    continue
                lbls = list(input_list)
                if mask:
                    lbls[:target_start_index] = [IGNORE_INDEX] * target_start_index
                input_ids_out.append(input_list)
                labels_out.append(lbls)
            return {"input_ids": input_ids_out, "labels": labels_out}

        for ds_name, ds in raw_datasets.items():
            mapped = ds.map(
                process_batch,
                batched=True,
                batch_size=batch_size,
                remove_columns=ds.column_names,
                desc=f"Tokenising {ds_name}",
            )
            mapped.set_format(type="torch", columns=["input_ids", "labels"])
            self[ds_name] = mapped
            print(f"Loaded {len(mapped)} data points from {ds_name} dataset.")


if __name__ == "__main__":
    # Example usage
    with open("configs/character/train/sft_instruct_fsdp_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token=os.getenv("HUGGINGFACE_TOKEN"),  # or pass directly
        local_files_only=False  # default
    )

    if tokenizer.pad_token is None:
        print("Setting pad token to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens_dict = {'additional_special_tokens': ["<|capitalism|>", "<|communism|>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('Added', num_added_toks, 'new special tokens to the tokenizer.')
    
    dataset_config = config["train_dataset_config"]

    dataset_config = DatasetConfig(**dataset_config)
    print(dataset_config)
    dataset_dict = CustomDatasetDict(
        config=dataset_config,
        tokenizer=tokenizer,
        test_fold=0,
        revert_special_tokens=False
    )