from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import LabelSmoother

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

class DatasetConfig(BaseModel):
    mask_untrainable_tokens: bool = True
    new_special_tokens: List[str]
    data_path: str
    parser_config: ParserConfig
    test_fold: Optional[int] = 0

    

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

    def _load_datasets(self, tokenizer: PreTrainedTokenizerBase):
        # Load datasets using the paths from the config
        raw_datasets: DatasetDict = self.load_from_disk(self.config.data_path, self.test_fold)
        
        parser = Parser(self.config.parser_config)
        for ds_name, ds in raw_datasets.items():
            # New format
            qas = []     
            for data in ds:
                dict_to_validate = {}
                for k, v in data.items():
                    dict_to_validate[k] = v
                    # turn special tokens into normal strings if flagged
                    if self.revert_special_tokens and k == "signifiers":
                        dict_to_validate["signifiers"] = self._revert_new_tokens(dict_to_validate["signifiers"])
                # validate
                qa = QA(**dict_to_validate)
                qas.append(qa)
            self[ds_name] = QADataset.from_qas(
                qas,
                tokenizer,
                parser,
                self.config.mask_untrainable_tokens
            )
        print(f"Loaded {len(qas)} data points from {ds_name} dataset.") 


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