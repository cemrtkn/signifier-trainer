# coding=utf-8
# Copyright 2024 David Carreto Fidalgo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

from dataclasses import dataclass, field

from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer


@dataclass
class Arguments:
    model_name: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Model identifier from 'huggingface.co/models'."},
    )

    hf_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Your Hugging Face Token in case you want to access a gated model."
        },
    )

if __name__ == "__main__":
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()

    # download model weights and tokenizer, and cache it.
    AutoTokenizer.from_pretrained(
        args.model_name, token=args.hf_token
    )

    AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        token=args.hf_token,
    )
