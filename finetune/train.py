"""Mode-dispatching entry point: runs sft (default) or dpo per config.mode."""

import argparse

from finetune.utils.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.mode == "dpo":
        from finetune.dpo import run_dpo

        run_dpo(config=config)
    else:
        from finetune.sft import run_sft

        run_sft(config=config)
