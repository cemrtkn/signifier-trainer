# signifier-trainer

FSDP-based SFT trainer for signifier-conditioned language models. Extracted from [`cemrtkn/collectively-grounded-llms`](https://github.com/cemrtkn/collectively-grounded-llms) so the trainer can evolve independently of any one downstream project.

The trainer expects a HuggingFace `DatasetDict` of rows shaped:

```
{ "signifiers": "<|token|>",
  "question":   "<user-side prompt content>",
  "answer":     "<assistant target>" }
```

A YAML `parser_config` block controls how those three fields are spliced into the model's chat template; special tokens (`new_special_tokens`) are added to the tokenizer at the top of training. The full config schema lives in `finetune.sft_types.TrainingConfig`.

## Quickstart

```bash
uv venv && uv pip install -e .
python -m finetune.sft --config <your-config>.yaml
```

For multi-GPU FSDP:

```bash
torchrun --nproc_per_node=<N> -m finetune.sft --config <your-config>.yaml
```

## Layout

- `finetune/` — the trainer package (`sft.py` entry point, `sft_types.TrainingConfig` schema, dataset / parser / training-mode utilities).
- `tests/` — unit + integration tests.

## Using as a submodule

```bash
git submodule add https://github.com/cemrtkn/signifier-trainer external/signifier-trainer
git -C external/signifier-trainer checkout v0.1.2   # or another pinned tag
```

Then expose `external/signifier-trainer` on `PYTHONPATH` when invoking `python -m finetune.sft`.

## License

MIT — see [LICENSE](LICENSE).
