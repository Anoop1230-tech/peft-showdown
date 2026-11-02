# Results JSON format (schema version 1)

Owner: Anoop. Every run writes **one JSON file** through `src/results.py` (`save_result`). Do not write result files by hand. `save_result` validates the file and refuses to save it if a required field is missing.

## Where files go

```
results/
├── final/       72 runs:   {method}_{task}_{regime}_seed{seed}.json
├── lr_search/   LR search: {method}_{task}_{regime}_lr{lr}_seed{seed}.json
└── ablation/    ablations: {method}_{task}_{regime}_{name}-{value}_seed{seed}.json
```

Examples: `final/lora_rte_1k_seed1.json`, `lr_search/lora_rte_full_lr0.0003_seed1.json`, `ablation/lora_rte_full_lora_rank-4_seed1.json`.

The subfolders keep LR-search and ablation runs from overwriting final runs. `save_result` also refuses to overwrite an existing file unless you pass `overwrite=True`.

## Allowed values

| Field | Values |
|---|---|
| `method` | `full_ft`, `lora`, `adapter`, `prompt_tuning` |
| `task` | `sst2`, `mrpc`, `rte` |
| `regime` | `full`, `1k` |
| `stage` | `final`, `lr_search`, `ablation` |
| `seed` | `1`, `2`, `3` for final runs |
| `meta.owner` | `deepak`, `anoop`, `chiranjeet`, `praveen` |

Use exactly these strings, including in `train.py --method`.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | Always `1`. Bump if the format changes. |
| `run_id` | str | Same as the file name without `.json`. Generated automatically. |
| `stage`, `method`, `task`, `regime`, `seed` | | See table above. |
| `model_name` | str | `roberta-base` unless it's the BERT ablation. |
| `ablation` | object or null | `null` for final and LR-search runs. Otherwise `{"name": "lora_rank", "value": 4}`. |
| `hyperparams.learning_rate` | float | |
| `hyperparams.epochs` | int | |
| `hyperparams.batch_size` | int | 32 |
| `hyperparams.max_length` | int | 128 |
| `hyperparams.warmup_ratio` | float | 0.06 |
| `hyperparams.weight_decay` | float | 0.01 |
| `hyperparams.fp16` | bool | |
| `method_config` | object | Method-specific settings only. LoRA: `r`, `lora_alpha`, `lora_dropout`, `target_modules`. Adapters: `config`, `reduction_factor`. Prompt tuning: `num_virtual_tokens`, `init`. Full FT: `{}` (or `frozen_layers` for the ablation). |
| `data.n_train` | int | Training examples actually used (1000 in the 1k regime). |
| `data.n_dev` | int | Size of our 10% dev split. |
| `data.n_eval` | int | Size of the GLUE validation split we report on. |
| `metrics.eval_split` | str | Always `validation`. |
| `metrics.accuracy` | float 0–1 | On GLUE validation, using the best checkpoint. |
| `metrics.f1` | float 0–1 or null | Required for MRPC, `null` otherwise. |
| `metrics.dev_accuracy` | float 0–1 | Best score on our dev split (used for model selection and LR search). |
| `metrics.dev_f1` | float 0–1 or null | MRPC only. |
| `metrics.best_epoch` | number | Epoch of the selected checkpoint. |
| `efficiency.trainable_params` | int | `requires_grad` params, **classifier head included**. |
| `efficiency.total_params` | int | All params in the built model (base + added params + head). |
| `efficiency.trainable_pct` | float | `100 * trainable / total`. |
| `efficiency.peak_gpu_mem_mb` | float or null | `torch.cuda.max_memory_allocated()` in MiB after `reset_peak_memory_stats()` before training. |
| `efficiency.train_time_s` | float | Wall-clock around `trainer.train()` only. |
| `efficiency.checkpoint_size_mb` | float | Size on disk of the saved trainable weights (see below). |
| `environment` | object | GPU name, platform (colab/kaggle/local), library versions. Filled automatically. |
| `meta.owner`, `meta.timestamp_utc`, `meta.git_commit`, `meta.notes` | | Timestamp and commit are filled automatically. |

Metrics are fractions (0.874), not percentages (87.4). The analysis notebook converts them to percentages.

## What to save for `checkpoint_size_mb`

Save only what you would need to ship for this task, then pass the folder to `dir_size_mb(...)`:

| Method | What to save |
|---|---|
| Full FT | `model.save_pretrained(dir)`: the whole model |
| LoRA, prompt tuning | `model.save_pretrained(dir)` on the PEFT model: LoRA/prompt weights + classifier head |
| Adapters | `model.save_adapter(dir, "glue")` (it includes the head by default) |

Delete large checkpoints after measuring. Model weights are in `.gitignore` and must not be pushed.

## Rules for comparable numbers

1. Call `torch.cuda.reset_peak_memory_stats()` right before training. Run **one job per process** (a fresh `python src/train.py ...`), not several runs in one notebook loop.
2. Compare training times only between runs with the same `environment.gpu_name`.
3. Push the JSON to the repo as soon as a run finishes.

## Checking files

```
python src/results.py results/final/*.json
```

prints `ok` or the list of problems for each file.

See `docs/example_result.json` for a complete example.
