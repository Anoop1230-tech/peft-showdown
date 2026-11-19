"""Train and evaluate one run, then save its result JSON.

Run from the repo root (otherwise `from src...` imports fail):

    python -m src.train --method full_ft --task rte --regime 1k --seed 1

Writes results/final/{method}_{task}_{regime}_seed{seed}.json. If that file
already exists the run is skipped (use --overwrite to redo it), so a restarted
run_method.sh picks up where it stopped.

One run per process: peak GPU memory is only comparable that way.
"""

from __future__ import annotations

import os

# Use a single GPU. On a 2-GPU machine (Kaggle "T4 x2") the Trainer would
# otherwise split each batch across both GPUs and the memory numbers break.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import shutil
import tempfile
import time

import torch
from transformers import (AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from src.data import NUM_LABELS, load_task
from src.metrics import compute_metrics
from src.models import MODEL_NAME, METHOD_CONFIGS, build_model
from src.results import (METHODS, REGIMES, SEEDS, STAGES, TASKS, count_params,
                         dir_size_mb, new_result, peak_gpu_mem_mb, result_path,
                         save_result)

# Fixed settings, same for every method (issue #24). No LR search.
LEARNING_RATES = {"full_ft": 2e-5, "lora": 5e-4, "adapter": 3e-4,
                  "prompt_tuning": 1e-2}
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
MAX_LENGTH = 128
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01

OWNERS = {"full_ft": "deepak", "lora": "anoop", "adapter": "chiranjeet",
          "prompt_tuning": "praveen"}


def num_epochs(task: str, regime: str) -> int:
    if regime == "1k":
        return 20
    return 3 if task == "sst2" else 10


def save_trainable_weights(model, method: str, out_dir: str) -> None:
    """Save what you'd ship for this task (see docs/results_schema.md)."""
    # full_ft: whole model. lora / prompt_tuning: PEFT weights + head.
    # adapter: save adapter weights + head.
    if method == "adapter":
        model.save_adapter(out_dir, "glue")
    else:
        model.save_pretrained(out_dir)


def best_epoch(trainer) -> float | None:
    """Epoch of the checkpoint picked by load_best_model_at_end."""
    ckpt = trainer.state.best_model_checkpoint
    if not ckpt:
        return None
    step = int(ckpt.rstrip("/").split("-")[-1])
    for entry in trainer.state.log_history:
        if entry.get("step") == step and "eval_accuracy" in entry:
            return entry["epoch"]
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--regime", required=True, choices=REGIMES)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--stage", choices=STAGES, default="final",
                   help="stage: final (default), lr_search, or ablation")
    p.add_argument("--ablation-name", default=None,
                   help="name of ablation parameter (e.g. reduction_factor, config)")
    p.add_argument("--ablation-value", default=None,
                   help="value of ablation parameter (e.g. 8, seq_bn)")
    p.add_argument("--owner", default=None,
                   help="defaults to the method's owner (meta.owner)")
    p.add_argument("--notes", default="", help="free text saved in meta.notes")
    p.add_argument("--overwrite", action="store_true",
                   help="redo the run even if its result JSON exists")
    return p.parse_args(argv)


def run(args) -> None:
    fp16 = torch.cuda.is_available()  # fp16 needs a GPU
    hyperparams = {
        "learning_rate": LEARNING_RATES[args.method],
        "epochs": num_epochs(args.task, args.regime),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "fp16": fp16,
    }

    ablation = None
    method_config = dict(METHOD_CONFIGS[args.method])
    model_kwargs = {}
    if args.stage == "ablation" and args.ablation_name:
        val = args.ablation_value
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        ablation = {"name": args.ablation_name, "value": val}
        if args.method == "adapter":
            if args.ablation_name == "reduction_factor":
                method_config["reduction_factor"] = val
                model_kwargs["reduction_factor"] = val
            elif args.ablation_name in ("config", "adapter_config"):
                method_config["config"] = val
                model_kwargs["config"] = val
        elif args.method == "prompt_tuning":
            if args.ablation_name in ("num_virtual_tokens", "n_tokens", "virtual_tokens"):
                method_config["num_virtual_tokens"] = val
                model_kwargs["num_virtual_tokens"] = val
            elif args.ablation_name in ("init", "prompt_tuning_init"):
                method_config["prompt_tuning_init"] = str(val)
                model_kwargs["prompt_tuning_init"] = str(val)

    result = new_result(method=args.method, task=args.task, regime=args.regime,
                        seed=args.seed, stage=args.stage,
                        owner=args.owner or OWNERS[args.method],
                        hyperparams=hyperparams,
                        method_config=method_config,
                        ablation=ablation,
                        model_name=MODEL_NAME)
    result["meta"]["notes"] = args.notes

    out_path = result_path(result)
    if out_path.exists() and not args.overwrite:
        print(f"skip: {out_path} already exists")
        return

    print(f"=== {result['run_id']}  lr={hyperparams['learning_rate']} "
          f"epochs={hyperparams['epochs']} fp16={fp16}")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train, dev, validation = load_task(args.task, tokenizer, regime=args.regime,
                                       seed=args.seed, max_len=MAX_LENGTH)
    result["data"].update(n_train=len(train), n_dev=len(dev),
                          n_eval=len(validation))

    model = build_model(args.method, NUM_LABELS[args.task], **model_kwargs)
    result["efficiency"].update(count_params(model))
    print(f"trainable params: {result['efficiency']['trainable_params']:,} "
          f"({result['efficiency']['trainable_pct']}%)")

    work_dir = tempfile.mkdtemp(prefix=f"{result['run_id']}_")
    try:
        training_args = TrainingArguments(
            output_dir=os.path.join(work_dir, "trainer"),
            num_train_epochs=hyperparams["epochs"],
            learning_rate=hyperparams["learning_rate"],
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            fp16=fp16,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            seed=args.seed,
        )
        if args.method == "adapter":
            from adapters import AdapterTrainer
            trainer_cls = AdapterTrainer
        else:
            trainer_cls = Trainer

        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=dev,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics(args.task),
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        trainer.train()
        train_time = time.time() - t0
        peak_mem = peak_gpu_mem_mb()

        dev_m = trainer.evaluate(dev)                # best checkpoint, our 10% dev split
        val_m = trainer.evaluate(validation)         # GLUE validation, the reported number

        result["metrics"].update(
            accuracy=val_m["eval_accuracy"],
            f1=val_m.get("eval_f1"),
            dev_accuracy=dev_m["eval_accuracy"],
            dev_f1=dev_m.get("eval_f1"),
            best_epoch=best_epoch(trainer),
        )

        ckpt_dir = os.path.join(work_dir, "weights")
        save_trainable_weights(trainer.model, args.method, ckpt_dir)
        result["efficiency"].update(
            peak_gpu_mem_mb=peak_mem,
            train_time_s=round(train_time, 1),
            checkpoint_size_mb=dir_size_mb(ckpt_dir),
        )
    finally:
        # Kaggle has ~20 GB of disk; never keep checkpoints around
        shutil.rmtree(work_dir, ignore_errors=True)

    path = save_result(result, overwrite=args.overwrite)
    m = result["metrics"]
    f1 = f"  f1={m['f1']:.4f}" if m["f1"] is not None else ""
    print(f"saved {path}\n  val acc={m['accuracy']:.4f}{f1}  "
          f"dev acc={m['dev_accuracy']:.4f}  best epoch={m['best_epoch']}  "
          f"time={result['efficiency']['train_time_s']}s  "
          f"peak mem={result['efficiency']['peak_gpu_mem_mb']} MiB")


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
