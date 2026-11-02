"""Results JSON format for every run in the project.

Spec: docs/results_schema.md. Every run (final, LR search, ablation) must be
saved through `save_result` so that all files share one format and the
analysis notebook can load them all with `load_results`.

Typical use inside train.py:

    from src.results import (new_result, count_params, dir_size_mb,
                             environment_info, save_result)

    result = new_result(method="lora", task="rte", regime="1k", seed=1,
                        stage="final", owner="anoop",
                        hyperparams={...}, method_config={"r": 8, ...})
    result["efficiency"].update(count_params(model))
    ...train, evaluate, save weights...
    result["metrics"].update(accuracy=0.71, f1=None, ...)
    result["efficiency"].update(peak_gpu_mem_mb=..., train_time_s=...,
                                checkpoint_size_mb=dir_size_mb(save_dir))
    result["environment"] = environment_info()
    save_result(result)

torch / transformers / peft / adapters are imported lazily, so this module can
also be used on a machine without them (e.g. for analysis or validation).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

METHODS = ("full_ft", "lora", "adapter", "prompt_tuning")
TASKS = ("sst2", "mrpc", "rte")
REGIMES = ("full", "1k")
STAGES = ("final", "lr_search", "ablation")
SEEDS = (1, 2, 3)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

MB = 1024 ** 2  # all "_mb" fields are MiB (2**20 bytes)


# ---------------------------------------------------------------------------
# Building a result
# ---------------------------------------------------------------------------

def new_result(*, method: str, task: str, regime: str, seed: int,
               stage: str = "final", owner: str,
               hyperparams: dict[str, Any],
               method_config: dict[str, Any] | None = None,
               ablation: dict[str, Any] | None = None,
               model_name: str = "roberta-base") -> dict[str, Any]:
    """Return a result dict with identity fields filled and metrics empty."""
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": None,  # filled below
        "stage": stage,
        "method": method,
        "task": task,
        "regime": regime,
        "seed": seed,
        "model_name": model_name,
        "ablation": ablation,  # e.g. {"name": "lora_rank", "value": 4}
        "hyperparams": {
            "learning_rate": None,
            "epochs": None,
            "batch_size": None,
            "max_length": None,
            "warmup_ratio": None,
            "weight_decay": None,
            "fp16": None,
            **hyperparams,
        },
        "method_config": method_config or {},
        "data": {"n_train": None, "n_dev": None, "n_eval": None},
        "metrics": {
            "eval_split": "validation",
            "accuracy": None,
            "f1": None,             # MRPC only, null for SST-2 / RTE
            "dev_accuracy": None,   # best score on our 10% dev split
            "dev_f1": None,         # MRPC only
            "best_epoch": None,
        },
        "efficiency": {
            "trainable_params": None,
            "total_params": None,
            "trainable_pct": None,
            "peak_gpu_mem_mb": None,
            "train_time_s": None,
            "checkpoint_size_mb": None,
        },
        "environment": {},
        "meta": {
            "owner": owner,
            "timestamp_utc": None,
            "git_commit": None,
            "notes": "",
        },
    }
    result["run_id"] = make_run_id(result)
    return result


def make_run_id(result: dict[str, Any]) -> str:
    """Unique, filesystem-safe id. Also used as the file name (without .json).

    final:     lora_rte_1k_seed1
    lr_search: lora_rte_full_lr0.0003_seed1
    ablation:  lora_rte_full_lora_rank-4_seed1
    """
    base = f"{result['method']}_{result['task']}_{result['regime']}"
    stage = result["stage"]
    if stage == "lr_search":
        lr = result["hyperparams"]["learning_rate"]
        base += f"_lr{lr:g}"
    elif stage == "ablation":
        abl = result["ablation"] or {}
        value = str(abl.get("value")).replace("/", "-").replace(" ", "")
        base += f"_{abl.get('name')}-{value}"
    return f"{base}_seed{result['seed']}"


def result_path(result: dict[str, Any]) -> Path:
    """results/<stage>/<run_id>.json"""
    return RESULTS_DIR / result["stage"] / f"{result['run_id']}.json"


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def count_params(model) -> dict[str, Any]:
    """Trainable / total parameter counts, classifier head included.

    total_params is every parameter in the model as built (frozen base +
    any added LoRA / adapter / prompt parameters + head).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": round(100.0 * trainable / total, 4),
    }


def dir_size_mb(path: str | os.PathLike) -> float:
    """Size of a saved checkpoint directory (or single file) in MiB."""
    p = Path(path)
    if p.is_file():
        return round(p.stat().st_size / MB, 3)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / MB, 3)


def peak_gpu_mem_mb() -> float | None:
    """torch.cuda.max_memory_allocated() in MiB, or None without a GPU.

    Call torch.cuda.reset_peak_memory_stats() right before training.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / MB, 1)


def _version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


def _detect_platform() -> str:
    # check Kaggle first: Kaggle's image also sets COLAB_* variables
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle"):
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


def environment_info() -> dict[str, Any]:
    """GPU name and library versions. Times are only comparable on the same GPU."""
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "gpu_name": gpu,
        "platform": _detect_platform(),
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "adapters": _version("adapters"),
        "datasets": _version("datasets"),
    }


def _git_commit() -> str | None:
    # set by scripts/kaggle_push.py, where the code runs without a .git folder
    if os.environ.get("PEFT_GIT_COMMIT"):
        return os.environ["PEFT_GIT_COMMIT"]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Validation, saving, loading
# ---------------------------------------------------------------------------

_REQUIRED_NUMBERS = {
    "metrics": ("accuracy", "dev_accuracy"),
    "efficiency": ("trainable_params", "total_params", "trainable_pct",
                   "train_time_s", "checkpoint_size_mb"),
    "hyperparams": ("learning_rate", "epochs", "batch_size", "max_length"),
    "data": ("n_train", "n_eval"),
}


def validate(result: dict[str, Any]) -> list[str]:
    """Return a list of problems. Empty list means the result is valid."""
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(result.get("schema_version") == SCHEMA_VERSION,
          f"schema_version must be {SCHEMA_VERSION}")
    check(result.get("method") in METHODS, f"method must be one of {METHODS}")
    check(result.get("task") in TASKS, f"task must be one of {TASKS}")
    check(result.get("regime") in REGIMES, f"regime must be one of {REGIMES}")
    check(result.get("stage") in STAGES, f"stage must be one of {STAGES}")
    check(isinstance(result.get("seed"), int), "seed must be an int")
    if result.get("stage") == "final":
        check(result.get("seed") in SEEDS, f"final runs must use seeds {SEEDS}")
    if result.get("stage") == "ablation":
        abl = result.get("ablation")
        check(isinstance(abl, dict) and "name" in abl and "value" in abl,
              "ablation runs need ablation = {'name': ..., 'value': ...}")
    check(result.get("meta", {}).get("owner"), "meta.owner is required")

    for section, keys in _REQUIRED_NUMBERS.items():
        block = result.get(section, {})
        for k in keys:
            v = block.get(k)
            check(isinstance(v, (int, float)) and not isinstance(v, bool),
                  f"{section}.{k} must be a number, got {v!r}")

    m = result.get("metrics", {})
    for k in ("accuracy", "dev_accuracy", "f1", "dev_f1"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            check(0.0 <= v <= 1.0, f"metrics.{k} must be a fraction in [0, 1]")
    if result.get("task") == "mrpc":
        check(isinstance(m.get("f1"), (int, float)), "MRPC runs must report metrics.f1")

    if result.get("run_id") is not None and not errors:
        check(result["run_id"] == make_run_id(result),
              f"run_id {result['run_id']!r} does not match fields "
              f"(expected {make_run_id(result)!r})")
    return errors


def save_result(result: dict[str, Any], overwrite: bool = False) -> Path:
    """Validate and write results/<stage>/<run_id>.json atomically."""
    result["run_id"] = make_run_id(result)
    result["meta"]["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["meta"]["git_commit"] = result["meta"].get("git_commit") or _git_commit()
    if not result.get("environment"):
        result["environment"] = environment_info()

    errors = validate(result)
    if errors:
        raise ValueError("Invalid result:\n  - " + "\n  - ".join(errors))

    path = result_path(result)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # atomic: a Colab disconnect never leaves half a file
    return path


def load_results(stage: str | None = None, root: str | os.PathLike | None = None):
    """Load result files into a flat pandas DataFrame (one row per run).

    Nested fields become dotted columns, e.g. "metrics.accuracy".
    """
    import pandas as pd
    base = Path(root) if root else RESULTS_DIR
    pattern = f"{stage}/*.json" if stage else "*/*.json"
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(base.glob(pattern))]
    return pd.json_normalize(rows)


if __name__ == "__main__":
    # python src/results.py results/final/*.json  -> validates files
    bad = 0
    for f in sys.argv[1:]:
        errs = validate(json.loads(Path(f).read_text(encoding="utf-8")))
        if errs:
            bad += 1
            print(f"FAIL {f}")
            for e in errs:
                print(f"     - {e}")
        else:
            print(f"ok   {f}")
    sys.exit(1 if bad else 0)
