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
"""  # file description; no effect on running the code

from __future__ import annotations  # allows modern type hints like "str | None" on older Python

import json  # read / write JSON files
import os  # environment variables, atomic file replace
import platform  # Python version for the environment info
import subprocess  # runs "git" to get the commit hash
import sys  # command-line arguments and exit code
from datetime import datetime, timezone  # timestamp of when the result was saved
from pathlib import Path  # easy file paths
from typing import Any  # type hint meaning "any value"

SCHEMA_VERSION = 1  # version of the JSON layout; validate() rejects files with another version

METHODS = ("full_ft", "lora", "adapter", "prompt_tuning")  # allowed method names (also --method choices in train.py)
TASKS = ("sst2", "mrpc", "rte")  # allowed task names
REGIMES = ("full", "1k")  # allowed data regimes
STAGES = ("final", "lr_search", "ablation")  # allowed stages = results sub-folders
SEEDS = (1, 2, 3)  # allowed seeds for final runs

REPO_ROOT = Path(__file__).resolve().parent.parent  # this file is src/results.py, so 2 levels up = repo root; works from any working folder
RESULTS_DIR = REPO_ROOT / "results"  # all results go to results/<stage>/

MB = 1024 ** 2  # 1 MiB = 1,048,576 bytes; all "_mb" numbers use this


def new_result(*, method: str, task: str, regime: str, seed: int,  # creates the empty result dict; "*" forces keyword arguments so values can't be swapped by mistake
               stage: str = "final", owner: str,  # stage defaults to final; owner is required
               hyperparams: dict[str, Any],  # training settings
               method_config: dict[str, Any] | None = None,  # method settings (optional)
               ablation: dict[str, Any] | None = None,  # ablation info (None for normal runs)
               model_name: str = "roberta-base") -> dict[str, Any]:  # base model name
    """Return a result dict with identity fields filled and metrics empty."""  # help text
    result = {  # the full result structure; every run has exactly these keys
        "schema_version": SCHEMA_VERSION,  # layout version
        "run_id": None,  # filled just below by make_run_id
        "stage": stage,  # final / lr_search / ablation
        "method": method,  # which method
        "task": task,  # which task
        "regime": regime,  # full or 1k
        "seed": seed,  # which seed
        "model_name": model_name,  # roberta-base
        "ablation": ablation,  # e.g. {"name": "lora_rank", "value": 4}, else None
        "hyperparams": {  # training settings
            "learning_rate": None,  # placeholders so every key always exists ...
            "epochs": None,  # ...
            "batch_size": None,  # ...
            "max_length": None,  # ...
            "warmup_ratio": None,  # ...
            "weight_decay": None,  # ...
            "fp16": None,  # ...
            **hyperparams,  # ... then the real values overwrite the placeholders
        },  # end of hyperparams
        "method_config": method_config or {},  # method settings, or empty dict if None
        "data": {"n_train": None, "n_dev": None, "n_eval": None},  # dataset sizes, filled later
        "metrics": {  # scores, filled after training
            "eval_split": "validation",  # reported numbers come from GLUE validation
            "accuracy": None,  # validation accuracy
            "f1": None,  # MRPC only, null for SST-2 / RTE
            "dev_accuracy": None,  # best score on our 10% dev split
            "dev_f1": None,  # MRPC only
            "best_epoch": None,  # epoch of the best checkpoint
        },  # end of metrics
        "efficiency": {  # cost numbers, filled during / after training
            "trainable_params": None,  # params that are trained
            "total_params": None,  # all params
            "trainable_pct": None,  # trainable as a percent of total
            "peak_gpu_mem_mb": None,  # peak GPU memory
            "train_time_s": None,  # training seconds
            "checkpoint_size_mb": None,  # size of saved weights
        },  # end of efficiency
        "environment": {},  # GPU and library versions, filled by save_result
        "meta": {  # bookkeeping
            "owner": owner,  # who ran it
            "timestamp_utc": None,  # when it was saved
            "git_commit": None,  # which code version
            "notes": "",  # free text
        },  # end of meta
    }  # end of result
    result["run_id"] = make_run_id(result)  # build the unique name from the fields above
    return result  # give it back to train.py


def make_run_id(result: dict[str, Any]) -> str:  # builds the unique run name, which is also the file name
    """Unique, filesystem-safe id. Also used as the file name (without .json).

    final:     lora_rte_1k_seed1
    lr_search: lora_rte_full_lr0.0003_seed1
    ablation:  lora_rte_full_lora_rank-4_seed1
    """  # help text with examples
    base = f"{result['method']}_{result['task']}_{result['regime']}"  # e.g. "lora_rte_1k"
    stage = result["stage"]  # stage decides what extra part is added
    if stage == "lr_search":  # LR-search runs differ only by LR ...
        lr = result["hyperparams"]["learning_rate"]  # ... so the LR goes in the name
        base += f"_lr{lr:g}"  # ":g" = short form like 0.0003; without the LR, runs with different LRs would overwrite each other
    elif stage == "ablation":  # ablation runs differ by the changed setting
        abl = result["ablation"] or {}  # {} if missing, so .get below doesn't crash
        value = str(abl.get("value")).replace("/", "-").replace(" ", "")  # remove "/" and spaces, which are unsafe in file names
        base += f"_{abl.get('name')}-{value}"  # e.g. "_lora_rank-4"
    return f"{base}_seed{result['seed']}"  # add the seed, e.g. "lora_rte_1k_seed1"


def result_path(result: dict[str, Any]) -> Path:  # where the result file goes
    """results/<stage>/<run_id>.json"""  # help text
    return RESULTS_DIR / result["stage"] / f"{result['run_id']}.json"  # e.g. results/final/lora_rte_1k_seed1.json


def count_params(model) -> dict[str, Any]:  # counts trainable vs total parameters
    """Trainable / total parameter counts, classifier head included.

    total_params is every parameter in the model as built (frozen base +
    any added LoRA / adapter / prompt parameters + head).
    """  # help text
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)  # numbers in all weights that get gradients (are trained)
    total = sum(p.numel() for p in model.parameters())  # numbers in all weights, frozen or not
    return {  # result as a dict (merged into result["efficiency"])
        "trainable_params": int(trainable),  # int() so it is plain JSON
        "total_params": int(total),  # total
        "trainable_pct": round(100.0 * trainable / total, 4),  # percent trainable, 4 decimals
    }  # end of dict


def dir_size_mb(path: str | os.PathLike) -> float:  # size of a file or folder in MiB (checkpoint size)
    """Size of a saved checkpoint directory (or single file) in MiB."""  # help text
    p = Path(path)  # make it a Path object
    if p.is_file():  # a single file
        return round(p.stat().st_size / MB, 3)  # its size in MiB
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())  # add up every file in the folder and sub-folders
    return round(total / MB, 3)  # in MiB, 3 decimals


def peak_gpu_mem_mb() -> float | None:  # highest GPU memory PyTorch used since the last reset
    """torch.cuda.max_memory_allocated() in MiB, or None without a GPU.

    Call torch.cuda.reset_peak_memory_stats() right before training.
    """  # help text
    import torch  # imported here so analysis machines without torch can still use this file
    if not torch.cuda.is_available():  # no GPU (e.g. a laptop)
        return None  # no number instead of crashing
    return round(torch.cuda.max_memory_allocated() / MB, 1)  # peak memory in MiB, 1 decimal


def _version(pkg: str) -> str | None:  # installed version of a package
    try:  # the package may not be installed
        from importlib.metadata import version  # standard tool to read installed versions
        return version(pkg)  # e.g. "4.57.6"
    except Exception:  # not installed or any other problem
        return None  # None instead of crashing the run


def _detect_platform() -> str:  # guesses where the code runs
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle"):  # Kaggle signs; checked first because Kaggle also sets COLAB_* variables
        return "kaggle"  # on Kaggle
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:  # Colab signs
        return "colab"  # on Colab
    return "local"  # anything else = own computer


def environment_info() -> dict[str, Any]:  # GPU and library versions, so results can be reproduced
    """GPU name and library versions. Times are only comparable on the same GPU."""  # help text
    gpu = None  # default: no GPU
    try:  # torch may be missing
        import torch  # imported here so the file works without torch
        if torch.cuda.is_available():  # GPU present
            gpu = torch.cuda.get_device_name(0)  # e.g. "Tesla T4"; times are only comparable on the same GPU
    except ImportError:  # torch not installed
        pass  # keep gpu = None
    return {  # all the environment details
        "gpu_name": gpu,  # GPU model
        "platform": _detect_platform(),  # kaggle / colab / local
        "python": platform.python_version(),  # e.g. "3.11.13"
        "torch": _version("torch"),  # library versions ...
        "transformers": _version("transformers"),  # ...
        "peft": _version("peft"),  # ...
        "adapters": _version("adapters"),  # ...
        "datasets": _version("datasets"),  # ...
    }  # end of dict


def _git_commit() -> str | None:  # short git hash of the code version that made the result
    if os.environ.get("PEFT_GIT_COMMIT"):  # set by scripts/kaggle_push.py, because on Kaggle there is no .git folder
        return os.environ["PEFT_GIT_COMMIT"]  # use that value
    try:  # git may be missing
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],  # ask git for the current commit
                             cwd=REPO_ROOT, capture_output=True, text=True,  # run in the repo, capture text output
                             timeout=5)  # give up after 5 s so a stuck git never blocks saving
        return out.stdout.strip() or None  # e.g. "a1b2c3d", or None if empty
    except Exception:  # git not installed / not a repo
        return None  # None instead of crashing


_REQUIRED_NUMBERS = {  # fields that MUST be numbers in a valid result: section -> keys
    "metrics": ("accuracy", "dev_accuracy"),  # scores
    "efficiency": ("trainable_params", "total_params", "trainable_pct",  # params ...
                   "train_time_s", "checkpoint_size_mb"),  # ... time and size (peak memory not required: CPU runs have none)
    "hyperparams": ("learning_rate", "epochs", "batch_size", "max_length"),  # main settings
    "data": ("n_train", "n_eval"),  # dataset sizes
}  # end of _REQUIRED_NUMBERS


def validate(result: dict[str, Any]) -> list[str]:  # checks a result and returns a list of problems (empty = valid)
    """Return a list of problems. Empty list means the result is valid."""  # help text
    errors: list[str] = []  # collected problems

    def check(cond: bool, msg: str) -> None:  # small helper: record msg if cond is False
        if not cond:  # the condition failed
            errors.append(msg)  # save the message (we keep going to find ALL problems, not just the first)

    check(result.get("schema_version") == SCHEMA_VERSION,  # layout version must match
          f"schema_version must be {SCHEMA_VERSION}")  # message if not
    check(result.get("method") in METHODS, f"method must be one of {METHODS}")  # method must be allowed
    check(result.get("task") in TASKS, f"task must be one of {TASKS}")  # task must be allowed
    check(result.get("regime") in REGIMES, f"regime must be one of {REGIMES}")  # regime must be allowed
    check(result.get("stage") in STAGES, f"stage must be one of {STAGES}")  # stage must be allowed
    check(isinstance(result.get("seed"), int), "seed must be an int")  # seed must be a whole number
    if result.get("stage") == "final":  # final runs ...
        check(result.get("seed") in SEEDS, f"final runs must use seeds {SEEDS}")  # ... must use seed 1, 2 or 3
    if result.get("stage") == "ablation":  # ablation runs ...
        abl = result.get("ablation")  # ... must say what was changed
        check(isinstance(abl, dict) and "name" in abl and "value" in abl,  # needs both name and value
              "ablation runs need ablation = {'name': ..., 'value': ...}")  # message if not
    check(result.get("meta", {}).get("owner"), "meta.owner is required")  # someone must own the run

    for section, keys in _REQUIRED_NUMBERS.items():  # each section with required numbers
        block = result.get(section, {})  # that section ({} if missing)
        for k in keys:  # each required key
            v = block.get(k)  # its value
            check(isinstance(v, (int, float)) and not isinstance(v, bool),  # must be a number; bool excluded because True counts as 1 in Python
                  f"{section}.{k} must be a number, got {v!r}")  # message if not

    m = result.get("metrics", {})  # the metrics section
    for k in ("accuracy", "dev_accuracy", "f1", "dev_f1"):  # every score
        v = m.get(k)  # its value
        if isinstance(v, (int, float)):  # only check numbers (f1 may be None)
            check(0.0 <= v <= 1.0, f"metrics.{k} must be a fraction in [0, 1]")  # catches someone saving 85 instead of 0.85
    if result.get("task") == "mrpc":  # MRPC ...
        check(isinstance(m.get("f1"), (int, float)), "MRPC runs must report metrics.f1")  # ... must have F1 (its official metric)

    if result.get("run_id") is not None and not errors:  # only if everything else is fine
        check(result["run_id"] == make_run_id(result),  # the name must match the fields (catches hand-edited files)
              f"run_id {result['run_id']!r} does not match fields "  # message ...
              f"(expected {make_run_id(result)!r})")  # ... with the expected name
    return errors  # list of problems


def save_result(result: dict[str, Any], overwrite: bool = False) -> Path:  # fills final fields, validates and writes the JSON
    """Validate and write results/<stage>/<run_id>.json atomically."""  # help text
    result["run_id"] = make_run_id(result)  # rebuild the name in case fields changed
    result["meta"]["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")  # save time in UTC, e.g. 2026-10-12T08:30:00+00:00
    result["meta"]["git_commit"] = result["meta"].get("git_commit") or _git_commit()  # code version, unless already set
    if not result.get("environment"):  # environment not filled yet
        result["environment"] = environment_info()  # fill it

    errors = validate(result)  # check everything
    if errors:  # any problem?
        raise ValueError("Invalid result:\n  - " + "\n  - ".join(errors))  # stop and list them; a bad file never gets saved

    path = result_path(result)  # results/<stage>/<run_id>.json
    if path.exists() and not overwrite:  # file already there?
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace)")  # don't silently replace a finished result
    path.parent.mkdir(parents=True, exist_ok=True)  # create results/<stage>/ if missing; else writing fails
    tmp = path.with_suffix(".json.tmp")  # write to a temporary file first
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")  # pretty JSON with 2-space indent
    os.replace(tmp, path)  # rename tmp -> real name in one step; if the session dies mid-write you never get half a JSON
    return path  # where it was saved


def load_results(stage: str | None = None, root: str | os.PathLike | None = None):  # loads result JSONs into one table for analysis
    """Load result files into a flat pandas DataFrame (one row per run).

    Nested fields become dotted columns, e.g. "metrics.accuracy".
    """  # help text
    import pandas as pd  # imported here so training doesn't need pandas
    base = Path(root) if root else RESULTS_DIR  # folder to read from (default: results/)
    pattern = f"{stage}/*.json" if stage else "*/*.json"  # one stage, or all stages
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(base.glob(pattern))]  # read every matching file (sorted = same order every time)
    return pd.json_normalize(rows)  # flatten nested dicts into columns like "metrics.accuracy", one row per run


if __name__ == "__main__":  # only when run directly: python src/results.py results/final/*.json
    bad = 0  # number of invalid files
    for f in sys.argv[1:]:  # every file given on the command line
        errs = validate(json.loads(Path(f).read_text(encoding="utf-8")))  # read and check it
        if errs:  # problems found
            bad += 1  # count it
            print(f"FAIL {f}")  # report the file
            for e in errs:  # each problem
                print(f"     - {e}")  # print it
        else:  # no problems
            print(f"ok   {f}")  # report ok
    sys.exit(1 if bad else 0)  # exit code 1 if any file failed, so scripts can notice
