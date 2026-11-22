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
"""  # DOCSTRING: the description of this whole file. Python stores it as the module's help text (help(results)); it does not run anything. This file's job: define ONE common format for every result file, so all 122 runs can be loaded into one table.

from __future__ import annotations  # Makes Python treat type hints (like "str | None") as plain text instead of evaluating them. Without it, Python 3.8/3.9 would crash on the "|" syntax in hints. It does not change how the code runs.

import json  # Built-in library to convert Python dicts <-> JSON text. Used to WRITE result files (json.dumps) and READ them back (json.loads). Without it we could not save or load results.
import os  # Built-in library for operating-system tasks. Used here for: reading environment variables (os.environ), checking folders (os.path.isdir) and the atomic file rename (os.replace).
import platform  # Built-in library that reports system info. Used only for platform.python_version() (e.g. "3.11.13") stored in each result for reproducibility.
import subprocess  # Built-in library to run other programs. Used to run "git rev-parse" and get the current commit hash. Without it we could not record which code version made the result.
import sys  # Built-in library for interpreter access. Used at the bottom: sys.argv (file names given on the command line) and sys.exit (return success / failure code).
from datetime import datetime, timezone  # Date and time tools. Used to stamp each result with the exact save time in UTC (universal time, so team members in any time zone get comparable timestamps).
from pathlib import Path  # Object-style file paths. Lets us write RESULTS_DIR / "final" / "x.json" instead of joining strings by hand, and works on both Windows (\) and Linux (/).
from typing import Any  # Type hint meaning "any type of value". Used in hints like dict[str, Any] = a dict with string keys and values of any type. Only for readability; no effect at runtime.

SCHEMA_VERSION = 1  # Version number of the JSON layout. Written into every result file. If the layout is ever changed (new/renamed fields), this number is increased so old and new files can be told apart. validate() rejects any file whose version is different.

METHODS = ("full_ft", "lora", "adapter", "prompt_tuning")  # The 4 allowed method names. A tuple (cannot be changed by accident). Used by validate() to reject typos like "LoRA", and by train.py as the allowed values of --method.
TASKS = ("sst2", "mrpc", "rte")  # The 3 allowed GLUE task names. Any other task name in a result file makes validate() fail.
REGIMES = ("full", "1k")  # The 2 allowed data regimes: "full" = all training data, "1k" = only 1,000 training examples (low-data experiment).
STAGES = ("final", "lr_search", "ablation")  # The 3 kinds of experiment. The stage is also the sub-folder name: results/final/, results/lr_search/, results/ablation/. This keeps the 64 main runs separate from the extra experiments.
SEEDS = (1, 2, 3)  # Allowed random seeds for final runs. Each setting is run with 3 seeds so we can report mean ± standard deviation and see how stable a method is.

REPO_ROOT = Path(__file__).resolve().parent.parent  # Absolute path of the project folder. __file__ = this file (src/results.py); .resolve() makes it absolute; .parent = src/; .parent again = repo root. Doing it this way means paths are correct no matter which folder you run the code from.
RESULTS_DIR = REPO_ROOT / "results"  # The folder where all result JSON files are kept: <repo>/results/. Every save and load uses this, so all team members' results end up in the same place.

MB = 1024 ** 2  # Number of bytes in 1 MiB = 1024 x 1024 = 1,048,576. All "_mb" values (memory, checkpoint size) are divided by this. Using one constant avoids mixing up MB (1,000,000) and MiB (1,048,576).


def new_result(*, method: str, task: str, regime: str, seed: int,  # FUNCTION new_result: creates a new, mostly empty result dictionary for ONE run. The "*" means every argument must be given by name (method="lora", ...), so arguments can never be swapped by mistake (e.g. task and regime).
               stage: str = "final", owner: str,  # stage: which experiment type (default "final"). owner: the team member who ran it (required, no default, so it can't be forgotten).
               hyperparams: dict[str, Any],  # hyperparams: the training settings dict (learning_rate, epochs, batch_size, ...) built in train.py.
               method_config: dict[str, Any] | None = None,  # method_config: the method's own settings, e.g. {"r": 8, "lora_alpha": 16} for LoRA. Optional; None means "no extra settings" (full FT).
               ablation: dict[str, Any] | None = None,  # ablation: for ablation runs, which setting was changed, e.g. {"name": "lora_rank", "value": 4}. None for normal runs.
               model_name: str = "roberta-base") -> dict[str, Any]:  # model_name: the pretrained model used (always roberta-base here). "-> dict[str, Any]" says the function returns a dictionary.
    """Return a result dict with identity fields filled and metrics empty."""  # Short help text for the function.
    result = {  # Build the dictionary. Every result in the project has EXACTLY this structure, which is what lets the analysis load all files into one table.
        "schema_version": SCHEMA_VERSION,  # Layout version (1). Lets future code recognise which format the file uses.
        "run_id": None,  # Unique name of the run, e.g. "lora_rte_1k_seed1". Left as None here and filled in at the end of this function, because it is built FROM the other fields.
        "stage": stage,  # final / lr_search / ablation. Decides which sub-folder the file is saved in.
        "method": method,  # Which of the 4 methods was trained.
        "task": task,  # Which GLUE task (sst2 / mrpc / rte).
        "regime": regime,  # full data or 1k examples.
        "seed": seed,  # Random seed (1, 2 or 3). Same seed = same data split and same starting weights, so the run can be repeated exactly.
        "model_name": model_name,  # Base model ("roberta-base"). Stored so the file is self-explanatory even without the code.
        "ablation": ablation,  # None for normal runs; for ablations, the changed setting, e.g. {"name": "lora_rank", "value": 4}.
        "hyperparams": {  # Training settings section.
            "learning_rate": None,  # Placeholder. Every key is first set to None so that EVERY file has every key, even if a value is missing ...
            "epochs": None,  # ... number of passes over the training data ...
            "batch_size": None,  # ... examples per training step ...
            "max_length": None,  # ... maximum tokens per example ...
            "warmup_ratio": None,  # ... fraction of steps used to warm up the learning rate ...
            "weight_decay": None,  # ... AdamW weight decay ...
            "fp16": None,  # ... whether half-precision (16-bit) math was used ...
            **hyperparams,  # ... then "**" copies in the real values from train.py, overwriting the None placeholders. Any missing value stays None and validate() will catch it.
        },  # End of hyperparams section.
        "method_config": method_config or {},  # Method settings. "or {}" turns None into an empty dict, so code reading this field never has to check for None.
        "data": {"n_train": None, "n_dev": None, "n_eval": None},  # Dataset sizes (training, dev, validation examples). Filled in by train.py after loading the data; useful to confirm e.g. that 1k runs really used 1,000 examples.
        "metrics": {  # Scores section. All None now; filled in after training and evaluation.
            "eval_split": "validation",  # Records that the reported scores come from the official GLUE validation split (GLUE test labels are hidden, so validation is used for reporting).
            "accuracy": None,  # Validation accuracy as a fraction (0.76 = 76%). This is THE main reported number.
            "f1": None,  # Validation F1 score. Only for MRPC (its official GLUE metric); stays None (null in JSON) for SST-2 and RTE.
            "dev_accuracy": None,  # Accuracy on our own 10% dev split, for the best checkpoint. Used for choosing the best epoch and the best learning rate.
            "dev_f1": None,  # Dev-split F1, MRPC only.
            "best_epoch": None,  # Which epoch gave the best dev accuracy (the checkpoint that was kept). Shows whether a method was still improving at the end (e.g. prompt tuning).
        },  # End of metrics section.
        "efficiency": {  # Cost section: the "how expensive is this method" numbers.
            "trainable_params": None,  # Number of weights that were actually trained (requires_grad=True), e.g. 887,042 for LoRA.
            "total_params": None,  # Number of all weights in the model (frozen + trainable), about 125 million.
            "trainable_pct": None,  # trainable / total x 100, e.g. 0.71 (%). Makes methods easy to compare.
            "peak_gpu_mem_mb": None,  # Highest GPU memory used during training, in MiB (e.g. 3264 for full FT vs ~1900 for PEFT methods).
            "train_time_s": None,  # Wall-clock training time in seconds (only trainer.train(), not data loading or evaluation).
            "checkpoint_size_mb": None,  # Disk size of the saved trainable weights in MiB (475.5 for full FT vs 2-9 for PEFT). Shows the storage cost per task.
        },  # End of efficiency section.
        "environment": {},  # Software/hardware info (GPU name, library versions). Left empty here; save_result() fills it automatically by calling environment_info().
        "meta": {  # Bookkeeping section: who, when, which code version.
            "owner": owner,  # Team member responsible for this run (e.g. "praveen"). Required by validate().
            "timestamp_utc": None,  # Exact save time in UTC. Filled by save_result().
            "git_commit": None,  # Git hash of the code that produced the result. Filled by save_result(); lets anyone check out exactly that code version later.
            "notes": "",  # Free-text notes (from --notes in train.py). Empty string by default.
        },  # End of meta section.
    }  # End of the result dictionary.
    result["run_id"] = make_run_id(result)  # Now that all identity fields exist, build the unique run name from them (e.g. "lora_rte_1k_seed1") and store it.
    return result  # Give the dictionary back to train.py, which fills in data sizes, metrics and efficiency during the run.


def make_run_id(result: dict[str, Any]) -> str:  # FUNCTION make_run_id: builds the unique, human-readable name of a run. The same name is used as the JSON file name, so two different runs can never overwrite each other's file.
    """Unique, filesystem-safe id. Also used as the file name (without .json).

    final:     lora_rte_1k_seed1
    lr_search: lora_rte_full_lr0.0003_seed1
    ablation:  lora_rte_full_lora_rank-4_seed1
    """  # Help text with one example name for each stage.
    base = f"{result['method']}_{result['task']}_{result['regime']}"  # Start the name with method, task and regime joined by "_", e.g. "lora" + "rte" + "1k" -> "lora_rte_1k".
    stage = result["stage"]  # Read the stage, because LR-search and ablation runs need extra information in the name.
    if stage == "lr_search":  # LR-search runs: same method/task/regime/seed but DIFFERENT learning rates ...
        lr = result["hyperparams"]["learning_rate"]  # ... so read the learning rate of this run ...
        base += f"_lr{lr:g}"  # ... and add it to the name. ":g" prints the number in short form (0.0003, not 0.000300000). Without this, the 3 LR runs would get the same name and overwrite each other.
    elif stage == "ablation":  # Ablation runs: differ by the one setting that was changed ...
        abl = result["ablation"] or {}  # ... read that info; "or {}" avoids a crash if it is None (validate() reports that problem separately).
        value = str(abl.get("value")).replace("/", "-").replace(" ", "")  # Turn the changed value into text and remove "/" (would create a sub-folder) and spaces (awkward in file names). E.g. 4 -> "4", "seq_bn" -> "seq_bn".
        base += f"_{abl.get('name')}-{value}"  # Add "_<setting>-<value>" to the name, e.g. "_lora_rank-4" -> "lora_rte_full_lora_rank-4".
    return f"{base}_seed{result['seed']}"  # Finally add the seed, e.g. "lora_rte_1k_seed1". Seed must be in the name because the 3 seeds of one setting are 3 separate files.


def result_path(result: dict[str, Any]) -> Path:  # FUNCTION result_path: works out WHERE a result file is stored on disk.
    """results/<stage>/<run_id>.json"""  # Help text: the path pattern.
    return RESULTS_DIR / result["stage"] / f"{result['run_id']}.json"  # Join results folder + stage sub-folder + run name + ".json", e.g. results/final/lora_rte_1k_seed1.json. train.py uses this to check whether a run is already done (and skip it).


def count_params(model) -> dict[str, Any]:  # FUNCTION count_params: counts how many weights the model has and how many of them are trained. This is the "trainable parameters" efficiency metric.
    """Trainable / total parameter counts, classifier head included.

    total_params is every parameter in the model as built (frozen base +
    any added LoRA / adapter / prompt parameters + head).
    """  # Help text: the head is counted for every method, so the numbers are comparable.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Go through every weight tensor of the model; keep only those with requires_grad=True (the ones the optimizer updates); p.numel() = number of values in that tensor; sum them all. E.g. 607,490 for prompt tuning.
    total = sum(p.numel() for p in model.parameters())  # Same, but for ALL weight tensors, frozen or not. About 125 million for every method (PEFT methods add a little on top of RoBERTa).
    return {  # Return the three numbers as a dict, so train.py can merge them straight into result["efficiency"] with .update().
        "trainable_params": int(trainable),  # int() converts to a plain Python integer so json.dumps can write it.
        "total_params": int(total),  # Total weight count.
        "trainable_pct": round(100.0 * trainable / total, 4),  # Percentage trained, rounded to 4 decimals, e.g. 0.4873. 100.0 (a float) makes sure we get a decimal result.
    }  # End of returned dict.


def dir_size_mb(path: str | os.PathLike) -> float:  # FUNCTION dir_size_mb: measures how big a saved checkpoint is on disk, in MiB. This is the "checkpoint size" efficiency metric.
    """Size of a saved checkpoint directory (or single file) in MiB."""  # Help text.
    p = Path(path)  # Convert the input (text or path object) into a Path object so we can use .is_file(), .stat() and .rglob().
    if p.is_file():  # Case 1: the path is a single file ...
        return round(p.stat().st_size / MB, 3)  # ... return its size: st_size is in bytes, divide by MB to get MiB, round to 3 decimals.
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())  # Case 2: a folder. rglob("*") lists EVERYTHING inside it, including sub-folders; keep only files and add up their sizes in bytes.
    return round(total / MB, 3)  # Convert the total to MiB, round to 3 decimals, e.g. 3.384 for a LoRA checkpoint.


def peak_gpu_mem_mb() -> float | None:  # FUNCTION peak_gpu_mem_mb: returns the highest amount of GPU memory PyTorch used since the last reset. This is the "peak memory" efficiency metric.
    """torch.cuda.max_memory_allocated() in MiB, or None without a GPU.

    Call torch.cuda.reset_peak_memory_stats() right before training.
    """  # Help text: train.py resets the counter just before training, so the peak covers training only.
    import torch  # Import PyTorch HERE (inside the function), not at the top of the file. That way the analysis machine can import results.py to load/validate files even if torch is not installed.
    if not torch.cuda.is_available():  # No GPU (e.g. running on a laptop CPU) ...
        return None  # ... there is no GPU memory to measure, so return None instead of crashing. validate() allows this field to be None.
    return round(torch.cuda.max_memory_allocated() / MB, 1)  # Peak bytes allocated by PyTorch tensors -> MiB, rounded to 1 decimal, e.g. 1923.4. (nvidia-smi shows more because it also counts the CUDA context and cache.)


def _version(pkg: str) -> str | None:  # HELPER _version: returns the installed version of a Python package. The leading "_" means "internal helper, not meant to be used from other files".
    try:  # Wrap in try because the package may not be installed (e.g. "adapters" on an analysis machine).
        from importlib.metadata import version  # Standard-library function that reads a package's installed version from its metadata.
        return version(pkg)  # E.g. version("transformers") -> "4.57.6".
    except Exception:  # Package not installed, or any other error ...
        return None  # ... return None. Recording a missing version must NEVER crash a training run that has already finished.


def _detect_platform() -> str:  # HELPER _detect_platform: guesses which machine type the run happened on, stored in environment.platform.
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle"):  # Kaggle sets this environment variable and has a /kaggle folder. Checked FIRST because Kaggle's image also sets some COLAB_* variables, which would otherwise make us say "colab" wrongly.
        return "kaggle"  # Running on Kaggle (where all our real runs happened).
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:  # Environment variables that only Google Colab sets.
        return "colab"  # Running on Google Colab.
    return "local"  # Neither -> assume a personal computer.


def environment_info() -> dict[str, Any]:  # FUNCTION environment_info: collects hardware and software details, so anyone can see what a result was produced with (and reproduce it). Times are only comparable on the same GPU.
    """GPU name and library versions. Times are only comparable on the same GPU."""  # Help text.
    gpu = None  # Start with "no GPU"; changed below if one is found.
    try:  # torch might not be installed (analysis machine) ...
        import torch  # ... so import it inside try, here rather than at the top of the file.
        if torch.cuda.is_available():  # A CUDA GPU is present ...
            gpu = torch.cuda.get_device_name(0)  # ... read the name of GPU 0, e.g. "Tesla T4". Important because training time depends heavily on the GPU type.
    except ImportError:  # torch is not installed ...
        pass  # ... keep gpu = None and carry on.
    return {  # Return all environment details as one dict; save_result() stores it in result["environment"].
        "gpu_name": gpu,  # GPU model or None.
        "platform": _detect_platform(),  # "kaggle" / "colab" / "local".
        "python": platform.python_version(),  # Python version, e.g. "3.11.13".
        "torch": _version("torch"),  # PyTorch version (not pinned in requirements.txt, because Kaggle provides its own CUDA build, so it is recorded here instead).
        "transformers": _version("transformers"),  # Hugging Face Transformers version (model + Trainer).
        "peft": _version("peft"),  # Hugging Face PEFT version (LoRA + prompt tuning).
        "adapters": _version("adapters"),  # AdapterHub version (adapters method).
        "datasets": _version("datasets"),  # Hugging Face Datasets version (GLUE loading).
    }  # End of environment dict.


def _git_commit() -> str | None:  # HELPER _git_commit: returns the short git hash of the code version that produced the result, so a result can always be traced back to the exact code.
    if os.environ.get("PEFT_GIT_COMMIT"):  # On Kaggle there is no .git folder (kaggle_push.py uploads plain files), so kaggle_push.py passes the hash in this environment variable instead ...
        return os.environ["PEFT_GIT_COMMIT"]  # ... use that value, e.g. "04524b4" or "04524b4-dirty" (dirty = uncommitted changes were uploaded).
    try:  # Otherwise ask git directly; wrapped in try because git may be missing.
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],  # Run the command "git rev-parse --short HEAD", which prints the short hash of the current commit.
                             cwd=REPO_ROOT, capture_output=True, text=True,  # cwd: run it inside the repo folder; capture_output: catch what it prints instead of showing it; text: give us a string, not bytes.
                             timeout=5)  # Give up after 5 seconds, so a hanging git can never block saving a finished result.
        return out.stdout.strip() or None  # The printed hash without the trailing newline, e.g. "bec25ec". If git printed nothing (not a repo), "or None" returns None.
    except Exception:  # git not installed, timeout, or any other error ...
        return None  # ... record None rather than crash.


_REQUIRED_NUMBERS = {  # Table of fields that MUST contain a number in every valid result: section name -> keys in that section. validate() loops over it. A file missing any of these is rejected.
    "metrics": ("accuracy", "dev_accuracy"),  # Every run must have validation and dev accuracy (F1 is checked separately, only for MRPC).
    "efficiency": ("trainable_params", "total_params", "trainable_pct",  # Parameter counts are required ...
                   "train_time_s", "checkpoint_size_mb"),  # ... and time and checkpoint size. peak_gpu_mem_mb is NOT required, because a CPU-only run has no GPU memory.
    "hyperparams": ("learning_rate", "epochs", "batch_size", "max_length"),  # The core training settings must be recorded, otherwise the result can't be interpreted.
    "data": ("n_train", "n_eval"),  # Training and evaluation set sizes must be recorded.
}  # End of _REQUIRED_NUMBERS.


def validate(result: dict[str, Any]) -> list[str]:  # FUNCTION validate: checks one result against all the format rules and returns a LIST of every problem found. An empty list means the result is valid. Used before saving and by "python src/results.py files...".
    """Return a list of problems. Empty list means the result is valid."""  # Help text.
    errors: list[str] = []  # Start with an empty list; each failed rule adds one message to it.

    def check(cond: bool, msg: str) -> None:  # Small helper inside validate: "if this condition is false, record this message".
        if not cond:  # The rule is broken ...
            errors.append(msg)  # ... add the message. We do NOT stop at the first problem, so one run of validate() shows ALL problems at once.

    check(result.get("schema_version") == SCHEMA_VERSION,  # Rule: the file must use layout version 1. .get() returns None instead of crashing if the key is missing.
          f"schema_version must be {SCHEMA_VERSION}")  # Message if broken.
    check(result.get("method") in METHODS, f"method must be one of {METHODS}")  # Rule: method must be one of the 4 allowed names (catches typos like "lora_ft").
    check(result.get("task") in TASKS, f"task must be one of {TASKS}")  # Rule: task must be sst2, mrpc or rte.
    check(result.get("regime") in REGIMES, f"regime must be one of {REGIMES}")  # Rule: regime must be full or 1k.
    check(result.get("stage") in STAGES, f"stage must be one of {STAGES}")  # Rule: stage must be final, lr_search or ablation.
    check(isinstance(result.get("seed"), int), "seed must be an int")  # Rule: seed must be a whole number (not "1" as text, not 1.0).
    if result.get("stage") == "final":  # Extra rule only for final runs:
        check(result.get("seed") in SEEDS, f"final runs must use seeds {SEEDS}")  # final runs must use seed 1, 2 or 3, so every method is averaged over the same seeds.
    if result.get("stage") == "ablation":  # Extra rule only for ablation runs:
        abl = result.get("ablation")  # read the ablation info ...
        check(isinstance(abl, dict) and "name" in abl and "value" in abl,  # ... it must be a dict with both "name" and "value", otherwise we can't tell which setting was changed ...
              "ablation runs need ablation = {'name': ..., 'value': ...}")  # ... message if broken.
    check(result.get("meta", {}).get("owner"), "meta.owner is required")  # Rule: an owner must be set (non-empty). .get("meta", {}) avoids a crash if "meta" is missing entirely.

    for section, keys in _REQUIRED_NUMBERS.items():  # Go through each section listed in _REQUIRED_NUMBERS (metrics, efficiency, hyperparams, data) ...
        block = result.get(section, {})  # ... get that section of the result ({} if missing, so the checks below report the missing keys instead of crashing) ...
        for k in keys:  # ... and each required key in it:
            v = block.get(k)  # read its value (None if missing) ...
            check(isinstance(v, (int, float)) and not isinstance(v, bool),  # ... it must be an int or float. bool is excluded on purpose: in Python True/False count as int (True == 1), so without this a wrong "True" would pass.
                  f"{section}.{k} must be a number, got {v!r}")  # Message shows the bad value, e.g. "metrics.accuracy must be a number, got None".

    m = result.get("metrics", {})  # Get the metrics section for the score checks below.
    for k in ("accuracy", "dev_accuracy", "f1", "dev_f1"):  # Go through all four score fields ...
        v = m.get(k)  # ... read the value ...
        if isinstance(v, (int, float)):  # ... only check it if it is a number (f1 is legitimately None for SST-2 / RTE) ...
            check(0.0 <= v <= 1.0, f"metrics.{k} must be a fraction in [0, 1]")  # ... it must be between 0 and 1. This catches the common mistake of saving 76.1 (percent) instead of 0.761 (fraction).
    if result.get("task") == "mrpc":  # Extra rule for MRPC:
        check(isinstance(m.get("f1"), (int, float)), "MRPC runs must report metrics.f1")  # F1 is MRPC's official GLUE metric, so an MRPC result without F1 is incomplete.

    if result.get("run_id") is not None and not errors:  # Last rule, only checked when everything else passed (otherwise make_run_id might crash on bad fields):
        check(result["run_id"] == make_run_id(result),  # the stored run name must match the name rebuilt from the fields. This catches hand-edited or renamed files whose name no longer matches their content.
              f"run_id {result['run_id']!r} does not match fields "  # Message part 1: the wrong name ...
              f"(expected {make_run_id(result)!r})")  # ... part 2: the name it should have.
    return errors  # Return all problems found (empty list = valid).


def save_result(result: dict[str, Any], overwrite: bool = False) -> Path:  # FUNCTION save_result: the ONLY way results should be written. Fills in the last automatic fields, validates, and writes results/<stage>/<run_id>.json safely. Returns the path written.
    """Validate and write results/<stage>/<run_id>.json atomically."""  # Help text.
    result["run_id"] = make_run_id(result)  # Rebuild the run name, in case any identity field changed after new_result() (so the name always matches the content).
    result["meta"]["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")  # Record the current time in UTC as text, to the second, e.g. "2026-10-12T08:30:00+00:00".
    result["meta"]["git_commit"] = result["meta"].get("git_commit") or _git_commit()  # Record the code version: keep a hash if one was already set, otherwise look it up with _git_commit().
    if not result.get("environment"):  # If the environment section is still empty ...
        result["environment"] = environment_info()  # ... fill it with GPU name and library versions automatically, so nobody has to remember to do it.

    errors = validate(result)  # Run every format check on the finished result.
    if errors:  # If there is at least one problem ...
        raise ValueError("Invalid result:\n  - " + "\n  - ".join(errors))  # ... stop with an error listing every problem, one per line. This guarantees that a broken result file is NEVER written to disk.

    path = result_path(result)  # Work out the destination, e.g. results/final/lora_rte_1k_seed1.json.
    if path.exists() and not overwrite:  # If a file with that name already exists and overwriting was not requested ...
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace)")  # ... refuse, so a finished result is never replaced by accident.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create the folder (e.g. results/ablation/) if it doesn't exist yet. parents=True also creates results/ itself; exist_ok=True means no error if it already exists.
    tmp = path.with_suffix(".json.tmp")  # Name of a temporary file next to the real one: "...seed1.json.tmp". We write here FIRST ...
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")  # ... convert the dict to nicely indented JSON text (2 spaces, easy to read and diff in git) and write it. utf-8 keeps any special characters.
    os.replace(tmp, path)  # ... then rename the temp file to the real name in ONE step (atomic). If the machine dies mid-write, only the .tmp file is damaged; the real file is either complete or absent, never half-written.
    return path  # Return where the file was saved (train.py prints it).


def load_results(stage: str | None = None, root: str | os.PathLike | None = None):  # FUNCTION load_results: reads many result files into ONE pandas table for analysis (used by analysis.ipynb and extra_analysis.py).
    """Load result files into a flat pandas DataFrame (one row per run).

    Nested fields become dotted columns, e.g. "metrics.accuracy".
    """  # Help text.
    import pandas as pd  # Import pandas here (not at the top), so training machines don't need pandas just to save results.
    base = Path(root) if root else RESULTS_DIR  # Folder to read from: the given root, or by default <repo>/results.
    pattern = f"{stage}/*.json" if stage else "*/*.json"  # Which files: one stage only (e.g. "final/*.json") or, if no stage is given, all stages ("*/*.json").
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(base.glob(pattern))]  # Find all matching files, sort them (so the order is the same every time), read each one and convert its JSON text into a Python dict. Result: a list of dicts, one per run.
    return pd.json_normalize(rows)  # Turn the list of nested dicts into a flat table: one row per run, nested keys become columns like "metrics.accuracy" and "efficiency.peak_gpu_mem_mb". Easy to group, average and plot.


if __name__ == "__main__":  # This block runs ONLY when the file is executed directly (python src/results.py ...), not when another file imports it. It turns results.py into a small validation tool.
    bad = 0  # Counter of invalid files.
    for f in sys.argv[1:]:  # sys.argv = the command-line words; [1:] skips the script name, leaving the file names, e.g. results/final/*.json expands to 64 file names.
        errs = validate(json.loads(Path(f).read_text(encoding="utf-8")))  # Read one file, convert JSON to dict, and run all the checks on it.
        if errs:  # The file has problems ...
            bad += 1  # ... count it ...
            print(f"FAIL {f}")  # ... print FAIL and the file name ...
            for e in errs:  # ... then each problem ...
                print(f"     - {e}")  # ... indented on its own line.
        else:  # No problems ...
            print(f"ok   {f}")  # ... print "ok" and the file name (this is the "ok" the README tells you to look for).
    sys.exit(1 if bad else 0)  # End the program with exit code 1 if ANY file failed, else 0. Shell scripts (run_method.sh) and CI can use this code to detect failures automatically.
