#!/usr/bin/env bash
# LR search and ablation runs (PROJECT_PLAN.md steps 5 and 7).
#
#   bash scripts/run_extra.sh lr         # LR search, 16 runs
#   bash scripts/run_extra.sh ablation   # ablations, 42 runs
#   bash scripts/run_extra.sh check      # only build every ablation model, no training
#
# LR search: 3 LRs per method, seed 1, RTE and MRPC full data, picked on our
# 10% dev split (metrics.dev_accuracy). The LR used for the final runs is one
# of the 3; its seed-1 result is already in results/final/, so only the other
# two are run here.
#
# Ablations: RTE full data, seeds 1-3, one setting changed at a time. The
# default setting of each method is the final run, so it is not repeated.
#
# Runs whose JSON already exists are skipped; failed runs are logged.

set -u  # Safety setting: using a variable that was never set is an ERROR instead of silently becoming empty. Catches typos. (No "set -e": one failed run must not stop the others.)
PART="${1:-}"  # Which part to run = the first word after the script name: "lr", "ablation" or "check". ":-" gives an empty string if nothing was given.
cd "$(dirname "$0")/.." || exit 1  # Go to the repo root (this script is in scripts/, so one level up). Needed because "python -m src.train" only works from the repo root. Stop if cd fails.
PYTHON="${PYTHON:-python}"  # Which Python to use: the PYTHON environment variable if set, otherwise "python".

RUNS=()  # Empty list that will hold the command-line options for every run to do.
case "$PART" in  # Choose what to build depending on the part asked for:
  lr)  # PART = "lr": the learning-rate search (16 runs).
    for task in rte mrpc; do  # On both small tasks, RTE and MRPC (SST-2 left out: one full run takes about an hour).
      for lr in 1e-5 3e-5;  do RUNS+=("--method full_ft --task $task --stage lr_search --lr $lr --seed 1"); done  # Full FT: the 2 LRs around its final LR 2e-5 (2e-5 itself is already in results/final, so not repeated).
      for lr in 1e-4 3e-4;  do RUNS+=("--method lora --task $task --stage lr_search --lr $lr --seed 1"); done  # LoRA: 2 LRs below its final LR 5e-4.
      for lr in 1e-4 1e-3;  do RUNS+=("--method adapter --task $task --stage lr_search --lr $lr --seed 1"); done  # Adapters: the 2 LRs around its final LR 3e-4.
      for lr in 1e-3 3e-2;  do RUNS+=("--method prompt_tuning --task $task --stage lr_search --lr $lr --seed 1"); done  # Prompt tuning: the 2 LRs around its final LR 1e-2. 2 tasks x 4 methods x 2 LRs = 16 runs, all seed 1.
    done  # End of the task loop.
    ;;  # End of the "lr" case.
  ablation)  # PART = "ablation": change one setting at a time (42 runs).
    ABL=(  # List of the 14 ablation settings, each written as "method setting value":
      "full_ft frozen_layers 3" "full_ft frozen_layers 6" "full_ft frozen_layers 9"  # Full FT: freeze the bottom 3, 6 or 9 layers (default 0).
      "lora lora_rank 1" "lora lora_rank 4" "lora lora_rank 16" "lora lora_rank 64"  # LoRA: rank 1, 4, 16, 64 (default 8).
      "adapter config seq_bn" "adapter reduction_factor 8" "adapter reduction_factor 64"  # Adapters: Pfeiffer type (default Houlsby), reduction factor 8 or 64 (default 16).
      "prompt_tuning num_virtual_tokens 5" "prompt_tuning num_virtual_tokens 50"  # Prompt tuning: 5 or 50 virtual tokens (default 20) ...
      "prompt_tuning num_virtual_tokens 100" "prompt_tuning init TEXT"  # ... 100 tokens, and TEXT initialisation (default RANDOM).
    )  # End of the list. The default settings are NOT in it: their results come from the final runs.
    for seed in 1 2 3; do  # Every setting with 3 seeds, so mean ± std can be reported. Seed in the OUTER loop: if the session stops early, every setting already has at least one seed.
      for a in "${ABL[@]}"; do  # For each of the 14 settings:
        read -r m name value <<< "$a"  # split "lora lora_rank 4" into m=lora, name=lora_rank, value=4.
        RUNS+=("--method $m --task rte --stage ablation --ablation-name $name --ablation-value $value --seed $seed")  # Add the run's options. All ablations are on RTE. 14 settings x 3 seeds = 42 runs.
      done  # End of the settings loop.
    done  # End of the seed loop.
    ;;  # End of the "ablation" case.
  check)  # PART = "check": no training, just BUILD every ablation model once and print its parameter counts. A quick test that all ablation settings work before spending GPU hours.
    exec "$PYTHON" - <<'EOF'  # Replace this script with Python, reading the Python code from the lines below up to "EOF". The quotes around EOF mean bash does not change anything inside. "exec" means the script ends when Python ends.
from src.models import build_model  # Our function that builds a model for a method (src/models.py).
from src.results import count_params  # Our function that counts trainable / total parameters.
for method, kw in [("full_ft", {"frozen_layers": 6}), ("lora", {"r": 1}),  # Loop over a sample of ablation settings: full FT with 6 frozen layers, LoRA rank 1 ...
                   ("lora", {"r": 64}), ("adapter", {"config": "seq_bn"}),  # ... LoRA rank 64, Pfeiffer adapters ...
                   ("adapter", {"reduction_factor": 64}),  # ... adapters with reduction factor 64 ...
                   ("prompt_tuning", {"num_virtual_tokens": 100}),  # ... prompt tuning with 100 tokens ...
                   ("prompt_tuning", {"prompt_tuning_init": "TEXT"})]:  # ... and prompt tuning with TEXT init.
    print(method, kw, count_params(build_model(method, 2, **kw)), flush=True)  # Build each model with 2 classes and print its parameter counts. flush=True prints immediately, so you see progress even though building takes a few seconds.
EOF
    ;;  # End of the "check" case.
  *) echo "usage: bash scripts/run_extra.sh {lr|ablation|check}" >&2; exit 2 ;;  # Anything else (missing or misspelled): print how to use the script (to stderr) and stop with exit code 2.
esac  # End of the case block.

FAILED=()  # Empty list to remember which runs failed, for the summary at the end.
start=$(date +%s)  # Start time in seconds, to report the total duration.
i=0  # Run counter for the progress display.
for run in "${RUNS[@]}"; do  # Go through every planned run (each entry is the run's option text).
  i=$((i + 1))  # Increase the counter.
  echo  # Empty line to separate runs in the log.
  echo "[$i/${#RUNS[@]}] $run  ($(date '+%H:%M:%S'))"  # Progress line, e.g. "[5/42] --method lora --task rte ... (15:20:03)".
  # shellcheck disable=SC2086
  if ! "$PYTHON" -m src.train $run --regime full; then  # Start ONE training run in a new Python process. $run is deliberately NOT in quotes, so bash splits it into separate options (--method, lora, ...); the shellcheck line above tells the checker this is intended. All extra runs use full data. "if !" = if the run fails:
    echo "FAILED: $run"  # report it in the log ...
    FAILED+=("$run")  # ... and remember it. The loop continues with the next run.
  fi  # End of the failure check.
done  # End of the loop over all runs.

echo  # Empty line before the summary.
echo "done in $(( ($(date +%s) - start) / 60 )) min"  # Total time in minutes.
shopt -s nullglob  # A file pattern that matches nothing becomes an empty list (instead of the literal pattern text), so the count below is correct.
dir=$([ "$PART" = lr ] && echo lr_search || echo ablation)  # Results folder to check: "lr_search" for the LR part, otherwise "ablation".
files=(results/"$dir"/*.json)  # List of all result files in that folder.
[ ${#files[@]} -gt 0 ] && "$PYTHON" src/results.py "${files[@]}"  # If there are any, validate them all with results.py (prints "ok" or "FAIL" per file).

if [ ${#FAILED[@]} -gt 0 ]; then  # If any run failed ...
  echo "${#FAILED[@]} run(s) failed:"  # ... print how many ...
  printf '  %s\n' "${FAILED[@]}"  # ... list each failed run on its own line ...
  exit 1  # ... and finish with exit code 1, so the caller (Kaggle) can see something went wrong.
fi  # End. Otherwise the script ends normally with exit code 0.
