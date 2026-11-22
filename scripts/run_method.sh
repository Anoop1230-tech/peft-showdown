#!/usr/bin/env bash
# Run all 16 final runs for one method, one python process per run.
#
#   bash scripts/run_method.sh full_ft      (or lora / adapter / prompt_tuning)
#
# Order: all 1k runs, then RTE/MRPC full, then SST-2 full last (about 1 h),
# so most results exist early. Runs whose JSON already exists are skipped,
# so you can rerun this after a crash and it continues where it stopped.
# A failed run is logged and the script moves on to the next one.

set -u  # Safety setting: using a variable that was never set is an ERROR instead of silently becoming empty text. Catches typos in variable names. (No "set -e" on purpose: one failed run must not stop the remaining runs.)

METHOD="${1:-}"  # The method name = the first word after the script name (e.g. "lora"). ":-" gives an empty string if nothing was given, instead of an error from set -u.
case "$METHOD" in  # Check the method name against the allowed values:
  full_ft|lora|adapter|prompt_tuning) ;;  # one of the 4 valid names -> do nothing and continue.
  *) echo "usage: bash scripts/run_method.sh {full_ft|lora|adapter|prompt_tuning}" >&2  # Anything else (missing or misspelled) -> print how to use the script, to stderr (>&2) because it is an error message ...
     exit 2 ;;  # ... and stop with exit code 2 (the usual code for "used wrongly"), BEFORE wasting hours of GPU time on a wrong name.
esac  # End of the check ("case" written backwards).

cd "$(dirname "$0")/.." || exit 1   # repo root, needed for `python -m src.train` -- $0 is this script's path, dirname gives scripts/, "/.." goes one level up to the repo root. "|| exit 1" stops if cd fails, because nothing below would work from the wrong folder.
PYTHON="${PYTHON:-python}"  # Which Python program to use: the PYTHON environment variable if set (e.g. PYTHON=python3), otherwise just "python". Lets you choose a different interpreter without editing the script.

RUNS=()  # Empty list (bash array) that will hold all 16 runs, each written as "task regime seed".
for task in rte mrpc sst2; do  # First, the low-data runs for all 3 tasks. They are quick, so most results exist early even if the session stops later.
  for seed in 1 2 3; do RUNS+=("$task 1k $seed"); done  # Add 3 runs per task (seeds 1, 2, 3), e.g. "rte 1k 1". 3 tasks x 3 seeds = 9 runs.
done  # End of the 1k loop.
for task in rte mrpc; do  # Then the full-data runs for the two SMALL tasks.
  for seed in 1 2 3; do RUNS+=("$task full $seed"); done  # 3 seeds each, e.g. "mrpc full 2". 2 tasks x 3 seeds = 6 runs (15 so far).
done  # End of the full-data loop.
RUNS+=("sst2 full 1")   # one seed only, see issue #25 / report limitations -- SST-2 full is the slowest run (about 1 hour for full FT on a T4), so it goes LAST and has only seed 1. Total = 16 runs.

FAILED=()  # Empty list to remember which runs failed, so they can all be reported at the end.
start=$(date +%s)  # Current time in seconds since 1970, to measure how long the whole batch takes.
i=0  # Run counter, used to print progress like [3/16].
for run in "${RUNS[@]}"; do  # Go through every planned run. "${RUNS[@]}" with quotes keeps each "task regime seed" entry together as one item.
  i=$((i + 1))  # Increase the counter by 1 (bash arithmetic).
  read -r task regime seed <<< "$run"  # Split the entry "rte 1k 1" into three variables: task=rte, regime=1k, seed=1. "<<<" feeds the text to read; -r keeps backslashes as they are.
  echo  # Print an empty line, so runs are visually separated in the log.
  echo "[$i/${#RUNS[@]}] $METHOD $task $regime seed$seed  ($(date '+%H:%M:%S'))"  # Print a progress line, e.g. "[3/16] lora rte 1k seed3  (14:05:12)". ${#RUNS[@]} = number of runs; $(date ...) = current clock time.
  if ! "$PYTHON" -m src.train --method "$METHOD" --task "$task" \
         --regime "$regime" --seed "$seed"; then  # (These 2 lines are ONE command; the "\" joins them, so the comment goes here.) Start ONE training run in a NEW Python process, so peak GPU memory is measured cleanly for each run. "if !" = if the run FAILS (non-zero exit code), do the lines below.
    echo "FAILED: $METHOD $task $regime seed$seed"  # Report the failure in the log immediately ...
    FAILED+=("$task $regime seed$seed")  # ... and remember it for the summary at the end. The loop then CONTINUES with the next run, so one crash does not waste the whole session.
  fi  # End of the failure check.
done  # End of the loop over all runs.

echo  # Empty line before the summary.
echo "done in $(( ($(date +%s) - start) / 60 )) min"  # Print the total time: (now - start) seconds, divided by 60 = minutes.
shopt -s nullglob  # Bash option: a file pattern that matches NOTHING becomes an empty list instead of staying as the literal text "results/final/lora_*.json". Needed for the file count below to be correct.
files=(results/final/"${METHOD}"_*.json)  # List of all result files of this method, e.g. results/final/lora_rte_1k_seed1.json ... (should be 16).
if [ ${#files[@]} -gt 0 ]; then  # If there is at least one result file ...
  echo "${#files[@]} result file(s) for $METHOD:"  # ... say how many ...
  "$PYTHON" src/results.py "${files[@]}"  # ... and validate every one of them with results.py, which prints "ok" or "FAIL" with the problems for each file.
else  # No result files at all ...
  echo "no result files for $METHOD"  # ... say so (probably every run failed).
fi  # End of the validation step.

if [ ${#FAILED[@]} -gt 0 ]; then  # If any run failed ...
  echo "${#FAILED[@]} run(s) failed:"  # ... print how many ...
  printf '  %s\n' "${FAILED[@]}"  # ... and list each failed run on its own indented line, so you know exactly what to rerun.
  exit 1  # Finish with exit code 1 ("something went wrong"), so Kaggle / the caller can see that not everything succeeded.
fi  # End. If nothing failed, the script ends normally with exit code 0.
