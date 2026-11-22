#!/usr/bin/env bash
# Run all 16 final runs for one method, one python process per run.
#
#   bash scripts/run_method.sh full_ft      (or lora / adapter / prompt_tuning)
#
# Order: all 1k runs, then RTE/MRPC full, then SST-2 full last (about 1 h),
# so most results exist early. Runs whose JSON already exists are skipped,
# so you can rerun this after a crash and it continues where it stopped.
# A failed run is logged and the script moves on to the next one.

set -u

METHOD="${1:-}"
case "$METHOD" in
  full_ft|lora|adapter|prompt_tuning) ;;
  *) echo "usage: bash scripts/run_method.sh {full_ft|lora|adapter|prompt_tuning}" >&2
     exit 2 ;;
esac

cd "$(dirname "$0")/.." || exit 1   # repo root, needed for `python -m src.train`
PYTHON="${PYTHON:-python}"

# Order runs: 1k subsets first for quick iteration, then RTE/MRPC full data,
# and SST-2 full last (which takes ~45-60 mins on a single T4 GPU).
RUNS=()
for task in rte mrpc sst2; do
  for seed in 1 2 3; do RUNS+=("$task 1k $seed"); done
done
for task in rte mrpc; do
  for seed in 1 2 3; do RUNS+=("$task full $seed"); done
done
# SST-2 full data is evaluated on seed 1 only due to computational budget
# and because 67k examples has significantly lower variance than smaller datasets.
RUNS+=("sst2 full 1")

FAILED=()
start=$(date +%s)
i=0
for run in "${RUNS[@]}"; do
  i=$((i + 1))
  read -r task regime seed <<< "$run"
  echo
  echo "[$i/${#RUNS[@]}] $METHOD $task $regime seed$seed  ($(date '+%H:%M:%S'))"
  # Log failed runs but continue execution so remaining experiments finish
  if ! "$PYTHON" -m src.train --method "$METHOD" --task "$task" \
         --regime "$regime" --seed "$seed"; then
    echo "FAILED: $METHOD $task $regime seed$seed"
    FAILED+=("$task $regime seed$seed")
  fi
done

echo
echo "done in $(( ($(date +%s) - start) / 60 )) min"
# Validate all generated JSON result files against the schema to ensure integrity
shopt -s nullglob
files=(results/final/"${METHOD}"_*.json)
if [ ${#files[@]} -gt 0 ]; then
  echo "${#files[@]} result file(s) for $METHOD:"
  "$PYTHON" src/results.py "${files[@]}"
else
  echo "no result files for $METHOD"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "${#FAILED[@]} run(s) failed:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
