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

set -u
PART="${1:-}"
cd "$(dirname "$0")/.." || exit 1
PYTHON="${PYTHON:-python}"

RUNS=()
case "$PART" in
  lr)
    # Evaluate candidate learning rates on seed 1 (RTE and MRPC full data) to determine best default
    for task in rte mrpc; do
      for lr in 1e-5 3e-5;  do RUNS+=("--method full_ft --task $task --stage lr_search --lr $lr --seed 1"); done
      for lr in 1e-4 3e-4;  do RUNS+=("--method lora --task $task --stage lr_search --lr $lr --seed 1"); done
      for lr in 1e-4 1e-3;  do RUNS+=("--method adapter --task $task --stage lr_search --lr $lr --seed 1"); done
      for lr in 1e-3 3e-2;  do RUNS+=("--method prompt_tuning --task $task --stage lr_search --lr $lr --seed 1"); done
    done
    ;;
  ablation)
    # Ablation grid on RTE full data across 3 seeds to test parameter sensitivity
    ABL=(
      "full_ft frozen_layers 3" "full_ft frozen_layers 6" "full_ft frozen_layers 9"
      "lora lora_rank 1" "lora lora_rank 4" "lora lora_rank 16" "lora lora_rank 64"
      "adapter config seq_bn" "adapter reduction_factor 8" "adapter reduction_factor 64"
      "prompt_tuning num_virtual_tokens 5" "prompt_tuning num_virtual_tokens 50"
      "prompt_tuning num_virtual_tokens 100" "prompt_tuning init TEXT"
    )
    for seed in 1 2 3; do
      for a in "${ABL[@]}"; do
        read -r m name value <<< "$a"
        RUNS+=("--method $m --task rte --stage ablation --ablation-name $name --ablation-value $value --seed $seed")
      done
    done
    ;;
  check)
    # Sanity check: instantiate all models and verify parameter counts match expectations without training
    exec "$PYTHON" - <<'EOF'
from src.models import build_model
from src.results import count_params
for method, kw in [("full_ft", {"frozen_layers": 6}), ("lora", {"r": 1}),
                   ("lora", {"r": 64}), ("adapter", {"config": "seq_bn"}),
                   ("adapter", {"reduction_factor": 64}),
                   ("prompt_tuning", {"num_virtual_tokens": 100}),
                   ("prompt_tuning", {"prompt_tuning_init": "TEXT"})]:
    print(method, kw, count_params(build_model(method, 2, **kw)), flush=True)
EOF
    ;;
  *) echo "usage: bash scripts/run_extra.sh {lr|ablation|check}" >&2; exit 2 ;;
esac

FAILED=()
start=$(date +%s)
i=0
for run in "${RUNS[@]}"; do
  i=$((i + 1))
  echo
  echo "[$i/${#RUNS[@]}] $run  ($(date '+%H:%M:%S'))"
  # shellcheck disable=SC2086
  if ! "$PYTHON" -m src.train $run --regime full; then
    echo "FAILED: $run"
    FAILED+=("$run")
  fi
done

echo
echo "done in $(( ($(date +%s) - start) / 60 )) min"
shopt -s nullglob
dir=$([ "$PART" = lr ] && echo lr_search || echo ablation)
files=(results/"$dir"/*.json)
[ ${#files[@]} -gt 0 ] && "$PYTHON" src/results.py "${files[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "${#FAILED[@]} run(s) failed:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
