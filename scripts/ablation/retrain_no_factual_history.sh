#!/usr/bin/env bash
# The retrained no-factual-history row: the shared-successor variant trained without factual history.
# Usage: retrain_no_factual_history.sh DATASET SEED
source scripts/common.sh
resolve_dataset "$1"
check_seed "$2"
seed=$2
model=$run/ablation/retrained_no_factual_history/model_seed$seed
resume=()
[[ -f "$model/latest.pt" ]] && resume=(--resume)
torchrun --standalone --nproc-per-node="$gpus" -m provia.launch train-single-shared-successor --no-factual-history \
  --examples "$run/examples.pt" --prefix-evidence "$run/prefix_evidence" \
  --expert-effect-registry "$run/step_effects/registry.json" --language-registry "$run/automaton/registry.json" \
  --output-dir "$model" --device "$device" --seed "$seed" "${resume[@]}"
python -m provia.launch evaluate-single --examples "$run/examples.pt" --checkpoint "$model/best.pt" --split validation \
  --output "$model/validation.json" --device "$device"
python -m provia.launch evaluate-single --examples "$run/examples.pt" --checkpoint "$model/best.pt" \
  --query-examples "$run/test_query.pt" --query-prefix-evidence "$run/test_prefix_evidence" \
  --frozen-validation "$model/validation.json" --split test --output "$model/test.json" --device "$device"
