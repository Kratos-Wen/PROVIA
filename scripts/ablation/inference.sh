#!/usr/bin/env bash
# Ablations of the trained model, validation then test. Usage: inference.sh DATASET SEED [CHECKPOINT]
# A given CHECKPOINT, such as a released provia_seedN.pt, is read with the inputs built by evaluate_checkpoints.sh.
source scripts/common.sh
resolve_dataset "$1"
check_seed "$2"
seed=$2
checkpoint=${3:-$run/model_seed$seed/best.pt}
rebuilt=()
[[ -n "${3:-}" ]] && rebuilt=(--rebuilt-inputs --prefix-evidence "$run/prefix_evidence")
for name in no_factual_history no_execution_mode_persistence no_procedure_frontier no_causal_decision_clock; do
  out=$run/ablation/$name/model_seed$seed
  mkdir -p "$out"
  python -m provia.launch evaluate-single --examples "$run/examples.pt" --checkpoint "$checkpoint" \
    --inference-ablation "$name" --split validation --output "$out/validation.json" --device "$device" "${rebuilt[@]}"
  python -m provia.launch evaluate-single --examples "$run/examples.pt" --checkpoint "$checkpoint" \
    --query-examples "$run/test_query.pt" --query-prefix-evidence "$run/test_prefix_evidence" \
    --inference-ablation "$name" --frozen-validation "$out/validation.json" --split test --output "$out/test.json" --device "$device" "${rebuilt[@]}"
done
