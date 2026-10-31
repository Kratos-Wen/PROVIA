#!/usr/bin/env bash
# Train PROVIA on the two-hand stream and evaluate on validation. Usage: train_provia.sh SEED
source scripts/common.sh
resolve_dataset impact_ego
check_seed "$1"
seed=$1
model=$run/model_seed$seed
resume=()
[[ -f "$model/latest.pt" ]] && resume=(--resume)
torchrun --standalone --nproc-per-node="$gpus" -m provia.launch train-bimanual \
  --examples "$run/examples.pt" --prefix-evidence "$run/prefix_evidence" --expert-effect "$run/step_effects.pt" \
  --language-registry "$run/automaton/registry.json" --output-dir "$model" --device "$device" --seed "$seed" "${resume[@]}"
python -m provia.launch evaluate-bimanual --examples "$run/examples.pt" --checkpoint "$model/best.pt" --split validation \
  --output "$model/validation.json" --score-output "$model/validation_scores.npz" \
  --candidate-score-output "$model/validation_candidates.npz" --device "$device"
