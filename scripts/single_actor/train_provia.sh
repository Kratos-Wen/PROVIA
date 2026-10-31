#!/usr/bin/env bash
# Train PROVIA and evaluate on validation. Usage: train_provia.sh DATASET SEED
source scripts/common.sh
resolve_dataset "$1"
check_seed "$2"
seed=$2
model=$run/model_seed$seed
resume=()
[[ -f "$model/latest.pt" ]] && resume=(--resume)
torchrun --standalone --nproc-per-node="$gpus" -m provia.launch train-single \
  --examples "$run/examples.pt" --prefix-evidence "$run/prefix_evidence" \
  --expert-effect-registry "$run/step_effects/registry.json" --language-registry "$run/automaton/registry.json" \
  --output-dir "$model" --device "$device" --seed "$seed" "${resume[@]}"
python -m provia.launch evaluate-single --examples "$run/examples.pt" --checkpoint "$model/best.pt" --split validation \
  --output "$model/validation.json" --score-output "$model/validation_scores.npz" \
  --candidate-score-output "$model/validation_candidates.npz" --device "$device"
