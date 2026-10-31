#!/usr/bin/env bash
# Online step segmentation: train one model per learning rate, select by validation likelihood.
# Usage: train_observer.sh DATASET
source scripts/common.sh
resolve_dataset "$1"
runs=()
for lr in 1e-4 3e-4 1e-3; do
  python -m provia.training.train_single_actor_observer --manifest "$manifest" --feature-root "$features" \
    --output-dir "$run/observer_lr$lr" --fold 0 --seed 0 --event-learning-rate "$lr" --semantic-learning-rate "$lr" \
    --weight-decay 1e-2 --epochs 40 --early-stopping-epochs 6 --gradient-clip 1.0 --device "$device"
  runs+=(--run "$run/observer_lr$lr")
done
python -m provia.training.select_single_actor_observer "${runs[@]}" --output-dir "$run/observer" --device "$device"
