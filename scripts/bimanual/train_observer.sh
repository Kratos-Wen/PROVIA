#!/usr/bin/env bash
# Two-hand step segmentation: train one model per learning rate, select by validation likelihood.
source scripts/common.sh
resolve_dataset impact_ego
runs=()
for lr in 1e-4 3e-4 1e-3; do
  python -m provia.training.train_bimanual_observer --manifest "$manifest" --feature-root "$features" \
    --output-dir "$run/observer_lr$lr" --fold 0 --seed 0 --learning-rate "$lr" --weight-decay 1e-2 \
    --epochs 80 --early-stopping-epochs 10 --gradient-clip 1.0 --device "$device"
  runs+=(--run "$run/observer_lr$lr")
done
python -m provia.training.select_bimanual_observer "${runs[@]}" --output-dir "$run/observer" --device "$device"
