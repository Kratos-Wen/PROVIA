#!/usr/bin/env bash
# Causal-TCN, Controlled PREGO and MistSense-RGB-style on the shared features and predicted segments.
# Usage: single_actor.sh DATASET METHOD SEED, METHOD in {causal_tcn, prego, mistsense_rgb}
source scripts/common.sh
resolve_dataset "$1"
method=$2
check_seed "$3"
seed=$3
out=$run/baselines/${method}_seed$seed
readout=native_discrete
[[ "$method" == prego ]] && readout=posterior_continuous
python -m provia.baselines.run_single_actor --method "$method" --manifest "$manifest" --feature-root "$features" \
  --query-feature-root "$test_features" --observer-checkpoint "$run/observer/best.pt" --output-dir "$out" \
  --fold 0 --seed "$seed" --batch-size 32 --device "$device"
python -m provia.evaluation.evaluate_baseline_single_actor --run-dir "$out" --manifest "$manifest" --feature-root "$features" \
  --output "$out/validation_$readout.json" --split validation --readout "$readout" \
  --candidate-score-output "$out/validation_candidates.npz" --device "$device"
python -m provia.evaluation.evaluate_baseline_single_actor --run-dir "$out" --manifest "$manifest" \
  --training-feature-root "$features" --feature-root "$test_features" --frozen-validation "$out/validation_$readout.json" \
  --output "$out/test_$readout.json" --split test --readout "$readout" \
  --candidate-score-output "$out/test_candidates.npz" --device "$device"
