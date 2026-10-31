#!/usr/bin/env bash
# Baselines on the two-hand stream, scored on the same open-step candidates as PROVIA.
# Usage: bimanual.sh METHOD SEED, METHOD in {causal_tcn, prego, mistsense_rgb}
source scripts/common.sh
resolve_dataset impact_ego
method=$1
check_seed "$2"
seed=$2
out=$run/baselines/${method}_seed$seed
readout=native_discrete
[[ "$method" == prego ]] && readout=posterior_continuous
python -m provia.baselines.run_bimanual --method "$method" --manifest "$manifest" --feature-root "$features" \
  --query-feature-root "$test_features" --observer-checkpoint "$run/observer/best.pt" --output-dir "$out" \
  --fold 0 --seed "$seed" --batch-size 32 --device "$device"
python -m provia.evaluation.evaluate_baseline_bimanual --run-dir "$out" --examples "$run/examples.pt" --manifest "$manifest" \
  --feature-root "$features" --output "$out/validation_$readout.json" \
  --split validation --readout "$readout" --candidate-score-output "$out/validation_candidates.npz" --batch-size 128 --device "$device"
python -m provia.evaluation.evaluate_baseline_bimanual --run-dir "$out" --examples "$run/examples.pt" \
  --query-examples "$run/test_query.pt" --manifest "$manifest" --training-feature-root "$features" --feature-root "$test_features" \
  --frozen-validation "$out/validation_$readout.json" --output "$out/test_$readout.json" \
  --split test --readout "$readout" --candidate-score-output "$out/test_candidates.npz" --batch-size 128 --device "$device"
