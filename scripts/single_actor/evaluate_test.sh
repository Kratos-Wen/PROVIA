#!/usr/bin/env bash
# Build the test query without annotations, then score seeds 0-2 once at their validation thresholds.
# Usage: evaluate_test.sh DATASET
source scripts/common.sh
resolve_dataset "$1"
python -m provia.launch materialize-single-query --source-examples "$run/examples.pt" --manifest "$manifest" \
  --feature-root "$test_features" --output "$run/test_query.pt" --split test --device "$device"
python -m provia.launch materialize-single-prefix --source-examples "$run/examples.pt" --query-examples "$run/test_query.pt" \
  --output-dir "$run/test_prefix_evidence" --split test --device "$device"
for seed in 0 1 2; do
  model=$run/model_seed$seed
  python -m provia.launch evaluate-single --examples "$run/examples.pt" --checkpoint "$model/best.pt" \
    --query-examples "$run/test_query.pt" --query-prefix-evidence "$run/test_prefix_evidence" \
    --frozen-validation "$model/validation.json" --split test --output "$model/test.json" \
    --score-output "$model/test_scores.npz" --candidate-score-output "$model/test_candidates.npz" --device "$device"
done
