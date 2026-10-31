#!/usr/bin/env bash
# Evaluate the released models: rebuild the inputs with the released step segmentation model, then score seeds 0-2 on
# validation and once on test at the validation thresholds. Usage: evaluate_checkpoints.sh DATASET CHECKPOINT_DIR
source scripts/common.sh
resolve_dataset "$1"
checkpoints=$2
mkdir -p "$run/observer"
cp "$checkpoints/observer.pt" "$run/observer/best.pt"
if [[ "$dataset" == impact_ego ]]; then
  cp "$checkpoints/intent_space.json" "$run/observer/intent_space.json"
  mode=bimanual
  bash scripts/bimanual/build_automaton.sh
  python -m provia.launch materialize-bimanual-query --source-examples "$run/examples.pt" --feature-root "$test_features" \
    --output "$run/test_query.pt" --split test --device "$device"
  python -m provia.launch materialize-bimanual-prefix --source-examples "$run/examples.pt" --query-examples "$run/test_query.pt" \
    --output-dir "$run/test_prefix_evidence" --split test --device "$device"
else
  mode=single
  bash scripts/single_actor/build_automaton.sh "$dataset"
  python -m provia.launch materialize-single-query --source-examples "$run/examples.pt" --manifest "$manifest" \
    --feature-root "$test_features" --output "$run/test_query.pt" --split test --device "$device"
  python -m provia.launch materialize-single-prefix --source-examples "$run/examples.pt" --query-examples "$run/test_query.pt" \
    --output-dir "$run/test_prefix_evidence" --split test --device "$device"
fi
for seed in 0 1 2; do
  model=$run/model_seed$seed
  mkdir -p "$model"
  cp "$checkpoints/provia_seed$seed.pt" "$model/best.pt"
  python -m provia.launch "evaluate-$mode" --examples "$run/examples.pt" --checkpoint "$checkpoints/provia_seed$seed.pt" \
    --rebuilt-inputs --prefix-evidence "$run/prefix_evidence" --split validation --output "$model/validation.json" \
    --score-output "$model/validation_scores.npz" --candidate-score-output "$model/validation_candidates.npz" --device "$device"
  python -m provia.launch "evaluate-$mode" --examples "$run/examples.pt" --checkpoint "$checkpoints/provia_seed$seed.pt" \
    --rebuilt-inputs --query-examples "$run/test_query.pt" --query-prefix-evidence "$run/test_prefix_evidence" \
    --frozen-validation "$model/validation.json" --split test --output "$model/test.json" \
    --score-output "$model/test_scores.npz" --candidate-score-output "$model/test_candidates.npz" --device "$device"
done
python - "$run" <<'PY'
import json, sys
import numpy as np
metrics = [json.load(open(f"{sys.argv[1]}/model_seed{seed}/test.json"))["metrics"]["checkpoints"]["anytime"]["native_event"] for seed in range(3)]
ap, auroc = [m["average_precision"] for m in metrics], [m["auroc"] for m in metrics]
for seed, (seed_ap, seed_auroc) in enumerate(zip(ap, auroc)):
    print(f"seed {seed}: test AP {seed_ap:.4f}   AUROC {seed_auroc:.4f}")
print(f"test AP {np.mean(ap):.3f} ± {np.std(ap, ddof=1):.3f}   AUROC {np.mean(auroc):.3f} ± {np.std(auroc, ddof=1):.3f}")
PY
