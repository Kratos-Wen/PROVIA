#!/usr/bin/env bash
# Reference traces from correct training executions, the two-hand language, training examples with open-step
# candidates, prefix evidence, step effects and the predictive-state automaton.
source scripts/common.sh
resolve_dataset impact_ego
observer=$run/observer
reference=$run/references
reference_features=cache/$dataset/reference
mkdir -p "$reference/raw" "$reference/aligned"
mapfile -t videos < <(python - "$manifest" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    entry = json.loads(line)
    roles = {r["fold"]: r["partition"] for r in entry["attributes"]["cross_validation_roles"]}
    if roles.get(0) == "train" and all(s["correctness"] == "correct" for s in entry["segments"]):
        print(entry["video_id"])
PY
)
printf '%s\n' "${videos[@]}" > "$reference/video_ids.txt"
python scripts/features/build_hand_object_role_caches.py --mode expert --hand-object "cache/$dataset/hand_object/finetuned" \
  --source "cache/$dataset/hand_object/frames" --output "$reference_features" --references "$reference/video_ids.txt"
raw=(); aligned=()
for video in "${videos[@]}"; do
  python -m provia.automaton.export_reference_traces --features "$reference_features/${video}_stream_features.npz" \
    --feature-manifest "$reference_features/manifest.json" --checkpoint "$observer/best.pt" \
    --intent-space "$observer/intent_space.json" --video-id "$video" --output "$reference/raw/$video.json" --device "$device"
  raw+=(--reference "$reference/raw/$video.json")
  aligned+=(--reference "$reference/aligned/$video.json")
done
python -m provia.automaton.align_references "${raw[@]}" --output-dir "$reference/aligned" \
  --audit "$reference/aligned/alignment_audit.json"
python -m provia.automaton.interval_language --intent-space "$observer/intent_space.json" "${aligned[@]}" \
  --output "$reference/language.json" --audit "$reference/language_audit.json"
prepare() {
  python -m provia.training.prepare_bimanual_examples --manifest "$manifest" --feature-root "$features" \
    --observer-checkpoint "$observer/best.pt" --procedure-language "$reference/language.json" --fold 0 --device "$device" "$@"
}
prepare --output "$run/onsets.pt"
python -m provia.automaton.causal_windows --examples "$run/onsets.pt" --output "$run/base_examples.pt"
prepare --output "$run/onsets_open_step.pt" --gap-candidates 8 --gap-base-examples "$run/base_examples.pt"
python -m provia.automaton.causal_windows --examples "$run/onsets_open_step.pt" --output "$run/examples.pt"
python -m provia.automaton.prefix_evidence_bimanual --examples "$run/base_examples.pt" --output-dir "$run/prefix_evidence" --device "$device"
python -m provia.automaton.step_effects role --language "$reference/language.json" "${aligned[@]}" \
  --feature-root "$reference_features" --output "$run/step_effects.pt"
python -m provia.automaton.compile_bimanual_automaton --examples "$run/base_examples.pt" \
  --reference-audit "$reference/language_audit.json" --output-dir "$run/automaton"
