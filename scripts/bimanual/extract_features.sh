#!/usr/bin/env bash
# IMPACT-ego hand-object stream at 15 Hz: frame features, hand boxes, a DINOv2-S/14 fine-tuned on training-fold step
# labels, per-hand descriptors and the two-hand cache. Usage: extract_features.sh DETECTOR_WEIGHTS [ENCODER]
# A given ENCODER, such as the released hand_object_encoder.pth, replaces the fine-tuning step.
source scripts/common.sh
resolve_dataset impact_ego
detector=$1
hand=cache/$dataset/hand_object
finetuned=$run/hand_object_encoder
encoder=${2:-$finetuned/dinov2_vits14_hand_object_finetuned.pth}
python -m provia.training.extract_stream_features --manifest "$manifest" --output-root "$hand/frames" \
  --partition train --partition validation --sample-rate-hz 15 --batch-size 256 --device "$device"
python -m provia.training.extract_stream_features --manifest "$manifest" --output-root "$hand/frames_test" \
  --partition test --sample-rate-hz 15 --batch-size 256 --device "$device"
python scripts/features/extract_hand_object_features.py --manifest "$manifest" --source-cache "$hand/frames" \
  --output-root "$hand/frozen" --partitions train validation --detector "$detector" --device "$device"
[[ -n "${2:-}" ]] || python scripts/features/finetune_crop_backbone.py --manifest "$manifest" --cache "$hand/frozen" \
  --output "$finetuned" --device "$device"
python scripts/features/extract_hand_object_features.py --manifest "$manifest" --source-cache "$hand/frames" \
  --output-root "$hand/finetuned" --partitions train validation --reuse-boxes "$hand/frozen" \
  --checkpoint "$encoder" --device "$device"
python scripts/features/extract_hand_object_features.py --manifest "$manifest" --source-cache "$hand/frames_test" \
  --output-root "$hand/finetuned_test" --partitions test --detector "$detector" \
  --checkpoint "$encoder" --device "$device"
python scripts/features/build_hand_object_role_caches.py --mode query --hand-object "$hand/finetuned" \
  --source "$hand/frames" --output "$features"
python scripts/features/build_hand_object_role_caches.py --mode query --hand-object "$hand/finetuned_test" \
  --source "$hand/frames_test" --output "$test_features"
