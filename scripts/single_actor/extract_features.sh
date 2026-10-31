#!/usr/bin/env bash
# Frozen DINOv2-S/14 frame features at 8 Hz. Usage: extract_features.sh DATASET {train_val|test}
source scripts/common.sh
resolve_dataset "$1"
case "$2" in
  train_val) output=$features; partitions=(--partition train --partition validation) ;;
  test) output=$test_features; partitions=(--partition test) ;;
  *) echo "split must be train_val or test" >&2; exit 2 ;;
esac
python -m provia.training.extract_stream_features --manifest "$manifest" --output-root "$output" "${partitions[@]}" \
  --sample-rate-hz 8 --batch-size 256 --device "$device"
