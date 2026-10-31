#!/usr/bin/env bash
# Alarm-level results: thresholds fixed on validation, test recordings scored once at them.
source scripts/common.sh
for dataset_name in captaincook4d industreal holoassist impact_ego; do
  python scripts/alarm/build_training_position_prior.py --manifest "data/$dataset_name.jsonl" \
    --output "runs/$dataset_name/training_position_prior.json"
done
python scripts/alarm/build_tables.py
python scripts/alarm/build_latex_table.py
