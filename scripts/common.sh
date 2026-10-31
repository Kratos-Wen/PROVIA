# Paths shared by the pipeline scripts. Source this file from the repository root.
set -euo pipefail
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH} PYTHONUNBUFFERED=1
device=${PROVIA_DEVICE:-cuda}
gpus=${PROVIA_GPUS:-4}

resolve_dataset() {
  dataset=$1
  case "$dataset" in
    captaincook4d|industreal|holoassist|impact_ego) ;;
    *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
  esac
  manifest=data/$dataset.jsonl
  features=cache/$dataset/train_val
  test_features=cache/$dataset/test
  run=runs/$dataset
  mkdir -p "$run"
}

check_seed() {
  case "$1" in
    0|1|2) ;;
    *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;;
  esac
}
