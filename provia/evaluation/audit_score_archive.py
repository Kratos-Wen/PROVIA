"""Audit a native-event score archive against its reported metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from .score_archive import read_native_event_score_archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    *, metrics_path: Path, scores_path: Path, expected_device: str
) -> dict[str, object]:
    report = json.loads(metrics_path.read_text())
    if report.get("evaluation_device") != expected_device:
        raise ValueError("evaluation device does not match the device")
    if report.get("split") != "test" or report.get("test_split_accessed") is not True:
        raise ValueError("metrics artifact is not a test evaluation")

    metrics = report.get("metrics", {})
    primary = metrics.get("primary_streaming_checkpoint")
    native = metrics.get("checkpoints", {}).get(primary, {}).get("native_event")
    if not isinstance(native, dict):
        raise ValueError("primary native-event metrics are missing")

    target, score, video_id, has_mode = read_native_event_score_archive(scores_path)
    observed_ap = float(average_precision_score(target, score))
    observed_auroc = float(roc_auc_score(target, score))
    expected_hash = metrics.get("score_archive_sha256")
    observed_hash = _sha256(scores_path)
    checks = {
        "archive_sha256": observed_hash == expected_hash,
        "support": int(len(target)) == int(native.get("support", -1)),
        "positive_events": int(target.sum()) == int(native.get("positives", -1)),
        "average_precision": abs(observed_ap - float(native["average_precision"]))
        <= 1e-12,
        "auroc": abs(observed_auroc - float(native["auroc"])) <= 1e-12,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"native-event score audit failed: {failed}")

    return {
        "schema": "native_event_score_audit",
        "metrics": str(metrics_path.resolve()),
        "scores": str(scores_path.resolve()),
        "evaluation_device": expected_device,
        "primary_streaming_checkpoint": primary,
        "support": int(len(target)),
        "positive_events": int(target.sum()),
        "video_clusters": int(len(set(video_id.tolist()))),
        "average_precision": observed_ap,
        "auroc": observed_auroc,
        "archive_has_correctness_mode": has_mode,
        "checks": checks,
        "audit": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        metrics_path=args.metrics,
        scores_path=args.scores,
        expected_device=args.expected_device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"audit": "passed", "support": result["support"]}))


if __name__ == "__main__":
    main()
