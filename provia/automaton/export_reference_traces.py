"""Decode an annotation-free expert stream with a frozen renewal observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from provia.automaton.event_decoding import decode_posterior_median_events
from provia.data.role_stream import _validate_cache
from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES
from provia.observer.build import ROLE_EVENT_OBSERVER_ARCHITECTURES, build_role_event_observer
from provia.model.intents import CompositionalIntentSpace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _factor_priors(metadata: Mapping[str, object]) -> dict[str, torch.Tensor]:
    raw = metadata.get("event_balanced_factor_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no event-balanced factor counts")
    result = {}
    for name in EVENT_FACTORS:
        counts = torch.tensor(raw[name], dtype=torch.float32)  # type: ignore[arg-type]
        result[name] = (counts + 0.5) / (counts + 0.5).sum()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--intent-space", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("observer_architecture") not in ROLE_EVENT_OBSERVER_ARCHITECTURES
        or metadata.get("test_split_accessed") is not False
    ):
        raise ValueError("checkpoint is not a role event observer")
    with open(args.feature_manifest, encoding="utf-8") as handle:
        feature_manifest = json.load(handle)
    cache = metadata.get("cache")
    if not isinstance(cache, Mapping) or not isinstance(feature_manifest, Mapping):
        raise ValueError("feature provenance is missing")
    if (
        feature_manifest.get("contains_annotations") is not False
        or feature_manifest.get("uses_expert_annotations") is not False
        or feature_manifest.get("observer_sha256") != cache.get("observer_sha256")
        or float(feature_manifest.get("sample_rate_hz", -1))
        != float(cache.get("sample_rate_hz", -2))
    ):
        raise ValueError("expert and query feature protocols are incompatible")
    with open(args.intent_space, encoding="utf-8") as handle:
        intent_space = CompositionalIntentSpace.from_dict(json.load(handle))
    frame_ids, timestamps, global_features, role_features, role_change = (
        _validate_cache(args.features)
    )
    enabled_change = metadata.get("role_change_evidence")
    if enabled_change not in (True, False):
        raise ValueError("checkpoint does not declare role-change evidence")
    if enabled_change and role_change is None:
        raise ValueError("checkpoint requires unavailable role-change evidence")
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    model = build_role_event_observer(metadata, int(global_features.shape[-1]), labels)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()
    priors = _factor_priors(metadata)
    with torch.inference_mode():
        evidence, posterior = model(
            torch.from_numpy(global_features).to(device),
            torch.from_numpy(role_features).to(device),
            torch.from_numpy(timestamps).to(device=device, dtype=torch.float32),
            (
                None
                if not enabled_change
                else torch.from_numpy(role_change).to(device)  # type: ignore[arg-type]
            ),
        )
        events = decode_posterior_median_events(
            posterior.conditional_mark_probability.cpu(),
            {
                name: values.float().cpu()
                for name, values in evidence.factor_logits.items()
            },
            priors,
            model.initial_phase_probability.cpu(),
            semantic_statistic=str(metadata.get("semantic_mark_statistic") or "frame-evidence-mean"),
        )
    if not events:
        raise RuntimeError("expert decoder produced no complete events")
    rows = []
    for index, event in enumerate(events):
        if event.right_censored:
            continue
        rows.append(
            {
                "event_id": f"{EVENT_ROLES[event.role]}_{index:04d}",
                "hand": EVENT_ROLES[event.role],
                "start_frame": int(frame_ids[event.start_index]),
                "onset_frame": int(frame_ids[event.onset_index]),
                "end_frame": int(frame_ids[event.end_index]),
                "start_sec": float(timestamps[event.start_index]),
                "onset_sec": float(timestamps[event.onset_index]),
                "end_sec": float(timestamps[event.end_index]),
                "right_censored": False,
                "argmax": {
                    name: labels[name][
                        int(torch.tensor(event.factor_log_probability[name]).argmax())
                    ]
                    for name in EVENT_FACTORS
                },
                "factor_distributions": {
                    name: {
                        "labels": list(labels[name]),
                        "log_probability": list(event.factor_log_probability[name]),
                    }
                    for name in EVENT_FACTORS
                },
            }
        )
    result = {
        "schema": "provia_uncertain_expert_reference",
        "video_id": args.video_id,
        "right_censored_events_excluded": int(
            sum(event.right_censored for event in events)
        ),
        "decoder": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "uses_expert_annotations": False,
            "query_mode": True,
            "role_change_evidence": bool(enabled_change),
            "semantic_mark_statistic": metadata.get("semantic_mark_statistic"),
        },
        "feature_cache": str(args.features.resolve()),
        "feature_sha256": _sha256(args.features),
        "feature_manifest": str(args.feature_manifest.resolve()),
        "feature_manifest_sha256": _sha256(args.feature_manifest),
        "events": rows,
    }
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {"video_id": args.video_id, "events": len(rows), "output": str(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
