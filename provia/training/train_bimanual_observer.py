"""Train the role-wise marked-renewal observer by exact boundary likelihood.

Boundary times are optimized as interval-censored competing risks, not as
imbalanced frame occupations.  Event semantics form a conditionally separate
mark likelihood evaluated only on the causal start-to-onset prefix.  No
correctness, anomaly, recovery, or reserved-test label is read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from provia.automaton.event_decoding import decode_posterior_median_events
from provia.data.io import read_jsonl
from provia.data.role_stream import CompetitiveRoleStream, load_competitive_role_streams
from provia.observer.factors import EVENT_FACTORS, EVENT_PHASES, EVENT_ROLES
from provia.observer.marked_renewal import (
    END,
    ONSET,
    RENEW_ONSET,
    RENEW_START,
    START,
    START_ONSET,
    MarkedRenewalEvidence,
    MarkedRenewalRoleObserver,
    estimate_renewal_statistics,
    renewal_mark_targets,
)
from provia.training.event_metrics import _summarize
from provia.training.common import (
    _append_jsonl,
    _atomic_json,
    _cache_provenance,
    _jeffreys_prior,
    _partitions,
    _prior_corrected_nll,
    _seed_everything,
    _statistics,
)
from provia.model.intents import CompositionalIntentSpace


BOUNDARY_PREFIXES = ("boundary_map.", "intensity_head.")
SEMANTIC_PREFIXES = ("semantic_map.", "factor_heads.")


def _component_state(
    model: MarkedRenewalRoleObserver, prefixes: Sequence[str]
) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(tuple(prefixes))
    }


def _restore_components(
    model: MarkedRenewalRoleObserver,
    boundary: Mapping[str, Tensor],
    semantics: Mapping[str, Tensor],
) -> None:
    state = model.state_dict()
    expected_boundary = {
        name for name in state if name.startswith(BOUNDARY_PREFIXES)
    }
    expected_semantics = {
        name for name in state if name.startswith(SEMANTIC_PREFIXES)
    }
    if set(boundary) != expected_boundary or set(semantics) != expected_semantics:
        raise ValueError("a marked-renewal component checkpoint is incomplete")
    state.update(boundary)
    state.update(semantics)
    model.load_state_dict(state, strict=True)


def _deltas(stream: CompetitiveRoleStream, device: torch.device) -> Tensor:
    timestamps = stream.timestamps_sec.to(device=device, dtype=torch.float32)
    return torch.cat(
        (torch.zeros_like(timestamps[:1]), timestamps[1:] - timestamps[:-1])
    )


def _emit(
    model: MarkedRenewalRoleObserver,
    stream: CompetitiveRoleStream,
    device: torch.device,
) -> MarkedRenewalEvidence:
    return model.emit_stream(
        stream.global_features.to(device),
        stream.role_features.to(device),
        stream.timestamps_sec.to(device=device, dtype=torch.float32),
        (
            None
            if stream.role_change_features is None
            else stream.role_change_features.to(device)
        ),
    )


def _boundary_nll(
    model: MarkedRenewalRoleObserver,
    stream: CompetitiveRoleStream,
    evidence: MarkedRenewalEvidence,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if len(stream.frame_ids) < 2:
        raise ValueError("boundary likelihood requires at least two frames")
    conditional = model.conditional_mark_probabilities(
        evidence.intensity_per_second, _deltas(stream, device)
    )
    source = stream.phase_targets[:-1].to(device=device, dtype=torch.long)
    marks = renewal_mark_targets(
        stream.intervals, len(stream.frame_ids)
    )[1:].to(device)
    source_probability = conditional[1:].gather(
        2,
        source.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, conditional.shape[-1]
        ),
    ).squeeze(2)
    probability = source_probability.gather(
        -1, marks.unsqueeze(-1)
    ).squeeze(-1)
    nll = -probability.clamp_min(
        torch.finfo(probability.dtype).tiny
    ).log().mean()
    return nll, conditional


def _semantic_event_tensors(
    model: MarkedRenewalRoleObserver,
    evidence: MarkedRenewalEvidence,
    stream: CompetitiveRoleStream,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    logits = {name: [] for name in EVENT_FACTORS}
    targets = {name: [] for name in EVENT_FACTORS}
    for interval in stream.intervals:
        event_logits = {
            name: evidence.factor_logits[name][
                interval.start : interval.onset + 1, interval.role
            ].mean(dim=0)
            for name in EVENT_FACTORS
        }
        for name in EVENT_FACTORS:
            logits[name].append(event_logits[name])
            targets[name].append(
                stream.factor_targets[name][interval.onset, interval.role]
            )
    if not stream.intervals:
        raise ValueError("semantic likelihood requires at least one event")
    return (
        {name: torch.stack(values) for name, values in logits.items()},
        {
            name: torch.stack(values).to(device=device, dtype=torch.long)
            for name, values in targets.items()
        },
    )


def _objective(
    model: MarkedRenewalRoleObserver,
    stream: CompetitiveRoleStream,
    factor_counts: Mapping[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor], MarkedRenewalEvidence, Tensor]:
    evidence = _emit(model, stream, device)
    boundary, conditional = _boundary_nll(model, stream, evidence, device)
    event_logits, event_targets = _semantic_event_tensors(
        model, evidence, stream, device
    )
    semantics = {
        name: _prior_corrected_nll(
            event_logits[name], event_targets[name], factor_counts[name]
        )
        for name in EVENT_FACTORS
    }
    # The parameter sets are disjoint. This is the negative log likelihood of
    # the factorized boundary-and-mark model, with no scalarized trade-off.
    total = boundary + torch.stack(tuple(semantics.values())).sum()
    return total, {"boundary": boundary, **semantics}, evidence, conditional


def _boundary_scores(mark_probability: Tensor) -> dict[str, Tensor]:
    return {
        "start": mark_probability[..., [START, START_ONSET, RENEW_START, RENEW_ONSET]].sum(dim=-1),
        "onset": mark_probability[..., [START_ONSET, ONSET, RENEW_ONSET]].sum(dim=-1),
        "end": mark_probability[..., [END, RENEW_START, RENEW_ONSET]].sum(dim=-1),
    }


def _boundary_targets(stream: CompetitiveRoleStream) -> dict[str, Tensor]:
    marks = renewal_mark_targets(stream.intervals, len(stream.frame_ids))
    return {
        "start": (
            (marks == START)
            | (marks == START_ONSET)
            | (marks == RENEW_START)
            | (marks == RENEW_ONSET)
        ),
        "onset": (
            (marks == START_ONSET) | (marks == ONSET) | (marks == RENEW_ONSET)
        ),
        "end": (
            (marks == END) | (marks == RENEW_START) | (marks == RENEW_ONSET)
        ),
    }


@torch.no_grad()
def _evaluate(
    model: MarkedRenewalRoleObserver,
    streams: Sequence[CompetitiveRoleStream],
    factor_counts: Mapping[str, Tensor],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    components = {name: [] for name in ("boundary", *EVENT_FACTORS)}
    phase_targets = []
    phase_predictions = []
    factor_correct = {name: 0 for name in EVENT_FACTORS}
    factor_total = 0
    boundary_target = {name: [] for name in ("start", "onset", "end")}
    boundary_score = {name: [] for name in boundary_target}
    decoded = []
    priors = {
        name: _jeffreys_prior(
            factor_counts[name], device=torch.device("cpu"), dtype=torch.float32
        )
        for name in EVENT_FACTORS
    }
    for stream in streams:
        _, values, evidence, conditional = _objective(
            model, stream, factor_counts, device
        )
        for name, value in values.items():
            components[name].append(float(value))
        posterior = model.filter(conditional)
        phase_targets.append(stream.phase_targets.reshape(-1))
        phase_predictions.append(
            posterior.state_probability.argmax(dim=-1).cpu().reshape(-1)
        )
        event_logits, event_targets = _semantic_event_tensors(
            model, evidence, stream, device
        )
        factor_total += len(stream.intervals)
        for name in EVENT_FACTORS:
            prior = _jeffreys_prior(
                factor_counts[name], device=device, dtype=event_logits[name].dtype
            )
            prediction = (event_logits[name] + prior.log()).argmax(dim=-1)
            factor_correct[name] += int(
                (prediction == event_targets[name]).sum()
            )
        scores = {
            name: value.cpu()
            for name, value in _boundary_scores(
                posterior.mark_probability
            ).items()
        }
        targets = _boundary_targets(stream)
        for name in boundary_target:
            boundary_target[name].append(targets[name].reshape(-1).numpy())
            boundary_score[name].append(scores[name].reshape(-1).numpy())
        factor_logits_cpu = {
            name: value.cpu() for name, value in evidence.factor_logits.items()
        }
        predictions = decode_posterior_median_events(
            conditional.cpu(),
            factor_logits_cpu,
            priors,
            model.initial_phase_probability.cpu(),
        )
        decoded.append((stream, predictions))
    target_values = torch.cat(phase_targets).numpy()
    predicted_values = torch.cat(phase_predictions).numpy()
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        target_values,
        predicted_values,
        labels=range(len(EVENT_PHASES)),
        zero_division=0,
    )
    component_nll = {
        name: float(np.mean(values)) for name, values in components.items()
    }
    return {
        "boundary_nll": component_nll["boundary"],
        "mean_semantic_nll": float(
            np.mean([component_nll[name] for name in EVENT_FACTORS])
        ),
        "component_nll": component_nll,
        "phase_accuracy": float(np.mean(target_values == predicted_values)),
        "phase_macro_f1": float(
            f1_score(
                target_values,
                predicted_values,
                labels=range(len(EVENT_PHASES)),
                average="macro",
                zero_division=0,
            )
        ),
        "phase_by_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(EVENT_PHASES)
        },
        "event_factor_accuracy": {
            name: factor_correct[name] / max(factor_total, 1)
            for name in EVENT_FACTORS
        },
        "boundary_auprc": {
            name: float(
                average_precision_score(
                    np.concatenate(boundary_target[name]).astype(np.int64),
                    np.concatenate(boundary_score[name]),
                )
            )
            for name in boundary_target
        },
        "boundary_prevalence": {
            name: float(np.concatenate(boundary_target[name]).mean())
            for name in boundary_target
        },
        "posterior_median_event_support": _summarize(decoded),
        "frames": int(len(target_values) // len(EVENT_ROLES)),
        "events": int(sum(len(stream.intervals) for stream in streams)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early-stopping-epochs", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.epochs <= 0 or args.early_stopping_epochs <= 0:
        parser.error("epoch counts must be positive")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    episodes = tuple(read_jsonl(args.manifest))
    train_native, validation_native = _partitions(episodes, args.fold)
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    train = load_competitive_role_streams(
        train_native, args.feature_root, intent_space
    )
    validation = load_competitive_role_streams(
        validation_native, args.feature_root, intent_space
    )
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    _, factor_counts = _statistics(train, labels)
    base_rates, initial, mark_counts, exposure = estimate_renewal_statistics(train)
    model = MarkedRenewalRoleObserver(
        int(train[0].global_features.shape[-1]), labels, base_rates, initial
    ).to(device)
    boundary_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(BOUNDARY_PREFIXES)
    ]
    semantic_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(SEMANTIC_PREFIXES)
    ]
    if (
        not boundary_parameters
        or not semantic_parameters
        or len(boundary_parameters) + len(semantic_parameters)
        != sum(1 for _ in model.parameters())
    ):
        raise ValueError("renewal parameters are not exactly factorized")
    boundary_optimizer = torch.optim.AdamW(
        boundary_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    semantic_optimizer = torch.optim.AdamW(
        semantic_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    cache = _cache_provenance(args.feature_root, args.fold)
    metadata = {
        "schema": "marked_renewal_role_observer_training",
        "observer_architecture": "marked_renewal_observer",
        "semantic_mark_statistic": "frame-evidence-mean",
        "decoder": (
            "causal posterior median of each renewal time; no tuned score, "
            "duration, gap, or suppression threshold"
        ),
        "compound_marks": ["end+start", "end+start+onset"],
        "visual_change_units": "per_second",
        "role_change_evidence": all(
            stream.role_change_features is not None for stream in train
        ),
        "anomaly_labels_used_for_optimization": False,
        "anomaly_metrics_used_for_selection": False,
        "test_split_accessed": False,
        "fold": args.fold,
        "seed": args.seed,
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "train_frames": sum(len(stream.frame_ids) for stream in train),
        "validation_frames": sum(len(stream.frame_ids) for stream in validation),
        "train_events": sum(len(stream.intervals) for stream in train),
        "validation_events": sum(len(stream.intervals) for stream in validation),
        "base_rates_per_second": base_rates.tolist(),
        "initial_phase_probability": initial.tolist(),
        "training_mark_counts": mark_counts.tolist(),
        "training_phase_exposure_sec": exposure.tolist(),
        "event_balanced_factor_counts": {
            name: factor_counts[name].tolist() for name in EVENT_FACTORS
        },
        "parameters": {
            "total": sum(value.numel() for value in model.parameters()),
            "trainable": sum(
                value.numel() for value in model.parameters() if value.requires_grad
            ),
        },
        "cache": cache,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _atomic_json(args.output_dir / "run.json", metadata)
    _atomic_json(args.output_dir / "intent_space.json", intent_space.to_dict())

    initial_validation = _evaluate(
        model, validation, factor_counts, device
    )
    best_boundary_value = float(initial_validation["boundary_nll"])
    best_semantic_value = float(initial_validation["mean_semantic_nll"])
    best_boundary_epoch = 0
    best_semantic_epoch = 0
    best_boundary = _component_state(model, BOUNDARY_PREFIXES)
    best_semantics = _component_state(model, SEMANTIC_PREFIXES)
    boundary_stale = 0
    semantic_stale = 0
    _append_jsonl(
        args.output_dir / "metrics.jsonl",
        {"epoch": 0, "validation": initial_validation},
    )

    generator = torch.Generator().manual_seed(args.seed)
    started = time.monotonic()
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        train_values = []
        train_components = {name: [] for name in ("boundary", *EVENT_FACTORS)}
        gradient_norms = {"boundary": [], "semantics": []}
        for stream_index in order.tolist():
            boundary_optimizer.zero_grad(set_to_none=True)
            semantic_optimizer.zero_grad(set_to_none=True)
            loss, values, _, _ = _objective(
                model,
                train[stream_index],
                factor_counts,
                device,
            )
            loss.backward()
            boundary_gradient_norm = clip_grad_norm_(
                boundary_parameters, args.gradient_clip
            )
            semantic_gradient_norm = clip_grad_norm_(
                semantic_parameters, args.gradient_clip
            )
            boundary_optimizer.step()
            semantic_optimizer.step()
            train_values.append(float(loss.detach()))
            gradient_norms["boundary"].append(float(boundary_gradient_norm))
            gradient_norms["semantics"].append(float(semantic_gradient_norm))
            for name, value in values.items():
                train_components[name].append(float(value.detach()))
        validation_result = _evaluate(
            model,
            validation,
            factor_counts,
            device,
        )
        boundary_value = float(validation_result["boundary_nll"])
        semantic_value = float(validation_result["mean_semantic_nll"])
        if boundary_value < best_boundary_value - 1e-12:
            best_boundary_value = boundary_value
            best_boundary_epoch = epoch
            best_boundary = _component_state(model, BOUNDARY_PREFIXES)
            boundary_stale = 0
        else:
            boundary_stale += 1
        if semantic_value < best_semantic_value - 1e-12:
            best_semantic_value = semantic_value
            best_semantic_epoch = epoch
            best_semantics = _component_state(model, SEMANTIC_PREFIXES)
            semantic_stale = 0
        else:
            semantic_stale += 1
        record = {
            "epoch": epoch,
            "elapsed_sec": time.monotonic() - started,
            "train_loss": float(np.mean(train_values)),
            "train_component_nll": {
                name: float(np.mean(values))
                for name, values in train_components.items()
            },
            "gradient_norm": {
                name: float(np.mean(values))
                for name, values in gradient_norms.items()
            },
            "validation": validation_result,
            "best_boundary_epoch": best_boundary_epoch,
            "best_semantic_epoch": best_semantic_epoch,
        }
        _append_jsonl(args.output_dir / "metrics.jsonl", record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if (
            boundary_stale >= args.early_stopping_epochs
            and semantic_stale >= args.early_stopping_epochs
        ):
            break

    _restore_components(model, best_boundary, best_semantics)
    final_validation = _evaluate(
        model, validation, factor_counts, device
    )
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": metadata,
            "epoch": {
                "boundary": best_boundary_epoch,
                "semantics": best_semantic_epoch,
            },
            "validation": final_validation,
        },
        args.output_dir / "best.pt",
    )
    _atomic_json(args.output_dir / "validation.json", final_validation)
    _atomic_json(
        args.output_dir / "complete.json",
        {
            "status": "complete",
            "epochs_completed": epoch,
            "best_boundary_epoch": best_boundary_epoch,
            "best_boundary_nll": best_boundary_value,
            "best_semantic_epoch": best_semantic_epoch,
            "best_mean_semantic_nll": best_semantic_value,
            "elapsed_sec": time.monotonic() - started,
            "test_split_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
