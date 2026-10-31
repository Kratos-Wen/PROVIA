"""Evaluate PROVIA on single-actor streams under native event protocols.

Every score is produced from a causal prefix of a fully predicted candidate.
The completion checkpoint uses only the candidate endpoint.  The anytime
checkpoint takes the maximum score seen up to that endpoint and records the
first threshold crossing relative to the annotated native completion time.
No evaluation threshold is used for model or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.evaluation.operating_points import (
    _checkpoint_metrics,
    _free_response_false_alarm_scores,
)
from provia.evaluation.score_archive import (
    write_native_event_score_archive,
)
from provia.evaluation.single_actor_query import (
    completion_inference_view,
    load_completion_query_artifact,
    load_completion_query_prefix_registry,
)
from provia.training.common import _atomic_json, _intact, _sha256
from provia.training.completion_inputs import (
    _load_expert_effects,
    _load_transports,
)
from provia.training.train_single_actor import (
    _build_model,
    _candidate_context,
    _endpoint_factors,
    _global_stream,
    _load_prefix_registry,
    _prefix_evidence,
    _source_with_predictive_language_registry,
)
from provia.model.mode_filter import ERROR
from provia.model.intent_state import (
    CAUSAL_CANDIDATE_DECISION_CLOCK,
    MARGINAL_PREFIX_ERROR_READOUT,
)
from provia.model.step_effects import ExpertEffectPosterior
from provia.model.single_actor import (
    PROVIA_SINGLE_ACTOR_ARCHITECTURE,
    PROVIA_SHARED_SUCCESSOR_ARCHITECTURE,
)
from provia.model.ablation import (
    FULL_INFERENCE,
    INFERENCE_ABLATION_DEFINITIONS,
    NO_CAUSAL_DECISION_CLOCK,
    NO_FACTUAL_HISTORY,
    NO_PROCEDURE_FRONTIER,
    NO_PERSISTENT_PROCEDURE_FRONTIER,
    configure_model_ablation,
    configure_transport_ablation,
    parse_inference_ablations,
    project_to_static_frontier,
    static_frontier_probability,
)


FROC_BUDGETS = ("0.1", "0.5", "1.0", "2.0")


SUPPORTED_COMPLETION_ARCHITECTURES = frozenset(
    {PROVIA_SINGLE_ACTOR_ARCHITECTURE, PROVIA_SHARED_SUCCESSOR_ARCHITECTURE}
)


def _load_frozen_operating_thresholds(
    path: Path,
    *,
    checkpoint_sha256: str,
    examples_sha256: str,
    inference_ablation: str,
    anytime_readout: str | None = None,
) -> dict[str, dict[str, float]]:
    """Load validation operating points for a matching test evaluation."""

    value = json.loads(path.read_text())
    if (
        not isinstance(value, Mapping)
        or value.get("split") != "validation"
        or value.get("test_split_accessed") is not False
        or value.get("checkpoint_sha256") != checkpoint_sha256
        or value.get("examples_sha256") != examples_sha256
        or value.get("inference_ablation") != inference_ablation
        or (
            anytime_readout is not None
            and value.get("anytime_readout") != anytime_readout
        )
    ):
        raise ValueError(
            "frozen validation artifact is incompatible with test evaluation"
        )
    metrics = value.get("metrics")
    checkpoints = metrics.get("checkpoints") if isinstance(metrics, Mapping) else None
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("future_frames_used_by_score") is not False
        or not isinstance(checkpoints, Mapping)
    ):
        raise ValueError(
            "frozen validation artifact lacks causal checkpoint metrics"
        )
    result: dict[str, dict[str, float]] = {}
    for checkpoint in ("completion", "anytime"):
        entry = checkpoints.get(checkpoint)
        response = entry.get("free_response") if isinstance(entry, Mapping) else None
        points = response.get("operating_points") if isinstance(response, Mapping) else None
        if not isinstance(points, Mapping):
            raise ValueError(
                f"frozen validation artifact lacks {checkpoint} operating points"
            )
        thresholds: dict[str, float] = {}
        for budget in FROC_BUDGETS:
            point = points.get(budget)
            if not isinstance(point, Mapping) or "validation_threshold" not in point:
                raise ValueError(
                    f"frozen validation artifact lacks {checkpoint}/{budget} threshold"
                )
            threshold = float(point["validation_threshold"])
            if np.isnan(threshold):
                raise ValueError("a frozen validation threshold cannot be NaN")
            thresholds[budget] = threshold
        result[checkpoint] = thresholds
    return result


def _pool_native_events(
    *,
    assignment: np.ndarray,
    candidate_score: np.ndarray,
    candidate_options: Sequence[np.ndarray],
    native_count: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    score = np.zeros(native_count, dtype=np.float64)
    covered = np.zeros(native_count, dtype=bool)
    options: list[np.ndarray] = []
    for native_index in range(native_count):
        candidates = np.flatnonzero(assignment == native_index)
        if len(candidates):
            score[native_index] = float(candidate_score[candidates].max())
            covered[native_index] = True
            options.append(
                np.concatenate(
                    [candidate_options[int(index)] for index in candidates],
                    axis=0,
                )
            )
        else:
            options.append(np.empty((0, 2), dtype=np.float64))
    return score, covered, options


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    sequences: Sequence[Mapping[str, object]],
    feature_root: Path,
    prefix_root: Path | None,
    prefix_registry: Mapping[str, object] | None,
    transports: Sequence[torch.nn.Module],
    expert_effects: Sequence[ExpertEffectPosterior] | None,
    device: torch.device,
    inference_ablation: str = FULL_INFERENCE,
    score_output: Path | None = None,
    frozen_operating_thresholds: Mapping[str, Mapping[str, float]] | None = None,
    candidate_dump: list | None = None,
) -> dict[str, object]:
    ablation_parts = parse_inference_ablations(inference_ablation)
    completion_candidate_scores = []
    anytime_candidate_scores = []
    candidate_targets = []
    origin_scores = []
    origin_targets = []
    completion_false_alarm_scores = []
    anytime_false_alarm_scores = []
    completion_native_scores = []
    anytime_native_scores = []
    native_modes = []
    native_covered = []
    completion_detection_options: list[np.ndarray] = []
    anytime_detection_options: list[np.ndarray] = []
    native_categories: list[tuple[str, ...]] = []
    native_video_ids: list[str] = []
    duration_minutes = 0.0
    per_video = []

    model.eval()
    for sequence in sequences:
        # Construct an immutable allowlisted view before any score is
        # produced.  Query labels remain reachable only from ``sequence``
        # below the scoring block, where they are used for metrics.
        inference = completion_inference_view(sequence)
        task = int(inference["task_index"])
        video_id = str(inference["video_id"])
        feature, timestamp = _global_stream(feature_root, video_id)
        _readout_root = getattr(model.energy, "_readout_root", None)
        model.energy._readout_stream = None if _readout_root is None else _global_stream(Path(_readout_root), video_id)[0]
        if prefix_root is not None and prefix_registry is not None:
            prefix_event, prefix_factor = _prefix_evidence(
                prefix_root, prefix_registry, video_id, timestamp
            )
        else:
            prefix_event = None
            prefix_factor = None
        encoded_stream = (
            model.energy.window_encoder.forward_stream(
                feature.to(device=device, dtype=torch.float32)
            )
            if prefix_event is not None
            else None
        )
        transport = transports[task]
        expert_effect = (
            None if expert_effects is None else expert_effects[task]
        )
        decoded = inference["decoded_index"]
        if not isinstance(decoded, torch.Tensor) or decoded.dtype != torch.long:
            raise ValueError("decoded candidate indices are malformed")
        candidate_state_paths: tuple[torch.Tensor, ...] | None = None
        if (
            encoded_stream is None
            or prefix_event is None
            or prefix_factor is None
        ):
            raise ValueError(
                "coupled candidate batching requires causal prefix evidence"
            )
        encoded_candidates = []
        factor_candidates = []
        event_candidates = []
        previous_candidate_end = 0
        for candidate_index in range(len(decoded)):
            candidate_end = int(decoded[candidate_index])
            candidate_start = previous_candidate_end + 1
            endpoint = _endpoint_factors(
                inference, candidate_index, device
            )
            prefix = {
                name: value[candidate_start : candidate_end + 1].to(
                    device=device, dtype=torch.float32
                )
                for name, value in prefix_factor.items()
            }
            encoded_candidates.append(
                encoded_stream[candidate_start : candidate_end + 1]
            )
            factor_candidates.append(
                model._endpoint_causal_factors(prefix, endpoint)
            )
            event_candidates.append(
                prefix_event[candidate_start : candidate_end + 1].to(
                    device=device, dtype=torch.float32
                )
            )
            previous_candidate_end = candidate_end
        candidate_state_paths = model.energy.encode_candidate_state_paths(
            encoded_candidates,
            factor_candidates,
            event_candidates,
        )
        belief = model.filter.initial_belief(
            transport.initial_belief.to(device)
        )
        static_frontier = (
            static_frontier_probability(transport, belief)
            if any(
                part in {NO_PROCEDURE_FRONTIER, NO_PERSISTENT_PROCEDURE_FRONTIER}
                for part in ablation_parts
            )
            else None
        )
        history = model.energy.initial_history(feature[:1].to(device))
        endpoint_score = []
        anytime_score = []
        _explanations = []
        _belief_in = []
        genuine_score = []
        completion_options = []
        anytime_options = []
        _expected_verb = []
        previous_end = 0
        for index in range(len(decoded)):
            if static_frontier is not None:
                belief = project_to_static_frontier(belief, static_frontier)
            # expected next-step verb distribution under the current belief (procedure-state marginal ×
            # successor-node law × node-verb possibility), recorded for belief-conditioned observations
            try:
                _state = belief.reshape(belief.shape[0], -1).sum(-1)
                _node = _state.to(transport.frontier_node_probability) @ transport.frontier_node_probability
                _verb = _node @ transport.node_verb_possibility
                _verb = _verb / _verb.sum().clamp_min(1e-12)
                _expected_verb.append(_verb.detach().float().cpu().numpy())
            except Exception:
                _expected_verb.append(None)
            end = int(decoded[index])
            start = previous_end + 1
            context, offset = _candidate_context(feature, start, end, device)
            factor = _endpoint_factors(inference, index, device)
            effect = (
                inference["statistic"][index].to(  # type: ignore[index,union-attr]
                    device=device, dtype=torch.float32
                )
                if expert_effect is not None
                else None
            )
            if prefix_event is None or prefix_factor is None:
                raise ValueError("evaluation requires causal prefix evidence")
            output = model.forward_candidate(
                context,
                offset,
                {
                    name: value[start : end + 1].to(
                        device=device, dtype=torch.float32
                    )
                    for name, value in prefix_factor.items()
                },
                prefix_event[start : end + 1].to(
                    device=device, dtype=torch.float32
                ),
                factor,
                belief,
                history,
                transport,
                effect_statistic=effect,
                expert_effect=expert_effect,
                encoded_prefix_hidden=encoded_stream[start : end + 1],
                candidate_state_hidden=candidate_state_paths[index],
                marginalize_procedure_frontier=(
                    NO_PROCEDURE_FRONTIER in ablation_parts
                ),
            )
            prefix = output.prefix_anomaly_probability.detach().cpu().numpy()
            if candidate_dump is not None:
                _b = belief.detach().float().cpu()
                _explanations.append(output.explanation_probability.detach().float().cpu().numpy().copy())
                _belief_in.append((_b.reshape(_b.shape[0], -1).sum(0) if _b.ndim >= 2 else _b).numpy().copy())
            prefix_time = timestamp[start : end + 1].numpy()
            if prefix.shape != prefix_time.shape:
                raise RuntimeError("prefix scores and causal timestamps disagree")
            native_time_placeholder = np.zeros_like(prefix_time)
            endpoint_score.append(float(output.anomaly_probability.cpu()))
            anytime_score.append(float(prefix.max()))
            genuine_score.append(
                float(output.explanation_probability[1:].sum().cpu())
            )
            completion_options.append(
                np.asarray(
                    ((float(output.anomaly_probability.cpu()), float(timestamp[end])),),
                    dtype=np.float64,
                )
            )
            anytime_options.append(
                np.column_stack((prefix, prefix_time - native_time_placeholder))
            )
            belief = output.next_procedure_mode_lifecycle_probability
            history = (
                model.energy.initial_history(feature[:1].to(device))
                if NO_FACTUAL_HISTORY in ablation_parts
                else output.next_factual_history
            )
            previous_end = end

        endpoint_score_array = np.asarray(endpoint_score, dtype=np.float64)
        anytime_score_array = np.asarray(anytime_score, dtype=np.float64)
        genuine_score_array = np.asarray(genuine_score, dtype=np.float64)
        origin = sequence["origin_target"].numpy().astype(bool)  # type: ignore[union-attr]
        mode = sequence["mode_target"].numpy()  # type: ignore[union-attr]
        candidate_target = origin & (mode == ERROR)
        assignment = sequence["native_assignment"].numpy()  # type: ignore[union-attr]
        native_mode = sequence["native_mode"].numpy()  # type: ignore[union-attr]
        native_time = sequence["native_time_sec"].numpy()  # type: ignore[union-attr]
        if assignment.shape != endpoint_score_array.shape:
            raise ValueError("native assignment and candidate scores disagree")

        # Convert absolute causal times into delays relative to the assigned
        # native completion.  This is only evaluation bookkeeping and never
        # enters the model or checkpoint selection.
        for candidate_index, native_index in enumerate(assignment):
            if native_index >= 0:
                reference = float(native_time[int(native_index)])
                completion_options[candidate_index][:, 1] -= reference
                anytime_options[candidate_index][:, 1] -= reference
            else:
                completion_options[candidate_index][:, 1] = np.nan
                anytime_options[candidate_index][:, 1] = np.nan

        completion_native, covered, completion_native_options = (
            _pool_native_events(
                assignment=assignment,
                candidate_score=endpoint_score_array,
                candidate_options=completion_options,
                native_count=len(native_mode),
            )
        )
        anytime_native, anytime_covered, anytime_native_options = (
            _pool_native_events(
                assignment=assignment,
                candidate_score=anytime_score_array,
                candidate_options=anytime_options,
                native_count=len(native_mode),
            )
        )
        if not np.array_equal(covered, anytime_covered):
            raise RuntimeError("checkpoint coverage must be score-independent")

        completion_false_alarm_scores.extend(
            _free_response_false_alarm_scores(
                endpoint_score_array, assignment, native_mode
            )
        )
        anytime_false_alarm_scores.extend(
            _free_response_false_alarm_scores(
                anytime_score_array, assignment, native_mode
            )
        )
        completion_candidate_scores.append(endpoint_score_array)
        anytime_candidate_scores.append(anytime_score_array)
        if candidate_dump is not None:
            candidate_dump.append(
                {
                    "video_id": video_id,
                    "endpoint_score": endpoint_score_array.copy(),
                    "anytime_score": anytime_score_array.copy(),
                    "explanation": np.asarray(_explanations, dtype=np.float64) if _explanations else np.zeros((0, 0)),
                    "belief_in_mode_marginal": np.asarray(_belief_in, dtype=np.float64) if _belief_in else np.zeros((0, 0)),
                    "prefix_options": [np.asarray(o, dtype=np.float64) for o in anytime_options],
                    "expected_verb": (np.stack([v for v in _expected_verb]) if _expected_verb and all(v is not None for v in _expected_verb) else np.zeros((0, 0))),
                    "assignment": assignment.copy(),
                    "candidate_mode": mode.copy(),
                    "origin": origin.copy(),
                    "native_mode": native_mode.copy(),
                    "native_time": native_time.copy(),
                    "decoded_time": np.asarray(sequence["decoded_time_sec"], dtype=np.float64),
                    "native_error_types": [tuple(t) for t in sequence.get("native_error_types", ())],
                }
            )
        candidate_targets.append(candidate_target.astype(np.int64))
        origin_scores.append(genuine_score_array)
        origin_targets.append(origin.astype(np.int64))
        completion_native_scores.extend(completion_native)
        anytime_native_scores.extend(anytime_native)
        native_modes.extend(native_mode)
        native_covered.extend(covered)
        completion_detection_options.extend(completion_native_options)
        anytime_detection_options.extend(anytime_native_options)
        native_categories.extend(sequence["native_error_types"])  # type: ignore[arg-type]
        native_video_ids.extend([video_id] * len(native_mode))
        video_minutes = (
            float(sequence["stream_end_sec"])
            - float(sequence["stream_start_sec"])
        ) / 60.0
        duration_minutes += video_minutes
        per_video.append(
            {
                "video_id": sequence["video_id"],
                "duration_minutes": video_minutes,
                "native_events": len(native_mode),
                "native_errors": int(
                    np.sum(native_mode == CORRECTNESS_INDEX["error"])
                ),
                "decoded_candidates": len(endpoint_score_array),
                "native_event_coverage": float(covered.mean()),
            }
        )

    completion_candidate = np.concatenate(completion_candidate_scores)
    anytime_candidate = np.concatenate(anytime_candidate_scores)
    candidate_target = np.concatenate(candidate_targets)
    origin_score = np.concatenate(origin_scores)
    origin_target = np.concatenate(origin_targets)
    native_mode = np.asarray(native_modes, dtype=np.int64)
    covered = np.asarray(native_covered, dtype=bool)
    anytime_native_score = np.asarray(anytime_native_scores, dtype=np.float64)
    native_video_id = np.asarray(native_video_ids, dtype=np.str_)
    if not (
        native_mode.shape == covered.shape == anytime_native_score.shape
        == native_video_id.shape
    ):
        raise RuntimeError("native score archive fields are not aligned")
    score_archive = None
    if score_output is not None:
        score_archive = write_native_event_score_archive(
            score_output,
            native_mode=native_mode,
            native_score=anytime_native_score,
            video_id=native_video_id,
        )
    common = {
        "candidate_target": candidate_target,
        "origin_score": origin_score,
        "origin_target": origin_target,
        "native_mode": native_mode,
        "covered": covered,
        "native_categories": native_categories,
        "duration_minutes": duration_minutes,
        "delay_reference": "native_completion",
    }
    completion = {
        "causal_checkpoint": "predicted candidate completion",
        "frames_after_checkpoint_used_by_score": False,
        **_checkpoint_metrics(
            candidate_score=completion_candidate,
            native_score=np.asarray(completion_native_scores, dtype=np.float64),
            delay=np.zeros(len(native_mode), dtype=np.float64),
            false_alarm_score=np.asarray(
                completion_false_alarm_scores, dtype=np.float64
            ),
            detection_options=completion_detection_options,
            frozen_operating_thresholds=(
                None
                if frozen_operating_thresholds is None
                else frozen_operating_thresholds["completion"]
            ),
            **common,
        ),
    }
    anytime = {
        "causal_checkpoint": "first warning over every causal candidate prefix",
        "frames_after_checkpoint_used_by_score": False,
        **_checkpoint_metrics(
            candidate_score=anytime_candidate,
            native_score=anytime_native_score,
            delay=np.zeros(len(native_mode), dtype=np.float64),
            false_alarm_score=np.asarray(
                anytime_false_alarm_scores, dtype=np.float64
            ),
            detection_options=anytime_detection_options,
            frozen_operating_thresholds=(
                None
                if frozen_operating_thresholds is None
                else frozen_operating_thresholds["anytime"]
            ),
            **common,
        ),
    }
    result = {
        "causal_checkpoint": anytime["causal_checkpoint"],
        "future_frames_used_by_score": False,
        "checkpoints": {"completion": completion, "anytime": anytime},
        "primary_streaming_checkpoint": "anytime",
        "duration_minutes": duration_minutes,
        "per_video": per_video,
    }
    if score_archive is not None:
        assert score_output is not None
        result["score_archive"] = score_archive
        result["score_archive_sha256"] = _sha256(score_output)
    return result


def _checkpoint_relative(checkpoint: Path, value: object) -> Path:
    """A file recorded in checkpoint metadata; released checkpoints record it relative to their own directory."""

    path = Path(str(value))
    return path if path.is_absolute() or path.exists() else checkpoint.parent / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument(
        "--query-examples",
        type=Path,
        help="query-only examples linked to frozen training/validation examples",
    )
    parser.add_argument(
        "--query-prefix-evidence",
        type=Path,
        help="annotation-free prefix evidence compiled for --query-examples",
    )
    parser.add_argument(
        "--frozen-validation",
        type=Path,
        help="matching validation metrics that fix test-time FROC thresholds",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--score-output",
        type=Path,
        help="optional aligned native-event anytime score archive",
    )
    parser.add_argument("--readout-feature-root", type=Path, default=None, help="frame-aligned feature root for the two-stream readout on the query split (defaults to the training-time root recorded in the checkpoint)")
    parser.add_argument(
        "--candidate-score-output",
        type=Path,
        default=None,
        help="optional per-video candidate score sequences (evaluation bookkeeping only)",
    )
    parser.add_argument(
        "--inference-ablation",
        type=str,
        default=FULL_INFERENCE,
        help="frozen-emission ablation; several registered "
        "interventions may be composed with '+'",
    )
    parser.add_argument(
        "--rebuilt-inputs",
        action="store_true",
        help="examples and prefix evidence were rebuilt with the released observer; skip their byte-level "
        "bindings to the training-time files (the event ontology is still checked)",
    )
    parser.add_argument(
        "--prefix-evidence",
        type=Path,
        default=None,
        help="prefix evidence compiled for --examples (validation split with --rebuilt-inputs)",
    )
    args = parser.parse_args()
    ablation_parts = parse_inference_ablations(args.inference_ablation)
    if args.rebuilt_inputs and args.split != "test" and args.prefix_evidence is None:
        parser.error("--rebuilt-inputs requires --prefix-evidence outside the test split")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.score_output is not None and args.score_output.exists():
        raise FileExistsError(
            f"refusing to overwrite score archive {args.score_output}"
        )
    if args.split == "test" and args.frozen_validation is None:
        parser.error("test evaluation requires --frozen-validation")
    if args.split != "test" and args.frozen_validation is not None:
        parser.error("--frozen-validation is accepted only for test evaluation")
    candidate_dump = [] if args.candidate_score_output is not None else None
    artifact = torch.load(args.examples, map_location="cpu", weights_only=False)
    source = artifact.get("metadata")
    query_source = None
    if args.query_examples is None:
        if args.split == "test" or args.query_prefix_evidence is not None:
            parser.error("test evaluation requires query examples and query prefix evidence")
        sequences = artifact.get(args.split)
    else:
        if args.split != "test" or args.query_prefix_evidence is None:
            parser.error("query examples are accepted only for a complete test query")
        query_source, sequences = load_completion_query_artifact(
            args.query_examples,
            source_examples=args.examples,
            split=args.split,
        )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_metadata = checkpoint.get("metadata")
    checkpoint_sha256 = _sha256(args.checkpoint)
    examples_sha256 = _sha256(args.examples)
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "completion_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(sequences, Sequence)
        or not sequences
        or not isinstance(checkpoint_metadata, Mapping)
        or checkpoint_metadata.get("architecture")
        not in SUPPORTED_COMPLETION_ARCHITECTURES
        or checkpoint_metadata.get("test_split_accessed") is not False
        or (not args.rebuilt_inputs and checkpoint_metadata.get("examples_sha256") != examples_sha256)
    ):
        raise ValueError("single-actor evaluation artifacts are incompatible")
    device = torch.device(args.device)
    anytime_readout = (
        MARGINAL_PREFIX_ERROR_READOUT
        if NO_CAUSAL_DECISION_CLOCK in ablation_parts
        else CAUSAL_CANDIDATE_DECISION_CLOCK
    )
    frozen_operating_thresholds = (
        None
        if args.frozen_validation is None
        else _load_frozen_operating_thresholds(
            args.frozen_validation,
            checkpoint_sha256=checkpoint_sha256,
            examples_sha256=examples_sha256,
            inference_ablation=args.inference_ablation,
            anytime_readout=anytime_readout,
        )
    )
    language_registry = checkpoint_metadata.get("language_registry")
    if language_registry is not None:
        registry_path = _checkpoint_relative(args.checkpoint, language_registry)
        if not _intact(registry_path, checkpoint_metadata.get("language_registry_sha256")):
            raise ValueError("predictive-language registry changed after training")
        source, _ = _source_with_predictive_language_registry(
            source, registry_path, args.examples, rebuilt_examples=args.rebuilt_inputs
        )
    model = _build_model(
        source,
        device,
        independent_successors=(
            checkpoint_metadata["architecture"] == PROVIA_SINGLE_ACTOR_ARCHITECTURE
        ),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    for part in ablation_parts:
        configure_model_ablation(model, part)
    model.eval()
    transports = _load_transports(source, device, posterior_predictive_innovation=True)
    for transport in transports:
        for part in ablation_parts:
            configure_transport_ablation(transport, part)
    prefix_root = None
    prefix_registry = None
    if args.rebuilt_inputs:
        prefix_root = args.prefix_evidence
    else:
        raw_prefix_root = checkpoint_metadata.get("prefix_evidence")
        raw_prefix_sha256 = checkpoint_metadata.get(
            "prefix_evidence_registry_sha256"
        )
        if not isinstance(raw_prefix_root, str) or not isinstance(
            raw_prefix_sha256, str
        ):
            raise ValueError("checkpoint lacks prefix evidence")
        prefix_root = Path(raw_prefix_root)
        if _sha256(prefix_root / "registry.json") != raw_prefix_sha256:
            raise ValueError("prefix evidence changed after training")
    if query_source is None:
        prefix_registry = _load_prefix_registry(prefix_root, args.examples)
    else:
        assert args.query_examples is not None
        assert args.query_prefix_evidence is not None
        prefix_root = args.query_prefix_evidence
        prefix_registry = load_completion_query_prefix_registry(
            prefix_root,
            query_examples=args.query_examples,
            source_examples=args.examples,
            split=args.split,
        )
    registry = checkpoint_metadata.get("expert_effect_registry")
    expert_effects = None
    if registry is not None:
        if not isinstance(registry, str):
            raise ValueError("expert-effect registry metadata are malformed")
        registry_path = _checkpoint_relative(args.checkpoint, registry)
        if not _intact(registry_path, checkpoint_metadata.get("expert_effect_registry_sha256")):
            raise ValueError("expert-effect registry changed after training")
        expert_effects = _load_expert_effects(
            source, registry_path, args.examples, transports, device, rebuilt_examples=args.rebuilt_inputs
        )
    feature_root = Path(
        str(
            source["feature_root"]
            if query_source is None
            else query_source["feature_root"]
        )
    )
    result = {
        "schema": "single_actor_evaluation",
        "split": args.split,
        "anytime_readout": anytime_readout,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "examples": str(args.examples.resolve()),
        "examples_sha256": examples_sha256,
        "query_examples": (
            None if args.query_examples is None else str(args.query_examples.resolve())
        ),
        "query_examples_sha256": (
            None if args.query_examples is None else _sha256(args.query_examples)
        ),
        "frozen_validation": (
            None
            if args.frozen_validation is None
            else str(args.frozen_validation.resolve())
        ),
        "frozen_validation_sha256": (
            None
            if args.frozen_validation is None
            else _sha256(args.frozen_validation)
        ),
        "inference_ablation": args.inference_ablation,
        "inference_ablation_definition": " + ".join(
            INFERENCE_ABLATION_DEFINITIONS[part] for part in ablation_parts
        ),
        "frozen_learned_emissions": True,
        "evaluation_device": device.type,
        "evaluation_dtype": "float32",
        "metrics": _evaluate(
            model,
            sequences,
            feature_root,
            prefix_root,
            prefix_registry,
            transports,
            expert_effects,
            device,
            args.inference_ablation,
            args.score_output,
            frozen_operating_thresholds,
            candidate_dump=candidate_dump,
        ),
        "test_split_accessed": args.split == "test",
    }
    _atomic_json(args.output, result)
    if args.candidate_score_output is not None:
        np.savez(args.candidate_score_output, videos=np.array(candidate_dump, dtype=object))
    print(json.dumps(result["metrics"]["checkpoints"], sort_keys=True))


if __name__ == "__main__":
    main()
