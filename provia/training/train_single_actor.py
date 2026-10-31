"""Train latent causal-decision paths on fully predicted completion streams."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from torch.nn.utils import clip_grad_norm_

from provia.observer.factors import EVENT_FACTORS
from provia.training.common import (
    _atomic_json,
    _seed_everything,
    _intact,
    _sha256,
)
from provia.training.completion_inputs import (
    EARLY_STOPPING_EPOCHS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIM,
    TARGET_CANDIDATES_PER_UPDATE,
    WEIGHT_DECAY,
    _correction_is_supervised,
    _load_expert_effects,
    _load_transports,
)
from provia.training.distributed_runtime import (
    DistributedRuntime,
    atomic_torch_save,
    balanced_rank_indices,
    computational_resume_contract,
    cpu_state_dict,
    gather_objects,
    gather_rng_states,
    initialize_distributed,
    optimizer_to_device,
    restore_rng_state,
    sum_scalars,
    update_groups,
    wrap_distributed,
)
from provia.model.completion_transport import FactorizedCompletionTransport
from provia.model.mode_filter import ERROR
from provia.model.step_effects import ExpertEffectPosterior
from provia.model.interval_transport import IntervalContinuationLanguage
from provia.model.ablation import (
    FULL_INFERENCE,
    NO_PROCEDURE_FRONTIER,
    NO_PERSISTENT_PROCEDURE_FRONTIER,
    project_to_static_frontier,
    static_frontier_probability,
    validate_inference_ablation,
)
from provia.model.observer_state import CausalWindowEncoder
from provia.model.single_actor import (
    PROVIA_SINGLE_ACTOR_ARCHITECTURE,
    PARTITION_CANDIDATE_TOPOLOGY,
    PROVIA_SHARED_SUCCESSOR_ARCHITECTURE,
    ProviaSingleActor,
    ProviaSingleActorSharedSuccessor,
    jeffreys_partition_mode_probability,
    partition_mode_nll,
    prior_preserving_partition_mode_log_score,
)
from provia.model.state_mark import RECURRENT_CANDIDATE_ENCODER






def _partition_class_probability(
    sequences: Sequence[Mapping[str, object]],
    *,
    correction_supervised: bool,
) -> Tensor:
    """Fold-training mode law for exhaustive non-overlapping candidate cells."""

    origin = torch.cat(
        [sequence["origin_target"].bool() for sequence in sequences]  # type: ignore[union-attr]
    )
    mode = torch.cat(
        [sequence["mode_target"].long() for sequence in sequences]  # type: ignore[union-attr]
    )
    return jeffreys_partition_mode_probability(
        origin,
        mode,
        correction_supervised=correction_supervised,
    )


FIXED_OPERATING_PROBABILITY = 0.5




class _DistributedSequenceLoss(nn.Module):
    """Expose one complete causal video as one DDP forward pass."""

    def __init__(self, model: ProviaSingleActorSharedSuccessor) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        sequence: Mapping[str, object] | None,
        *arguments: object,
        **keywords: object,
    ) -> object:
        if sequence is None:
            zero = next(self.model.parameters()).new_zeros(())
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    zero = zero + parameter.reshape(-1)[0] * 0.0
            return zero
        return _sequence_forward(
            self.model,
            sequence,
            *arguments,
            **keywords,
        )


def _global_stream(
    feature_root: Path, video_id: str
) -> tuple[Tensor, Tensor]:
    path = feature_root / f"{video_id}_stream_features.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        global_feature = np.asarray(
            payload["global_features"], dtype=np.float32
        ).copy()
        timestamp = np.asarray(payload["timestamps_sec"], dtype=np.float64).copy()
    if (
        global_feature.ndim != 2
        or timestamp.shape != (len(global_feature),)
        or not len(global_feature)
        or not np.isfinite(global_feature).all()
        or not np.isfinite(timestamp).all()
        or np.any(np.diff(timestamp) <= 0)
    ):
        raise ValueError(f"{path}: malformed causal stream")
    return torch.from_numpy(global_feature), torch.from_numpy(timestamp)


def _endpoint_factors(
    sequence: Mapping[str, object], index: int, device: torch.device
) -> dict[str, Tensor]:
    return {
        name: sequence[f"factor_{name}"][index].to(  # type: ignore[index,union-attr]
            device=device, dtype=torch.float32
        )
        for name in EVENT_FACTORS
    }


def _candidate_context(
    global_feature: Tensor, start: int, end: int, device: torch.device
) -> tuple[Tensor, int]:
    if not 1 <= start <= end < len(global_feature):
        raise ValueError("decoded candidate lies outside its causal stream")
    context_start = max(0, start - CausalWindowEncoder.RECEPTIVE_FIELD + 1)
    return (
        global_feature[context_start : end + 1].to(
            device=device, dtype=torch.float32
        ),
        start - context_start,
    )


def _safe_ranking(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(target) == 0 or len(np.unique(target)) < 2:
        return {"average_precision": float("nan"), "auroc": float("nan")}
    return {
        "average_precision": float(average_precision_score(target, score)),
        "auroc": float(roc_auc_score(target, score)),
    }


def _examples_binding_hashes(examples: Path) -> set[str]:
    """Hashes under which registries may be bound to ``examples``: its own sha256 and, for gap-candidate
    examples (every hand always carries an open step), the sha256 of the base examples whose prefix evidence,
    effects and languages they inherit (recorded by the examples builder)."""
    hashes = {_sha256(examples)}
    try:
        meta = torch.load(examples, map_location="cpu", weights_only=False).get("metadata", {})
        base = meta.get("gap_base_examples_sha256") if isinstance(meta, Mapping) else None
        if base: hashes.add(str(base))
    except Exception:
        pass
    return hashes


def _load_prefix_registry(
    root: Path, examples: Path
) -> Mapping[str, object]:
    path = root / "registry.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    registry = json.loads(path.read_text())
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema")
        != "prefix_observer_evidence"
        or registry.get("examples_sha256") not in _examples_binding_hashes(examples)
        or registry.get("causal") is not True
        or registry.get("parameters_fitted") is not False
        or registry.get("test_split_accessed") is not False
        or not isinstance(registry.get("videos"), Mapping)
    ):
        raise ValueError("prefix-evidence registry is malformed or incompatible")
    for video_id, raw in registry["videos"].items():  # type: ignore[union-attr]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{video_id}: prefix-evidence record is malformed")
        file = root / str(raw.get("file"))
        if not file.is_file() or _sha256(file) != raw.get("sha256"):
            raise ValueError(f"{video_id}: prefix evidence changed after compilation")
    return registry


def _same_node_possibility(
    first: Mapping[str, object], second: Mapping[str, object], exact: bool = True
) -> bool:
    """Node possibilities of one ontology; a rebuilt observer pass may move them by floating-point rounding."""

    if exact or set(first) != set(second):
        return first == second
    for name in first:
        a, b = np.asarray(first[name], dtype=np.float64), np.asarray(second[name], dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-4, atol=1e-8):
            return False
    return True


def _source_with_predictive_language_registry(
    source: Mapping[str, object],
    registry_path: Path,
    examples_path: Path,
    rebuilt_examples: bool = False,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Replace the continuation process, preserving its ontology.

    ``rebuilt_examples``: the examples were rebuilt from a released observer, so the byte-level bindings to the
    examples and languages the registry was compiled from are skipped; the event ontology is still checked.
    """

    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = json.loads(registry_path.read_text())
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema") != "bayesian_predictive_language_registry"
        or (not rebuilt_examples and registry.get("source_examples_sha256") not in _examples_binding_hashes(examples_path))
        or int(registry.get("fold", -1)) != int(source["fold"])
        or registry.get("anomaly_labels_used") is not False
        or registry.get("expert_event_annotations_used") is not False
        or registry.get("validation_videos_used") is not False
        or registry.get("test_videos_used") is not False
        or registry.get("similarity_threshold") is not False
        or registry.get("selected_state_count") is not False
    ):
        raise ValueError("predictive-language registry is malformed")
    task_ids = tuple(str(value) for value in source["task_ids"])  # type: ignore[index]
    source_records = source.get("languages")
    records = registry.get("tasks")
    if not isinstance(source_records, Mapping) or not isinstance(records, Mapping):
        raise ValueError("predictive-language registry lacks task records")
    languages: dict[str, dict[str, object]] = {}
    for task in task_ids:
        original = source_records.get(task)
        record = records.get(task)
        if not isinstance(original, Mapping) or not isinstance(record, Mapping):
            raise ValueError(f"task {task} lacks predictive-language provenance")
        original_path = Path(str(original["path"]))
        path = registry_path.parent / str(record["language"])
        if (
            not original_path.is_file()
            or _sha256(original_path) != original.get("sha256")
            or (not rebuilt_examples and record.get("source_language_sha256") != original.get("sha256"))
            or not path.is_file()
            or not _intact(path, record.get("sha256"))
        ):
            raise ValueError(f"task {task} language provenance changed")
        original_language = IntervalContinuationLanguage.from_dict(
            json.loads(original_path.read_text())
        )
        predictive_language = IntervalContinuationLanguage.from_dict(
            json.loads(path.read_text())
        )
        if (
            predictive_language.factor_labels != original_language.factor_labels
            or predictive_language.node_ids != original_language.node_ids
            or predictive_language.node_roles != original_language.node_roles
            or not _same_node_possibility(
                predictive_language.node_possibility,
                original_language.node_possibility,
                exact=not rebuilt_examples,
            )
            or predictive_language.source_trace_count
            != len(record.get("expert_videos", ()))
        ):
            raise ValueError(
                f"task {task} predictive process changed the frozen event ontology"
            )
        languages[task] = {
            "path": str(path.resolve()),
            "sha256": record.get("sha256"),
        }
    if set(records) != set(task_ids):
        raise ValueError("predictive-language registry task set changed")
    overridden = dict(source)
    overridden["languages"] = languages
    return overridden, registry


def _prefix_evidence(
    root: Path,
    registry: Mapping[str, object],
    video_id: str,
    timestamp: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    records = registry["videos"]
    if not isinstance(records, Mapping) or video_id not in records:
        raise ValueError(f"{video_id}: no prefix evidence")
    record = records[video_id]
    if not isinstance(record, Mapping):
        raise ValueError(f"{video_id}: malformed prefix-evidence record")
    path = root / str(record["file"])
    with np.load(path, allow_pickle=False) as payload:
        evidence_timestamp = np.asarray(
            payload["timestamps_sec"], dtype=np.float64
        ).copy()
        event = np.asarray(payload["event_probability"], dtype=np.float32).copy()
        factors = {
            name: np.asarray(
                payload[f"factor_{name}_probability"], dtype=np.float32
            ).copy()
            for name in EVENT_FACTORS
        }
    reference_time = timestamp.numpy()
    if (
        evidence_timestamp.shape != reference_time.shape
        or not np.array_equal(evidence_timestamp, reference_time)
        or event.shape != reference_time.shape
        or any(len(value) != len(reference_time) for value in factors.values())
        or not np.isfinite(event).all()
        or np.any((event < 0) | (event > 1))
        or any(
            not np.isfinite(value).all() or np.any(value < 0)
            for value in factors.values()
        )
    ):
        raise ValueError(f"{video_id}: prefix evidence and feature stream disagree")
    return torch.from_numpy(event), {
        name: torch.from_numpy(value) for name, value in factors.items()
    }


def _sequence_forward(
    model: ProviaSingleActorSharedSuccessor,
    sequence: Mapping[str, object],
    global_feature: Tensor,
    timestamp: Tensor,
    prefix_event: Tensor | None,
    prefix_factor: Mapping[str, Tensor] | None,
    transport: FactorizedCompletionTransport,
    expert_effect: ExpertEffectPosterior | None,
    device: torch.device,
    *,
    training: bool,
    class_probability: Tensor | None = None,
    correction_supervised: bool = False,
    deletion_probability: Tensor | float | None = None,
    report_prefix: bool = True,
    inference_ablation: str = FULL_INFERENCE,
) -> dict[str, object]:
    inference_ablation = validate_inference_ablation(inference_ablation)
    if training and inference_ablation != FULL_INFERENCE:
        raise ValueError("inference-law ablations require a frozen model")
    decoded = sequence["decoded_index"]
    origin = sequence["origin_target"]
    mode = sequence["mode_target"]
    if not isinstance(decoded, Tensor) or decoded.ndim != 1 or decoded.dtype != torch.long:
        raise ValueError("decoded candidate indices are malformed")
    events = len(decoded)
    if (
        not isinstance(origin, Tensor)
        or not isinstance(mode, Tensor)
        or origin.shape != (events,)
        or mode.shape != (events,)
        or torch.any(decoded[1:] <= decoded[:-1])
        or int(decoded[-1]) >= len(global_feature)
        or len(timestamp) != len(global_feature)
    ):
        raise ValueError("candidate sequence is malformed")
    encoded_stream = model.energy.window_encoder.forward_stream(
            global_feature.to(device=device, dtype=torch.float32)
        )
    expert_intent_key = model.energy.build_frontier_key(transport)
    candidate_state_paths: tuple[Tensor, ...] | None = None
    if encoded_stream is None or prefix_event is None or prefix_factor is None:
        raise ValueError("coupled candidate batching requires causal prefix evidence")
    encoded_candidates = []
    factor_candidates = []
    event_candidates = []
    previous_candidate_end = 0
    for candidate_index in range(events):
        candidate_end = int(decoded[candidate_index])
        candidate_start = previous_candidate_end + 1
        endpoint = _endpoint_factors(sequence, candidate_index, device)
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
        if inference_ablation
        in {NO_PROCEDURE_FRONTIER, NO_PERSISTENT_PROCEDURE_FRONTIER}
        else None
    )
    history = model.energy.initial_history(global_feature[:1].to(device))
    endpoint_scores = []
    maximum_prefix_scores = []
    endpoint_probabilities = []
    targets = []
    origins = []
    first_alarm_lead_sec = []
    detected_error = 0
    error_events = 0
    false_alarms = 0
    loss_numerator = global_feature.new_zeros((), device=device)
    loss_exposure = global_feature.new_zeros((), device=device)
    joint_loss_sum = global_feature.new_zeros((), device=device)
    previous_end = 0
    for index in range(events):
        if static_frontier is not None:
            belief = project_to_static_frontier(belief, static_frontier)
        end = int(decoded[index])
        start = previous_end + 1
        context, offset = _candidate_context(global_feature, start, end, device)
        factors = _endpoint_factors(sequence, index, device)
        effect = (
            sequence["statistic"][index].to(  # type: ignore[index,union-attr]
                device=device, dtype=torch.float32
            )
            if expert_effect is not None
            else None
        )
        if prefix_event is None or prefix_factor is None:
            raise ValueError(
                "procedure-conditioned streaming inference requires prefix evidence"
            )
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
            factors,
            belief,
            history,
            transport,
            effect_statistic=effect,
            expert_effect=expert_effect,
            encoded_prefix_hidden=encoded_stream[start : end + 1],
            candidate_state_hidden=candidate_state_paths[index],
            frontier_key=expert_intent_key,
            endpoint_only=training or not report_prefix,
            marginalize_procedure_frontier=inference_ablation == NO_PROCEDURE_FRONTIER,
        )
        target_origin = origin[index].to(device=device, dtype=torch.bool)
        target_mode = mode[index].to(device=device, dtype=torch.long)
        target_error = target_origin & (target_mode == ERROR)
        probability = output.explanation_probability.unsqueeze(0)
        if class_probability is not None:
            numerator, exposure = prior_preserving_partition_mode_log_score(
                probability,
                target_origin.unsqueeze(0),
                target_mode.unsqueeze(0),
                class_probability,
                correction_supervised=correction_supervised,
            )
            joint_loss = partition_mode_nll(
                probability,
                target_origin.unsqueeze(0),
                target_mode.unsqueeze(0),
                correction_supervised=correction_supervised,
            )
            loss_numerator = loss_numerator + numerator
            loss_exposure = loss_exposure + exposure
            joint_loss_sum = joint_loss_sum + joint_loss
        if not training:
            prefix_score = output.prefix_anomaly_probability
            maximum = prefix_score.max()
            alarm = torch.nonzero(
                prefix_score >= FIXED_OPERATING_PROBABILITY, as_tuple=False
            ).flatten()
            is_error = bool(target_error.detach().cpu())
            if is_error:
                error_events += 1
                if len(alarm):
                    detected_error += 1
                    alarm_index = start + int(alarm[0])
                    if not report_prefix:
                        alarm_index = end
                    first_alarm_lead_sec.append(
                        float(timestamp[end] - timestamp[alarm_index])
                    )
            elif len(alarm):
                false_alarms += 1
            endpoint_scores.append(output.anomaly_probability.detach().cpu())
            maximum_prefix_scores.append(maximum.detach().cpu())
            endpoint_probabilities.append(
                output.explanation_probability.detach().cpu()
            )
            targets.append(target_error.detach().cpu())
            origins.append(target_origin.detach().cpu())
        belief = output.next_procedure_mode_lifecycle_probability.detach()
        if training:
            # The event boundary is the model's natural sufficient-state
            # boundary.  This is one-step prequential differentiation, not a
            # tuned truncation length; forward filtering remains exact.
            history = model.completion_history(
                history.detach(),
                output.prefix_hidden[-1].detach(),
                factors,
                output.explanation_probability.detach(),
            )
        else:
            history = output.next_factual_history
        if getattr(model, "reset_factual_history", False):
            history = model.energy.initial_history(global_feature[:1].to(device))
        previous_end = end
    if not events:
        raise ValueError("candidate sequence is empty")
    result: dict[str, object] = {
        "loss_numerator": loss_numerator,
        "loss_exposure": loss_exposure,
        "joint_loss_sum": joint_loss_sum,
        "events": events,
    }
    if training:
        return result
    duration = max(
        float(timestamp[-1] - timestamp[0]),
        torch.finfo(torch.float64).eps,
    )
    result.update({
        "endpoint_score": torch.stack(endpoint_scores),
        "maximum_prefix_score": torch.stack(maximum_prefix_scores),
        "explanation_probability": torch.stack(endpoint_probabilities),
        "target": torch.stack(targets).bool(),
        "origin": torch.stack(origins).bool(),
        "error_events": error_events,
        "detected_error": detected_error,
        "false_alarms": false_alarms,
        "duration_sec": duration,
        "first_alarm_lead_sec": tuple(first_alarm_lead_sec),
    })
    native_assignment = sequence.get("native_assignment")
    native_mode = sequence.get("native_mode")
    if isinstance(native_assignment, Tensor) and isinstance(native_mode, Tensor):
        if native_assignment.shape != (events,) or native_mode.ndim != 1:
            raise ValueError("native event evaluation metadata is malformed")
        candidate_anomaly = torch.stack(endpoint_probabilities)[:, 1 + ERROR]
        native_maximum = candidate_anomaly.new_zeros(len(native_mode))
        native_covered = torch.zeros(len(native_mode), dtype=torch.bool)
        assignment_cpu = native_assignment.cpu()
        for native_position in range(len(native_mode)):
            candidate_position = torch.nonzero(
                assignment_cpu == native_position, as_tuple=False
            ).flatten()
            if len(candidate_position):
                native_maximum[native_position] = candidate_anomaly.index_select(
                    0, candidate_position
                ).max()
                native_covered[native_position] = True
        result["native_maximum_score"] = native_maximum
        result["native_target"] = (native_mode == ERROR).cpu()
        result["native_covered"] = native_covered
    return result


@torch.no_grad()
def _evaluate(
    model: ProviaSingleActorSharedSuccessor,
    sequences: Sequence[Mapping[str, object]],
    feature_root: Path,
    prefix_root: Path | None,
    prefix_registry: Mapping[str, object] | None,
    transports: Sequence[FactorizedCompletionTransport],
    expert_effects: Sequence[ExpertEffectPosterior] | None,
    device: torch.device,
    class_probability: Tensor,
    correction_supervised: bool,
    runtime: DistributedRuntime | None = None,
    deletion_probability: Tensor | float | None = None,
    report_prefix: bool = True,
) -> dict[str, object]:
    model.eval()
    balanced_sum = 0.0
    balanced_exposure = 0.0
    joint_sum = 0.0
    events = 0
    score_events = 0
    candidate_covered_events = 0
    exact_one_sum = 0.0
    endpoint = []
    maximum = []
    target = []
    origin = []
    error_events = 0
    detected_error = 0
    false_alarms = 0
    duration_sec = 0.0
    lead = []
    native_conditional = []
    native_joint = []
    native_maximum = []
    native_target = []
    native_covered = []
    local_sequences = (
        sequences
        if runtime is None
        else sequences[runtime.rank :: runtime.world_size]
    )
    for sequence in local_sequences:
        task_index = int(sequence["task_index"])
        video_id = str(sequence["video_id"])
        feature, timestamp = _global_stream(feature_root, video_id)
        if prefix_root is not None and prefix_registry is not None:
            prefix_event, prefix_factor = _prefix_evidence(
                prefix_root, prefix_registry, video_id, timestamp
            )
        else:
            prefix_event = None
            prefix_factor = None
        result = _sequence_forward(
            model,
            sequence,
            feature,
            timestamp,
            prefix_event,
            prefix_factor,
            transports[task_index],
            (
                None
                if expert_effects is None
                else expert_effects[task_index]
            ),
            device,
            training=False,
            class_probability=class_probability,
            correction_supervised=correction_supervised,
            deletion_probability=deletion_probability,
            report_prefix=report_prefix,
        )
        balanced_sum += float(result["loss_numerator"])
        balanced_exposure += float(result["loss_exposure"])
        joint_sum += float(result["joint_loss_sum"])
        events += int(result["events"])
        score_events += int(
            result.get("covered_native_events", result["events"])
        )
        candidate_covered_events += int(
            result.get("candidate_covered_native_events", result["events"])
        )
        exact_one_sum += float(
            result.get("mean_event_observation_probability", 0.0)
        ) * int(result.get("covered_native_events", 0))
        endpoint.append(result["endpoint_score"])
        maximum.append(result["maximum_prefix_score"])
        target.append(result["target"])
        origin.append(result["origin"])
        error_events += int(result["error_events"])
        detected_error += int(result["detected_error"])
        false_alarms += int(result["false_alarms"])
        duration_sec += float(result["duration_sec"])
        lead.extend(result["first_alarm_lead_sec"])
        if "native_maximum_score" in result:
            native_maximum.append(result["native_maximum_score"])
            native_target.append(result["native_target"])
            native_covered.append(result["native_covered"])
    payload: dict[str, object] = {
        "balanced_sum": balanced_sum,
        "balanced_exposure": balanced_exposure,
        "joint_sum": joint_sum,
        "events": events,
        "score_events": score_events,
        "candidate_covered_events": candidate_covered_events,
        "exact_one_sum": exact_one_sum,
        "endpoint": (
            torch.cat(endpoint).numpy()
            if endpoint
            else np.empty(0, dtype=np.float32)
        ),
        "maximum": (
            torch.cat(maximum).numpy()
            if maximum
            else np.empty(0, dtype=np.float32)
        ),
        "target": (
            torch.cat(target).numpy().astype(np.int64)
            if target
            else np.empty(0, dtype=np.int64)
        ),
        "origin": (
            torch.cat(origin).numpy().astype(np.int64)
            if origin
            else np.empty(0, dtype=np.int64)
        ),
        "error_events": error_events,
        "detected_error": detected_error,
        "false_alarms": false_alarms,
        "duration_sec": duration_sec,
        "lead": np.asarray(lead, dtype=np.float64),
        "native_conditional": (
            torch.cat(native_conditional).numpy()
            if native_conditional
            else np.empty(0, dtype=np.float32)
        ),
        "native_joint": (
            torch.cat(native_joint).numpy()
            if native_joint
            else np.empty(0, dtype=np.float32)
        ),
        "native_maximum": (
            torch.cat(native_maximum).numpy()
            if native_maximum
            else np.empty(0, dtype=np.float32)
        ),
        "native_target": (
            torch.cat(native_target).numpy().astype(np.int64)
            if native_target
            else np.empty(0, dtype=np.int64)
        ),
        "native_covered": (
            torch.cat(native_covered).numpy().astype(np.int64)
            if native_covered
            else np.empty(0, dtype=np.int64)
        ),
    }
    payloads = (
        (payload,) if runtime is None else gather_objects(runtime, payload)
    )
    balanced_sum = sum(float(value["balanced_sum"]) for value in payloads)
    balanced_exposure = sum(
        float(value["balanced_exposure"]) for value in payloads
    )
    joint_sum = sum(float(value["joint_sum"]) for value in payloads)
    events = sum(int(value["events"]) for value in payloads)
    score_events = sum(int(value["score_events"]) for value in payloads)
    candidate_covered_events = sum(
        int(value["candidate_covered_events"]) for value in payloads
    )
    exact_one_sum = sum(float(value["exact_one_sum"]) for value in payloads)
    endpoint_array = np.concatenate(
        [np.asarray(value["endpoint"]) for value in payloads]
    )
    maximum_array = np.concatenate(
        [np.asarray(value["maximum"]) for value in payloads]
    )
    target_array = np.concatenate(
        [np.asarray(value["target"], dtype=np.int64) for value in payloads]
    )
    origin_array = np.concatenate(
        [np.asarray(value["origin"], dtype=np.int64) for value in payloads]
    )
    error_events = sum(int(value["error_events"]) for value in payloads)
    detected_error = sum(int(value["detected_error"]) for value in payloads)
    false_alarms = sum(int(value["false_alarms"]) for value in payloads)
    duration_sec = sum(float(value["duration_sec"]) for value in payloads)
    lead_array = np.concatenate(
        [np.asarray(value["lead"], dtype=np.float64) for value in payloads]
    )
    native_maximum_array = np.concatenate(
        [np.asarray(value["native_maximum"]) for value in payloads]
    )
    native_target_array = np.concatenate(
        [np.asarray(value["native_target"], dtype=np.int64) for value in payloads]
    )
    native_covered_array = np.concatenate(
        [np.asarray(value["native_covered"], dtype=np.int64) for value in payloads]
    )
    return {
        "prior_preserving_case_control_log_score": (
            balanced_sum / balanced_exposure
        ),
        "joint_negative_log_likelihood": joint_sum / score_events,
        "decoded_candidates": events,
        "covered_native_events": (
            int(len(native_target_array)) if len(native_target_array) else None
        ),
        "candidate_covered_native_events": (
            int(native_covered_array.sum()) if len(native_covered_array) else None
        ),
        "native_event_coverage": (
            float(native_covered_array.mean())
            if len(native_covered_array)
            else None
        ),
        "mean_event_observation_probability": None,
        "genuine_candidates": int(origin_array.sum()),
        "error_candidates": int(target_array.sum()),
        "completion_ranking": _safe_ranking(target_array, endpoint_array),
        "best_causal_prefix_ranking": _safe_ranking(
            target_array, maximum_array
        ),
        "native_event_conditional_ranking": None,
        "native_event_joint_ranking": None,
        "native_event_maximum_ranking": (
            _safe_ranking(native_target_array, native_maximum_array)
            if len(native_target_array)
            else None
        ),
        "fixed_bayes_operation": {
            "probability": FIXED_OPERATING_PROBABILITY,
            "error_event_recall": (
                detected_error / error_events if error_events else float("nan")
            ),
            "detected_error_events": detected_error,
            "error_events": error_events,
            "false_alarms": false_alarms,
            "false_alarms_per_minute": false_alarms / (duration_sec / 60.0),
            "median_lead_to_predicted_completion_sec": (
                float(np.median(lead_array)) if len(lead_array) else float("nan")
            ),
            "mean_lead_to_predicted_completion_sec": (
                float(lead_array.mean()) if len(lead_array) else float("nan")
            ),
        },
    }


def _build_model(
    source: Mapping[str, object],
    device: torch.device,
    *,
    independent_successors: bool,
) -> ProviaSingleActorSharedSuccessor:
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }
    origin = source.get("origin_channel")
    cache = source.get("cache")
    if not isinstance(origin, Mapping) or not isinstance(cache, Mapping):
        raise ValueError("examples lack frozen origin or feature metadata")
    model_type = (
        ProviaSingleActor
        if independent_successors
        else ProviaSingleActorSharedSuccessor
    )
    return model_type(
        int(cache["global_dim"]),
        labels,
        torch.tensor(source["mode_transition_probability"], dtype=torch.float32),
        torch.tensor(source["mode_initial_probability"], dtype=torch.float32),
        model_dim=MODEL_DIM,
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--prefix-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-effect-registry", type=Path)
    parser.add_argument(
        "--language-registry",
        type=Path,
        help=(
            "successful-demonstration predictive-state registry; "
            "changes procedure state identity without changing event ontology"
        ),
    )
    parser.add_argument(
        "--no-factual-history",
        action="store_true",
        help=(
            "reset the factual history before every candidate during training "
            "and validation; a capacity-matched variant with identical parameters"
        ),
    )
    parser.add_argument(
        "--shared-successor",
        action="store_true",
        help="share one latent successor between emission and transport",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume exactly from output-dir/latest.pt at an epoch boundary",
    )
    args = parser.parse_args()
    independent_successors = not args.shared_successor
    if args.prefix_evidence is None:
        raise ValueError("training requires causal prefix evidence")
    _seed_everything(args.seed)
    runtime = initialize_distributed(args.device)
    device = runtime.device
    output_nonempty = args.output_dir.exists() and any(
        args.output_dir.iterdir()
    )
    if output_nonempty and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite non-empty {args.output_dir}; pass --resume "
            "only for an exact latest.pt checkpoint"
        )
    if args.resume and not (args.output_dir / "latest.pt").is_file():
        raise FileNotFoundError(
            f"cannot resume without {args.output_dir / 'latest.pt'}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime.barrier()
    artifact = torch.load(args.examples, map_location="cpu", weights_only=False)
    source = artifact.get("metadata")
    train = artifact.get("train")
    validation = artifact.get("validation")
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "completion_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(train, Sequence)
        or not isinstance(validation, Sequence)
        or not train
        or not validation
    ):
        raise ValueError("single-actor examples are malformed")
    language_registry = None
    if args.language_registry is not None:
        source, language_registry = _source_with_predictive_language_registry(
            source, args.language_registry, args.examples
        )
    feature_root = Path(str(source["feature_root"]))
    prefix_registry = (
        _load_prefix_registry(args.prefix_evidence, args.examples)
        if args.prefix_evidence is not None
        else None
    )
    transports = _load_transports(
        source,
        device,
        posterior_predictive_innovation=True,
    )
    expert_effects = (
        _load_expert_effects(
            source,
            args.expert_effect_registry,
            args.examples,
            transports,
            device,
        )
        if args.expert_effect_registry is not None
        else None
    )
    correction_supervised = _correction_is_supervised(train)
    model = _build_model(
        source,
        device,
        independent_successors=independent_successors,
    )
    if args.no_factual_history:
        model.reset_factual_history = True
    class_probability = _partition_class_probability(
        train, correction_supervised=correction_supervised
    ).to(device)
    deletion_probability = float(
        source.get("missed_event_probability", float("nan"))
    )
    if not 0.0 < deletion_probability < 1.0:
        raise ValueError("examples lack a valid fold-training deletion channel")
    sequence_trainer = wrap_distributed(
        _DistributedSequenceLoss(model), runtime
    )
    # Rank-local stochastic streams are checkpointed independently. The model
    # parameters themselves have already been broadcast by DDP from rank zero.
    _seed_everything(args.seed + 104729 * runtime.rank)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    metadata = {
        "schema": "single_actor_training",
        "architecture": (
            PROVIA_SINGLE_ACTOR_ARCHITECTURE
            if independent_successors
            else PROVIA_SHARED_SUCCESSOR_ARCHITECTURE
        ),
        "candidate_state_encoder": RECURRENT_CANDIDATE_ENCODER,
        "candidate_state_receptive_field": None,
        "candidate_topology": PARTITION_CANDIDATE_TOPOLOGY,
        "null_origin_state": False,
        "candidate_identity_targets_used_by_loss": True,
        "deletion_probability": deletion_probability,
        "alignment_timing_tolerance": None,
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "target_candidates_per_update": TARGET_CANDIDATES_PER_UPDATE,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_epochs": EARLY_STOPPING_EPOCHS,
            "ranking_or_latency_used_for_selection": False,
        },
        "model_dim": MODEL_DIM,
        "task_or_dataset_embedding": False,
        "dataset_specific_head": False,
        "score_fusion": False,
        "manual_threshold_selection": False,
        "decision_time_target": False,
        "latency_loss_or_reward": False,
        "fixed_report_probability": FIXED_OPERATING_PROBABILITY,
        "examples": str(args.examples.resolve()),
        "examples_sha256": _sha256(args.examples),
        "prefix_evidence": (
            None
            if args.prefix_evidence is None
            else str(args.prefix_evidence.resolve())
        ),
        "prefix_evidence_registry_sha256": (
            None
            if args.prefix_evidence is None
            else _sha256(args.prefix_evidence / "registry.json")
        ),
        "expert_effect_registry": (
            None
            if args.expert_effect_registry is None
            else str(args.expert_effect_registry.resolve())
        ),
        "expert_effect_registry_sha256": (
            None
            if args.expert_effect_registry is None
            else _sha256(args.expert_effect_registry)
        ),
        "expert_prefix_dynamics": None,
        "language_registry": (
            None
            if args.language_registry is None
            else str(args.language_registry.resolve())
        ),
        "language_registry_sha256": (
            None
            if args.language_registry is None
            else _sha256(args.language_registry)
        ),
        "procedure_state_induction": (
            None
            if language_registry is None
            else {
                "method": language_registry["method"],
                "model_prior_odds": language_registry["model_prior_odds"],
                "categorical_prior": language_registry["categorical_prior"],
                "anomaly_labels_used": False,
                "validation_or_test_used": False,
            }
        ),
        # Frontier-conditioned emission energies are shrunk by the language's
        # Jeffreys posterior-predictive data fraction when the compiled
        # languages retain reference evidence counts; otherwise the reliability is one.
        "structural_evidence_reliability": bool(
            all(
                transport.language.state_evidence_count is not None
                for transport in transports
            )
        ),
        "fold": int(source["fold"]),
        "seed": args.seed,
        "parameters": {
            "total": sum(value.numel() for value in model.parameters()),
            "trainable": sum(
                value.numel()
                for value in model.parameters()
                if value.requires_grad
            ),
        },
        "distributed": {
            "world_size": runtime.world_size,
            "backend": (
                torch.distributed.get_backend() if runtime.enabled else None
            ),
            
        },
        "test_split_accessed": False,
    }
    if runtime.is_main and not args.resume:
        _atomic_json(args.output_dir / "run.json", metadata)
    generator = torch.Generator().manual_seed(args.seed)
    started = time.monotonic()
    elapsed_before = 0.0
    latest_path = args.output_dir / "latest.pt"
    records: list[dict[str, object]]

    def write_metrics() -> None:
        if not runtime.is_main:
            return
        path = args.output_dir / "metrics.jsonl"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(value, sort_keys=True) + "\n" for value in records
            )
        )
        temporary.replace(path)

    if args.resume:
        checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )
        saved_metadata = checkpoint.get("metadata") if isinstance(checkpoint, Mapping) else None
        saved_contract = (
            checkpoint.get("resume_contract")
            if isinstance(checkpoint, Mapping)
            else None
        )
        if saved_contract is None and isinstance(saved_metadata, Mapping):
            saved_contract = computational_resume_contract(saved_metadata)
        current_contract = computational_resume_contract(metadata)
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("schema") != "distributed_epoch_resume"
            or saved_contract != current_contract
            or int(checkpoint.get("world_size", -1)) != runtime.world_size
        ):
            raise ValueError(
                "latest.pt is incompatible with this model, data, seed, or "
                "distributed world size"
            )
        model.load_state_dict(checkpoint["model"], strict=True)  # type: ignore[arg-type]
        optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore[arg-type]
        optimizer_to_device(optimizer, device)
        generator.set_state(checkpoint["generator_state"])  # type: ignore[arg-type]
        rng_states = checkpoint.get("rng_states")
        if not isinstance(rng_states, Sequence) or len(rng_states) != runtime.world_size:
            raise ValueError("latest.pt has incomplete per-rank RNG states")
        restore_rng_state(rng_states[runtime.rank], runtime)  # type: ignore[arg-type]
        best_value = float(checkpoint["best_value"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state"]
        best_validation = checkpoint["best_validation"]
        stale = int(checkpoint["stale"])
        final_epoch = int(checkpoint["epoch_completed"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
        records = list(checkpoint["records"])  # type: ignore[arg-type]
        write_metrics()
        if bool(checkpoint.get("training_complete")):
            if runtime.is_main:
                print(
                    json.dumps(
                        {
                            "output": str(args.output_dir),
                            "status": "already_complete",
                            "best_epoch": best_epoch,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            runtime.close()
            return
        start_epoch = final_epoch + 1
    else:
        initial = _evaluate(
            model,
            validation,
            feature_root,
            args.prefix_evidence,
            prefix_registry,
            transports,
            expert_effects,
            device,
            class_probability,
            correction_supervised,
            runtime=runtime,
            deletion_probability=deletion_probability,
            report_prefix=False,
        )
        best_value = float(
            initial["prior_preserving_case_control_log_score"]
        )
        best_epoch = 0
        best_state = cpu_state_dict(model)
        best_validation = initial
        stale = 0
        final_epoch = 0
        start_epoch = 1
        records = [{"epoch": 0, "validation": initial}]

    def elapsed_seconds() -> float:
        return elapsed_before + time.monotonic() - started

    def persist_best(epochs_completed: int, *, training_complete: bool) -> None:
        if not runtime.is_main:
            return
        checkpoint_metadata = {
            **metadata,
            "best_epoch": best_epoch,
            "epochs_completed": int(epochs_completed),
            "best_validation_prior_preserving_case_control_log_score": best_value,
            "final_validation": best_validation,
            "elapsed_seconds": elapsed_seconds(),
            "training_complete": bool(training_complete),
            "test_split_accessed": False,
        }
        atomic_torch_save(
            args.output_dir / "best.pt",
            {"metadata": checkpoint_metadata, "model": best_state},
        )
        _atomic_json(args.output_dir / "summary.json", checkpoint_metadata)

    def persist_latest(
        epoch_completed: int, *, training_complete: bool
    ) -> None:
        rng_states = gather_rng_states(runtime)
        if runtime.is_main:
            if rng_states is None:
                raise RuntimeError("rank zero did not receive distributed RNG states")
            atomic_torch_save(
                latest_path,
                {
                    "schema": "distributed_epoch_resume",
                    "metadata": metadata,
                    "resume_contract": computational_resume_contract(metadata),
                    "world_size": runtime.world_size,
                    "epoch_completed": int(epoch_completed),
                    "stale": stale,
                    "model": cpu_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "best_value": best_value,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "best_validation": best_validation,
                    "generator_state": generator.get_state(),
                    "rng_states": rng_states,
                    "records": records,
                    "elapsed_seconds": elapsed_seconds(),
                    "training_complete": bool(training_complete),
                },
            )
            write_metrics()
        runtime.barrier()

    if not args.resume:
        persist_best(0, training_complete=False)
        persist_latest(0, training_complete=False)

    train_sizes = []
    train_costs = []
    for sequence in train:
        decoded = sequence.get("decoded_index")
        if not isinstance(decoded, Tensor) or decoded.ndim != 1 or not len(decoded):
            raise ValueError("training sequence has no decoded candidates")
        train_sizes.append(len(decoded))
        train_costs.append(int(decoded[-1]) + 1)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        final_epoch = epoch
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        groups = update_groups(
            order, train_sizes, TARGET_CANDIDATES_PER_UPDATE
        )
        optimizer.zero_grad(set_to_none=True)
        epoch_numerator = 0.0
        epoch_exposure = 0.0
        gradient_norm = []
        for group in groups:
            local_numerator = 0.0
            local_exposure = 0.0
            local_candidates = 0
            for sequence_index in balanced_rank_indices(
                group, train_costs, runtime
            ):
                sequence = train[sequence_index]
                task_index = int(sequence["task_index"])
                video_id = str(sequence["video_id"])
                feature, timestamp = _global_stream(feature_root, video_id)
                if (
                    args.prefix_evidence is not None
                    and prefix_registry is not None
                ):
                    prefix_event, prefix_factor = _prefix_evidence(
                        args.prefix_evidence,
                        prefix_registry,
                        video_id,
                        timestamp,
                    )
                else:
                    prefix_event = None
                    prefix_factor = None
                if runtime.enabled:
                    if not isinstance(
                        sequence_trainer, DistributedDataParallel
                    ):
                        raise RuntimeError("distributed wrapper was not created")
                    with sequence_trainer.no_sync():
                        raw_result = sequence_trainer(
                            sequence,
                            feature,
                            timestamp,
                            prefix_event,
                            prefix_factor,
                            transports[task_index],
                            (
                                None
                                if expert_effects is None
                                else expert_effects[task_index]
                            ),
                            device,
                            training=True,
                            class_probability=class_probability,
                            correction_supervised=correction_supervised,
                            deletion_probability=deletion_probability,
                        )
                        if not isinstance(raw_result, Mapping):
                            raise RuntimeError("sequence loss returned no mapping")
                        numerator = raw_result["loss_numerator"]
                        if not isinstance(numerator, Tensor):
                            raise RuntimeError("sequence numerator is not a tensor")
                        numerator.backward()
                else:
                    raw_result = sequence_trainer(
                        sequence,
                        feature,
                        timestamp,
                        prefix_event,
                        prefix_factor,
                        transports[task_index],
                        (
                            None
                            if expert_effects is None
                            else expert_effects[task_index]
                        ),
                        device,
                        training=True,
                        class_probability=class_probability,
                        correction_supervised=correction_supervised,
                        deletion_probability=deletion_probability,
                    )
                    if not isinstance(raw_result, Mapping):
                        raise RuntimeError("sequence loss returned no mapping")
                    numerator = raw_result["loss_numerator"]
                    if not isinstance(numerator, Tensor):
                        raise RuntimeError("sequence numerator is not a tensor")
                    numerator.backward()
                exposure = float(raw_result["loss_exposure"])
                local_numerator += float(numerator.detach())
                local_exposure += exposure
                local_candidates += int(raw_result["events"])
            if runtime.enabled:
                synchronization_loss = sequence_trainer(None)
                if not isinstance(synchronization_loss, Tensor):
                    raise RuntimeError("DDP synchronization loss is not a tensor")
                synchronization_loss.backward()
            (
                global_numerator,
                global_exposure,
                global_candidates,
            ) = sum_scalars(
                runtime,
                local_numerator,
                local_exposure,
                float(local_candidates),
            )
            if global_exposure <= 0 or global_candidates <= 0:
                raise RuntimeError("optimizer group has no effective exposure")
            gradient_divisor = global_exposure / runtime.world_size
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(gradient_divisor)
            gradient_norm.append(
                float(
                    clip_grad_norm_(
                        model.parameters(),
                        GRADIENT_CLIP,
                        error_if_nonfinite=True,
                    )
                )
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_numerator += global_numerator
            epoch_exposure += global_exposure
        validation_result = _evaluate(
            model,
            validation,
            feature_root,
            args.prefix_evidence,
            prefix_registry,
            transports,
            expert_effects,
            device,
            class_probability,
            correction_supervised,
            runtime=runtime,
            deletion_probability=deletion_probability,
            report_prefix=False,
        )
        value = float(
            validation_result["prior_preserving_case_control_log_score"]
        )
        if value < best_value - 1e-12:
            best_value = value
            best_epoch = epoch
            best_validation = validation_result
            best_state = cpu_state_dict(model)
            stale = 0
            persist_best(epoch, training_complete=False)
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_prior_preserving_case_control_log_score": (
                epoch_numerator / epoch_exposure
            ),
            "mean_preclip_gradient_norm": float(np.mean(gradient_norm)),
            "validation": validation_result,
        }
        records.append(record)
        persist_latest(epoch, training_complete=False)
        if runtime.is_main:
            print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= EARLY_STOPPING_EPOCHS:
            break
    model.load_state_dict(best_state, strict=True)  # type: ignore[arg-type]
    final_validation = _evaluate(
        model,
        validation,
        feature_root,
        args.prefix_evidence,
        prefix_registry,
        transports,
        expert_effects,
        device,
        class_probability,
        correction_supervised,
        runtime=runtime,
        deletion_probability=deletion_probability,
    )
    best_validation = final_validation
    persist_best(final_epoch, training_complete=True)
    persist_latest(final_epoch, training_complete=True)
    if runtime.is_main:
        print(
            json.dumps(
                {
                    "output": str(args.output_dir),
                    "best_epoch": best_epoch,
                    "validation_case_control_log_score": best_value,
                    "elapsed_seconds": elapsed_seconds(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    runtime.close()


if __name__ == "__main__":
    main()
