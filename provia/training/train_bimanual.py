"""Train PROVIA on bimanual streams.

The query stream is consumed one frame at a time.  Every event starts
unresolved, receives a class-agnostic decision clock, and is forced to resolve
only at its predicted endpoint.  Current-prefix soft semantics define the
normal procedure operator; endpoint action-effect evidence is unavailable and
unused before that endpoint.  Same-frame left/right transitions are
marginalized in both orders.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from provia.training.train_single_actor import _examples_binding_hashes, _same_node_possibility
from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES
from provia.training.common import (
    _atomic_json,
    _seed_everything,
    _intact,
    _sha256,
)
from provia.training.completion_inputs import (
    _class_probability,
    _correction_is_supervised,
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
from provia.training.role_inputs import (
    EARLY_STOPPING_EPOCHS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIM,
    TARGET_CHECKPOINTS_PER_UPDATE,
    WEIGHT_DECAY,
    _load_expert_effect,
    _load_transport,
)
from provia.training.role_examples import (
    _load_prefix_registry,
    _prefix_evidence,
    _role_stream,
    _validate_sequence,
)
from provia.training.architectures import (
    FIXED_EQUAL_COST_PROBABILITY,
)
from provia.model.mode_filter import (
    ERROR,
    NORMAL,
    joint_explanation_nll,
    prior_preserving_case_control_log_score,
)
from provia.model.step_effects import ExpertEffectPosterior
from provia.model.interval_transport import (
    ContinuationTransport,
    IntervalContinuationLanguage,
)
from provia.model.state_mark import RECURRENT_CANDIDATE_ENCODER
from provia.model.bimanual import (
    PROVIA_BIMANUAL_ARCHITECTURE,
    RoleIntentStateModel,
    ProviaBimanual,
)



def _source_with_predictive_role_language_registry(
    source: Mapping[str, object],
    registry_path: Path,
    examples_path: Path,
    rebuilt_examples: bool = False,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = json.loads(registry_path.read_text())
    original_path = Path(str(source["procedure_language"]))
    original_hash = str(source["procedure_language_sha256"])
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema")
        != "bayesian_predictive_role_language_registry"
        or (not rebuilt_examples and registry.get("source_examples_sha256") not in _examples_binding_hashes(examples_path))
        or (not rebuilt_examples and registry.get("source_language_sha256") != original_hash)
        or int(registry.get("fold", -1)) != int(source["fold"])
        or registry.get("anomaly_labels_used") is not False
        or registry.get("expert_event_annotations_used") is not False
        or registry.get("validation_videos_used") is not False
        or registry.get("test_videos_used") is not False
        or registry.get("similarity_threshold") is not False
        or registry.get("selected_state_count") is not False
        or not original_path.is_file()
        or _sha256(original_path) != original_hash
    ):
        raise ValueError("predictive role-language registry is malformed")
    path = registry_path.parent / str(registry["language"])
    if not path.is_file() or not _intact(path, registry.get("language_sha256")):
        raise ValueError("predictive role language changed after compilation")
    original = IntervalContinuationLanguage.from_dict(
        json.loads(original_path.read_text())
    )
    predictive = IntervalContinuationLanguage.from_dict(json.loads(path.read_text()))
    if (
        predictive.factor_labels != original.factor_labels
        or predictive.node_ids != original.node_ids
        or predictive.node_roles != original.node_roles
        or not _same_node_possibility(predictive.node_possibility, original.node_possibility, exact=not rebuilt_examples)
        or predictive.source_trace_count != original.source_trace_count
    ):
        raise ValueError("predictive role process changed the frozen event ontology")
    overridden = dict(source)
    overridden["expert_effect_source_language_sha256"] = original_hash
    overridden["procedure_language"] = str(path.resolve())
    overridden["procedure_language_sha256"] = registry.get("language_sha256")
    return overridden, registry


class _DistributedRoleSequenceLoss(nn.Module):
    """Expose one complete bimanual video as one DDP forward pass."""

    def __init__(self, model: RoleIntentStateModel) -> None:
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


@dataclass
class _ActiveCandidate:
    index: int
    start: int
    onset: int
    end: int
    role: int
    effect_statistic: Tensor
    previous_cumulative_evidence: Tensor | None = None
    predictive_state_path: Tensor | None = None


def _safe_ranking(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(target) == 0 or len(np.unique(target)) < 2:
        return {"average_precision": float("nan"), "auroc": float("nan")}
    return {
        "average_precision": float(average_precision_score(target, score)),
        "auroc": float(roc_auc_score(target, score)),
    }


def _current_factors(
    factor_probability: Mapping[str, Tensor],
    frame: int,
    role: int,
    device: torch.device,
) -> dict[str, Tensor]:
    return {
        name: value[frame, role].to(device=device, dtype=torch.float32)
        for name, value in factor_probability.items()
    }


def _sequence_forward(
    model: RoleIntentStateModel,
    sequence: Mapping[str, object],
    frame: Tensor,
    timestamp: Tensor,
    global_feature: Tensor,
    role_feature: Tensor,
    event_probability: Tensor,
    factor_probability: Mapping[str, Tensor],
    labels: Mapping[str, Sequence[str]],
    transport: ContinuationTransport,
    expert_effect: ExpertEffectPosterior,
    device: torch.device,
    class_probability: Tensor,
    correction_supervised: bool,
    deletion_probability: Tensor | None = None,
    *,
    training: bool = False,
    report_prefix: bool = True,
    scoring_only: bool = False,
) -> dict[str, object]:
    source_candidates = _validate_sequence(sequence, len(frame), labels)
    candidates = [
        _ActiveCandidate(
            index=value.index,
            start=value.start,
            onset=value.onset,
            end=value.end,
            role=value.role,
            effect_statistic=value.effect_statistic,
        )
        for value in source_candidates
    ]
    input_stream = [
        torch.cat((global_feature, role_feature[:, role]), dim=-1).to(
            device=device, dtype=torch.float32
        )
        for role in range(len(EVENT_ROLES))
    ]
    encoded = tuple(
        model.energy.window_encoder.forward_stream(value) for value in input_stream
    )
    expert_intent_key = tuple(
            model.energy.build_frontier_key(transport, role)
            for role in range(len(EVENT_ROLES))
        )
    initial_procedure = global_feature.new_zeros(
        transport.language.state_count, device=device
    )
    initial_procedure[transport.language.start_state] = 1.0
    belief = model.filter.initial_belief(initial_procedure)
    history = model.energy.initial_history(encoded[0][:1])
    starts: dict[int, list[int]] = {}
    ends: dict[int, list[int]] = {}
    for candidate in candidates:
        starts.setdefault(candidate.start, []).append(candidate.index)
        ends.setdefault(candidate.end, []).append(candidate.index)
    active: dict[int, int] = {}
    probability_slots: list[list[Tensor | None]] = (
        [
            [None] * (candidate.end - candidate.start + 1)
            for candidate in candidates
        ]
        if not training and report_prefix
        else [[None] for _ in candidates]
    )

    for current in range(len(frame)):
        ending_now = sorted(
            ends.get(current, ()), key=lambda index: candidates[index].role
        )
        beginning = sorted(
            starts.get(current, ()), key=lambda index: candidates[index].role
        )
        # Every candidate reads the same factual filtration at this timestamp;
        # no arbitrary same-frame completion/start order is injected.
        for index in beginning:
            candidate = candidates[index]
            if candidate.role in active:
                raise ValueError("a role cannot carry two active candidates")
            candidate.predictive_state_path = (
                model.energy.forward_state_path(
                    encoded[candidate.role][
                        candidate.start : candidate.end + 1
                    ],
                    {
                        name: value[
                            candidate.start : candidate.end + 1,
                            candidate.role,
                        ].to(device=device, dtype=torch.float32)
                        for name, value in factor_probability.items()
                    },
                    event_probability[
                        candidate.start : candidate.end + 1,
                        candidate.role,
                    ].to(device=device, dtype=torch.float32),
                )
            )
            active[candidate.role] = index

        if active:
            mark_updates = {}
            readout_observations = {}
            for role, index in sorted(active.items()):
                candidate = candidates[index]
                if (
                    not ending_now
                    and (training or not report_prefix)
                ):
                    continue
                factors = _current_factors(
                    factor_probability, current, role, device
                )
                force = current == candidate.end
                if force and isinstance(expert_effect, ExpertEffectPosterior):
                    statistic = candidate.effect_statistic.to(
                        device=device, dtype=torch.float32
                    )
                    effect = model.node_effect_log_bayes_factor(
                            statistic, expert_effect, transport
                        )
                else:
                    effect = None
                if candidate.predictive_state_path is None:
                    raise RuntimeError(
                        "active role candidate lacks a predictive state"
                    )
                offset = current - candidate.start
                mark_hidden = candidate.predictive_state_path[
                    offset : offset + 1
                ]
                mark_evidence = model.energy.forward_mark(
                    mark_hidden,
                    {
                        name: value.unsqueeze(0)
                        for name, value in factors.items()
                    },
                    event_probability[current : current + 1, role].to(
                        device=device, dtype=torch.float32
                    ),
                    history,
                    role,
                    transport,
                    frontier_key=expert_intent_key[role],
                )
                mark_updates[role] = (
                    mark_evidence.explanation_log_likelihood[0]
                )
                learned_normal = (
                    mark_evidence.node_log_bayes_factor[
                        0, :, 1 + NORMAL
                    ]
                )
                learned_transport = (
                    model.transport_node_log_bayes_factor(
                        learned_normal
                    )
                )
                readout_effect = (
                    effect
                    if learned_transport is None
                    else learned_transport
                    if effect is None
                    else learned_transport + effect
                )
                readout_observations[role] = (factors, readout_effect)
            score_belief = (
                model.filter.mark_joint(belief, mark_updates)
                if mark_updates
                else belief
            )
            score_belief = model.condition_for_readout(
                    score_belief, readout_observations, transport
                )
            for role, index in sorted(active.items()):
                candidate = candidates[index]
                if (training or not report_prefix) and current != candidate.end:
                    continue
                summary = model.filter.summarize(score_belief, role)
                position = (
                    0 if training or not report_prefix else current - candidate.start
                )
                if probability_slots[index][position] is not None:
                    raise RuntimeError("a causal candidate prefix was scored twice")
                probability_slots[index][position] = summary.explanation_probability

        ending = ending_now
        if ending:
            ending_marks = {
                candidates[index].role: mark_updates[candidates[index].role]
                for index in ending
            }
            marked = model.filter.mark_joint(belief, ending_marks)
            completion_observations = {
                candidates[index].role: readout_observations[
                    candidates[index].role
                ]
                for index in ending
            }
            belief = model.filter.finalize_joint(
                    marked, completion_observations, transport
                )
        for index in ending:
            candidate = candidates[index]
            if active.get(candidate.role) != index:
                raise RuntimeError("ending candidate is not the active role event")
            endpoint_hidden = (
                candidate.predictive_state_path[-1]
                if candidate.predictive_state_path is not None
                else encoded[candidate.role][current]
            )
            endpoint_factors = _current_factors(
                factor_probability, current, candidate.role, device
            )
            history = model.energy.update_history(
                history,
                endpoint_hidden,
                endpoint_factors,
                candidate.role,
            )
            del active[candidate.role]

        # This is one-step prequential differentiation at the model's natural
        # step boundary. The forward filtering distribution is exact;
        # only gradients across already closed steps are truncated. If
        # another hand remains active, its concurrent step is retained.
        if training and ending and not active:
            belief = belief.detach()
            history = history.detach()

    if active:
        raise RuntimeError("video ended with an unresolved role step")
    probabilities = []
    for candidate, values in zip(candidates, probability_slots):
        if any(value is None for value in values):
            missing = [index for index, value in enumerate(values) if value is None]
            raise RuntimeError(
                f"candidate {candidate.index} has unscored prefixes {missing[:5]}"
            )
        probabilities.append(torch.stack(values))  # type: ignore[arg-type]

    detached_probabilities = tuple(
        value.detach().cpu() for value in probabilities
    )
    if scoring_only:
        return {
            "events": len(candidates),
            "prefix_explanation_probability": detached_probabilities,
        }

    origin = sequence["origin_target"].to(  # type: ignore[union-attr]
        device=device, dtype=torch.bool
    )
    mode = sequence["mode_target"].to(  # type: ignore[union-attr]
        device=device, dtype=torch.long
    )
    latent_native_events = 0
    loss_numerator = belief.new_zeros(())
    loss_exposure = belief.new_zeros(())
    joint_sum = belief.new_zeros(())
    endpoint_score = []
    maximum_score = []
    target = []
    origins = []
    first_alarm_time = []
    error_events = detected_error = false_alarms = 0
    for candidate, probability in zip(candidates, probabilities):
        endpoint = probability[-1:].contiguous()
        numerator, exposure = prior_preserving_case_control_log_score(
            endpoint,
            origin[candidate.index : candidate.index + 1],
            mode[candidate.index : candidate.index + 1],
            class_probability,
            correction_supervised=correction_supervised,
        )
        loss_numerator = loss_numerator + numerator
        loss_exposure = loss_exposure + exposure
        if not training:
            joint_sum = joint_sum + joint_explanation_nll(
                endpoint,
                origin[candidate.index : candidate.index + 1],
                mode[candidate.index : candidate.index + 1],
            )
        if not training:
            anomaly = probability[:, 1 + ERROR]
            candidate_target = origin[candidate.index] & (
                mode[candidate.index] == ERROR
            )
            alarm = torch.nonzero(
                anomaly >= FIXED_EQUAL_COST_PROBABILITY, as_tuple=False
            ).flatten()
            if bool(candidate_target.detach().cpu()):
                error_events += 1
                if len(alarm):
                    detected_error += 1
                    first = (
                        candidate.end
                        if not report_prefix
                        else candidate.start + int(alarm[0])
                    )
                    first_alarm_time.append(
                        (
                            float(timestamp[first] - timestamp[candidate.onset]),
                            float(timestamp[candidate.end] - timestamp[first]),
                        )
                    )
            elif len(alarm):
                false_alarms += 1
            endpoint_score.append(anomaly[-1].detach().cpu())
            maximum_score.append(anomaly.max().detach().cpu())
            target.append(candidate_target.detach().cpu())
            origins.append(origin[candidate.index].detach().cpu())
    result: dict[str, object] = {
        "loss_numerator": loss_numerator,
        "loss_exposure": loss_exposure,
        "endpoint_joint_sum": joint_sum,
        "events": len(candidates),
        "latent_native_events": latent_native_events,
        "prefix_explanation_probability": detached_probabilities,
    }
    if training:
        return result
    duration = max(
        float(timestamp[-1] - timestamp[0]), torch.finfo(torch.float64).eps
    )
    endpoint_tensor = torch.stack(endpoint_score)
    maximum_tensor = torch.stack(maximum_score)
    native = sequence.get("native_intervals")
    assignment = sequence.get("native_assignment")
    if not isinstance(native, Sequence) or not isinstance(assignment, Tensor):
        raise ValueError("native role evaluation metadata is unavailable")
    native_endpoint = endpoint_tensor.new_zeros(len(native))
    native_maximum = maximum_tensor.new_zeros(len(native))
    native_covered = torch.zeros(len(native), dtype=torch.bool)
    for native_index in range(len(native)):
        position = torch.nonzero(
            assignment == native_index, as_tuple=False
        ).flatten()
        if len(position):
            native_endpoint[native_index] = endpoint_tensor.index_select(
                0, position
            ).max()
            native_maximum[native_index] = maximum_tensor.index_select(
                0, position
            ).max()
            native_covered[native_index] = True
    native_target = torch.tensor(
        [int(interval["mode"]) == ERROR for interval in native],
        dtype=torch.bool,
    )
    result.update({
        "endpoint_score": endpoint_tensor,
        "maximum_prefix_score": maximum_tensor,
        "target": torch.stack(target).bool(),
        "origin": torch.stack(origins).bool(),
        "native_endpoint_score": native_endpoint,
        "native_maximum_score": native_maximum,
        "native_target": native_target,
        "native_covered": native_covered,
        "error_events": error_events,
        "detected_error": detected_error,
        "false_alarms": false_alarms,
        "duration_sec": duration,
        "first_alarm_time": tuple(first_alarm_time),
    })
    return result


@torch.no_grad()
def _evaluate(
    model: RoleIntentStateModel,
    sequences: Sequence[Mapping[str, object]],
    feature_root: Path,
    prefix_root: Path,
    prefix_registry: Mapping[str, object],
    labels: Mapping[str, Sequence[str]],
    transport: ContinuationTransport,
    expert_effect: ExpertEffectPosterior,
    device: torch.device,
    class_probability: Tensor,
    correction_supervised: bool,
    deletion_probability: Tensor,
    runtime: DistributedRuntime | None = None,
    report_prefix: bool = True,
) -> dict[str, object]:
    model.eval()
    numerator = exposure = joint = 0.0
    events = native_events = error_events = detected_error = false_alarms = 0
    duration = 0.0
    endpoint = []
    maximum = []
    targets = []
    origins = []
    native_endpoint = []
    native_maximum = []
    native_targets = []
    native_covered = []
    delay = []
    lead = []
    local_sequences = (
        sequences
        if runtime is None
        else sequences[runtime.rank :: runtime.world_size]
    )
    for sequence in local_sequences:
        video_id = str(sequence["video_id"])
        frame, timestamp, global_feature, role_feature = _role_stream(
            feature_root, video_id
        )
        event, factor = _prefix_evidence(
            prefix_root,
            prefix_registry,
            video_id,
            frame,
            timestamp,
            labels,
        )
        result = _sequence_forward(
            model,
            sequence,
            frame,
            timestamp,
            global_feature,
            role_feature,
            event,
            factor,
            labels,
            transport,
            expert_effect,
            device,
            class_probability,
            correction_supervised,
            deletion_probability,
            training=False,
            report_prefix=report_prefix,
        )
        numerator += float(result["loss_numerator"])
        exposure += float(result["loss_exposure"])
        joint += float(result["endpoint_joint_sum"])
        events += int(result["events"])
        native_events += int(result["latent_native_events"])
        endpoint.append(result["endpoint_score"])
        maximum.append(result["maximum_prefix_score"])
        targets.append(result["target"])
        origins.append(result["origin"])
        native_endpoint.append(result["native_endpoint_score"])
        native_maximum.append(result["native_maximum_score"])
        native_targets.append(result["native_target"])
        native_covered.append(result["native_covered"])
        error_events += int(result["error_events"])
        detected_error += int(result["detected_error"])
        false_alarms += int(result["false_alarms"])
        duration += float(result["duration_sec"])
        for after_onset, before_end in result["first_alarm_time"]:
            delay.append(after_onset)
            lead.append(before_end)
    payload: dict[str, object] = {
        "numerator": numerator,
        "exposure": exposure,
        "joint": joint,
        "events": events,
        "native_events": native_events,
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
            torch.cat(targets).numpy().astype(np.int64)
            if targets
            else np.empty(0, dtype=np.int64)
        ),
        "origin": (
            torch.cat(origins).numpy().astype(np.int64)
            if origins
            else np.empty(0, dtype=np.int64)
        ),
        "native_endpoint": torch.cat(native_endpoint).numpy(),
        "native_maximum": torch.cat(native_maximum).numpy(),
        "native_target": torch.cat(native_targets).numpy().astype(np.int64),
        "native_covered": torch.cat(native_covered).numpy().astype(np.int64),
        "error_events": error_events,
        "detected_error": detected_error,
        "false_alarms": false_alarms,
        "duration": duration,
        "delay": np.asarray(delay, dtype=np.float64),
        "lead": np.asarray(lead, dtype=np.float64),
    }
    payloads = (
        (payload,) if runtime is None else gather_objects(runtime, payload)
    )
    numerator = sum(float(value["numerator"]) for value in payloads)
    exposure = sum(float(value["exposure"]) for value in payloads)
    joint = sum(float(value["joint"]) for value in payloads)
    events = sum(int(value["events"]) for value in payloads)
    native_events = sum(int(value["native_events"]) for value in payloads)
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
    native_endpoint_array = np.concatenate(
        [np.asarray(value["native_endpoint"]) for value in payloads]
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
    error_events = sum(int(value["error_events"]) for value in payloads)
    detected_error = sum(int(value["detected_error"]) for value in payloads)
    false_alarms = sum(int(value["false_alarms"]) for value in payloads)
    duration = sum(float(value["duration"]) for value in payloads)
    delay_array = np.concatenate(
        [np.asarray(value["delay"], dtype=np.float64) for value in payloads]
    )
    lead_array = np.concatenate(
        [np.asarray(value["lead"], dtype=np.float64) for value in payloads]
    )
    log_score = numerator / exposure
    return {
        "prior_preserving_log_score": log_score,
        "endpoint_joint_negative_log_likelihood": joint / events,
        "decoded_candidates": events,
        "genuine_candidates": int(origin_array.sum()),
        "error_candidates": int(target_array.sum()),
        "completion_ranking": _safe_ranking(target_array, endpoint_array),
        "anytime_prefix_ranking": _safe_ranking(target_array, maximum_array),
        "native_endpoint_ranking": _safe_ranking(
            native_target_array, native_endpoint_array
        ),
        "native_anytime_ranking": _safe_ranking(
            native_target_array, native_maximum_array
        ),
        "native_event_coverage": float(native_covered_array.mean()),
        "native_events": int(len(native_target_array)),
        "fixed_equal_cost_operation": {
            "probability": FIXED_EQUAL_COST_PROBABILITY,
            "error_event_recall": (
                detected_error / error_events if error_events else float("nan")
            ),
            "detected_error_events": detected_error,
            "error_events": error_events,
            "false_alarms": false_alarms,
            "false_alarms_per_minute": false_alarms / (duration / 60.0),
            "median_delay_from_predicted_onset_sec": (
                float(np.median(delay_array))
                if len(delay_array)
                else float("nan")
            ),
            "median_lead_to_predicted_end_sec": (
                float(np.median(lead_array))
                if len(lead_array)
                else float("nan")
            ),
        },
    }


def _build_model(
    source: Mapping[str, object],
    device: torch.device,
    *,
    correction_supervised: bool = False,
) -> ProviaBimanual:
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }
    origin = source.get("origin_channel")
    if not isinstance(origin, Mapping):
        raise ValueError("role examples lack a origin channel")
    return ProviaBimanual(
        int(source["causal_window_dim"]),
        labels,
        torch.tensor(source["mode_transition_probability"], dtype=torch.float32),
        torch.tensor(source["mode_initial_probability"], dtype=torch.float32),
        float(origin["genuine_probability"]),
        model_dim=MODEL_DIM,
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--prefix-evidence", type=Path, required=True)
    parser.add_argument("--expert-effect", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--language-registry",
        type=Path,
        help="expert-only predictive quotient of the bimanual language",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume exactly from output-dir/latest.pt at an epoch boundary",
    )
    args = parser.parse_args()
    if args.expert_effect is None:
        raise ValueError("training requires --expert-effect")
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
        or source.get("schema") != "role_windowed_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(train, Sequence)
        or not isinstance(validation, Sequence)
        or not train
        or not validation
    ):
        raise ValueError("bimanual examples are malformed")
    language_registry = None
    if args.language_registry is not None:
        source, language_registry = _source_with_predictive_role_language_registry(
            source, args.language_registry, args.examples
        )
    if (
        source.get("schema") != "role_windowed_examples"
        or source.get("effect_statistic_interval")
        != "current predicted onset-to-end interval"
    ):
        raise ValueError(
            "predictive reliability requires non-censored current-event "
            "onset-to-end effect statistics"
        )
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }
    prefix_registry = _load_prefix_registry(args.prefix_evidence, args.examples)
    feature_root = Path(str(source["feature_root"]))
    transport = _load_transport(
        source, device, posterior_predictive_innovation=True
    )
    expert_effect: ExpertEffectPosterior
    expert_effect = _load_expert_effect(
        source, args.expert_effect, transport, device
    )
    correction_supervised = _correction_is_supervised(train)
    model = _build_model(
        source,
        device,
        correction_supervised=correction_supervised,
    )
    class_probability = _class_probability(
            train, correction_supervised=correction_supervised
        ).to(device)
    deletion_source = source.get("missed_event_probability")
    if (
        not isinstance(deletion_source, Sequence)
        or len(deletion_source) != len(EVENT_ROLES)
    ):
        raise ValueError("role examples lack one train-fold deletion channel per role")
    deletion_probability = torch.tensor(
        [float(value) for value in deletion_source],
        dtype=torch.float32,
        device=device,
    )
    if torch.any(deletion_probability <= 0) or torch.any(
        deletion_probability >= 1
    ):
        raise ValueError("role deletion probabilities must lie strictly in (0,1)")
    selection_key = "prior_preserving_log_score"
    sequence_trainer = wrap_distributed(
        _DistributedRoleSequenceLoss(model), runtime
    )
    _seed_everything(args.seed + 104729 * runtime.rank)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    metadata = {
        "schema": "bimanual_training",
        "architecture": PROVIA_BIMANUAL_ARCHITECTURE,
        "candidate_state_encoder": RECURRENT_CANDIDATE_ENCODER,
        "candidate_state_receptive_field": None,
        "method": "role-preserving decoupled-intent predictive state",
        "simultaneous_decisions": (
            "same-frame left/right operators are marginalized in both orders"
        ),
        "candidate_identity_targets_used_by_loss": True,
        "alignment_timing_tolerance": None,
        "deletion_probability": deletion_probability.detach().cpu().tolist(),
        "selection_metric": selection_key,
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "target_candidates_per_update": TARGET_CHECKPOINTS_PER_UPDATE,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_epochs": EARLY_STOPPING_EPOCHS,
            "ranking_latency_or_threshold_used_for_selection": False,
        },
        "model_dim": MODEL_DIM,
        "task_or_dataset_embedding": False,
        "role_specific_head": False,
        "dataset_specific_head": False,
        "score_fusion": False,
        "semantic_temperature": False,
        "manual_threshold_selection": False,
        "decision_time_target": False,
        "latency_loss_or_reward": False,
        "fixed_equal_cost_probability": FIXED_EQUAL_COST_PROBABILITY,
        "examples": str(args.examples.resolve()),
        "examples_sha256": _sha256(args.examples),
        "prefix_evidence": str(args.prefix_evidence.resolve()),
        "prefix_evidence_registry_sha256": _sha256(
            args.prefix_evidence / "registry.json"
        ),
        "expert_effect_artifact": (
            None
            if args.expert_effect is None
            else str(args.expert_effect.resolve())
        ),
        "expert_effect_artifact_sha256": (
            None
            if args.expert_effect is None
            else _sha256(args.expert_effect)
        ),
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
        # language retains reference evidence counts; otherwise the reliability is one.
        "fold": int(source["fold"]),
        "seed": args.seed,
        "parameters": {
            "total": sum(value.numel() for value in model.parameters()),
            "trainable": sum(
                value.numel() for value in model.parameters() if value.requires_grad
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
            labels,
            transport,
            expert_effect,
            device,
            class_probability,
            correction_supervised,
            deletion_probability,
            runtime=runtime,
            report_prefix=False,
        )
        best_value = float(initial[selection_key])
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
            "best_validation_selection_log_score": best_value,
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
        origin_target = sequence.get("origin_target")
        end_index = sequence.get("end_index")
        if (
            not isinstance(origin_target, Tensor)
            or origin_target.ndim != 1
            or not len(origin_target)
            or not isinstance(end_index, Tensor)
            or end_index.shape != origin_target.shape
        ):
            raise ValueError("training sequence has malformed role candidates")
        train_sizes.append(len(origin_target))
        train_costs.append(int(end_index.max()) + 1)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        final_epoch = epoch
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        groups = update_groups(
            order, train_sizes, TARGET_CHECKPOINTS_PER_UPDATE
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
                video_id = str(sequence["video_id"])
                frame, timestamp, global_feature, role_feature = _role_stream(
                    feature_root, video_id
                )
                event, factor = _prefix_evidence(
                    args.prefix_evidence,
                    prefix_registry,
                    video_id,
                    frame,
                    timestamp,
                    labels,
                )
                arguments = (
                    sequence,
                    frame,
                    timestamp,
                    global_feature,
                    role_feature,
                    event,
                    factor,
                    labels,
                    transport,
                    expert_effect,
                    device,
                    class_probability,
                    correction_supervised,
                    deletion_probability,
                )
                if runtime.enabled:
                    if not isinstance(
                        sequence_trainer, DistributedDataParallel
                    ):
                        raise RuntimeError("distributed wrapper was not created")
                    with sequence_trainer.no_sync():
                        raw_result = sequence_trainer(
                            *arguments,
                            training=True,
                            )
                        if not isinstance(raw_result, Mapping):
                            raise RuntimeError("sequence loss returned no mapping")
                        numerator = raw_result["loss_numerator"]
                        if not isinstance(numerator, Tensor):
                            raise RuntimeError("sequence numerator is not a tensor")
                        numerator.backward()
                else:
                    raw_result = sequence_trainer(
                        *arguments,
                        training=True,
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
            labels,
            transport,
            expert_effect,
            device,
            class_probability,
            correction_supervised,
            deletion_probability,
            runtime=runtime,
            report_prefix=False,
        )
        value = float(validation_result[selection_key])
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
            "train_selection_log_score": epoch_numerator / epoch_exposure,
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
        labels,
        transport,
        expert_effect,
        device,
        class_probability,
        correction_supervised,
        deletion_probability,
        runtime=runtime,
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
                    "validation_selection_log_score": best_value,
                    "elapsed_seconds": elapsed_seconds(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    runtime.close()


if __name__ == "__main__":
    main()
