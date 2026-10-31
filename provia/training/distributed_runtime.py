"""Deterministic distributed runtime helpers for sequence-level training."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


_RESUME_SCALAR_KEYS = (
    "schema",
    "architecture",
    "fold",
    "seed",
    "model_dim",
    "parameters",
    "deletion_probability",
    "incidence_prior",
    "semantic_temperature",
    "fixed_equal_cost_probability",
    "fixed_report_probability",
    "simultaneous_decisions",
    "role_specific_head",
    "task_or_dataset_embedding",
    "dataset_specific_head",
    "score_fusion",
    "manual_gate",
    "learned_applicability_gate",
    "manual_threshold_selection",
    "decision_time_target",
    "latency_loss_or_reward",
    "selection_metric",
)


def computational_resume_contract(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Extract fields that can change optimization from descriptive metadata.

    Run metadata deliberately contains human-readable method explanations.  Those
    strings may be clarified without changing a checkpoint's computation and
    therefore must not be used as a byte-for-byte resume guard.  The contract
    below retains architecture, artifact hashes, model/optimizer constants,
    machine-readable switches, and the distributed execution shape.
    """

    contract = {
        key: metadata.get(key)
        for key in _RESUME_SCALAR_KEYS
        if key in metadata
    }
    for key, value in metadata.items():
        if key.endswith("_sha256") and value is not None:
            contract[key] = value
    optimization = metadata.get("optimization")
    if isinstance(optimization, Mapping):
        contract["optimization"] = {
            key: optimization.get(key)
            for key in (
                "optimizer",
                "learning_rate",
                "weight_decay",
                "gradient_clip",
                "target_candidates_per_update",
                "target_checkpoints_per_update",
                "maximum_epochs",
                "early_stopping_epochs",
            )
            if key in optimization
        }
    distributed = metadata.get("distributed")
    if isinstance(distributed, Mapping):
        contract["distributed"] = {
            key: distributed.get(key)
            for key in ("world_size", "backend")
            if key in distributed
        }
    return contract


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(device_specification: str) -> DistributedRuntime:
    """Initialize torchrun while preserving ordinary single-device use."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = torch.device(device_specification)
    if world_size <= 1:
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but unavailable")
        return DistributedRuntime(0, 1, 0, requested)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "distributed CUDA training was requested but unavailable"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif requested.type == "cpu":
        # Gloo is retained for deterministic CPU equivalence tests. Production
        # jobs use one NCCL process per GPU.
        device = requested
        backend = "gloo"
    else:
        raise ValueError("distributed training supports CUDA or CPU devices")
    dist.init_process_group(backend=backend, init_method="env://")
    if dist.get_world_size() != world_size or dist.get_rank() != rank:
        raise RuntimeError("torch.distributed disagrees with the torchrun environment")
    return DistributedRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def wrap_distributed(module: nn.Module, runtime: DistributedRuntime) -> nn.Module:
    if not runtime.enabled:
        return module
    if runtime.device.type == "cpu":
        return DistributedDataParallel(
            module,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    return DistributedDataParallel(
        module,
        device_ids=[runtime.local_rank],
        output_device=runtime.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def update_groups(
    order: Sequence[int], sizes: Sequence[int], target_size: int
) -> tuple[tuple[int, ...], ...]:
    """Reproduce the single-process optimizer boundaries before sharding."""

    if (
        target_size <= 0
        or any(index < 0 for index in order)
        or len(sizes) <= max(order, default=-1)
    ):
        raise ValueError("invalid global update grouping inputs")
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    accumulated = 0
    for index in order:
        size = int(sizes[index])
        if size <= 0:
            raise ValueError("every sequence must expose a positive training size")
        current.append(int(index))
        accumulated += size
        if accumulated >= target_size:
            groups.append(tuple(current))
            current = []
            accumulated = 0
    if current:
        groups.append(tuple(current))
    if not groups:
        raise ValueError("the training order produced no optimizer updates")
    return tuple(groups)


def balanced_rank_indices(
    group: Sequence[int], costs: Sequence[int], runtime: DistributedRuntime
) -> tuple[int, ...]:
    """Deterministically balance whole independent videos across ranks."""

    if (
        any(index < 0 for index in group)
        or len(costs) <= max(group, default=-1)
    ):
        raise ValueError("sequence costs do not align with the update group")
    assignments: list[list[int]] = [[] for _ in range(runtime.world_size)]
    loads = [0] * runtime.world_size
    ordered = sorted(group, key=lambda index: (-int(costs[index]), int(index)))
    for index in ordered:
        rank = min(range(runtime.world_size), key=lambda value: (loads[value], value))
        assignments[rank].append(int(index))
        loads[rank] += int(costs[index])
    return tuple(assignments[runtime.rank])


def sum_scalars(
    runtime: DistributedRuntime, *values: float
) -> tuple[float, ...]:
    tensor = torch.tensor(values, dtype=torch.float64, device=runtime.device)
    if runtime.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tuple(float(value) for value in tensor.cpu())


T = TypeVar("T")


def gather_objects(
    runtime: DistributedRuntime, value: T
) -> tuple[T, ...]:
    """Gather a small serializable evaluation payload on every rank."""

    if not runtime.enabled:
        return (value,)
    gathered: list[T | None] = [None] * runtime.world_size
    dist.all_gather_object(gathered, value)
    if any(item is None for item in gathered):
        raise RuntimeError("one distributed evaluation payload was not gathered")
    return tuple(item for item in gathered if item is not None)


def capture_rng_state(runtime: DistributedRuntime) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if runtime.device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(runtime.device)
    return state


def gather_rng_states(
    runtime: DistributedRuntime,
) -> tuple[Mapping[str, object], ...] | None:
    local = capture_rng_state(runtime)
    if not runtime.enabled:
        return (local,)
    gathered: list[Mapping[str, object] | None] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local)
    if runtime.is_main:
        if any(value is None for value in gathered):
            raise RuntimeError("one distributed RNG state was not gathered")
        return tuple(value for value in gathered if value is not None)
    return None


def restore_rng_state(
    state: Mapping[str, object], runtime: DistributedRuntime
) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    if runtime.device.type == "cuda":
        cuda_state = state.get("torch_cuda")
        if not isinstance(cuda_state, Tensor):
            raise ValueError("resume checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state.cpu(), runtime.device)


def cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, Tensor):
                state[name] = value.to(device)
