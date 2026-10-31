from __future__ import annotations

import os
import random
import socket
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from provia.training.distributed_runtime import (
    DistributedRuntime,
    atomic_torch_save,
    balanced_rank_indices,
    capture_rng_state,
    computational_resume_contract,
    initialize_distributed,
    optimizer_to_device,
    restore_rng_state,
    sum_scalars,
    update_groups,
    wrap_distributed,
)


class _QuadraticSequenceLoss(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, item: tuple[Tensor, Tensor, float] | None
    ) -> dict[str, Tensor | float] | Tensor:
        if item is None:
            zero = next(self.model.parameters()).new_zeros(())
            for parameter in self.model.parameters():
                zero = zero + parameter.reshape(-1)[0] * 0.0
            return zero
        feature, target, exposure = item
        residual = self.model(feature) - target
        return {
            "numerator": residual.square().sum(),
            "exposure": exposure,
        }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])


def _distributed_update_worker(
    rank: int, world_size: int, port: int
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    runtime = initialize_distributed("cpu")
    torch.manual_seed(17)
    model = nn.Linear(3, 1, bias=True)
    initial = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    wrapped = wrap_distributed(_QuadraticSequenceLoss(model), runtime)
    assert isinstance(wrapped, DistributedDataParallel)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
    items = (
        (torch.tensor([[1.0, -1.0, 0.5]]), torch.tensor([[0.2]]), 1.0),
        (torch.tensor([[0.0, 2.0, -0.5]]), torch.tensor([[-0.4]]), 2.0),
        (torch.tensor([[1.5, 0.2, 1.0]]), torch.tensor([[0.7]]), 1.5),
    )
    costs = (4, 7, 3)
    local_exposure = 0.0
    for index in balanced_rank_indices((0, 1, 2), costs, runtime):
        with wrapped.no_sync():
            result = wrapped(items[index])
            assert isinstance(result, dict)
            numerator = result["numerator"]
            assert isinstance(numerator, Tensor)
            numerator.backward()
            local_exposure += float(result["exposure"])
    synchronization = wrapped(None)
    assert isinstance(synchronization, Tensor)
    synchronization.backward()
    (global_exposure,) = sum_scalars(runtime, local_exposure)
    for parameter in model.parameters():
        assert parameter.grad is not None
        parameter.grad.div_(global_exposure / world_size)
    optimizer.step()

    expected = nn.Linear(3, 1, bias=True)
    expected.load_state_dict(initial)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.07)
    expected_loss = expected.weight.new_zeros(())
    for feature, target, _ in items:
        expected_loss = expected_loss + (expected(feature) - target).square().sum()
    (expected_loss / sum(value[2] for value in items)).backward()
    expected_optimizer.step()
    for actual_parameter, expected_parameter in zip(
        model.parameters(), expected.parameters()
    ):
        torch.testing.assert_close(
            actual_parameter, expected_parameter, rtol=1e-6, atol=1e-7
        )
        gathered = [torch.empty_like(actual_parameter) for _ in range(world_size)]
        dist.all_gather(gathered, actual_parameter.detach())
        for peer in gathered:
            torch.testing.assert_close(peer, actual_parameter)
    runtime.close()


def test_global_update_groups_are_preserved_and_balanced() -> None:
    order = (3, 0, 2, 1)
    sizes = (2, 5, 3, 4)
    groups = update_groups(order, sizes, target_size=6)
    assert groups == ((3, 0), (2, 1))
    assignments = []
    for rank in range(2):
        runtime = DistributedRuntime(rank, 2, rank, torch.device("cpu"))
        assignments.extend(balanced_rank_indices(groups[0], sizes, runtime))
    assert sorted(assignments) == sorted(groups[0])


def test_two_rank_update_matches_global_objective() -> None:
    mp.spawn(
        _distributed_update_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def test_checkpoint_restores_optimizer_and_all_rng_streams(tmp_path: Path) -> None:
    runtime = DistributedRuntime(0, 1, 0, torch.device("cpu"))
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    optimizer.step()
    state = capture_rng_state(runtime)
    path = tmp_path / "latest.pt"
    atomic_torch_save(
        path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": state,
        },
    )
    expected = (random.random(), np.random.rand(), torch.rand(3))

    restored_model = nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored_model.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_to_device(restored_optimizer, torch.device("cpu"))
    restore_rng_state(checkpoint["rng"], runtime)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_resume_contract_ignores_prose_but_rejects_computational_changes() -> None:
    metadata = {
        "schema": "training",
        "architecture": "shared_latent_successor",
        "objective": "original prose",
        "examples_sha256": "examples",
        "prefix_evidence_registry_sha256": "prefixes",
        "model_dim": 192,
        "parameters": {"total": 816202, "trainable": 816202},
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-2,
            "gradient_clip": 1.0,
            "target_candidates_per_update": 2048,
            "maximum_epochs": 80,
            "early_stopping_epochs": 10,
            "prequential_gradient_boundary": "descriptive prose",
        },
        "distributed": {
            "world_size": 4,
            "backend": "nccl",
            "sharding": "descriptive prose",
        },
    }
    rewritten = {
        **metadata,
        "objective": "clearer prose",
        "unused_optional_registry_sha256": None,
        "optimization": {
            **metadata["optimization"],
            "prequential_gradient_boundary": "clearer prose",
        },
    }
    assert computational_resume_contract(metadata) == computational_resume_contract(
        rewritten
    )
    changed_data = {**rewritten, "examples_sha256": "different"}
    assert computational_resume_contract(metadata) != computational_resume_contract(
        changed_data
    )
    changed_optimizer = {
        **rewritten,
        "optimization": {**rewritten["optimization"], "learning_rate": 1e-3},
    }
    assert computational_resume_contract(metadata) != computational_resume_contract(
        changed_optimizer
    )
    changed_world = {
        **rewritten,
        "distributed": {**rewritten["distributed"], "world_size": 8},
    }
    assert computational_resume_contract(metadata) != computational_resume_contract(
        changed_world
    )
