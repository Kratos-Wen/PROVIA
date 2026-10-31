"""Command-line entry points for PROVIA training."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


def _training_parser(*, bimanual: bool) -> argparse.ArgumentParser:
    stream = "two-hand" if bimanual else "single-actor"
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=f"Train PROVIA on a {stream} benchmark.",
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--prefix-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    if bimanual:
        parser.add_argument("--expert-effect", required=True)
    else:
        parser.add_argument("--expert-effect-registry", required=True)
        parser.add_argument("--no-factual-history", action="store_true")
    parser.add_argument("--language-registry", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser


def _run_registered_model(
    main: Callable[[], None],
    registered_option: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    parser.parse_args(sys.argv[1:])
    if registered_option is not None:
        sys.argv.append(registered_option)
    main()


def train_single_actor() -> None:
    """Train PROVIA on a single-actor benchmark."""

    from .training.train_single_actor import main

    _run_registered_model(main, None, _training_parser(bimanual=False))


def train_single_actor_shared_successor() -> None:
    """Train the shared-successor variant used by the retrained ablation."""

    from .training.train_single_actor import main

    _run_registered_model(main, "--shared-successor", _training_parser(bimanual=False))


def train_bimanual() -> None:
    """Train PROVIA on a two-hand benchmark."""

    from .training.train_bimanual import main

    _run_registered_model(main, None, _training_parser(bimanual=True))


__all__ = [
    "train_bimanual",
    "train_single_actor",
    "train_single_actor_shared_successor",
]
