"""Unified command dispatcher for data, training, evaluation, and profiling."""

from __future__ import annotations

import argparse
import importlib
import sys


def _commands() -> dict[str, tuple[str, str]]:
    return {
        "data": ("convert a dataset to the manifest format", "provia.data.cli:main"),
        "train-single": (
            "train PROVIA on a single-actor benchmark",
            "provia.cli:train_single_actor",
        ),
        "train-single-shared-successor": (
            "train the shared-successor variant used by the retrained ablation",
            "provia.cli:train_single_actor_shared_successor",
        ),
        "train-bimanual": (
            "train PROVIA on a two-hand benchmark",
            "provia.cli:train_bimanual",
        ),
        "materialize-single-query": (
            "decode an annotation-free single-actor test query",
            "provia.evaluation.materialize_single_actor_query:main",
        ),
        "materialize-single-prefix": (
            "compile annotation-free prefix evidence for a single-actor test query",
            "provia.evaluation.materialize_single_actor_prefix:main",
        ),
        "materialize-bimanual-query": (
            "decode an annotation-free two-hand test query",
            "provia.evaluation.materialize_bimanual_query:main",
        ),
        "materialize-bimanual-prefix": (
            "compile annotation-free prefix evidence for a two-hand test query",
            "provia.evaluation.materialize_bimanual_prefix:main",
        ),
        "evaluate-single": (
            "evaluate a single-actor checkpoint",
            "provia.evaluation.evaluate_single_actor:main",
        ),
        "evaluate-bimanual": (
            "evaluate a two-hand checkpoint",
            "provia.evaluation.evaluate_bimanual:main",
        ),
        "benchmark-single": (
            "time single-actor streaming inference on CPU",
            "provia.evaluation.benchmark_single_actor:main",
        ),
        "benchmark-bimanual": (
            "time two-hand streaming inference on CPU",
            "provia.evaluation.benchmark_bimanual:main",
        ),
    }


def main() -> object:
    commands = _commands()
    parser = argparse.ArgumentParser(
        prog="provia",
        description="PROVIA command-line interface",
    )
    parser.add_argument("command", nargs="?", choices=tuple(commands))
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = sys.argv[1]
    if command not in commands:
        parser.error(f"invalid command: {command!r}")
    sys.argv = [f"provia {command}", *sys.argv[2:]]
    module_name, function_name = commands[command][1].split(":", maxsplit=1)
    function = getattr(importlib.import_module(module_name), function_name)
    return function()


if __name__ == "__main__":
    main()
