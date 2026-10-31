"""Command-line conversion and validation entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .adapters import ADAPTERS
from .adapters.base import read_config
from .io import write_jsonl


def _convert(args: argparse.Namespace) -> int:
    config = read_config(args.config)
    dataset = args.dataset.lower()
    configured = str(config.get("dataset", dataset)).lower()
    if configured != dataset:
        raise ValueError(
            f"CLI dataset {dataset!r} disagrees with config dataset {configured!r}"
        )
    adapter = ADAPTERS[dataset](config)
    report = write_jsonl(
        adapter.convert(),
        args.out,
        require_paths=args.require_paths,
        enforce_subject_disjoint=args.subject_disjoint,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert native procedural datasets into the PROVIA schema"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="convert one native dataset")
    convert.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    convert.add_argument("--config", type=Path, required=True)
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--require-paths", action="store_true")
    convert.add_argument("--subject-disjoint", action="store_true")
    convert.set_defaults(function=_convert)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
