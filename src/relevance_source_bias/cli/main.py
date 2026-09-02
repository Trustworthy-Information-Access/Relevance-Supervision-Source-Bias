"""Top-level command parser for the reproducibility workflows."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from . import analysis, interventions, retrieval, training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsb", description="Relevance supervision and source-bias experiments"
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieval.register(subparsers)
    analysis.register(subparsers)
    training.register(subparsers)
    interventions.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
