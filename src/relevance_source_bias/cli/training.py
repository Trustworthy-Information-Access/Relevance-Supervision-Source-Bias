"""CLI commands for controlled training and experiment matrices."""

from __future__ import annotations

import argparse

from ..core.config import load_yaml
from ..training.plans import prepare_training_plan
from ..training.runner import train_from_config
from ..workflows.aggregation import aggregate_matrix
from ..workflows.matrix import run_matrix
from .common import emit


def command_prepare_training(args: argparse.Namespace) -> None:
    emit(prepare_training_plan(load_yaml(args.config)))


def command_train(args: argparse.Namespace) -> None:
    emit(train_from_config(load_yaml(args.config)))


def command_run_matrix(args: argparse.Namespace) -> None:
    emit(
        run_matrix(
            load_yaml(args.config),
            force=args.force,
            evaluation_only=args.evaluation_only,
            fail_fast=args.fail_fast,
        )
    )


def command_aggregate_matrix(args: argparse.Namespace) -> None:
    emit(aggregate_matrix(load_yaml(args.config)))


def register(subparsers) -> None:
    prepare = subparsers.add_parser(
        "prepare-training", help="Build a deterministic controlled training plan"
    )
    prepare.add_argument("--config", required=True)
    prepare.set_defaults(func=command_prepare_training)

    train = subparsers.add_parser("train", help="Fine-tune a retriever from a training plan")
    train.add_argument("--config", required=True)
    train.set_defaults(func=command_train)

    matrix = subparsers.add_parser(
        "run-matrix", help="Run or resume a dataset-by-retriever evaluation matrix"
    )
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--force", action="store_true")
    matrix.add_argument("--evaluation-only", action="store_true")
    matrix.add_argument("--fail-fast", action="store_true")
    matrix.set_defaults(func=command_run_matrix)

    aggregate = subparsers.add_parser(
        "aggregate-matrix", help="Regenerate JSON, CSV, and LaTeX matrix summaries"
    )
    aggregate.add_argument("--config", required=True)
    aggregate.set_defaults(func=command_aggregate_matrix)
