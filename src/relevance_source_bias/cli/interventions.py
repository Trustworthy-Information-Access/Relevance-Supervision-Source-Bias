"""CLI commands for training- and inference-time interventions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.artifacts import load_feature_values
from ..core.config import load_yaml
from ..core.data import (
    EmbeddingTable,
    load_corpus,
    load_qrels,
    load_run,
    write_json,
    write_jsonl,
)
from ..interventions.cdc import apply_cdc_correction, estimate_cdc_coefficient
from ..interventions.length_control import (
    corpus_length_statistics,
    length_controlled_rewrite_rows,
)
from ..interventions.positive_pool import (
    positive_pool_document_ids,
    select_positive_pool_negatives,
)
from ..interventions.projection import (
    CalibrationGroup,
    calibrate_leave_one_out,
    project_out,
)
from .common import emit, named_paths


def _load_calibration_groups(path: str | Path) -> dict[str, CalibrationGroup]:
    config = load_yaml(path)
    raw_groups = config.get("groups")
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError("Calibration YAML must contain a non-empty 'groups' mapping")
    defaults = config.get("suffixes", {})
    groups: dict[str, CalibrationGroup] = {}
    for name, value in raw_groups.items():
        if not isinstance(value, dict):
            raise ValueError(f"Calibration group {name!r} must be a mapping")
        groups[str(name)] = CalibrationGroup(
            datasets=tuple(str(item) for item in value.get("datasets", [name])),
            human=EmbeddingTable.load(value["human"]),
            llm=EmbeddingTable.load(value["llm"]),
            human_suffix=str(value.get("human_suffix", defaults.get("human", "-human"))),
            llm_suffix=str(value.get("llm_suffix", defaults.get("llm", "-llama-2-7b-chat-tmp0.2"))),
        )
    return groups


def command_debias(args: argparse.Namespace) -> None:
    if len(args.passages) != len(args.output):
        raise ValueError("--passages and --output must contain the same number of paths")
    direction, metadata = calibrate_leave_one_out(
        _load_calibration_groups(args.calibration),
        target_dataset=args.target,
        samples_per_group=args.samples_per_group,
        seed=args.seed,
    )
    written: list[str] = []
    for input_path, output_path in zip(args.passages, args.output, strict=True):
        table = EmbeddingTable.load(input_path)
        EmbeddingTable(table.ids, project_out(table.vectors, direction)).save(output_path)
        written.append(str(output_path))
    metadata.update(
        {
            "calibration": str(args.calibration),
            "inputs": [str(item) for item in args.passages],
            "outputs": written,
            "post_projection_normalization": False,
        }
    )
    metadata_path = args.metadata or str(Path(args.output[0]).with_suffix(".calibration.json"))
    write_json(metadata, metadata_path)
    print(f"Wrote {len(written)} projected embedding table(s); metadata: {metadata_path}")


def command_cdc(args: argparse.Namespace) -> None:
    estimate = estimate_cdc_coefficient(
        load_run(args.calibration_run),
        load_qrels(args.calibration_qrels),
        load_feature_values(args.calibration_features, args.field),
        human_source=args.human_source,
        llm_source=args.llm_source,
        budget=args.budget,
        seed=args.seed,
    )
    corrected = apply_cdc_correction(
        load_run(args.run),
        load_feature_values(args.features, args.field),
        float(estimate["coefficient"]),
    )
    write_json(corrected, args.output)
    metadata = {
        "calibration_run": args.calibration_run,
        "calibration_qrels": args.calibration_qrels,
        "calibration_features": args.calibration_features,
        "run": args.run,
        "features": args.features,
        "output": args.output,
        "human_source": args.human_source,
        "llm_source": args.llm_source,
        "estimate": estimate,
    }
    metadata_path = args.metadata or str(args.output) + ".metadata.json"
    write_json(metadata, metadata_path)
    print(f"Wrote CDC-corrected run to {args.output}; metadata: {metadata_path}")


def command_build_positive_pool(args: argparse.Namespace) -> None:
    qrels = load_qrels(args.qrels)
    corpus = load_corpus(args.corpus)
    document_ids = sorted(positive_pool_document_ids(qrels))
    missing = [document_id for document_id in document_ids if document_id not in corpus]
    if missing:
        raise KeyError(f"{len(missing)} positive-pool documents are missing from the corpus")
    write_jsonl(
        ({"_id": document_id, **corpus[document_id]} for document_id in document_ids),
        args.output,
    )
    print(f"Wrote {len(document_ids)} unique positive-pool passages to {args.output}")


def command_prepare_positive_pool_negatives(args: argparse.Namespace) -> None:
    pairs = select_positive_pool_negatives(
        load_run(args.run), load_qrels(args.qrels), seed=args.seed
    )
    write_jsonl(pairs, args.output)
    write_json(
        {
            "run": args.run,
            "qrels": args.qrels,
            "seed": args.seed,
            "num_pairs": len(pairs),
            "output": args.output,
        },
        str(args.output) + ".metadata.json",
    )
    print(f"Wrote {len(pairs)} positive-pool training pairs to {args.output}")


def command_length_rewrite(args: argparse.Namespace) -> None:
    rows = length_controlled_rewrite_rows(
        args.corpus,
        model_name=args.model,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
    )
    write_jsonl(rows, args.output)
    metadata = {
        "corpus": args.corpus,
        "output": args.output,
        "model": args.model,
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "device": args.device,
    }
    metadata_path = args.metadata or str(args.output) + ".metadata.json"
    write_json(metadata, metadata_path)
    print(f"Wrote length-controlled rewrites to {args.output}; metadata: {metadata_path}")


def command_length_stats(args: argparse.Namespace) -> None:
    emit(
        corpus_length_statistics(
            named_paths(args.corpus, option="--corpus"),
            tokenizer=args.tokenizer,
            include_title=args.include_title,
        ),
        args.output,
    )


def register(subparsers) -> None:
    debias = subparsers.add_parser("debias", help="Calibrate and project passage embeddings")
    debias.add_argument("--calibration", required=True)
    debias.add_argument("--target", required=True)
    debias.add_argument("--passages", nargs="+", required=True)
    debias.add_argument("--output", nargs="+", required=True)
    debias.add_argument("--samples-per-group", type=int, default=100)
    debias.add_argument("--seed", type=int, default=42)
    debias.add_argument("--metadata")
    debias.set_defaults(func=command_debias)

    cdc = subparsers.add_parser("cdc", help="Apply the PPL-based CDC score correction")
    cdc.add_argument("--calibration-run", required=True)
    cdc.add_argument("--calibration-qrels", required=True)
    cdc.add_argument("--calibration-features", required=True)
    cdc.add_argument("--run", required=True)
    cdc.add_argument("--features", required=True)
    cdc.add_argument("--field", default="value")
    cdc.add_argument("--human-source", default="human")
    cdc.add_argument("--llm-source", default="llama-2-7b-chat-tmp0.2")
    cdc.add_argument("--budget", type=int, default=128)
    cdc.add_argument("--seed", type=int, default=42)
    cdc.add_argument("--output", required=True)
    cdc.add_argument("--metadata")
    cdc.set_defaults(func=command_cdc)

    build_pool = subparsers.add_parser(
        "build-positive-pool", help="Extract the unique annotated-positive passage pool"
    )
    build_pool.add_argument("--corpus", required=True)
    build_pool.add_argument("--qrels", required=True)
    build_pool.add_argument("--output", required=True)
    build_pool.set_defaults(func=command_build_positive_pool)

    pool_negatives = subparsers.add_parser(
        "prepare-positive-pool-negatives",
        help="Select each query's top BM25-ranked other-query positive",
    )
    pool_negatives.add_argument("--run", required=True)
    pool_negatives.add_argument("--qrels", required=True)
    pool_negatives.add_argument("--seed", type=int, default=42)
    pool_negatives.add_argument("--output", required=True)
    pool_negatives.set_defaults(func=command_prepare_positive_pool_negatives)

    length_rewrite = subparsers.add_parser(
        "length-rewrite", help="Create the paper's length-controlled corpus rewrite"
    )
    length_rewrite.add_argument("--corpus", required=True)
    length_rewrite.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    length_rewrite.add_argument("--batch-size", type=int, default=8)
    length_rewrite.add_argument("--max-input-length", type=int, default=4096)
    length_rewrite.add_argument("--max-new-tokens", type=int, default=2048)
    length_rewrite.add_argument("--temperature", type=float, default=0.2)
    length_rewrite.add_argument("--top-p", type=float, default=1.0)
    length_rewrite.add_argument("--seed", type=int, default=42)
    length_rewrite.add_argument("--device")
    length_rewrite.add_argument("--output", required=True)
    length_rewrite.add_argument("--metadata")
    length_rewrite.set_defaults(func=command_length_rewrite)

    length_stats = subparsers.add_parser(
        "length-stats", help="Compare Lucene-token lengths across aligned corpora"
    )
    length_stats.add_argument(
        "--corpus", action="append", required=True, help="Named corpus as name=path"
    )
    length_stats.add_argument("--tokenizer", choices=["lucene", "regex"], default="lucene")
    length_stats.add_argument(
        "--include-title", action=argparse.BooleanOptionalAction, default=False
    )
    length_stats.add_argument("--output")
    length_stats.set_defaults(func=command_length_stats)
