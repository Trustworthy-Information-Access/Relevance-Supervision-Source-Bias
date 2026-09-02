"""CLI commands for linguistic and embedding-space analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..analysis.artifacts import (
    build_idf,
    calculate_ppl_rows,
    compare_feature_groups,
    correlate_scores_with_feature,
    load_feature_values,
    score_idf_rows,
)
from ..analysis.geometry import (
    cosine_similarity,
    direction_cosine_matrix,
    estimate_direction,
    paired_direction_from_tables,
    paired_vectors_from_records,
    sample_positive_negative_pairs,
    summarize_pair_ids,
)
from ..analysis.significance import paired_sign_flip_validation, random_cosine_reference
from ..core.config import load_yaml
from ..core.data import (
    EmbeddingTable,
    add_source_suffix,
    iter_jsonl,
    load_qrels,
    load_run,
    pair_embedding_tables,
    write_json,
    write_jsonl,
)
from .common import emit


def command_ppl(args: argparse.Namespace) -> None:
    rows = calculate_ppl_rows(
        args.corpus,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_title=args.include_title,
        device=args.device,
    )
    if args.source:
        rows = ({**row, "_id": add_source_suffix(str(row["_id"]), args.source)} for row in rows)
    write_jsonl(rows, args.output)
    print(f"Wrote PPL features to {args.output}")


def command_idf_build(args: argparse.Namespace) -> None:
    result = build_idf(args.corpus, tokenizer=args.tokenizer, include_title=args.include_title)
    write_json(result, args.output)
    print(f"Wrote {len(result['idf'])} IDF values to {args.output}")


def command_idf_score(args: argparse.Namespace) -> None:
    with Path(args.idf).open("r", encoding="utf-8") as handle:
        idf = json.load(handle)
    rows = score_idf_rows(
        args.corpus,
        idf,
        tokenizer=args.tokenizer,
        include_title=args.include_title,
    )
    if args.source:
        rows = ({**row, "_id": add_source_suffix(str(row["_id"]), args.source)} for row in rows)
    write_jsonl(rows, args.output)
    print(f"Wrote passage IDF features to {args.output}")


def command_compare_features(args: argparse.Namespace) -> None:
    left = load_feature_values(args.left, args.field)
    right = load_feature_values(args.right, args.field)
    result = compare_feature_groups(list(left.values()), list(right.values()))
    result.update({"left": args.left, "right": args.right, "field": args.field})
    emit(result, args.output)


def command_correlate_scores(args: argparse.Namespace) -> None:
    sources = args.source or ["human", "llama-2-7b-chat-tmp0.2"]
    result = correlate_scores_with_feature(
        load_run(args.run),
        load_feature_values(args.features, args.field),
        sources=sources,
    )
    result.update(
        {
            "run": args.run,
            "features": args.features,
            "field": args.field,
            "sources": sources,
        }
    )
    emit(result, args.output)


def command_direction(args: argparse.Namespace) -> None:
    left, right = EmbeddingTable.load(args.left), EmbeddingTable.load(args.right)
    if args.pairs:
        records: list[dict[str, object]] = []
        for row in iter_jsonl(args.pairs):
            nested = row.get("samples")
            if isinstance(nested, list):
                records.extend(item for item in nested if isinstance(item, dict))
            else:
                records.append(row)
        left_vectors, right_vectors, ids = paired_vectors_from_records(
            left,
            right,
            records,
            left_id_field=args.left_id_field,
            right_id_field=args.right_id_field,
            left_suffix=args.left_suffix,
            right_suffix=args.right_suffix,
        )
        result = estimate_direction(
            left_vectors, right_vectors, permutations=args.permutations, seed=args.seed
        )
        result.update(summarize_pair_ids(ids))
    else:
        result = paired_direction_from_tables(
            left,
            right,
            left_suffix=args.left_suffix,
            right_suffix=args.right_suffix,
            permutations=args.permutations,
            seed=args.seed,
        )
    result.update(
        {
            "left": args.left,
            "right": args.right,
            "displacement": "right_minus_left",
            "left_suffix": args.left_suffix,
            "right_suffix": args.right_suffix,
            "pairs": args.pairs,
        }
    )
    emit(result, args.output)


def command_prepare_pn_pairs(args: argparse.Namespace) -> None:
    pairs = sample_positive_negative_pairs(
        load_run(args.run), load_qrels(args.qrels), top_k=args.top_k, seed=args.seed
    )
    write_jsonl(pairs, args.output)
    metadata = {
        "run": args.run,
        "qrels": args.qrels,
        "top_k": args.top_k,
        "seed": args.seed,
        "num_pairs": len(pairs),
        "output": args.output,
    }
    metadata_path = str(args.output) + ".metadata.json"
    write_json(metadata, metadata_path)
    print(f"Wrote {len(pairs)} PN pairs to {args.output}; metadata: {metadata_path}")


def command_compare_directions(args: argparse.Namespace) -> None:
    directions: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for item in args.direction:
        if "=" not in item:
            raise ValueError("Each --direction must have the form name=path")
        name, path = item.split("=", 1)
        if not name or name in directions:
            raise ValueError(f"Direction names must be non-empty and unique: {name!r}")
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        vector = value.get("mean_direction", value.get("unit_direction"))
        if vector is None:
            raise ValueError(f"Direction result {path} has no mean_direction or unit_direction")
        directions[name] = vector
        paths[name] = path
    result: dict[str, Any] = {
        "directions": paths,
        "cosine_matrix": direction_cosine_matrix(directions),
    }
    if args.reference:
        with Path(args.reference).open("r", encoding="utf-8") as handle:
            reference_data = json.load(handle)
        reference = reference_data.get("mean_direction", reference_data.get("unit_direction"))
        if reference is None:
            raise ValueError("Reference result has no mean_direction or unit_direction")
        result["reference"] = args.reference
        result["reference_alignment"] = {
            name: cosine_similarity(vector, reference) for name, vector in directions.items()
        }
    emit(result, args.output)


def command_sign_flip(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    raw_datasets = config.get("datasets")
    if not isinstance(raw_datasets, dict) or len(raw_datasets) < 2:
        raise ValueError("Sign-flip YAML must contain at least two datasets")
    defaults = config.get("suffixes", {})
    displacements = {}
    inputs: dict[str, dict[str, object]] = {}
    for name, value in raw_datasets.items():
        if not isinstance(value, dict) or "left" not in value or "right" not in value:
            raise ValueError(f"Sign-flip dataset {name!r} requires left and right paths")
        left_suffix = str(value.get("left_suffix", defaults.get("left", "-human")))
        right_suffix = str(
            value.get("right_suffix", defaults.get("right", "-llama-2-7b-chat-tmp0.2"))
        )
        left_path, right_path = str(value["left"]), str(value["right"])
        left_vectors, right_vectors, pair_ids = pair_embedding_tables(
            EmbeddingTable.load(left_path),
            EmbeddingTable.load(right_path),
            left_suffix=left_suffix,
            right_suffix=right_suffix,
        )
        displacements[str(name)] = right_vectors - left_vectors
        inputs[str(name)] = {
            "left": left_path,
            "right": right_path,
            "left_suffix": left_suffix,
            "right_suffix": right_suffix,
            "num_pairs": len(pair_ids),
            **summarize_pair_ids(pair_ids),
        }

    reference_path = config.get("reference")
    if not reference_path:
        raise ValueError("Sign-flip YAML must provide a PN direction as 'reference'")
    with Path(reference_path).open("r", encoding="utf-8") as handle:
        reference_data = json.load(handle)
    reference = reference_data.get("mean_direction", reference_data.get("unit_direction"))
    if reference is None:
        raise ValueError("The PN reference has no mean_direction or unit_direction")
    result = paired_sign_flip_validation(
        displacements,
        reference_direction=reference,
        permutations=args.permutations,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    result.update({"config": str(args.config), "reference": str(reference_path), "inputs": inputs})
    emit(result, args.output)


def command_random_reference(args: argparse.Namespace) -> None:
    emit(
        random_cosine_reference(
            args.dimension,
            sigma_multiplier=args.sigma_multiplier,
            samples=args.samples,
            seed=args.seed,
        ),
        args.output,
    )


def register(subparsers) -> None:
    ppl = subparsers.add_parser("ppl", help="Compute per-document causal-LM perplexity")
    ppl.add_argument("--corpus", required=True)
    ppl.add_argument("--model", required=True)
    ppl.add_argument("--batch-size", type=int, default=8)
    ppl.add_argument("--max-length", type=int, default=512)
    ppl.add_argument("--include-title", action=argparse.BooleanOptionalAction, default=False)
    ppl.add_argument("--device")
    ppl.add_argument("--source", help="Optional source suffix for output document IDs")
    ppl.add_argument("--output", required=True)
    ppl.set_defaults(func=command_ppl)

    idf_build = subparsers.add_parser("idf-build", help="Build full-collection IDF values")
    idf_build.add_argument("--corpus", required=True)
    idf_build.add_argument("--tokenizer", choices=["lucene", "regex"], default="lucene")
    idf_build.add_argument("--include-title", action=argparse.BooleanOptionalAction, default=False)
    idf_build.add_argument("--output", required=True)
    idf_build.set_defaults(func=command_idf_build)

    idf_score = subparsers.add_parser("idf-score", help="Compute passage median IDF")
    idf_score.add_argument("--corpus", required=True)
    idf_score.add_argument("--idf", required=True)
    idf_score.add_argument("--tokenizer", choices=["lucene", "regex"])
    idf_score.add_argument("--include-title", action=argparse.BooleanOptionalAction, default=None)
    idf_score.add_argument("--source", help="Optional source suffix for output document IDs")
    idf_score.add_argument("--output", required=True)
    idf_score.set_defaults(func=command_idf_score)

    compare = subparsers.add_parser("compare-features", help="Compare two feature files")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--field", default="value")
    compare.add_argument("--output")
    compare.set_defaults(func=command_compare_features)

    correlate = subparsers.add_parser(
        "correlate-scores", help="Correlate features with within-query normalized scores"
    )
    correlate.add_argument("--run", required=True)
    correlate.add_argument("--features", required=True)
    correlate.add_argument("--field", default="value")
    correlate.add_argument("--source", action="append")
    correlate.add_argument("--output")
    correlate.set_defaults(func=command_correlate_scores)

    direction = subparsers.add_parser("direction", help="Estimate a paired displacement direction")
    direction.add_argument("--left", "--human", dest="left", required=True)
    direction.add_argument("--right", "--llm", dest="right", required=True)
    direction.add_argument("--left-suffix", "--human-suffix", dest="left_suffix", default="")
    direction.add_argument("--right-suffix", "--llm-suffix", dest="right_suffix", default="")
    direction.add_argument("--pairs", help="Optional JSONL with explicit left/right IDs")
    direction.add_argument("--left-id-field", default="hard_negative_id")
    direction.add_argument("--right-id-field", default="positive_id")
    direction.add_argument("--permutations", type=int, default=0)
    direction.add_argument("--seed", type=int, default=42)
    direction.add_argument("--output")
    direction.set_defaults(func=command_direction)

    pn_pairs = subparsers.add_parser(
        "prepare-pn-pairs", help="Sample positives and BM25 top-k negatives"
    )
    pn_pairs.add_argument(
        "--run", required=True, help="BM25 run in nested JSON or six-column TREC format"
    )
    pn_pairs.add_argument("--qrels", required=True)
    pn_pairs.add_argument("--top-k", type=int, default=10)
    pn_pairs.add_argument("--seed", type=int, default=42)
    pn_pairs.add_argument("--output", required=True)
    pn_pairs.set_defaults(func=command_prepare_pn_pairs)

    compare_directions = subparsers.add_parser(
        "compare-directions", help="Compare saved mean directions across datasets/models"
    )
    compare_directions.add_argument(
        "--direction", action="append", required=True, help="Named result as name=path"
    )
    compare_directions.add_argument("--reference", help="Optional PN direction result")
    compare_directions.add_argument("--output")
    compare_directions.set_defaults(func=command_compare_directions)

    sign_flip = subparsers.add_parser(
        "sign-flip", help="Run the appendix paired sign-flip validation"
    )
    sign_flip.add_argument("--config", required=True)
    sign_flip.add_argument("--permutations", type=int, default=1000)
    sign_flip.add_argument("--seed", type=int, default=42)
    sign_flip.add_argument("--batch-size", type=int, default=16)
    sign_flip.add_argument("--output", required=True)
    sign_flip.set_defaults(func=command_sign_flip)

    random_reference = subparsers.add_parser(
        "random-reference", help="Compute the random high-dimensional cosine reference"
    )
    random_reference.add_argument("--dimension", type=int, default=768)
    random_reference.add_argument("--sigma-multiplier", type=float, default=3.0)
    random_reference.add_argument("--samples", type=int, default=0)
    random_reference.add_argument("--seed", type=int, default=42)
    random_reference.add_argument("--output")
    random_reference.set_defaults(func=command_random_reference)
