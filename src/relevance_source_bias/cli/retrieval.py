"""CLI commands for dense retrieval, BM25, and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..core.config import load_yaml
from ..core.data import EmbeddingTable, load_qrels, load_run, write_json
from ..retrieval.dense import exact_search, load_model_registry, retrieve_dataset
from ..retrieval.lexical import run_pyserini_bm25
from ..retrieval.metrics import evaluate_run
from .common import emit


def command_evaluate(args: argparse.Namespace) -> None:
    result = evaluate_run(
        load_run(args.run),
        human_source=args.human_source,
        llm_source=args.llm_source,
        k_values=args.k,
        qrels=load_qrels(args.qrels) if args.qrels else None,
    )
    emit(result, args.output)


def _dataset_paths(config: dict[str, Any], dataset: str) -> tuple[Path, Path, dict[str, Path]]:
    data_root = Path(config["data_root"])
    output_root = Path(config.get("output_root", "outputs"))
    dataset_root = data_root / dataset
    query_relative = str(config.get("queries", "queries.jsonl"))
    sources = config.get("sources", ["human", "llama-2-7b-chat-tmp0.2"])
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("Evaluation config must contain exactly two sources")
    corpus_pattern = str(config.get("corpus_pattern", "corpus/{source}.jsonl"))
    corpus_paths = {
        str(source): dataset_root / corpus_pattern.format(source=source) for source in sources
    }
    return dataset_root / query_relative, output_root, corpus_paths


def command_retrieve(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    registry = load_model_registry(args.models)
    if args.model not in registry:
        raise KeyError(f"Unknown model {args.model!r}; choices: {', '.join(sorted(registry))}")
    query_path, output_root, corpus_paths = _dataset_paths(config, args.dataset)
    output_dir = output_root / args.dataset / args.model
    run = retrieve_dataset(
        spec=registry[args.model],
        query_path=query_path,
        corpus_paths=corpus_paths,
        output_dir=output_dir,
        batch_size=int(config.get("batch_size", 128)),
        top_k=int(config.get("top_k", 200)),
        device=args.device or config.get("device"),
        corpus_batch_size=int(config.get("corpus_batch_size", 50_000)),
        remove_self_matches=bool(config.get("remove_self_matches", True)),
    )
    print(f"Wrote {len(run)} ranked lists to {output_dir / 'run.json'}")


def command_search(args: argparse.Namespace) -> None:
    run = exact_search(
        EmbeddingTable.load(args.queries),
        [EmbeddingTable.load(path) for path in args.corpus],
        top_k=args.top_k,
        score_function=args.score,
        query_batch_size=args.query_batch_size,
        corpus_batch_size=args.corpus_batch_size,
    )
    write_json(run, args.output)
    print(f"Wrote {len(run)} ranked lists to {args.output}")


def command_bm25(args: argparse.Namespace) -> None:
    result = run_pyserini_bm25(
        args.corpus,
        args.queries,
        args.work_dir,
        args.output,
        include_title=args.include_title,
        hits=args.hits,
        k1=args.k1,
        b=args.b,
        threads=args.threads,
        batch_size=args.batch_size,
        force_index=args.force_index,
    )
    metadata_path = args.metadata or str(args.output) + ".metadata.json"
    write_json(result, metadata_path)
    print(f"Wrote BM25 run to {args.output}; metadata: {metadata_path}")


def register(subparsers) -> None:
    evaluate = subparsers.add_parser("evaluate", help="Evaluate Delta-NDSR and NDCG")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--qrels")
    evaluate.add_argument("--human-source", default="human")
    evaluate.add_argument("--llm-source", default="llama-2-7b-chat-tmp0.2")
    evaluate.add_argument("--k", nargs="+", type=int, default=[5])
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=command_evaluate)

    retrieve = subparsers.add_parser("retrieve", help="Encode text and run exact retrieval")
    retrieve.add_argument("--config", required=True)
    retrieve.add_argument("--models", required=True)
    retrieve.add_argument("--dataset", required=True)
    retrieve.add_argument("--model", required=True)
    retrieve.add_argument("--device")
    retrieve.set_defaults(func=command_retrieve)

    search = subparsers.add_parser("search", help="Search precomputed embedding tables")
    search.add_argument("--queries", required=True)
    search.add_argument("--corpus", required=True, action="append")
    search.add_argument("--score", choices=["dot", "cosine", "cos_sim"], default="dot")
    search.add_argument("--top-k", type=int, default=200)
    search.add_argument("--query-batch-size", type=int, default=128)
    search.add_argument("--corpus-batch-size", type=int, default=50_000)
    search.add_argument("--output", required=True)
    search.set_defaults(func=command_search)

    bm25 = subparsers.add_parser(
        "bm25", help="Build or reuse a Pyserini index and retrieve a BEIR corpus"
    )
    bm25.add_argument("--corpus", required=True)
    bm25.add_argument("--queries", required=True)
    bm25.add_argument("--work-dir", required=True)
    bm25.add_argument("--hits", type=int, default=100)
    bm25.add_argument("--k1", type=float, default=0.82)
    bm25.add_argument("--b", type=float, default=0.68)
    bm25.add_argument("--threads", type=int, default=8)
    bm25.add_argument("--batch-size", type=int, default=128)
    bm25.add_argument("--include-title", action=argparse.BooleanOptionalAction, default=False)
    bm25.add_argument("--force-index", action="store_true")
    bm25.add_argument("--output", required=True)
    bm25.add_argument("--metadata")
    bm25.set_defaults(func=command_bm25)
