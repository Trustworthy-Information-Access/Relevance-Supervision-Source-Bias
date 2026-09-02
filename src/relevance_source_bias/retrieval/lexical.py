"""Small Pyserini wrapper for the BM25 controls used in the appendix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from ..core.data import format_document, iter_jsonl, load_queries, write_jsonl


def prepare_pyserini_corpus(
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    include_title: bool = False,
) -> int:
    """Convert a BEIR JSONL corpus into Pyserini's `id`/`contents` JSONL form."""
    count = 0

    def rows():
        nonlocal count
        for row in iter_jsonl(corpus_path):
            document_id = row.get("_id", row.get("id", row.get("docid")))
            if document_id is None:
                raise ValueError("A corpus row has no _id, id, or docid")
            count += 1
            yield {
                "id": str(document_id),
                "contents": format_document(row, include_title=include_title),
            }

    write_jsonl(rows(), output_path)
    return count


def prepare_pyserini_topics(queries_path: str | Path, output_path: str | Path) -> int:
    """Write query-ID/text topics accepted by `pyserini.search.lucene`."""
    queries = load_queries(queries_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id, query in queries.items():
            cleaned = " ".join(query.split())
            handle.write(f"{query_id}\t{cleaned}\n")
    return len(queries)


def pyserini_commands(
    *,
    input_dir: str | Path,
    index_dir: str | Path,
    topics_path: str | Path,
    output_path: str | Path,
    hits: int = 100,
    k1: float = 0.82,
    b: float = 0.68,
    threads: int = 8,
    batch_size: int = 128,
) -> tuple[list[str], list[str]]:
    if hits <= 0 or threads <= 0 or batch_size <= 0:
        raise ValueError("hits, threads, and batch_size must be positive")
    if k1 < 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
    index = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(input_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    search = [
        sys.executable,
        "-m",
        "pyserini.search.lucene",
        "--index",
        str(index_dir),
        "--topics",
        str(topics_path),
        "--output",
        str(output_path),
        "--hits",
        str(hits),
        "--bm25",
        "--k1",
        str(k1),
        "--b",
        str(b),
        "--threads",
        str(threads),
        "--batch-size",
        str(batch_size),
    ]
    return index, search


def run_pyserini_bm25(
    corpus_path: str | Path,
    queries_path: str | Path,
    work_dir: str | Path,
    output_path: str | Path,
    *,
    include_title: bool = False,
    hits: int = 100,
    k1: float = 0.82,
    b: float = 0.68,
    threads: int = 8,
    batch_size: int = 128,
    force_index: bool = False,
) -> dict[str, Any]:
    """Prepare inputs, build/reuse a Lucene index, and write a six-column TREC run."""
    work_dir = Path(work_dir)
    input_dir = work_dir / "input"
    corpus_output = input_dir / "docs.jsonl"
    topics_output = work_dir / "topics.tsv"
    index_dir = work_dir / "index"
    output_path = Path(output_path)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document_count = prepare_pyserini_corpus(
        corpus_path, corpus_output, include_title=include_title
    )
    query_count = prepare_pyserini_topics(queries_path, topics_output)
    index_command, search_command = pyserini_commands(
        input_dir=input_dir,
        index_dir=index_dir,
        topics_path=topics_output,
        output_path=output_path,
        hits=hits,
        k1=k1,
        b=b,
        threads=threads,
        batch_size=batch_size,
    )
    reused_index = index_dir.is_dir() and any(index_dir.iterdir()) and not force_index
    if not reused_index:
        subprocess.run(index_command, check=True)
    subprocess.run(search_command, check=True)
    return {
        "corpus": str(corpus_path),
        "queries": str(queries_path),
        "work_dir": str(work_dir),
        "output": str(output_path),
        "num_documents": document_count,
        "num_queries": query_count,
        "include_title": include_title,
        "hits": hits,
        "k1": k1,
        "b": b,
        "threads": threads,
        "batch_size": batch_size,
        "reused_index": reused_index,
        "index_command": index_command,
        "search_command": search_command,
    }
