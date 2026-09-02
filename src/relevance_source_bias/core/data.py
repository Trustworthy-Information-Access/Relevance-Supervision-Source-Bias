"""Input/output helpers for BEIR-style text data, runs, and embedding tables."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects from *path* with useful line-number errors."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield value


def write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _record_id(row: Mapping[str, Any], path: Path) -> str:
    value = row.get("_id", row.get("id", row.get("docid")))
    if value is None:
        raise ValueError(f"A record in {path} has no _id, id, or docid field")
    return str(value)


def load_corpus(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a BEIR corpus JSONL keyed by document ID."""
    path = Path(path)
    corpus: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        doc_id = _record_id(row, path)
        if doc_id in corpus:
            raise ValueError(f"Duplicate document ID {doc_id!r} in {path}")
        corpus[doc_id] = {
            "title": str(row.get("title", "")),
            "text": str(row.get("text", row.get("contents", ""))),
            "metadata": row.get("metadata", {}),
        }
    return corpus


def load_queries(path: str | Path) -> dict[str, str]:
    """Load BEIR queries from JSONL (`_id`, `text`) into an ID-to-text mapping."""
    path = Path(path)
    queries: dict[str, str] = {}
    for row in iter_jsonl(path):
        query_id = _record_id(row, path)
        if query_id in queries:
            raise ValueError(f"Duplicate query ID {query_id!r} in {path}")
        queries[query_id] = str(row.get("text", row.get("query", "")))
    return queries


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load three-column BEIR or four-column TREC qrels."""
    path = Path(path)
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = line.rstrip("\n").split("\t") if "\t" in line else line.split()
            if row_number == 1 and row[0].lower().replace("_", "-") in {
                "query-id",
                "queryid",
            }:
                continue
            if len(row) < 3:
                raise ValueError(f"Expected at least 3 qrels columns at {path}:{row_number}")
            if len(row) >= 4 and row[1].upper() in {"Q0", "0"}:
                query_id, document_id, raw_relevance = row[0], row[2], row[3]
            else:
                query_id, document_id, raw_relevance = row[0], row[1], row[2]
            try:
                relevance = int(float(raw_relevance))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid relevance at {path}:{row_number}: {raw_relevance!r}"
                ) from exc
            qrels.setdefault(str(query_id), {})[str(document_id)] = relevance
    return qrels


def load_run(path: str | Path) -> dict[str, dict[str, float]]:
    """Load a nested JSON run or a standard six-column TREC run."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(4096).lstrip()
        handle.seek(0)
        if prefix.startswith("{"):
            raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError(f"Expected an object at the root of {path}")
            run: dict[str, dict[str, float]] = {}
            for query_id, scores in raw.items():
                if not isinstance(scores, dict):
                    raise ValueError(f"Run entry {query_id!r} is not an object")
                run[str(query_id)] = {str(doc_id): float(score) for doc_id, score in scores.items()}
            return run

        run = {}
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = line.split()
            if len(row) < 6:
                raise ValueError(f"Expected a six-column TREC run at {path}:{row_number}")
            query_id, document_id = str(row[0]), str(row[2])
            try:
                score = float(row[4])
            except ValueError as exc:
                raise ValueError(f"Invalid run score at {path}:{row_number}: {row[4]!r}") from exc
            previous = run.setdefault(query_id, {}).get(document_id)
            if previous is None or score > previous:
                run[query_id][document_id] = score
        return run


def format_document(document: Mapping[str, Any], include_title: bool = True) -> str:
    text = str(document.get("text", document.get("contents", ""))).strip()
    title = str(document.get("title", "")).strip()
    if include_title and title:
        return f"{title} {text}".strip()
    return text


def add_source_suffix(document_id: str, source: str) -> str:
    suffix = f"-{source}"
    return document_id if document_id.endswith(suffix) else document_id + suffix


def remove_suffix(value: str, suffix: str) -> str | None:
    """Remove *suffix*; an empty suffix means that IDs are already base IDs."""
    if not suffix:
        return value
    return value[: -len(suffix)] if value.endswith(suffix) else None


@dataclass(frozen=True)
class EmbeddingTable:
    """A row-aligned collection of string IDs and float embeddings."""

    ids: tuple[str, ...]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        vectors = np.asarray(self.vectors)
        if vectors.ndim != 2:
            raise ValueError(f"Embedding vectors must be 2-D, got {vectors.shape}")
        if len(self.ids) != vectors.shape[0]:
            raise ValueError(f"Got {len(self.ids)} IDs for {vectors.shape[0]} vectors")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Embedding IDs must be unique")
        object.__setattr__(self, "vectors", vectors.astype(np.float32, copy=False))
        object.__setattr__(self, "ids", tuple(str(item) for item in self.ids))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            np.savez_compressed(path, ids=np.asarray(self.ids, dtype=str), embeddings=self.vectors)
            return
        if suffix in {".h5", ".hdf5"}:
            string_type = h5py.string_dtype(encoding="utf-8")
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "ids", data=np.asarray(self.ids, dtype=object), dtype=string_type
                )
                handle.create_dataset("embeddings", data=self.vectors, compression="gzip")
            return
        raise ValueError(f"Unsupported embedding extension {suffix!r}; use .npz or .h5")

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingTable:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "embeddings" not in data:
                    raise ValueError(f"{path} has no 'embeddings' array")
                vectors = data["embeddings"]
                ids = data["ids"] if "ids" in data else np.arange(len(vectors)).astype(str)
            return cls(tuple(str(item) for item in ids.tolist()), vectors)
        if suffix in {".h5", ".hdf5"}:
            with h5py.File(path, "r") as handle:
                if "embeddings" not in handle:
                    raise ValueError(f"{path} has no 'embeddings' dataset")
                vectors = handle["embeddings"][:]
                if "ids" in handle:
                    raw_ids = handle["ids"][:]
                    ids = tuple(
                        item.decode("utf-8") if isinstance(item, bytes) else str(item)
                        for item in raw_ids
                    )
                elif "id_to_idx" in handle:
                    raw = handle["id_to_idx"][0]
                    mapping = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                    ordered: list[str | None] = [None] * len(mapping)
                    for item, index in mapping.items():
                        ordered[int(index)] = str(item)
                    if any(item is None for item in ordered):
                        raise ValueError(f"Invalid id_to_idx mapping in {path}")
                    ids = tuple(item for item in ordered if item is not None)
                else:
                    ids = tuple(str(index) for index in range(len(vectors)))
            return cls(ids, vectors)
        raise ValueError(f"Unsupported embedding extension {suffix!r}; use .npz or .h5")


def pair_embedding_tables(
    left: EmbeddingTable,
    right: EmbeddingTable,
    *,
    left_suffix: str = "",
    right_suffix: str = "",
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Align two embedding tables by base ID after removing source suffixes."""

    def index(table: EmbeddingTable, suffix: str) -> dict[str, int]:
        output: dict[str, int] = {}
        for row, full_id in enumerate(table.ids):
            base_id = remove_suffix(full_id, suffix)
            if base_id is None:
                continue
            if base_id in output:
                raise ValueError(f"Duplicate base ID {base_id!r} after stripping {suffix!r}")
            output[base_id] = row
        return output

    left_index, right_index = index(left, left_suffix), index(right, right_suffix)
    common = tuple(sorted(left_index.keys() & right_index.keys()))
    if not common:
        raise ValueError("No paired embedding IDs were found")
    left_rows = np.asarray([left_index[item] for item in common], dtype=np.int64)
    right_rows = np.asarray([right_index[item] for item in common], dtype=np.int64)
    return left.vectors[left_rows], right.vectors[right_rows], common


def concatenate_tables(tables: Sequence[EmbeddingTable]) -> EmbeddingTable:
    if not tables:
        raise ValueError("At least one embedding table is required")
    dimensions = {table.vectors.shape[1] for table in tables}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions differ: {sorted(dimensions)}")
    ids = tuple(item for table in tables for item in table.ids)
    return EmbeddingTable(ids, np.vstack([table.vectors for table in tables]))
