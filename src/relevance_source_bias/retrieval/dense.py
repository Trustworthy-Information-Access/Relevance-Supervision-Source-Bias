"""Dense encoding and memory-bounded exact retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import load_yaml
from ..core.data import (
    EmbeddingTable,
    add_source_suffix,
    format_document,
    load_corpus,
    load_queries,
    write_json,
)


def l2_normalize(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), epsilon)
    return vectors / norms


def exact_search(
    queries: EmbeddingTable,
    corpora: Sequence[EmbeddingTable],
    *,
    top_k: int = 200,
    score_function: str = "dot",
    query_batch_size: int = 128,
    corpus_batch_size: int = 50_000,
) -> dict[str, dict[str, float]]:
    """Search concatenated embedding tables without materializing the full score matrix."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if query_batch_size <= 0 or corpus_batch_size <= 0:
        raise ValueError("query_batch_size and corpus_batch_size must be positive")
    if score_function not in {"dot", "cosine", "cos_sim"}:
        raise ValueError("score_function must be 'dot' or 'cosine'")
    if not corpora:
        raise ValueError("At least one corpus embedding table is required")
    corpus_size = sum(len(table.ids) for table in corpora)
    if corpus_size == 0:
        raise ValueError("Corpus embeddings are empty")
    dimensions = {table.vectors.shape[1] for table in corpora}
    if len(dimensions) != 1 or queries.vectors.shape[1] not in dimensions:
        raise ValueError("Query and corpus embedding dimensions differ")
    query_vectors = queries.vectors
    if score_function in {"cosine", "cos_sim"}:
        query_vectors = l2_normalize(query_vectors)
    corpus_ids = tuple(document_id for table in corpora for document_id in table.ids)
    keep = min(top_k, corpus_size)
    output: dict[str, dict[str, float]] = {}
    for query_start in range(0, len(queries.ids), query_batch_size):
        query_end = min(query_start + query_batch_size, len(queries.ids))
        block_queries = query_vectors[query_start:query_end]
        best_scores = np.full((len(block_queries), keep), -np.inf, dtype=np.float32)
        best_indices = np.full((len(block_queries), keep), -1, dtype=np.int64)
        corpus_offset = 0
        for table in corpora:
            for corpus_start in range(0, len(table.ids), corpus_batch_size):
                corpus_end = min(corpus_start + corpus_batch_size, len(table.ids))
                block_corpus = table.vectors[corpus_start:corpus_end]
                if score_function in {"cosine", "cos_sim"}:
                    block_corpus = l2_normalize(block_corpus)
                scores = block_queries @ block_corpus.T
                scores = np.where(np.isfinite(scores), scores, -np.inf)
                indices = np.arange(
                    corpus_offset + corpus_start,
                    corpus_offset + corpus_end,
                    dtype=np.int64,
                )
                indices = np.broadcast_to(indices, scores.shape)
                merged_scores = np.concatenate((best_scores, scores), axis=1)
                merged_indices = np.concatenate((best_indices, indices), axis=1)
                selected = np.argpartition(merged_scores, -keep, axis=1)[:, -keep:]
                best_scores = np.take_along_axis(merged_scores, selected, axis=1)
                best_indices = np.take_along_axis(merged_indices, selected, axis=1)
            corpus_offset += len(table.ids)
        for row, query_id in enumerate(queries.ids[query_start:query_end]):
            pairs = [
                (corpus_ids[int(index)], float(score))
                for index, score in zip(best_indices[row], best_scores[row], strict=True)
                if index >= 0
            ]
            pairs.sort(key=lambda item: (-item[1], item[0]))
            output[query_id] = dict(pairs)
    return output


def remove_query_self_matches(
    run: Mapping[str, Mapping[str, float]],
    *,
    sources: Sequence[str],
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Remove source copies whose base document ID equals the query ID."""
    cleaned: dict[str, dict[str, float]] = {}
    for query_id, document_scores in run.items():
        excluded = {add_source_suffix(query_id, source) for source in sources}
        ranked = sorted(document_scores.items(), key=lambda item: (-item[1], item[0]))
        cleaned[query_id] = {
            document_id: score for document_id, score in ranked if document_id not in excluded
        }
        cleaned[query_id] = dict(list(cleaned[query_id].items())[:top_k])
    return cleaned


@dataclass(frozen=True)
class ModelSpec:
    name: str
    document_model: str
    query_model: str | None = None
    query_prefix: str = ""
    passage_prefix: str = ""
    score_function: str = "dot"
    normalize_embeddings: bool = False
    include_title: bool = True
    max_query_length: int = 32
    max_passage_length: int = 512

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> ModelSpec:
        if "model" not in value and "document_model" not in value:
            raise ValueError(f"Model {name!r} has no model/document_model")
        return cls(
            name=name,
            document_model=str(value.get("document_model", value.get("model"))),
            query_model=(str(value["query_model"]) if value.get("query_model") else None),
            query_prefix=str(value.get("query_prefix", "")),
            passage_prefix=str(value.get("passage_prefix", "")),
            score_function=str(value.get("score_function", "dot")),
            normalize_embeddings=bool(value.get("normalize_embeddings", False)),
            include_title=bool(value.get("include_title", True)),
            max_query_length=int(value.get("max_query_length", 32)),
            max_passage_length=int(
                value.get("max_passage_length", value.get("max_sequence_length", 512))
            ),
        )


def load_model_registry(path: str | Path) -> dict[str, ModelSpec]:
    raw = load_yaml(path)
    models = raw.get("models", raw)
    if not isinstance(models, dict):
        raise ValueError("The model registry must contain a 'models' mapping")
    return {name: ModelSpec.from_mapping(name, value) for name, value in models.items()}


class SentenceTransformerEncoder:
    """Thin role-aware wrapper around SentenceTransformer."""

    def __init__(self, spec: ModelSpec, *, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install model dependencies with: pip install -e '.[models]'"
            ) from exc
        self.spec = spec
        self.query_encoder = SentenceTransformer(
            spec.query_model or spec.document_model, device=device
        )
        if spec.query_model and spec.query_model != spec.document_model:
            self.document_encoder = SentenceTransformer(spec.document_model, device=device)
        else:
            self.document_encoder = self.query_encoder

    def encode_queries(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        self.query_encoder.max_seq_length = self.spec.max_query_length
        values = [self.spec.query_prefix + text for text in texts]
        return np.asarray(
            self.query_encoder.encode(
                values,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.spec.normalize_embeddings,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

    def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        self.document_encoder.max_seq_length = self.spec.max_passage_length
        values = [self.spec.passage_prefix + text for text in texts]
        return np.asarray(
            self.document_encoder.encode(
                values,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.spec.normalize_embeddings,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )


def encode_dataset(
    *,
    spec: ModelSpec,
    query_path: str | Path,
    corpus_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    batch_size: int = 128,
    device: str | None = None,
) -> tuple[EmbeddingTable, list[EmbeddingTable]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformerEncoder(spec, device=device)
    queries = load_queries(query_path)
    query_ids = tuple(queries)
    query_table = EmbeddingTable(
        query_ids,
        encoder.encode_queries([queries[item] for item in query_ids], batch_size=batch_size),
    )
    query_table.save(output_dir / "queries.npz")
    corpus_tables: list[EmbeddingTable] = []
    for source, path in corpus_paths.items():
        corpus = load_corpus(path)
        base_ids = tuple(corpus)
        vectors = encoder.encode_documents(
            [format_document(corpus[item], spec.include_title) for item in base_ids],
            batch_size=batch_size,
        )
        table = EmbeddingTable(tuple(add_source_suffix(item, source) for item in base_ids), vectors)
        table.save(output_dir / f"{source}.npz")
        corpus_tables.append(table)
    return query_table, corpus_tables


def retrieve_dataset(
    *,
    spec: ModelSpec,
    query_path: str | Path,
    corpus_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    batch_size: int = 128,
    top_k: int = 200,
    device: str | None = None,
    corpus_batch_size: int = 50_000,
    remove_self_matches: bool = True,
) -> dict[str, dict[str, float]]:
    query_table, corpus_tables = encode_dataset(
        spec=spec,
        query_path=query_path,
        corpus_paths=corpus_paths,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
    )
    run = exact_search(
        query_table,
        corpus_tables,
        top_k=top_k + len(corpus_paths) if remove_self_matches else top_k,
        score_function=spec.score_function,
        corpus_batch_size=corpus_batch_size,
    )
    if remove_self_matches:
        run = remove_query_self_matches(run, sources=tuple(corpus_paths), top_k=top_k)
    write_json(run, Path(output_dir) / "run.json")
    return run
