"""Embedding-direction analyses and inference-time projection mitigation."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence

import numpy as np

from ..core.data import EmbeddingTable, pair_embedding_tables, remove_suffix

EPSILON = 1e-12


def summarize_pair_ids(ids: Sequence[str], examples: int = 10) -> dict[str, object]:
    """Return compact provenance without serializing millions of pair IDs."""
    digest = hashlib.sha256()
    for pair_id in ids:
        digest.update(str(pair_id).encode("utf-8"))
        digest.update(b"\n")
    return {
        "paired_id_examples": [str(item) for item in ids[:examples]],
        "paired_ids_sha256": digest.hexdigest(),
    }


def sample_positive_negative_pairs(
    run: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    top_k: int = 10,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Sample one judged positive and one BM25 top-k negative for each query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []
    for query_id in sorted(set(run) & set(qrels)):
        positives = sorted(
            document_id for document_id, relevance in qrels[query_id].items() if int(relevance) > 0
        )
        if not positives:
            continue
        ranked = sorted(run[query_id].items(), key=lambda item: (-item[1], item[0]))
        candidates = [
            document_id
            for document_id, _ in ranked[:top_k]
            if int(qrels[query_id].get(document_id, 0)) <= 0
        ]
        if not candidates:
            continue
        pairs.append(
            {
                "query_id": query_id,
                "positive_id": rng.choice(positives),
                "hard_negative_id": rng.choice(candidates),
            }
        )
    if not pairs:
        raise ValueError("No positive/BM25-negative pairs could be sampled")
    return pairs


def unit_vector(vector: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < epsilon:
        raise ValueError("Cannot normalize a near-zero direction")
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit_vector(left), unit_vector(right)))


def displacement_vectors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise `right - left` displacement vectors."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            f"Paired matrices must have the same 2-D shape: {left.shape}, {right.shape}"
        )
    return right - left


def average_pairwise_cosine(vectors: np.ndarray, epsilon: float = EPSILON) -> float:
    """Compute the average cosine over all distinct pairs in O(n*d) time."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    normalized = vectors[norms > epsilon] / norms[norms > epsilon, None]
    count = len(normalized)
    if count < 2:
        return 0.0
    # ||sum u_i||^2 = n + 2 * sum_{i<j} <u_i, u_j>.
    summed = normalized.sum(axis=0)
    return float((np.dot(summed, summed) - count) / (count * (count - 1)))


def mean_alignment_to_mean(vectors: np.ndarray, epsilon: float = EPSILON) -> float:
    """Average cosine between pair displacements and their dataset-level mean.

    This is the within-dataset statistic used by the paired sign-flip validation in
    the paper appendix. It is intentionally reported alongside, rather than in place
    of, the average pairwise cosine used in the main embedding analysis.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("vectors must be a non-empty 2-D array")
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > epsilon
    if not np.any(valid):
        raise ValueError("No non-zero displacement vectors are available")
    mean = vectors.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm <= epsilon:
        return 0.0
    normalized = vectors[valid] / norms[valid, None]
    return float(np.mean(normalized @ (mean / mean_norm)))


def _within_sign_flip_null(
    vectors: np.ndarray,
    *,
    permutations: int,
    seed: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Null distribution for mean pair-to-recomputed-mean alignment."""
    if permutations <= 0:
        return np.empty(0, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > EPSILON
    vectors = vectors[valid]
    normalized = vectors / norms[valid, None]
    count = len(vectors)
    if not count:
        return np.zeros(permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty(permutations, dtype=np.float64)
    for start in range(0, permutations, batch_size):
        end = min(start + batch_size, permutations)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(end - start, count))
        signed_raw_sums = signs @ vectors
        signed_unit_sums = signs @ normalized
        mean_norms = np.linalg.norm(signed_raw_sums, axis=1)
        mean_directions = np.divide(
            signed_raw_sums,
            mean_norms[:, None],
            out=np.zeros_like(signed_raw_sums),
            where=mean_norms[:, None] > EPSILON,
        )
        output[start:end] = np.einsum("ij,ij->i", signed_unit_sums, mean_directions) / count
    return output


def estimate_direction(
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int = 0,
    seed: int = 42,
) -> dict[str, object]:
    """Estimate the mean `right - left` direction and consistency statistics."""
    vectors = displacement_vectors(left, right)
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    normalized = unit_vector(mean)
    within = average_pairwise_cosine(vectors)
    mean_alignment = mean_alignment_to_mean(vectors)
    null = _within_sign_flip_null(vectors, permutations=permutations, seed=seed)
    p_value = (
        float((1 + np.count_nonzero(null >= mean_alignment)) / (len(null) + 1))
        if len(null)
        else None
    )
    return {
        "num_pairs": int(len(vectors)),
        "dimension": int(vectors.shape[1]),
        "mean_direction": mean.tolist(),
        "unit_direction": normalized.tolist(),
        "mean_direction_norm": norm,
        "average_pairwise_cosine": within,
        "mean_alignment_to_dataset_mean": mean_alignment,
        "sign_flip_statistic": "mean_alignment_to_recomputed_dataset_mean",
        "sign_flip_permutations": int(permutations),
        "sign_flip_p_value": p_value,
        "sign_flip_null_mean": float(null.mean()) if len(null) else None,
        "sign_flip_null_std": float(null.std()) if len(null) else None,
        "seed": int(seed),
    }


def direction_cosine_matrix(directions: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    names = sorted(directions)
    return {
        left: {right: cosine_similarity(directions[left], directions[right]) for right in names}
        for left in names
    }


def paired_direction_from_tables(
    left: EmbeddingTable,
    right: EmbeddingTable,
    *,
    left_suffix: str = "",
    right_suffix: str = "",
    permutations: int = 0,
    seed: int = 42,
) -> dict[str, object]:
    left_vectors, right_vectors, ids = pair_embedding_tables(
        left, right, left_suffix=left_suffix, right_suffix=right_suffix
    )
    result = estimate_direction(left_vectors, right_vectors, permutations=permutations, seed=seed)
    result.update(summarize_pair_ids(ids))
    return result


def paired_vectors_from_records(
    left: EmbeddingTable,
    right: EmbeddingTable,
    records: Sequence[Mapping[str, object]],
    *,
    left_id_field: str,
    right_id_field: str,
    left_suffix: str = "",
    right_suffix: str = "",
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Align embeddings using explicit pair records (for PN rather than source pairs)."""

    def index(table: EmbeddingTable, suffix: str) -> dict[str, int]:
        output: dict[str, int] = {}
        for row, full_id in enumerate(table.ids):
            base_id = remove_suffix(full_id, suffix)
            if base_id is not None:
                if base_id in output:
                    raise ValueError(f"Duplicate base ID {base_id!r} after stripping {suffix!r}")
                output[base_id] = row
        return output

    left_index = index(left, left_suffix)
    right_index = index(right, right_suffix)
    left_rows: list[int] = []
    right_rows: list[int] = []
    labels: list[str] = []
    for number, record in enumerate(records):
        if left_id_field not in record or right_id_field not in record:
            raise ValueError(f"Pair record {number} lacks {left_id_field!r} or {right_id_field!r}")
        left_id, right_id = str(record[left_id_field]), str(record[right_id_field])
        if left_id not in left_index or right_id not in right_index:
            raise KeyError(f"Pair {left_id!r} -> {right_id!r} is missing from embedding tables")
        left_rows.append(left_index[left_id])
        right_rows.append(right_index[right_id])
        labels.append(f"{left_id}->{right_id}")
    if not left_rows:
        raise ValueError("The pair file contains no usable records")
    return (
        left.vectors[np.asarray(left_rows)],
        right.vectors[np.asarray(right_rows)],
        tuple(labels),
    )
