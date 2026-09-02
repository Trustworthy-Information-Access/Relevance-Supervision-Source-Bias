"""Positive-pool negative construction for the appendix training control."""

from __future__ import annotations

import random
from collections.abc import Mapping


def positive_pool_document_ids(qrels: Mapping[str, Mapping[str, int]]) -> set[str]:
    return {
        document_id
        for documents in qrels.values()
        for document_id, relevance in documents.items()
        if int(relevance) > 0
    }


def select_positive_pool_negatives(
    run: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Select the highest-ranked other-query positive as each query's negative."""
    pool = positive_pool_document_ids(qrels)
    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []
    for query_id in sorted(set(run) & set(qrels)):
        current = {
            document_id for document_id, relevance in qrels[query_id].items() if int(relevance) > 0
        }
        if not current:
            continue
        ranked = sorted(run[query_id].items(), key=lambda item: (-float(item[1]), item[0]))
        negative = next(
            (
                document_id
                for document_id, _ in ranked
                if document_id in pool and document_id not in current
            ),
            None,
        )
        if negative is None:
            continue
        pairs.append(
            {
                "query_id": query_id,
                "positive_id": rng.choice(sorted(current)),
                "hard_negative_id": negative,
            }
        )
    if not pairs:
        raise ValueError("No other-query positive-pool negatives could be selected")
    return pairs
