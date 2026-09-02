"""Source-preference and retrieval-effectiveness metrics used in the paper."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import ttest_1samp

Run = Mapping[str, Mapping[str, float]]
Qrels = Mapping[str, Mapping[str, int]]


def source_of(document_id: str, sources: Sequence[str]) -> str | None:
    """Return the longest matching `-<source>` suffix."""
    for source in sorted(sources, key=len, reverse=True):
        if document_id.endswith(f"-{source}"):
            return source
    return None


def _ranked_items(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    # The document ID is a deterministic tie-breaker.
    return sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))


def per_query_ndsr(
    scores: Mapping[str, float],
    *,
    sources: Sequence[str],
    k: int,
) -> dict[str, float]:
    """Compute normalized discounted source ratios for one ranked list."""
    if k <= 0:
        raise ValueError("k must be positive")
    totals = {source: 0.0 for source in sources}
    denominator = 0.0
    recognized_rank = 0
    for document_id, _ in _ranked_items(scores):
        source = source_of(document_id, sources)
        if source is None:
            continue
        recognized_rank += 1
        if recognized_rank > k:
            break
        weight = 1.0 / math.log2(1.0 + recognized_rank)
        totals[source] += weight
        denominator += weight
    if denominator == 0.0:
        return totals
    return {source: value / denominator for source, value in totals.items()}


def evaluate_source_preference(
    run: Run,
    *,
    human_source: str,
    llm_source: str,
    k_values: Iterable[int] = (5,),
) -> dict[str, object]:
    """Evaluate Delta-NDSR and its per-query two-sided t-test.

    Positive deltas indicate human preference; negative deltas indicate LLM preference.
    """
    sources = (human_source, llm_source)
    query_ids = sorted(run)
    output: dict[str, object] = {"num_queries": len(query_ids), "per_k": {}}
    for k in k_values:
        per_query: dict[str, float] = {}
        human_values: list[float] = []
        llm_values: list[float] = []
        for query_id in query_ids:
            ratios = per_query_ndsr(run[query_id], sources=sources, k=int(k))
            human = ratios[human_source]
            llm = ratios[llm_source]
            human_values.append(human)
            llm_values.append(llm)
            per_query[query_id] = human - llm
        human_mean = float(np.mean(human_values)) if human_values else 0.0
        llm_mean = float(np.mean(llm_values)) if llm_values else 0.0
        delta = human_mean - llm_mean
        values = np.asarray(list(per_query.values()), dtype=np.float64)
        if len(values) > 1 and not np.allclose(values, values[0]):
            p_value = float(ttest_1samp(values, popmean=0.0).pvalue)
        elif len(values) > 1 and values[0] != 0.0:
            p_value = 0.0
        else:
            p_value = 1.0
        output["per_k"][str(k)] = {
            "human_ndsr": human_mean,
            "llm_ndsr": llm_mean,
            "delta_ndsr": delta,
            "p_value": p_value,
            "per_query_delta": per_query,
        }
    return output


def _base_document_id(document_id: str, sources: Sequence[str]) -> str:
    source = source_of(document_id, sources)
    return document_id[: -(len(source) + 1)] if source is not None else document_id


def _dcg(relevances: Sequence[int], k: int) -> float:
    return sum((2.0**rel - 1.0) / math.log2(rank + 2.0) for rank, rel in enumerate(relevances[:k]))


def evaluate_ndcg(
    run: Run,
    qrels: Qrels,
    *,
    sources: Sequence[str],
    k_values: Iterable[int] = (5,),
) -> dict[str, object]:
    """Evaluate NDCG after sharing each base qrel across the requested sources."""
    output: dict[str, object] = {"num_queries": len(qrels), "per_k": {}}
    for k in k_values:
        per_query: dict[str, float] = {}
        for query_id, relevant in qrels.items():
            ranked = _ranked_items(run.get(query_id, {}))
            retrieved_rels = [
                int(relevant.get(_base_document_id(document_id, sources), 0))
                for document_id, _ in ranked[: int(k)]
            ]
            ideal_labels = sorted(
                [int(label) for label in relevant.values() for _ in sources], reverse=True
            )
            ideal = _dcg(ideal_labels, int(k))
            per_query[query_id] = _dcg(retrieved_rels, int(k)) / ideal if ideal > 0 else 0.0
        output["per_k"][str(k)] = {
            "ndcg": float(np.mean(list(per_query.values()))) if per_query else 0.0,
            "per_query": per_query,
        }
    return output


def evaluate_run(
    run: Run,
    *,
    human_source: str,
    llm_source: str,
    k_values: Iterable[int] = (5,),
    qrels: Qrels | None = None,
) -> dict[str, object]:
    k_values = tuple(int(k) for k in k_values)
    output = {
        "source_preference": evaluate_source_preference(
            run,
            human_source=human_source,
            llm_source=llm_source,
            k_values=k_values,
        )
    }
    if qrels is not None:
        output["retrieval"] = evaluate_ndcg(
            run, qrels, sources=(human_source, llm_source), k_values=k_values
        )
    return output
