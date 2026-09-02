"""Perplexity-based causal document correction (CDC)."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from ..core.data import add_source_suffix


def estimate_cdc_coefficient(
    run: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    perplexity: Mapping[str, float],
    *,
    human_source: str,
    llm_source: str,
    budget: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate CDC's PPL effect from paired relevant human/LLM documents.

    Following the released CDC implementation, the first stage replaces each source's
    document PPL with its group mean. The second stage regresses the paired retriever
    scores on those predicted PPL values. With a balanced paired design, the slope is
    the corresponding Wald/2SLS estimate.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    candidates: list[tuple[str, str, float, float, float, float]] = []
    for query_id in sorted(set(run) & set(qrels)):
        for document_id, relevance in sorted(qrels[query_id].items()):
            if int(relevance) <= 0:
                continue
            human_id = add_source_suffix(document_id, human_source)
            llm_id = add_source_suffix(document_id, llm_source)
            if (
                human_id in run[query_id]
                and llm_id in run[query_id]
                and human_id in perplexity
                and llm_id in perplexity
            ):
                candidates.append(
                    (
                        query_id,
                        document_id,
                        float(run[query_id][human_id]),
                        float(run[query_id][llm_id]),
                        float(perplexity[human_id]),
                        float(perplexity[llm_id]),
                    )
                )
    if len(candidates) < budget:
        raise ValueError(
            f"CDC requires {budget} paired relevant samples, "
            f"but only {len(candidates)} are available"
        )
    rng = random.Random(seed)
    selected = rng.sample(candidates, budget)
    human_ppl_mean = float(np.mean([item[4] for item in selected]))
    llm_ppl_mean = float(np.mean([item[5] for item in selected]))
    if math.isclose(human_ppl_mean, llm_ppl_mean):
        raise ValueError(
            "CDC cannot estimate an effect when source-level mean PPL values are equal"
        )
    predicted_ppl = np.asarray(
        [value for _ in selected for value in (human_ppl_mean, llm_ppl_mean)],
        dtype=np.float64,
    )
    relevance_scores = np.asarray(
        [value for item in selected for value in (item[2], item[3])], dtype=np.float64
    )
    regression = stats.linregress(predicted_ppl, relevance_scores)
    return {
        "coefficient": float(regression.slope),
        "intercept": float(regression.intercept),
        "p_value": float(regression.pvalue),
        "standard_error": float(regression.stderr),
        "human_ppl_mean": human_ppl_mean,
        "llm_ppl_mean": llm_ppl_mean,
        "human_score_mean": float(np.mean([item[2] for item in selected])),
        "llm_score_mean": float(np.mean([item[3] for item in selected])),
        "budget": int(budget),
        "available_pairs": int(len(candidates)),
        "seed": int(seed),
        "sampled_pairs": [{"query_id": item[0], "document_id": item[1]} for item in selected],
        "method": "paired_source_instrument_2sls",
    }


def apply_cdc_correction(
    run: Mapping[str, Mapping[str, float]],
    perplexity: Mapping[str, float],
    coefficient: float,
) -> dict[str, dict[str, float]]:
    """Apply CDC score correction: corrected = original - beta * PPL."""
    corrected: dict[str, dict[str, float]] = {}
    missing: set[str] = set()
    for query_id, document_scores in run.items():
        corrected[query_id] = {}
        for document_id, score in document_scores.items():
            if document_id not in perplexity:
                missing.add(document_id)
                continue
            corrected[query_id][document_id] = float(score) - coefficient * float(
                perplexity[document_id]
            )
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise KeyError(f"PPL features are missing for {len(missing)} run documents: {examples}")
    return corrected
