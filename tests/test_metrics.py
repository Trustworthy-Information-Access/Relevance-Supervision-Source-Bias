import math

import pytest

from relevance_source_bias.retrieval.metrics import evaluate_ndcg, evaluate_source_preference

HUMAN = "human"
LLM = "llama-2-7b-chat-tmp0.2"


def test_delta_ndsr_is_signed_human_minus_llm():
    run = {
        "q1": {f"d1-{HUMAN}": 2.0, f"d1-{LLM}": 1.0},
        "q2": {f"d2-{LLM}": 2.0, f"d2-{HUMAN}": 1.0},
    }
    result = evaluate_source_preference(run, human_source=HUMAN, llm_source=LLM, k_values=[1, 2])
    assert result["per_k"]["1"]["per_query_delta"] == {"q1": 1.0, "q2": -1.0}
    assert result["per_k"]["1"]["delta_ndsr"] == pytest.approx(0.0)
    expected_q1 = (1.0 - 1 / math.log2(3)) / (1.0 + 1 / math.log2(3))
    assert result["per_k"]["2"]["per_query_delta"]["q1"] == pytest.approx(expected_q1)


def test_mixed_corpus_ndcg_expands_base_qrels_to_both_sources():
    run = {"q1": {f"d1-{HUMAN}": 2.0, f"d1-{LLM}": 1.0}}
    qrels = {"q1": {"d1": 1}}
    result = evaluate_ndcg(run, qrels, sources=(HUMAN, LLM), k_values=[2])
    assert result["per_k"]["2"]["ndcg"] == pytest.approx(1.0)
