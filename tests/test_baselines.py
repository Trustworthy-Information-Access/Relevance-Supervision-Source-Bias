import numpy as np
import pytest

from relevance_source_bias.analysis.significance import random_cosine_reference
from relevance_source_bias.interventions.cdc import (
    apply_cdc_correction,
    estimate_cdc_coefficient,
)
from relevance_source_bias.interventions.positive_pool import select_positive_pool_negatives


def test_cdc_estimation_and_correction_remove_paired_ppl_effect():
    run = {
        "q1": {"d1-human": 1.0, "d1-llm": 2.0},
        "q2": {"d2-human": 3.0, "d2-llm": 4.0},
    }
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    perplexity = {
        "d1-human": 20.0,
        "d1-llm": 10.0,
        "d2-human": 20.0,
        "d2-llm": 10.0,
    }
    estimate = estimate_cdc_coefficient(
        run,
        qrels,
        perplexity,
        human_source="human",
        llm_source="llm",
        budget=2,
        seed=42,
    )
    assert estimate["coefficient"] == pytest.approx(-0.1)
    corrected = apply_cdc_correction(run, perplexity, estimate["coefficient"])
    assert corrected["q1"]["d1-human"] == pytest.approx(3.0)
    assert corrected["q1"]["d1-llm"] == pytest.approx(3.0)


def test_positive_pool_negative_is_another_queries_positive():
    qrels = {"q1": {"p1": 1}, "q2": {"p2": 1}}
    run = {"q1": {"other": 3.0, "p2": 2.0}, "q2": {"p1": 4.0}}
    assert select_positive_pool_negatives(run, qrels) == [
        {"query_id": "q1", "positive_id": "p1", "hard_negative_id": "p2"},
        {"query_id": "q2", "positive_id": "p2", "hard_negative_id": "p1"},
    ]


def test_random_cosine_reference_matches_768_dimensional_threshold():
    result = random_cosine_reference(768, sigma_multiplier=3)
    assert result["threshold"] == pytest.approx(3 / np.sqrt(768))
    assert result["exact_two_sided_tail_probability"] == pytest.approx(0.00266, abs=0.0001)
