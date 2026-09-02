import json

import pytest

from relevance_source_bias.analysis.artifacts import (
    build_idf,
    compare_feature_groups,
    correlate_scores_with_feature,
    score_idf_rows,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_idf_build_and_passage_median(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"_id": "a", "text": "rare common common"},
            {"_id": "b", "text": "common other"},
        ],
    )
    idf = build_idf(corpus, tokenizer="regex", include_title=False)
    assert idf["num_documents"] == 2
    assert idf["idf"]["rare"] > idf["idf"]["common"]
    rows = list(score_idf_rows(corpus, idf, tokenizer="regex", include_title=False))
    assert rows[0]["num_tokens"] == 3
    assert rows[0]["value"] == pytest.approx(idf["idf"]["common"])


def test_score_feature_correlation_normalizes_within_query():
    run = {
        "q1": {"a-human": 1.0, "b-human": 2.0},
        "q2": {"a-human": 10.0, "b-human": 20.0},
    }
    features = {"a": 1.0, "b": 2.0}
    result = correlate_scores_with_feature(run, features, sources=["human"])
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["num_pairs"] == 4


def test_feature_comparison_reports_hedges_g():
    result = compare_feature_groups([2.0, 3.0, 4.0], [0.0, 1.0, 2.0])
    assert result["cohens_d"] > 0
    assert 0 < result["hedges_g"] < result["cohens_d"]
