import pytest

from relevance_source_bias.core.data import write_jsonl
from relevance_source_bias.interventions.length_control import (
    LENGTH_CONTROL_USER_PROMPT,
    clean_rewrite_text,
    corpus_length_statistics,
)


def test_length_control_prompt_and_cleaning_match_expected_format():
    assert "10% longer" in LENGTH_CONTROL_USER_PROMPT
    assert clean_rewrite_text(" Rewritten Text:  A longer passage. ") == "A longer passage."
    assert clean_rewrite_text("plain output") == "plain output"


def test_corpus_length_statistics_aligns_document_ids(tmp_path):
    original = tmp_path / "original.jsonl"
    rewritten = tmp_path / "rewritten.jsonl"
    write_jsonl(
        [
            {"_id": "d1", "text": "one two"},
            {"_id": "d2", "text": "three four five"},
        ],
        original,
    )
    write_jsonl(
        [
            {"_id": "d1", "text": "one two extra"},
            {"_id": "d2", "text": "three four five extra"},
        ],
        rewritten,
    )
    result = corpus_length_statistics(
        {"original": original, "rewritten": rewritten}, tokenizer="regex"
    )
    assert result["summaries"]["original"]["mean_tokens"] == pytest.approx(2.5)
    comparison = result["comparisons"]["rewritten"]
    assert comparison["paired_documents"] == 2
    assert comparison["mean_token_change"] == pytest.approx(1.0)
    assert comparison["fraction_longer"] == pytest.approx(1.0)
