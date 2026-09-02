import numpy as np
import pytest

from relevance_source_bias.core.data import (
    EmbeddingTable,
    load_qrels,
    load_run,
    pair_embedding_tables,
)
from relevance_source_bias.retrieval.dense import exact_search, remove_query_self_matches


def test_embedding_round_trip_npz_and_hdf5(tmp_path):
    table = EmbeddingTable(("a", "b"), np.asarray([[1, 2], [3, 4]], dtype=np.float32))
    for filename in ("table.npz", "table.h5"):
        path = tmp_path / filename
        table.save(path)
        loaded = EmbeddingTable.load(path)
        assert loaded.ids == table.ids
        np.testing.assert_array_equal(loaded.vectors, table.vectors)


def test_pairing_strips_source_suffixes():
    human = EmbeddingTable(("b-human", "a-human"), np.asarray([[2, 0], [1, 0]]))
    llm = EmbeddingTable(("a-llm", "b-llm"), np.asarray([[1, 1], [2, 1]]))
    left, right, ids = pair_embedding_tables(human, llm, left_suffix="-human", right_suffix="-llm")
    assert ids == ("a", "b")
    np.testing.assert_array_equal(right - left, np.asarray([[0, 1], [0, 1]]))


def test_exact_search_across_multiple_corpora():
    queries = EmbeddingTable(("q",), np.asarray([[1.0, 0.0]]))
    human = EmbeddingTable(("h-human",), np.asarray([[0.5, 0.0]]))
    llm = EmbeddingTable(("x-llm", "y-llm"), np.asarray([[0.8, 0.0], [-1.0, 0.0]]))
    run = exact_search(queries, [human, llm], top_k=2, score_function="dot")
    assert list(run["q"]) == ["x-llm", "h-human"]


def test_exact_search_rejects_empty_corpus():
    queries = EmbeddingTable(("q",), np.asarray([[1.0, 0.0]]))
    empty = EmbeddingTable((), np.empty((0, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        exact_search(queries, [empty])


def test_query_self_matches_are_removed_and_refilled():
    run = {
        "q": {
            "q-human": 3.0,
            "q-llm": 2.0,
            "a-human": 1.0,
            "b-llm": 0.5,
        }
    }
    cleaned = remove_query_self_matches(run, sources=("human", "llm"), top_k=2)
    assert list(cleaned["q"]) == ["a-human", "b-llm"]


def test_trec_run_and_qrels_loaders(tmp_path):
    run_path = tmp_path / "bm25.txt"
    run_path.write_text("q1 Q0 d1 1 12.5 bm25\nq1 Q0 d2 2 10.0 bm25\n", encoding="utf-8")
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("q1 0 d1 1\n", encoding="utf-8")
    assert load_run(run_path) == {"q1": {"d1": 12.5, "d2": 10.0}}
    assert load_qrels(qrels_path) == {"q1": {"d1": 1}}
