import json

from relevance_source_bias.core.data import write_jsonl
from relevance_source_bias.retrieval.lexical import (
    prepare_pyserini_corpus,
    prepare_pyserini_topics,
    pyserini_commands,
)


def test_prepare_pyserini_inputs(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    docs = tmp_path / "input" / "docs.jsonl"
    topics = tmp_path / "topics.tsv"
    write_jsonl([{"_id": "d1", "title": "Title", "text": "Body"}], corpus)
    write_jsonl([{"_id": "q1", "text": "multi\nline query"}], queries)

    assert prepare_pyserini_corpus(corpus, docs, include_title=True) == 1
    assert json.loads(docs.read_text()) == {"id": "d1", "contents": "Title Body"}
    assert prepare_pyserini_topics(queries, topics) == 1
    assert topics.read_text() == "q1\tmulti line query\n"


def test_pyserini_commands_use_trec_output_and_paper_bm25_parameters(tmp_path):
    index, search = pyserini_commands(
        input_dir=tmp_path / "input",
        index_dir=tmp_path / "index",
        topics_path=tmp_path / "topics.tsv",
        output_path=tmp_path / "run.txt",
    )
    assert "pyserini.index.lucene" in index
    assert "pyserini.search.lucene" in search
    assert "--output-format" not in search
    assert search[search.index("--k1") + 1] == "0.82"
    assert search[search.index("--b") + 1] == "0.68"
