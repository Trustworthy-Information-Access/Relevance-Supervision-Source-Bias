from pathlib import Path

from relevance_source_bias.retrieval.dense import load_model_registry


def test_paper_model_registry():
    root = Path(__file__).resolve().parents[1]
    models = load_model_registry(root / "configs" / "models.yaml")
    assert len(models) == 14
    assert models["dragon"].query_model != models["dragon"].document_model
    assert models["simcse"].score_function == "cosine"
    assert models["e5"].query_prefix == "query: "
    assert models["e5"].passage_prefix == "passage: "
    assert models["dpr-nq"].query_model != models["dpr-nq"].document_model
