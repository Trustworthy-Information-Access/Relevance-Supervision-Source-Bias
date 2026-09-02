import gzip
import json

import numpy as np

from relevance_source_bias.core.data import iter_jsonl, write_jsonl
from relevance_source_bias.training import (
    aligned_positive_negative_scores,
    prepare_training_plan,
    prepare_training_plan_from_pairs,
)


def test_hard_negative_only_selects_only_aligned_pairs():
    positive = np.asarray([[10.0, 99.0], [98.0, 20.0]])
    negative = np.asarray([[-1.0, -99.0], [-98.0, -2.0]])
    positive_scores, negative_scores = aligned_positive_negative_scores(positive, negative)
    np.testing.assert_allclose(positive_scores, [10.0, 20.0])
    np.testing.assert_allclose(negative_scores, [-1.0, -2.0])


def test_training_plan_is_deterministic(tmp_path):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            [
                json.dumps({"_id": "q1", "text": "one"}),
                json.dumps({"_id": "q2", "text": "two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    negatives = tmp_path / "hard.jsonl.gz"
    rows = [
        {
            "qid": query_id,
            "pos": [{"pid": f"p{index}", "ce-score": 10.0}],
            "neg": {"bm25": [{"pid": f"n{index}", "ce-score": 0.0}]},
        }
        for index, query_id in enumerate(("q1", "q2"), 1)
    ]
    with gzip.open(negatives, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    outputs = []
    for name in ("first.jsonl", "second.jsonl"):
        output = tmp_path / name
        metadata = prepare_training_plan(
            {
                "data": {"queries": str(queries), "hard_negatives": str(negatives)},
                "output": str(output),
                "seed": 42,
                "epochs": 2,
                "batch_size": 1,
            }
        )
        assert metadata["total_samples"] == 4
        outputs.append(output.read_text(encoding="utf-8"))
    assert outputs[0] == outputs[1]


def test_prepare_plan_from_fixed_positive_pool_pairs(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    write_jsonl(
        [
            {"query_id": "q1", "positive_id": "p1", "hard_negative_id": "n1"},
            {"query_id": "q2", "positive_id": "p2", "hard_negative_id": "n2"},
        ],
        pairs,
    )
    plan = tmp_path / "plan.jsonl"
    metadata = prepare_training_plan_from_pairs(pairs, plan, seed=42, epochs=2, batch_size=1)
    rows = list(iter_jsonl(plan))
    assert metadata["total_samples"] == 4
    assert metadata["total_batches"] == 4
    assert {row["epoch"] for row in rows} == {0, 1}
