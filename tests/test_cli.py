import json

import numpy as np
import yaml

from relevance_source_bias.cli import main
from relevance_source_bias.core.data import EmbeddingTable


def test_direction_and_comparison_commands(tmp_path):
    table = EmbeddingTable(
        ("p1", "p2", "n1", "n2"),
        np.asarray([[1, 1], [2, 1], [1, 0], [2, 0]], dtype=np.float32),
    )
    embeddings = tmp_path / "corpus.npz"
    table.save(embeddings)
    run = tmp_path / "bm25.txt"
    run.write_text(
        "q1 Q0 p1 1 4.0 bm25\nq1 Q0 n1 2 3.0 bm25\nq2 Q0 n2 1 4.0 bm25\nq2 Q0 p2 2 3.0 bm25\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.txt"
    qrels.write_text("q1 Q0 p1 1\nq2 Q0 p2 1\n", encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    main(
        [
            "prepare-pn-pairs",
            "--run",
            str(run),
            "--qrels",
            str(qrels),
            "--output",
            str(pairs),
        ],
    )
    direction = tmp_path / "pn.json"
    main(
        [
            "direction",
            "--left",
            str(embeddings),
            "--right",
            str(embeddings),
            "--pairs",
            str(pairs),
            "--output",
            str(direction),
        ]
    )
    result = json.loads(direction.read_text(encoding="utf-8"))
    assert result["unit_direction"] == [0.0, 1.0]

    comparison = tmp_path / "comparison.json"
    main(
        [
            "compare-directions",
            "--direction",
            f"first={direction}",
            "--direction",
            f"second={direction}",
            "--reference",
            str(direction),
            "--output",
            str(comparison),
        ]
    )
    compared = json.loads(comparison.read_text(encoding="utf-8"))
    assert compared["cosine_matrix"]["first"]["second"] == 1.0
    assert compared["reference_alignment"]["first"] == 1.0


def test_sign_flip_command(tmp_path):
    datasets = {}
    for name, displacement in (("a", [1.0, 0.0]), ("b", [1.0, 1.0])):
        ids = tuple(f"{index}" for index in range(12))
        human = EmbeddingTable(
            tuple(f"{item}-human" for item in ids), np.zeros((12, 2), dtype=np.float32)
        )
        llm = EmbeddingTable(
            tuple(f"{item}-llm" for item in ids),
            np.tile(np.asarray(displacement, dtype=np.float32), (12, 1)),
        )
        human_path, llm_path = tmp_path / f"{name}-human.npz", tmp_path / f"{name}-llm.npz"
        human.save(human_path)
        llm.save(llm_path)
        datasets[name] = {"left": str(human_path), "right": str(llm_path)}

    reference = tmp_path / "pn.json"
    reference.write_text(json.dumps({"mean_direction": [1.0, 0.0]}), encoding="utf-8")
    config = tmp_path / "sign-flip.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "suffixes": {"left": "-human", "right": "-llm"},
                "reference": str(reference),
                "datasets": datasets,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sign-flip.json"
    main(
        [
            "sign-flip",
            "--config",
            str(config),
            "--permutations",
            "20",
            "--batch-size",
            "5",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["datasets"] == ["a", "b"]
    assert set(result) >= {
        "within_dataset_lh",
        "cross_dataset_lh",
        "pn_lh_alignment",
    }
