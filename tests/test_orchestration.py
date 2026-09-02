import json

import yaml

from relevance_source_bias.workflows.matrix import matrix_tasks, run_matrix


def test_matrix_expands_seeded_retrievers():
    tasks = matrix_tasks(
        {
            "datasets": ["a", "b"],
            "retrievers": [
                "ance",
                {
                    "template": "contriever-ft-seed{seed}",
                    "label": "contriever-ft",
                    "seeds": [42, 100],
                },
            ],
        }
    )
    assert len(tasks) == 6
    assert tasks[-1] == {
        "dataset": "b",
        "model": "contriever-ft-seed100",
        "label": "contriever-ft",
        "seed": 100,
    }


def test_evaluation_only_matrix_resumes_and_writes_aggregates(tmp_path):
    data_root, output_root = tmp_path / "data", tmp_path / "outputs"
    qrels = data_root / "toy" / "qrels" / "test.tsv"
    qrels.parent.mkdir(parents=True)
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    run_path = output_root / "toy" / "dummy" / "run.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(json.dumps({"q1": {"d1-human": 2.0, "d1-llm": 1.0}}), encoding="utf-8")
    evaluation = tmp_path / "evaluation.yaml"
    evaluation.write_text(
        yaml.safe_dump(
            {
                "data_root": str(data_root),
                "output_root": str(output_root),
                "sources": ["human", "llm"],
            }
        ),
        encoding="utf-8",
    )
    models = tmp_path / "models.yaml"
    models.write_text("models:\n  dummy:\n    model: unused\n", encoding="utf-8")
    aggregate_dir = output_root / "aggregate"
    config = {
        "evaluation_config": str(evaluation),
        "models": str(models),
        "datasets": ["toy"],
        "retrievers": ["dummy"],
        "k": [1],
        "aggregate_dir": str(aggregate_dir),
        "manifest": str(output_root / "manifest.json"),
    }
    first = run_matrix(config, evaluation_only=True, fail_fast=True)
    assert first["tasks"][0]["status"] == "completed"
    assert first["aggregate"]["records"][0]["delta_ndsr"] == 1.0
    assert (aggregate_dir / "results.csv").exists()
    assert (aggregate_dir / "delta_ndsr_at_1.tex").exists()

    second = run_matrix(config, evaluation_only=True, fail_fast=True)
    assert second["tasks"][0]["status"] == "skipped_complete"
