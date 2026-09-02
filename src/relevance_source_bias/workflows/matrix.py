"""Resumable dataset-by-retriever experiment matrices."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core.config import load_yaml
from ..core.data import load_qrels, load_run, write_json
from ..retrieval.dense import load_model_registry, retrieve_dataset
from ..retrieval.metrics import evaluate_run


def _expanded_retrievers(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("retrievers")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Matrix config must contain a non-empty retrievers list")
    expanded: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, str):
            expanded.append({"model": entry, "label": entry, "seed": None})
            continue
        if not isinstance(entry, Mapping) or "template" not in entry:
            raise ValueError("Retriever entries must be names or mappings with a template")
        template = str(entry["template"])
        label = str(entry.get("label", template))
        seeds = entry.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Templated retriever {template!r} requires a non-empty seeds list")
        for seed in seeds:
            numeric_seed = int(seed)
            expanded.append(
                {
                    "model": template.format(seed=numeric_seed),
                    "label": label,
                    "seed": numeric_seed,
                }
            )
    models = [item["model"] for item in expanded]
    if len(models) != len(set(models)):
        raise ValueError("Expanded retriever names must be unique")
    return expanded


def _qrels_path(matrix: Mapping[str, Any], evaluation: Mapping[str, Any], dataset: str) -> Path:
    qrels = matrix.get("qrels", {"default": "qrels/test.tsv"})
    if isinstance(qrels, str):
        relative = qrels
    elif isinstance(qrels, Mapping):
        relative = qrels.get(dataset, qrels.get("default", "qrels/test.tsv"))
    else:
        raise ValueError("qrels must be a path or a dataset-to-path mapping")
    path = Path(str(relative).format(dataset=dataset))
    if path.is_absolute():
        return path
    return Path(evaluation["data_root"]) / dataset / path


def matrix_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the configured dataset and retriever Cartesian product."""
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Matrix config must contain a non-empty datasets list")
    retrievers = _expanded_retrievers(config)
    return [
        {"dataset": str(dataset), **retriever} for retriever in retrievers for dataset in datasets
    ]


def run_matrix(
    config: Mapping[str, Any],
    *,
    force: bool = False,
    evaluation_only: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run or resume every task and aggregate completed results."""
    evaluation = load_yaml(config["evaluation_config"])
    registry = load_model_registry(config["models"])
    output_root = Path(evaluation.get("output_root", "outputs"))
    data_root = Path(evaluation["data_root"])
    sources = [str(item) for item in evaluation.get("sources", ["human", "llm"])]
    if len(sources) != 2:
        raise ValueError("Evaluation config must contain exactly two sources")
    tasks = matrix_tasks(config)
    manifest_path = Path(config.get("manifest", output_root / "matrix-manifest.json"))
    statuses: list[dict[str, Any]] = []
    for task in tasks:
        dataset, model = str(task["dataset"]), str(task["model"])
        output_dir = output_root / dataset / model
        run_path, metrics_path = output_dir / "run.json", output_dir / "metrics.json"
        status = {**task, "run": str(run_path), "metrics": str(metrics_path)}
        try:
            if metrics_path.exists() and not force:
                status["status"] = "skipped_complete"
            else:
                if not run_path.exists() or force:
                    if evaluation_only:
                        raise FileNotFoundError(f"Missing existing run: {run_path}")
                    if model not in registry:
                        raise KeyError(f"Retriever {model!r} is absent from the model registry")
                    query_path = (
                        data_root / dataset / str(evaluation.get("queries", "queries.jsonl"))
                    )
                    corpus_pattern = str(evaluation.get("corpus_pattern", "corpus/{source}.jsonl"))
                    corpus_paths = {
                        source: data_root / dataset / corpus_pattern.format(source=source)
                        for source in sources
                    }
                    retrieve_dataset(
                        spec=registry[model],
                        query_path=query_path,
                        corpus_paths=corpus_paths,
                        output_dir=output_dir,
                        batch_size=int(evaluation.get("batch_size", 128)),
                        top_k=int(evaluation.get("top_k", 200)),
                        device=evaluation.get("device"),
                        corpus_batch_size=int(evaluation.get("corpus_batch_size", 50_000)),
                        remove_self_matches=bool(evaluation.get("remove_self_matches", True)),
                    )
                qrels_path = _qrels_path(config, evaluation, dataset)
                metrics = evaluate_run(
                    load_run(run_path),
                    human_source=sources[0],
                    llm_source=sources[1],
                    k_values=[int(item) for item in config.get("k", [5])],
                    qrels=load_qrels(qrels_path),
                )
                metrics["provenance"] = {
                    "dataset": dataset,
                    "model": model,
                    "model_label": task["label"],
                    "seed": task.get("seed"),
                    "run": str(run_path),
                    "qrels": str(qrels_path),
                }
                write_json(metrics, metrics_path)
                status["status"] = "completed"
        except Exception as exc:
            status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            statuses.append(status)
            write_json({"tasks": statuses}, manifest_path)
            if fail_fast:
                raise
            continue
        statuses.append(status)
        write_json({"tasks": statuses}, manifest_path)

    from .aggregation import aggregate_matrix

    aggregate = aggregate_matrix(config)
    result = {"tasks": statuses, "aggregate": aggregate}
    write_json(result, manifest_path)
    return result
