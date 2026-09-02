"""Aggregate matrix results into machine-readable and paper-ready tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import load_yaml
from ..core.data import write_json
from .matrix import matrix_tasks


def _metric_records(
    tasks: Iterable[Mapping[str, Any]], *, output_root: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        metrics_path = output_root / str(task["dataset"]) / str(task["model"]) / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        preference = metrics["source_preference"]
        retrieval = metrics.get("retrieval", {}).get("per_k", {})
        for k, values in preference["per_k"].items():
            records.append(
                {
                    "dataset": str(task["dataset"]),
                    "model": str(task["model"]),
                    "model_label": str(task["label"]),
                    "seed": task.get("seed"),
                    "k": int(k),
                    "delta_ndsr": float(values["delta_ndsr"]),
                    "p_value": float(values["p_value"]),
                    "human_ndsr": float(values["human_ndsr"]),
                    "llm_ndsr": float(values["llm_ndsr"]),
                    "ndcg": (
                        float(retrieval[k]["ndcg"])
                        if k in retrieval and retrieval[k].get("ndcg") is not None
                        else None
                    ),
                    "metrics_path": str(metrics_path),
                }
            )
    return records


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _summaries(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_model: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[tuple[str, int | None, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[(record["model"], record["k"])].append(record)
        by_seed[(record["model_label"], record["seed"], record["k"])].append(record)

    model_summary = []
    for (model, k), rows in sorted(by_model.items()):
        model_summary.append(
            {
                "model": model,
                "model_label": rows[0]["model_label"],
                "seed": rows[0]["seed"],
                "k": k,
                "num_datasets": len(rows),
                "mean_delta_ndsr": _mean(row["delta_ndsr"] for row in rows),
                "mean_absolute_delta_ndsr": _mean(abs(row["delta_ndsr"]) for row in rows),
                "mean_ndcg": _mean(row["ndcg"] for row in rows),
            }
        )

    seed_summary = []
    for (label, seed, k), rows in sorted(
        by_seed.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])
    ):
        seed_summary.append(
            {
                "model_label": label,
                "seed": seed,
                "k": k,
                "num_datasets": len(rows),
                "mean_absolute_delta_ndsr": _mean(abs(row["delta_ndsr"]) for row in rows),
                "mean_ndcg": _mean(row["ndcg"] for row in rows),
            }
        )
    return {"by_model": model_summary, "by_seed": seed_summary}


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%")):
        text = text.replace(old, new)
    return text


def _render_metric_table(
    records: list[dict[str, Any]], metric: str, k: int, datasets: list[str]
) -> str:
    selected = [row for row in records if row["k"] == k]
    models = sorted({row["model"] for row in selected})
    values = {(row["dataset"], row["model"]): row for row in selected}
    lines = [
        r"\begin{tabular}{l" + "r" * len(models) + "}",
        r"\toprule",
        "Dataset & " + " & ".join(_latex_escape(model) for model in models) + r" \\",
        r"\midrule",
    ]
    for dataset in datasets:
        cells = []
        for model in models:
            row = values.get((dataset, model))
            value = None if row is None else row.get(metric)
            if value is None:
                cells.append("--")
            else:
                marker = "*" if metric == "delta_ndsr" and row["p_value"] < 0.05 else ""
                cells.append(f"{float(value):.3f}{marker}")
        lines.append(_latex_escape(dataset) + " & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _render_seed_table(seed_summary: list[dict[str, Any]], k: int) -> str:
    rows = [row for row in seed_summary if row["k"] == k and row["seed"] is not None]
    labels = sorted({row["model_label"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    values = {(row["model_label"], int(row["seed"])): row for row in rows}
    lines = [
        r"\begin{tabular}{l" + "r" * len(seeds) + "}",
        r"\toprule",
        "Model & " + " & ".join(str(seed) for seed in seeds) + r" \\",
        r"\midrule",
    ]
    for label in labels:
        cells = []
        for seed in seeds:
            row = values.get((label, seed))
            value = None if row is None else row["mean_absolute_delta_ndsr"]
            cells.append("--" if value is None else f"{float(value):.3f}")
        lines.append(_latex_escape(label) + " & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def aggregate_matrix(config: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate every completed task declared by a matrix configuration."""
    evaluation = load_yaml(config["evaluation_config"])
    output_root = Path(evaluation.get("output_root", "outputs"))
    tasks = matrix_tasks(config)
    records = _metric_records(tasks, output_root=output_root)
    summaries = _summaries(records)
    result = {
        "expected_tasks": len(tasks),
        "completed_tasks": len({(row["dataset"], row["model"]) for row in records}),
        "records": records,
        "summaries": summaries,
    }
    aggregate_dir = Path(config.get("aggregate_dir", output_root / "aggregate"))
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_json(result, aggregate_dir / "results.json")
    fields = [
        "dataset",
        "model",
        "model_label",
        "seed",
        "k",
        "delta_ndsr",
        "p_value",
        "human_ndsr",
        "llm_ndsr",
        "ndcg",
        "metrics_path",
    ]
    with (aggregate_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    datasets = [str(item) for item in config["datasets"]]
    for k in sorted({row["k"] for row in records}):
        (aggregate_dir / f"delta_ndsr_at_{k}.tex").write_text(
            _render_metric_table(records, "delta_ndsr", k, datasets), encoding="utf-8"
        )
        (aggregate_dir / f"ndcg_at_{k}.tex").write_text(
            _render_metric_table(records, "ndcg", k, datasets), encoding="utf-8"
        )
        (aggregate_dir / f"seed_summary_at_{k}.tex").write_text(
            _render_seed_table(summaries["by_seed"], k), encoding="utf-8"
        )
    return result
