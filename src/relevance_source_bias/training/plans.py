"""Controlled MS MARCO fine-tuning used by the RQ1 and RQ3 experiments."""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..core.data import (
    format_document,
    iter_jsonl,
    load_queries,
    write_json,
    write_jsonl,
)


def load_msmarco_hard_negatives(
    path: str | Path,
    *,
    ce_score_margin: float = 3.0,
    negatives_per_system: int = 5,
) -> dict[str, dict[str, list[str]]]:
    """Apply the SentenceTransformers MS MARCO hard-negative filtering protocol."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    queries: dict[str, dict[str, list[str]]] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                positives = [str(item["pid"]) for item in row["pos"]]
                threshold = min(float(item["ce-score"]) for item in row["pos"]) - ce_score_margin
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid hard-negative row at {path}:{line_number}") from exc
            negatives: list[str] = []
            seen: set[str] = set()
            for system_items in row.get("neg", {}).values():
                added = 0
                for item in system_items:
                    if float(item["ce-score"]) > threshold:
                        continue
                    document_id = str(item["pid"])
                    if document_id not in seen:
                        seen.add(document_id)
                        negatives.append(document_id)
                        added += 1
                    if added >= negatives_per_system:
                        break
            if positives and negatives:
                queries[str(row["qid"])] = {"positives": positives, "hard_negatives": negatives}
    return queries


def prepare_training_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create one deterministic plan shared by all negative-sampling conditions."""
    data = config["data"]
    output_path = Path(config["output"])
    seed = int(config.get("seed", 42))
    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch_size", 75))
    if data.get("pairs"):
        return prepare_training_plan_from_pairs(
            data["pairs"],
            output_path,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
        )
    hard_negatives = load_msmarco_hard_negatives(
        data["hard_negatives"],
        ce_score_margin=float(config.get("ce_score_margin", 3.0)),
        negatives_per_system=int(config.get("negatives_per_system", 5)),
    )
    available_queries = load_queries(data["queries"])
    # Preserve the hard-negative file order, matching the BEIR training pipeline.
    query_ids = [query_id for query_id in hard_negatives if query_id in available_queries]
    if not query_ids:
        raise ValueError("No hard-negative query IDs occur in the query file")
    rng = random.Random(seed)
    state: dict[str, dict[str, Any]] = {}
    for query_id in query_ids:
        positives = list(hard_negatives[query_id]["positives"])
        negatives = list(hard_negatives[query_id]["hard_negatives"])
        rng.shuffle(negatives)
        state[query_id] = {"positives": positives, "negatives": negatives, "p": 0, "n": 0}

    total_samples = 0
    total_batches = 0

    def rows() -> Iterable[dict[str, Any]]:
        nonlocal total_samples, total_batches
        for epoch in range(epochs):
            epoch_ids = list(query_ids)
            rng.shuffle(epoch_ids)
            for batch_index, start in enumerate(range(0, len(epoch_ids), batch_size)):
                samples: list[dict[str, str]] = []
                for query_id in epoch_ids[start : start + batch_size]:
                    item = state[query_id]
                    positive = item["positives"][item["p"] % len(item["positives"])]
                    negative = item["negatives"][item["n"] % len(item["negatives"])]
                    item["p"] += 1
                    item["n"] += 1
                    samples.append(
                        {
                            "query_id": query_id,
                            "positive_id": positive,
                            "hard_negative_id": negative,
                        }
                    )
                total_samples += len(samples)
                total_batches += 1
                yield {"epoch": epoch, "batch_idx": batch_index, "samples": samples}

    write_jsonl(rows(), output_path)
    metadata = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_queries": len(query_ids),
        "total_samples": total_samples,
        "total_batches": total_batches,
        "ce_score_margin": float(config.get("ce_score_margin", 3.0)),
        "negatives_per_system": int(config.get("negatives_per_system", 5)),
        "plan": str(output_path),
    }
    write_json(metadata, output_path.with_suffix(output_path.suffix + ".metadata.json"))
    return metadata


def prepare_training_plan_from_pairs(
    pairs_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 75,
) -> dict[str, Any]:
    """Batch a fixed query/positive/negative pair file into a deterministic plan."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    required = ("query_id", "positive_id", "hard_negative_id")
    samples: list[dict[str, str]] = []
    for line_number, row in enumerate(iter_jsonl(pairs_path), 1):
        if not all(field in row for field in required):
            raise ValueError(f"Missing pair fields at {pairs_path}:{line_number}")
        samples.append({field: str(row[field]) for field in required})
    if not samples:
        raise ValueError(f"Pair file {pairs_path} contains no samples")
    rng = random.Random(seed)
    total_batches = 0

    def rows() -> Iterable[dict[str, Any]]:
        nonlocal total_batches
        for epoch in range(epochs):
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            for batch_index, start in enumerate(range(0, len(indices), batch_size)):
                total_batches += 1
                yield {
                    "epoch": epoch,
                    "batch_idx": batch_index,
                    "samples": [samples[index] for index in indices[start : start + batch_size]],
                }

    output_path = Path(output_path)
    write_jsonl(rows(), output_path)
    metadata = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_queries": len({sample["query_id"] for sample in samples}),
        "samples_per_epoch": len(samples),
        "total_samples": epochs * len(samples),
        "total_batches": total_batches,
        "pair_source": str(pairs_path),
        "plan": str(output_path),
    }
    write_json(metadata, output_path.with_suffix(output_path.suffix + ".metadata.json"))
    return metadata


class ControlledPlanDataset:
    """Map-style dataset retaining exact batch boundaries from a plan JSONL."""

    def __init__(
        self,
        plan_path: str | Path,
        *,
        queries: Mapping[str, str],
        corpus: Mapping[str, Mapping[str, Any]],
        query_prefix: str = "",
        passage_prefix: str = "",
        include_title: bool = False,
    ) -> None:
        try:
            from sentence_transformers import InputExample
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install model dependencies with: pip install -e '.[models]'"
            ) from exc
        self.input_example = InputExample
        self.queries = queries
        self.corpus = corpus
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.include_title = include_title
        self.samples: list[dict[str, str]] = []
        self.batches: list[list[int]] = []
        with Path(plan_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                indices: list[int] = []
                for sample in row.get("samples", []):
                    required = {"query_id", "positive_id", "hard_negative_id"}
                    if not required.issubset(sample):
                        raise ValueError(f"Missing plan fields at {plan_path}:{line_number}")
                    indices.append(len(self.samples))
                    self.samples.append({key: str(sample[key]) for key in required})
                if indices:
                    self.batches.append(indices)
        if not self.samples:
            raise ValueError(f"Training plan {plan_path} contains no samples")
        self._validate()

    def _validate(self) -> None:
        for sample in self.samples:
            query_id = sample["query_id"]
            positive_id = sample["positive_id"]
            negative_id = sample["hard_negative_id"]
            if query_id not in self.queries:
                raise KeyError(f"Query {query_id!r} from the plan is missing")
            if positive_id not in self.corpus:
                raise KeyError(f"Positive {positive_id!r} from the plan is missing")
            if negative_id not in self.corpus:
                raise KeyError(f"Hard negative {negative_id!r} from the plan is missing")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        query = self.query_prefix + self.queries[sample["query_id"]]
        positive = self.passage_prefix + format_document(
            self.corpus[sample["positive_id"]], self.include_title
        )
        negative = self.passage_prefix + format_document(
            self.corpus[sample["hard_negative_id"]], self.include_title
        )
        return self.input_example(texts=[query, positive, negative])
