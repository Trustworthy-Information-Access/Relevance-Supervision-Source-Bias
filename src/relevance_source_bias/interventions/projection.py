"""Leave-one-group-out calibration and embedding projection mitigation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..analysis.geometry import unit_vector
from ..core.data import EmbeddingTable, pair_embedding_tables


def project_out(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component of every row along *direction* without renormalizing."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2-D array")
    normal = unit_vector(direction).astype(vectors.dtype, copy=False)
    if vectors.shape[1] != len(normal):
        raise ValueError(f"Dimension mismatch: vectors={vectors.shape[1]}, direction={len(normal)}")
    return vectors - (vectors @ normal)[:, None] * normal[None, :]


@dataclass(frozen=True)
class CalibrationGroup:
    """A unique corpus group used for leave-one-group-out calibration."""

    datasets: tuple[str, ...]
    human: EmbeddingTable
    llm: EmbeddingTable
    human_suffix: str = "-human"
    llm_suffix: str = "-llama-2-7b-chat-tmp0.2"


def calibrate_leave_one_out(
    groups: Mapping[str, CalibrationGroup],
    *,
    target_dataset: str,
    samples_per_group: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate an LH direction while excluding the target's shared collection group."""
    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive")
    excluded = [name for name, group in groups.items() if target_dataset in group.datasets]
    if len(excluded) != 1:
        raise ValueError(
            f"Target {target_dataset!r} must occur in exactly one calibration group; "
            f"found {excluded}"
        )
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    group_counts: dict[str, int] = {}
    for name, group in groups.items():
        if name == excluded[0]:
            continue
        human, llm, _ = pair_embedding_tables(
            group.human,
            group.llm,
            left_suffix=group.human_suffix,
            right_suffix=group.llm_suffix,
        )
        if len(human) < samples_per_group:
            raise ValueError(
                f"Calibration group {name!r} has {len(human)} pairs; "
                f"{samples_per_group} are required"
            )
        indices = rng.choice(len(human), size=samples_per_group, replace=False)
        sampled.append(llm[indices] - human[indices])
        group_counts[name] = int(samples_per_group)
    if not sampled:
        raise ValueError("No calibration pairs remain after excluding the target group")
    all_directions = np.vstack(sampled)
    direction = unit_vector(all_directions.mean(axis=0))
    metadata: dict[str, object] = {
        "target_dataset": target_dataset,
        "excluded_group": excluded[0],
        "samples_per_group": int(samples_per_group),
        "samples_used": int(len(all_directions)),
        "groups_used": group_counts,
        "seed": int(seed),
        "unit_direction": direction.tolist(),
    }
    return direction, metadata
