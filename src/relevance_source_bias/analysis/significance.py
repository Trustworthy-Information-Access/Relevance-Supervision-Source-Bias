"""Empirical and analytic significance references for embedding directions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import special, stats

from .geometry import (
    EPSILON,
    average_pairwise_cosine,
    cosine_similarity,
    mean_alignment_to_mean,
    unit_vector,
)


def _null_statistics(observed: float, null: np.ndarray) -> dict[str, float]:
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
    }


def paired_sign_flip_validation(
    displacements: Mapping[str, np.ndarray],
    *,
    reference_direction: np.ndarray,
    permutations: int = 1000,
    seed: int = 42,
    batch_size: int = 16,
) -> dict[str, object]:
    """Run the three paired sign-flip validations reported in the appendix."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(displacements) < 2:
        raise ValueError("At least two datasets are required for cross-dataset validation")

    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    dimension: int | None = None
    for name in sorted(displacements):
        vectors = np.asarray(displacements[name], dtype=np.float64)
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError(f"Dataset {name!r} must contain a non-empty 2-D array")
        if dimension is None:
            dimension = int(vectors.shape[1])
        elif vectors.shape[1] != dimension:
            raise ValueError("All displacement arrays must have the same dimension")
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > EPSILON
        vectors = vectors[valid]
        if not len(vectors):
            raise ValueError(f"Dataset {name!r} has no non-zero displacements")
        prepared[name] = (vectors, vectors / norms[valid, None])

    reference = unit_vector(reference_direction).astype(np.float64)
    if len(reference) != dimension:
        raise ValueError("The PN reference and LH displacements have different dimensions")

    names = sorted(prepared)
    observed_means = {name: prepared[name][0].mean(axis=0) for name in names}
    observed_within = {name: mean_alignment_to_mean(prepared[name][0]) for name in names}
    observed_units = np.vstack([unit_vector(observed_means[name]) for name in names])
    observed_cross = average_pairwise_cosine(observed_units)
    observed_pn = {name: cosine_similarity(reference, observed_means[name]) for name in names}
    observed_within_macro = float(np.mean(list(observed_within.values())))
    observed_pn_macro = float(np.mean(list(observed_pn.values())))

    within_null = np.empty(permutations, dtype=np.float64)
    cross_null = np.empty(permutations, dtype=np.float64)
    pn_null = np.empty(permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, permutations, batch_size):
        end = min(start + batch_size, permutations)
        current_batch = end - start
        batch_within: list[np.ndarray] = []
        batch_directions: list[np.ndarray] = []
        for name in names:
            vectors, normalized = prepared[name]
            count = len(vectors)
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=(current_batch, count))
            signed_raw_sums = signs @ vectors
            signed_unit_sums = signs @ normalized
            mean_norms = np.linalg.norm(signed_raw_sums, axis=1)
            directions = np.divide(
                signed_raw_sums,
                mean_norms[:, None],
                out=np.zeros_like(signed_raw_sums),
                where=mean_norms[:, None] > EPSILON,
            )
            batch_directions.append(directions)
            batch_within.append(np.einsum("ij,ij->i", signed_unit_sums, directions) / count)

        direction_stack = np.stack(batch_directions, axis=1)
        within_null[start:end] = np.mean(np.stack(batch_within, axis=1), axis=1)
        summed_directions = direction_stack.sum(axis=1)
        squared_sum_norms = np.einsum("ij,ij->i", summed_directions, summed_directions)
        individual_squared_norms = np.einsum("ijk,ijk->i", direction_stack, direction_stack)
        dataset_count = len(names)
        cross_null[start:end] = (squared_sum_norms - individual_squared_norms) / (
            dataset_count * (dataset_count - 1)
        )
        pn_null[start:end] = np.mean(direction_stack @ reference, axis=1)

    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "dimension": int(dimension),
        "datasets": names,
        "num_pairs": {name: int(len(prepared[name][0])) for name in names},
        "within_dataset_lh": {
            **_null_statistics(observed_within_macro, within_null),
            "per_dataset": observed_within,
            "aggregation": "macro_average_within_each_permutation",
        },
        "cross_dataset_lh": {
            **_null_statistics(observed_cross, cross_null),
            "aggregation": "average_pairwise_cosine_of_dataset_mean_directions",
        },
        "pn_lh_alignment": {
            **_null_statistics(observed_pn_macro, pn_null),
            "per_dataset": observed_pn,
            "aggregation": "macro_average_within_each_permutation",
        },
    }


def random_cosine_reference(
    dimension: int = 768,
    *,
    sigma_multiplier: float = 3.0,
    samples: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Return the exact and Gaussian random-vector cosine reference."""
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if sigma_multiplier <= 0 or samples < 0:
        raise ValueError("sigma_multiplier must be positive and samples non-negative")
    standard_deviation = 1.0 / math.sqrt(dimension)
    threshold = sigma_multiplier * standard_deviation
    if threshold >= 1.0:
        exact_two_sided_tail = 0.0
    else:
        exact_two_sided_tail = float(
            special.betainc((dimension - 1.0) / 2.0, 0.5, 1.0 - threshold**2)
        )
    result: dict[str, Any] = {
        "dimension": int(dimension),
        "mean": 0.0,
        "standard_deviation": standard_deviation,
        "sigma_multiplier": float(sigma_multiplier),
        "threshold": threshold,
        "exact_two_sided_tail_probability": exact_two_sided_tail,
        "gaussian_two_sided_tail_probability": float(2.0 * stats.norm.sf(sigma_multiplier)),
    }
    if samples:
        rng = np.random.default_rng(seed)
        left = rng.normal(size=(samples, dimension))
        right = rng.normal(size=(samples, dimension))
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        cosines = np.einsum("ij,ij->i", left, right)
        result["simulation"] = {
            "samples": int(samples),
            "seed": int(seed),
            "mean": float(cosines.mean()),
            "standard_deviation": float(cosines.std()),
            "two_sided_tail_probability": float(np.mean(np.abs(cosines) > threshold)),
        }
    return result
