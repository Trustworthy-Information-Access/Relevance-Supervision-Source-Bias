import numpy as np
import pytest

from relevance_source_bias.analysis.geometry import (
    average_pairwise_cosine,
    estimate_direction,
    mean_alignment_to_mean,
    paired_vectors_from_records,
    sample_positive_negative_pairs,
)
from relevance_source_bias.analysis.significance import paired_sign_flip_validation
from relevance_source_bias.core.data import EmbeddingTable
from relevance_source_bias.interventions.projection import (
    CalibrationGroup,
    calibrate_leave_one_out,
    project_out,
)


def test_direction_and_projection():
    human = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, -1.0]])
    llm = human + np.asarray([0.0, 2.0])
    result = estimate_direction(human, llm, permutations=100, seed=7)
    assert result["unit_direction"] == pytest.approx([0.0, 1.0])
    assert result["average_pairwise_cosine"] == pytest.approx(1.0)
    projected = project_out(llm, np.asarray(result["unit_direction"]))
    np.testing.assert_allclose(projected[:, 1], 0.0)
    np.testing.assert_allclose(projected[:, 0], llm[:, 0])


def test_average_pairwise_cosine_handles_opposites():
    vectors = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    assert average_pairwise_cosine(vectors) == pytest.approx(-1.0)


def test_appendix_within_statistic_uses_recomputed_mean_direction():
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert average_pairwise_cosine(vectors) == pytest.approx(0.0)
    assert mean_alignment_to_mean(vectors) == pytest.approx(2**-0.5)


def test_three_sign_flip_statistics_are_deterministic():
    first = np.tile(np.asarray([1.0, 0.0]), (12, 1))
    second = np.tile(np.asarray([1.0, 1.0]), (12, 1))
    arguments = {
        "displacements": {"a": first, "b": second},
        "reference_direction": np.asarray([1.0, 0.0]),
        "permutations": 40,
        "seed": 7,
        "batch_size": 8,
    }
    result = paired_sign_flip_validation(**arguments)
    repeated = paired_sign_flip_validation(**arguments)
    assert result == repeated
    assert result["within_dataset_lh"]["observed"] == pytest.approx(1.0)
    assert result["cross_dataset_lh"]["observed"] == pytest.approx(2**-0.5)
    expected_pn = (1.0 + 2**-0.5) / 2
    assert result["pn_lh_alignment"]["observed"] == pytest.approx(expected_pn)
    assert result["num_pairs"] == {"a": 12, "b": 12}


def test_calibration_excludes_target_collection_group():
    def group(datasets, displacement):
        human = EmbeddingTable(("a-human", "b-human"), np.zeros((2, 2)))
        llm = EmbeddingTable(("a-llm", "b-llm"), np.tile(np.asarray(displacement), (2, 1)))
        return CalibrationGroup(
            tuple(datasets), human, llm, human_suffix="-human", llm_suffix="-llm"
        )

    groups = {
        "shared": group(["msmarco", "dl19", "dl20"], [0.0, 10.0]),
        "a": group(["a"], [2.0, 0.0]),
        "b": group(["b"], [4.0, 0.0]),
    }
    direction, metadata = calibrate_leave_one_out(
        groups, target_dataset="dl19", samples_per_group=1, seed=42
    )
    np.testing.assert_allclose(direction, [1.0, 0.0])
    assert metadata["excluded_group"] == "shared"
    assert metadata["samples_used"] == 2


def test_explicit_positive_negative_pairs():
    corpus = EmbeddingTable(
        ("p1", "p2", "n1", "n2"),
        np.asarray([[1, 1], [2, 1], [1, 0], [2, 0]], dtype=np.float32),
    )
    records = [
        {"positive_id": "p1", "hard_negative_id": "n1"},
        {"positive_id": "p2", "hard_negative_id": "n2"},
    ]
    negative, positive, labels = paired_vectors_from_records(
        corpus,
        corpus,
        records,
        left_id_field="hard_negative_id",
        right_id_field="positive_id",
    )
    np.testing.assert_allclose(positive - negative, [[0, 1], [0, 1]])
    assert labels == ("n1->p1", "n2->p2")


def test_bm25_positive_negative_sampling_is_deterministic():
    run = {
        "q1": {"p1": 4.0, "n1": 3.0, "n2": 2.0},
        "q2": {"n3": 2.0, "p2": 1.0},
    }
    qrels = {"q1": {"p1": 1}, "q2": {"p2": 1}}
    first = sample_positive_negative_pairs(run, qrels, top_k=2, seed=42)
    second = sample_positive_negative_pairs(run, qrels, top_k=2, seed=42)
    assert first == second
    assert first == [
        {"query_id": "q1", "positive_id": "p1", "hard_negative_id": "n1"},
        {"query_id": "q2", "positive_id": "p2", "hard_negative_id": "n3"},
    ]
