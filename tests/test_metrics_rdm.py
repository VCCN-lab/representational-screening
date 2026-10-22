"""Tests for repscreen.metrics.rdm."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from repscreen.metrics.rdm import (
    column_correlations,
    compare_rdms,
    compare_rdms_global,
    compute_rdm,
)


def test_compute_rdm_l2squared_matches_manual_distances():
    vectors = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    result = compute_rdm(vectors, metric="L2squared")
    assert result[0, 1] == pytest.approx(25.0)
    assert result[0, 2] == pytest.approx(1.0)
    assert np.allclose(np.diag(result), 0.0)


def test_compute_rdm_l2_is_not_squared():
    vectors = np.array([[0.0, 0.0], [3.0, 4.0]])
    result = compute_rdm(vectors, metric="L2")
    assert result[0, 1] == pytest.approx(5.0)


def test_compute_rdm_pearson():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(5, 7))
    result = compute_rdm(vectors, metric="pearson")
    assert np.allclose(result, 1.0 - np.corrcoef(vectors))


def test_compute_rdm_dotproduct():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = compute_rdm(vectors, metric="dotproduct")
    assert np.allclose(result, 1.0 - vectors @ vectors.T)


def test_compute_rdm_normalize_suffix_scales_rows():
    vectors = np.array([[3.0, 4.0], [30.0, 40.0], [0.0, 1.0]])
    result = compute_rdm(vectors, metric="L2squared_normalize")
    assert result[0, 1] == pytest.approx(0.0)


def test_compute_rdm_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unknown dissimilarity metric"):
        compute_rdm(np.zeros((3, 2)), metric="manhattan")


def test_compute_rdm_rejects_non_2d_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        compute_rdm(np.zeros((2, 3, 4)))


def test_compare_rdms_cosine_of_identical_matrices(rdm_pair):
    first, _ = rdm_pair
    assert compare_rdms(first, first, metric="cosine") == pytest.approx(
        1.0
    )


def test_compare_rdms_pearson_matches_upper_triangle(rdm_pair):
    first, second = rdm_pair
    upper = np.triu_indices(len(first), k=1)
    expected = np.corrcoef(first[upper], second[upper])[0, 1]
    result = compare_rdms(first, second, metric="pearson")
    assert result == pytest.approx(expected)


def test_compare_rdms_spearman_matches_scipy(rdm_pair):
    first, second = rdm_pair
    upper = np.triu_indices(len(first), k=1)
    expected = spearmanr(first[upper], second[upper]).statistic
    result = compare_rdms(first, second, metric="spearman")
    assert result == pytest.approx(expected)


def test_compare_rdms_centering_changes_result(rdm_pair):
    first, second = rdm_pair
    plain = compare_rdms(first, second, metric="cosine")
    centred = compare_rdms(
        first, second, metric="cosine", center=True
    )
    assert plain != pytest.approx(centred)


def test_compare_rdms_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        compare_rdms(np.zeros((3, 3)), np.zeros((4, 4)))


def test_compare_rdms_rejects_unknown_metric(rdm_pair):
    first, second = rdm_pair
    with pytest.raises(ValueError, match="Unknown similarity metric"):
        compare_rdms(first, second, metric="kendall")


def test_compare_rdms_global_uses_supplied_statistics(rdm_pair):
    first, second = rdm_pair
    upper = np.triu_indices(len(first), k=1)
    mean_first = float(np.mean(first[upper]))
    mean_second = float(np.mean(second[upper]))
    norm = float(
        np.std(first[upper] - mean_first)
        * np.std(second[upper] - mean_second)
    )
    expected = float(
        np.mean(
            (first[upper] - mean_first) * (second[upper] - mean_second)
        )
        / norm
    )
    result = compare_rdms_global(
        first, second, mean_first, mean_second, norm
    )
    assert result == pytest.approx(expected)


def test_column_correlations_shape(rdm_pair):
    first, second = rdm_pair
    result = column_correlations(first, second)
    assert result.shape == (len(first),)


def test_column_correlations_of_identical_rdms_are_one(rdm_pair):
    first, _ = rdm_pair
    result = column_correlations(first, first)
    assert np.allclose(result, 1.0)


def test_column_correlations_match_explicit_computation(rdm_pair):
    first, second = rdm_pair
    size = len(first)
    expected = np.zeros(size)
    off_first = np.array(
        [np.delete(first[i], i) for i in range(size)]
    ).T
    off_second = np.array(
        [np.delete(second[i], i) for i in range(size)]
    ).T
    centred_first = off_first - np.mean(off_first)
    centred_second = off_second - np.mean(off_second)
    for column in range(size):
        left = centred_first[:, column]
        right = centred_second[:, column]
        expected[column] = np.sum(
            left
            / np.sqrt(np.sum(left**2))
            * right
            / np.sqrt(np.sum(right**2))
        )
    assert np.allclose(column_correlations(first, second), expected)


def test_column_correlations_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        column_correlations(np.zeros((3, 3)), np.zeros((4, 4)))
