"""Tests for repscreen.metrics.distances."""

import numpy as np
import pytest

from repscreen.metrics.distances import (
    cosine_similarity,
    dotproduct_dissimilarity,
    l2_dissimilarity,
)


def test_cosine_similarity_identical_vectors():
    vector = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_opposite_vectors():
    vector = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(vector, -vector) == pytest.approx(-1.0)


def test_cosine_similarity_orthogonal_vectors():
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])
    assert cosine_similarity(first, second) == pytest.approx(0.0)


def test_cosine_similarity_matches_manual_formula():
    rng = np.random.default_rng(0)
    first = rng.normal(size=10)
    second = rng.normal(size=10)
    expected = np.dot(first, second) / (
        np.linalg.norm(first) * np.linalg.norm(second)
    )
    assert cosine_similarity(first, second) == pytest.approx(expected)


def test_cosine_similarity_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        cosine_similarity(np.zeros(3), np.zeros(4))


def test_cosine_similarity_warns_on_zero_norm():
    with pytest.warns(UserWarning, match="zero norm"):
        result = cosine_similarity(np.zeros(3), np.ones(3))
    assert result == 0.0


def test_dotproduct_dissimilarity_shape_and_diagonal():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(5, 4))
    result = dotproduct_dissimilarity(vectors, normalize=True)
    assert result.shape == (5, 5)
    assert np.allclose(np.diag(result), 0.0)


def test_dotproduct_dissimilarity_without_normalisation():
    vectors = np.array([[1.0, 0.0], [0.0, 2.0]])
    expected = 1.0 - vectors @ vectors.T
    assert np.allclose(dotproduct_dissimilarity(vectors), expected)


def test_l2_dissimilarity_squared_flag():
    vectors = np.array([[0.0, 0.0], [3.0, 4.0]])
    plain = l2_dissimilarity(vectors)
    squared = l2_dissimilarity(vectors, squared=True)
    assert plain[0, 1] == pytest.approx(5.0)
    assert squared[0, 1] == pytest.approx(25.0)


def test_l2_dissimilarity_normalises_rows():
    vectors = np.array([[3.0, 4.0], [30.0, 40.0]])
    result = l2_dissimilarity(vectors, normalize=True)
    assert result[0, 1] == pytest.approx(0.0)


def test_l2_dissimilarity_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(6, 3))
    result = l2_dissimilarity(vectors)
    assert np.allclose(result, result.T)
    assert np.allclose(np.diag(result), 0.0)
