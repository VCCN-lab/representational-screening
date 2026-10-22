"""Tests for repscreen.metrics.cka."""

import numpy as np
import pytest

from repscreen.metrics.cka import (
    center_gram,
    center_matrix,
    kernel_cka,
    kernel_hsic,
    linear_cka,
    linear_hsic,
    rbf_kernel,
)


def test_center_matrix_removes_row_and_column_means():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(6, 6))
    result = center_matrix(matrix)
    assert np.allclose(result.mean(axis=1), 0.0)


def test_center_matrix_preserves_shape():
    matrix = np.arange(16.0).reshape(4, 4)
    assert center_matrix(matrix).shape == (4, 4)


def test_center_gram_matches_double_centering():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(5, 3))
    gram = features @ features.T
    size = len(gram)
    projector = np.eye(size) - np.ones((size, size)) / size
    expected = projector @ gram @ projector
    assert np.allclose(center_gram(gram), expected)


def test_rbf_kernel_diagonal_is_one():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(5, 4))
    kernel = rbf_kernel(features)
    assert np.allclose(np.diag(kernel), 1.0)


def test_rbf_kernel_is_symmetric():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(5, 4))
    kernel = rbf_kernel(features, sigma=2.0)
    assert np.allclose(kernel, kernel.T)


def test_rbf_kernel_decreases_with_distance():
    features = np.array([[0.0], [1.0], [5.0]])
    kernel = rbf_kernel(features, sigma=1.0)
    assert kernel[0, 1] > kernel[0, 2]


def test_linear_hsic_is_symmetric_in_its_arguments():
    rng = np.random.default_rng(4)
    first = rng.normal(size=(6, 3))
    second = rng.normal(size=(6, 4))
    assert linear_hsic(first, second) == pytest.approx(
        linear_hsic(second, first)
    )


def test_kernel_hsic_is_symmetric_in_its_arguments():
    rng = np.random.default_rng(5)
    first = rng.normal(size=(6, 3))
    second = rng.normal(size=(6, 4))
    assert kernel_hsic(first, second, 1.0) == pytest.approx(
        kernel_hsic(second, first, 1.0)
    )


def test_linear_cka_of_identical_features_is_one():
    rng = np.random.default_rng(6)
    features = rng.normal(size=(8, 5))
    assert linear_cka(features, features) == pytest.approx(1.0)


def test_linear_cka_is_invariant_to_isotropic_scaling():
    rng = np.random.default_rng(7)
    first = rng.normal(size=(8, 5))
    second = rng.normal(size=(8, 5))
    assert linear_cka(first, second) == pytest.approx(
        linear_cka(first, 3.0 * second)
    )


def test_linear_cka_is_bounded():
    rng = np.random.default_rng(8)
    first = rng.normal(size=(8, 5))
    second = rng.normal(size=(8, 5))
    value = linear_cka(first, second)
    assert 0.0 <= value <= 1.0


def test_kernel_cka_of_identical_features_is_one():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(8, 5))
    assert kernel_cka(features, features) == pytest.approx(1.0)


def test_kernel_cka_accepts_explicit_sigma():
    rng = np.random.default_rng(10)
    first = rng.normal(size=(8, 5))
    second = rng.normal(size=(8, 5))
    value = kernel_cka(first, second, sigma=1.5)
    assert 0.0 <= value <= 1.0
