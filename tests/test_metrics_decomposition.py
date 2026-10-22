"""Tests for repscreen.metrics.decomposition."""

import numpy as np
import pytest

from repscreen.metrics.decomposition import (
    classical_mds,
    principal_components,
    procrustes,
)


@pytest.fixture
def observations():
    """Observations in rows, variables in columns."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(40, 2))
    mixing = rng.normal(size=(2, 5))
    return latent @ mixing + 0.05 * rng.normal(size=(40, 5))


def test_principal_components_shapes(observations):
    result = principal_components(observations)
    n_variables = observations.shape[1]
    assert result.coefficients.shape == (n_variables, n_variables)
    assert result.scores.shape == (
        n_variables,
        observations.shape[0],
    )
    assert result.eigenvalues.shape == (n_variables,)
    assert result.explained.shape == (n_variables,)


def test_principal_components_eigenvalues_are_descending(
    observations,
):
    result = principal_components(observations)
    assert np.all(np.diff(result.eigenvalues) <= 1e-9)


def test_principal_components_eigenvalues_are_real(observations):
    result = principal_components(observations)
    assert not np.iscomplexobj(result.eigenvalues)
    assert not np.iscomplexobj(result.coefficients)


def test_principal_components_explained_sums_to_one_hundred(
    observations,
):
    result = principal_components(observations)
    assert np.sum(result.explained) == pytest.approx(100.0)


def test_principal_components_eigenvalues_match_the_covariance(
    observations,
):
    result = principal_components(observations)
    centred = observations - observations.mean(axis=0)
    expected = np.linalg.eigvalsh(np.cov(centred.T))[::-1]
    assert np.allclose(result.eigenvalues, expected)


def test_principal_components_coefficients_are_orthonormal(
    observations,
):
    result = principal_components(observations)
    product = result.coefficients.T @ result.coefficients
    assert np.allclose(
        product, np.eye(observations.shape[1]), atol=1e-8
    )


def test_principal_components_recovers_a_rank_two_structure(
    observations,
):
    result = principal_components(observations)
    leading = result.explained[0] + result.explained[1]
    assert leading > 99.0


def test_principal_components_scores_are_the_projection(
    observations,
):
    result = principal_components(observations)
    centred = (observations - observations.mean(axis=0)).T
    expected = result.coefficients.T @ centred
    assert np.allclose(result.scores, expected)


def test_principal_components_scores_are_uncorrelated(observations):
    result = principal_components(observations)
    correlation = np.corrcoef(result.scores[:2])
    assert correlation[0, 1] == pytest.approx(0.0, abs=1e-8)


def test_principal_components_rejects_non_2d_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        principal_components(np.zeros((4, 3, 2)))


def test_classical_mds_recovers_planar_distances():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(12, 2))
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :], axis=-1
    )
    embedding, eigenvalues = classical_mds(distances)
    recovered = np.linalg.norm(
        embedding[:, None, :] - embedding[None, :, :], axis=-1
    )
    assert np.allclose(recovered, distances, atol=1e-8)
    assert eigenvalues.shape == (12,)


def test_classical_mds_eigenvalues_are_descending():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(10, 3))
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :], axis=-1
    )
    _, eigenvalues = classical_mds(distances)
    assert np.all(np.diff(eigenvalues) <= 1e-9)


def test_classical_mds_keeps_only_positive_components():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(10, 3))
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :], axis=-1
    )
    embedding, eigenvalues = classical_mds(distances)
    assert embedding.shape[1] == int(np.sum(eigenvalues > 0))


def test_classical_mds_embedding_is_centred():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(10, 2))
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :], axis=-1
    )
    embedding, _ = classical_mds(distances)
    assert np.allclose(embedding.mean(axis=0), 0.0, atol=1e-8)


def test_classical_mds_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        classical_mds(np.zeros((3, 4)))


def test_procrustes_of_identical_configurations_is_exact():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(15, 3))
    disparity, transformed, transform = procrustes(points, points)
    assert disparity == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(transformed, points)
    assert transform["scale"] == pytest.approx(1.0)


def test_procrustes_undoes_a_rotation_and_scaling():
    rng = np.random.default_rng(6)
    target = rng.normal(size=(20, 2))
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    moved = 2.5 * target @ rotation + np.array([3.0, -1.0])
    disparity, transformed, _ = procrustes(target, moved)
    assert disparity == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(transformed, target, atol=1e-8)


def test_procrustes_disparity_is_scale_free():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(20, 3))
    moved = rng.normal(size=(20, 3))
    plain = procrustes(target, moved)[0]
    scaled = procrustes(target, 10.0 * moved)[0]
    assert plain == pytest.approx(scaled)


def test_procrustes_without_scaling_keeps_unit_scale():
    rng = np.random.default_rng(8)
    target = rng.normal(size=(20, 3))
    moved = rng.normal(size=(20, 3))
    _, _, transform = procrustes(target, moved, scaling=False)
    assert transform["scale"] == pytest.approx(1.0)


def test_procrustes_can_forbid_a_reflection():
    rng = np.random.default_rng(9)
    target = rng.normal(size=(20, 2))
    reflected = target @ np.array([[1.0, 0.0], [0.0, -1.0]])
    _, _, transform = procrustes(
        target, reflected, reflection=False
    )
    assert np.linalg.det(transform["rotation"]) > 0


def test_procrustes_can_force_a_reflection():
    rng = np.random.default_rng(10)
    target = rng.normal(size=(20, 2))
    moved = rng.normal(size=(20, 2))
    _, _, transform = procrustes(target, moved, reflection=True)
    assert np.linalg.det(transform["rotation"]) < 0


def test_procrustes_transform_reproduces_the_output():
    rng = np.random.default_rng(11)
    target = rng.normal(size=(20, 3))
    moved = rng.normal(size=(20, 3))
    _, transformed, transform = procrustes(target, moved)
    rebuilt = (
        transform["scale"] * moved @ transform["rotation"]
        + transform["translation"]
    )
    assert np.allclose(transformed, rebuilt)


def test_procrustes_pads_fewer_input_dimensions():
    rng = np.random.default_rng(12)
    target = rng.normal(size=(15, 3))
    moved = rng.normal(size=(15, 2))
    disparity, transformed, transform = procrustes(target, moved)
    assert transformed.shape == (15, 3)
    assert transform["rotation"].shape == (2, 3)
    assert np.isfinite(disparity)


def test_procrustes_rejects_differing_point_counts():
    with pytest.raises(ValueError, match="same number of points"):
        procrustes(np.zeros((10, 2)), np.zeros((9, 2)))


def test_procrustes_rejects_more_input_than_target_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        procrustes(np.zeros((10, 2)), np.zeros((10, 3)))
