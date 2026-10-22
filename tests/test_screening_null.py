"""Tests for repscreen.screening.null."""

import numpy as np
import pytest

from repscreen.metrics.rdm import compare_rdms, compute_rdm
from repscreen.screening.null import (
    sample_category_similarity,
    sample_image_similarity,
    sample_rdm_similarity,
)

from .conftest import MODELS, N_CATEGORIES, N_PER_CATEGORY

N_SAMPLES = 12
SUBSET = 4


def test_sample_rdm_similarity_shapes(rdm_pair):
    first, second = rdm_pair
    similarities, indices = sample_rdm_similarity(
        first,
        second,
        n_samples=N_SAMPLES,
        subset_size=SUBSET,
        batch_size=5,
        seed=0,
    )
    assert similarities.shape == (N_SAMPLES,)
    assert indices.shape == (N_SAMPLES, SUBSET)


def test_sample_rdm_similarity_is_reproducible(rdm_pair):
    first, second = rdm_pair
    kwargs = dict(
        n_samples=N_SAMPLES, subset_size=SUBSET, batch_size=5, seed=7
    )
    left = sample_rdm_similarity(first, second, **kwargs)
    right = sample_rdm_similarity(first, second, **kwargs)
    assert np.allclose(left[0], right[0])
    assert np.array_equal(left[1], right[1])


def test_sample_rdm_similarity_indices_are_unique_per_sample(
    rdm_pair,
):
    first, second = rdm_pair
    _, indices = sample_rdm_similarity(
        first,
        second,
        n_samples=N_SAMPLES,
        subset_size=SUBSET,
        seed=1,
    )
    for row in indices:
        assert len(set(row)) == SUBSET


def test_sample_rdm_similarity_matches_a_direct_comparison(rdm_pair):
    first, second = rdm_pair
    similarities, indices = sample_rdm_similarity(
        first, second, n_samples=3, subset_size=SUBSET, seed=2
    )
    for value, row in zip(similarities, indices):
        expected = compare_rdms(
            first[np.ix_(row, row)],
            second[np.ix_(row, row)],
            metric="pearson",
        )
        assert value == pytest.approx(expected)


def test_sample_rdm_similarity_of_identical_matrices_is_one(rdm_pair):
    first, _ = rdm_pair
    similarities, _ = sample_rdm_similarity(
        first, first, n_samples=4, subset_size=SUBSET, seed=3
    )
    assert np.allclose(similarities, 1.0)


def test_sample_rdm_similarity_rejects_oversized_subsets(rdm_pair):
    first, second = rdm_pair
    with pytest.raises(ValueError, match="subset_size"):
        sample_rdm_similarity(
            first, second, n_samples=2, subset_size=len(first) + 1
        )


def test_sample_image_similarity_shapes(flat_activations):
    similarities, indices = sample_image_similarity(
        flat_activations,
        MODELS,
        n_samples=N_SAMPLES,
        subset_size=SUBSET,
        batch_size=5,
        seed=0,
    )
    assert similarities.shape == (N_SAMPLES,)
    assert indices.shape == (N_SAMPLES, SUBSET)


def test_sample_image_similarity_matches_a_direct_comparison(
    flat_activations,
):
    similarities, indices = sample_image_similarity(
        flat_activations,
        MODELS,
        n_samples=3,
        subset_size=SUBSET,
        seed=4,
    )
    for value, row in zip(similarities, indices):
        rdms = [
            compute_rdm(flat_activations[model][row])
            for model in MODELS
        ]
        expected = compare_rdms(*rdms, metric="pearson")
        assert value == pytest.approx(expected)


def test_sample_image_similarity_rejects_oversized_subsets(
    flat_activations,
):
    size = len(flat_activations[MODELS[0]])
    with pytest.raises(ValueError, match="subset_size"):
        sample_image_similarity(
            flat_activations,
            MODELS,
            n_samples=2,
            subset_size=size + 1,
        )


def test_sample_category_similarity_shapes(cat_activations):
    similarities, indices = sample_category_similarity(
        cat_activations,
        MODELS,
        n_samples=N_SAMPLES,
        n_categories=3,
        batch_size=5,
        seed=0,
    )
    assert similarities.shape == (N_SAMPLES,)
    assert indices.shape == (N_SAMPLES, 3)


def test_sample_category_similarity_draws_category_indices(
    cat_activations,
):
    _, indices = sample_category_similarity(
        cat_activations, MODELS, n_samples=6, n_categories=3, seed=5
    )
    assert indices.max() < N_CATEGORIES
    for row in indices:
        assert len(set(row)) == 3


def test_sample_category_similarity_matches_a_direct_comparison(
    cat_activations,
):
    similarities, indices = sample_category_similarity(
        cat_activations, MODELS, n_samples=3, n_categories=3, seed=6
    )
    for value, row in zip(similarities, indices):
        rdms = []
        for model in MODELS:
            subset = cat_activations[model][row]
            rdms.append(
                compute_rdm(subset.reshape(3 * N_PER_CATEGORY, -1))
            )
        expected = compare_rdms(*rdms, metric="pearson")
        assert value == pytest.approx(expected)


def test_sample_category_similarity_rejects_oversized_requests(
    cat_activations,
):
    with pytest.raises(ValueError, match="n_categories"):
        sample_category_similarity(
            cat_activations,
            MODELS,
            n_samples=2,
            n_categories=N_CATEGORIES + 1,
        )
