"""Tests for repscreen.screening.exemplars."""

import numpy as np
import pytest

from repscreen.metrics.rdm import column_correlations, compute_rdm
from repscreen.screening.exemplars import screen_exemplars

from .conftest import MODELS, N_PER_CATEGORY

SELECTED = np.array([4, 1, 2])
N_EXEMPLARS = 2


def _selection(cat_activations):
    """Run the exemplar screening used across these tests."""
    return screen_exemplars(
        cat_activations,
        MODELS,
        SELECTED,
        n_exemplars=N_EXEMPLARS,
        n_per_category=N_PER_CATEGORY,
    )


def test_full_rdms_cover_every_exemplar_of_selected_categories(
    cat_activations,
):
    result = _selection(cat_activations)
    size = len(SELECTED) * N_PER_CATEGORY
    for model in MODELS:
        assert result.full_rdms[model].shape == (size, size)


def test_full_rdms_match_a_direct_computation(cat_activations):
    result = _selection(cat_activations)
    for model in MODELS:
        subset = cat_activations[model][SELECTED]
        expected = compute_rdm(
            subset.reshape(-1, subset.shape[-1]), metric="L2squared"
        )
        assert np.allclose(result.full_rdms[model], expected)


def test_selected_rdms_cover_only_the_kept_exemplars(
    cat_activations,
):
    result = _selection(cat_activations)
    size = len(SELECTED) * N_EXEMPLARS
    for model in MODELS:
        assert result.selected_rdms[model].shape == (size, size)


def test_one_global_index_per_kept_exemplar(cat_activations):
    result = _selection(cat_activations)
    assert result.global_indices.shape == (
        len(SELECTED) * N_EXEMPLARS,
    )


def test_global_indices_point_into_the_selected_categories(
    cat_activations,
):
    result = _selection(cat_activations)
    categories = result.global_indices // N_PER_CATEGORY
    assert set(categories) == set(SELECTED)


def test_global_indices_keep_the_selection_order(cat_activations):
    result = _selection(cat_activations)
    categories = result.global_indices // N_PER_CATEGORY
    assert list(categories) == [
        category
        for category in SELECTED
        for _ in range(N_EXEMPLARS)
    ]


def test_global_indices_are_unique(cat_activations):
    result = _selection(cat_activations)
    assert len(set(result.global_indices)) == len(
        result.global_indices
    )


def test_kept_exemplars_are_the_least_correlated_in_their_category(
    cat_activations,
):
    result = _selection(cat_activations)
    rdms = [result.full_rdms[model] for model in MODELS]
    correlations = column_correlations(*rdms)
    for position, category in enumerate(SELECTED):
        start = position * N_PER_CATEGORY
        within = correlations[start:start + N_PER_CATEGORY]
        expected = category * N_PER_CATEGORY + np.argsort(within)[
            :N_EXEMPLARS
        ]
        kept = result.global_indices[
            position * N_EXEMPLARS:(position + 1) * N_EXEMPLARS
        ]
        assert list(kept) == list(expected)


def test_local_indices_address_the_full_rdms(cat_activations):
    result = _selection(cat_activations)
    for model in MODELS:
        expected = result.full_rdms[model][
            np.ix_(result.local_indices, result.local_indices)
        ]
        assert np.allclose(result.selected_rdms[model], expected)


def test_rejects_more_exemplars_than_available(cat_activations):
    with pytest.raises(ValueError, match="n_exemplars"):
        screen_exemplars(
            cat_activations,
            MODELS,
            SELECTED,
            n_exemplars=N_PER_CATEGORY + 1,
            n_per_category=N_PER_CATEGORY,
        )


def test_rejects_an_empty_category_selection(cat_activations):
    with pytest.raises(ValueError, match="at least one category"):
        screen_exemplars(
            cat_activations,
            MODELS,
            np.array([], dtype=int),
            n_exemplars=N_EXEMPLARS,
            n_per_category=N_PER_CATEGORY,
        )


def test_rejects_a_category_index_out_of_range(cat_activations):
    with pytest.raises(IndexError, match="out of range"):
        screen_exemplars(
            cat_activations,
            MODELS,
            np.array([0, 99]),
            n_exemplars=N_EXEMPLARS,
            n_per_category=N_PER_CATEGORY,
        )
