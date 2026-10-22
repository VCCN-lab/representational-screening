"""Tests for repscreen.screening.categories."""

import numpy as np
import pytest

from repscreen.metrics.compactness import compute_compactness_pair
from repscreen.metrics.rdm import column_correlations, compute_rdm
from repscreen.screening.categories import (
    category_rdm_correlations,
    combined_category_score,
    compactness_difference,
    rank_categories,
    screen_categories,
)

from .conftest import MODELS, N_CATEGORIES


def test_category_rdm_correlations_match_mean_activation_rdms(
    cat_activations,
):
    rdms = [
        compute_rdm(
            cat_activations[model].mean(axis=1), metric="L2squared"
        )
        for model in MODELS
    ]
    expected = column_correlations(*rdms)
    result = category_rdm_correlations(cat_activations, MODELS)
    assert np.allclose(result, expected)


def test_category_rdm_correlations_has_one_value_per_category(
    cat_activations,
):
    result = category_rdm_correlations(cat_activations, MODELS)
    assert result.shape == (N_CATEGORIES,)


def test_category_rdm_correlations_of_a_model_with_itself_are_one(
    cat_activations,
):
    same = {
        MODELS[0]: cat_activations[MODELS[0]],
        MODELS[1]: cat_activations[MODELS[0]],
    }
    result = category_rdm_correlations(same, MODELS)
    assert np.allclose(result, 1.0)


def test_category_rdm_correlations_rejects_shape_mismatch(
    cat_activations,
):
    mismatched = {
        MODELS[0]: cat_activations[MODELS[0]],
        MODELS[1]: cat_activations[MODELS[1]][:-1],
    }
    with pytest.raises(ValueError, match="same category structure"):
        category_rdm_correlations(mismatched, MODELS)


def test_compactness_difference_normalised_is_centred_and_scaled(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    scaled = []
    for model in MODELS:
        values = compactness.compactness[model]
        centred = values - np.mean(values)
        scaled.append(centred / np.max(np.absolute(centred)))
    expected = scaled[1] - scaled[0]
    result = compactness_difference(
        compactness,
        category_names,
        MODELS,
        difference_measure="normalizedDiff",
    )
    assert np.allclose(result.difference, expected)


def test_compactness_difference_rank_subtracts_ranks(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    ranks = []
    for model in MODELS:
        lookup = {
            category: rank
            for rank, category in enumerate(
                compactness.sorted_categories[model]
            )
        }
        ranks.append(
            np.array([lookup[name] for name in category_names])
        )
    expected = ranks[1] - ranks[0]
    result = compactness_difference(
        compactness,
        category_names,
        MODELS,
        difference_measure="rank",
    )
    assert np.allclose(result.difference, expected)


def test_compactness_difference_is_in_category_order(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    result = compactness_difference(
        compactness, category_names, MODELS
    )
    assert result.difference.shape == (N_CATEGORIES,)


def test_compactness_difference_orders_by_absolute_difference(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    result = compactness_difference(
        compactness, category_names, MODELS
    )
    ranked = np.absolute(result.difference[result.order])
    assert np.all(np.diff(ranked) <= 1e-12)


def test_compactness_difference_labels_follow_the_order(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    result = compactness_difference(
        compactness, category_names, MODELS
    )
    expected = np.array(category_names)[result.order]
    assert list(result.categories) == list(expected)


def test_compactness_difference_rejects_unknown_measure(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    with pytest.raises(ValueError, match="Unknown difference measure"):
        compactness_difference(
            compactness,
            category_names,
            MODELS,
            difference_measure="ratio",
        )


def test_combined_score_rewards_difference_and_penalises_correlation():
    difference = np.array([0.5, -0.9, 0.0])
    correlations = np.array([0.1, 0.1, 0.8])
    expected = np.array([0.4, 0.8, -0.8])
    result = combined_category_score(correlations, difference)
    assert np.allclose(result, expected)


def test_combined_score_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        combined_category_score(np.zeros(3), np.zeros(4))


def test_rank_categories_is_descending_by_score():
    score = np.array([0.1, 0.9, 0.5])
    assert list(rank_categories(score)) == [1, 2, 0]


def test_rank_categories_can_truncate():
    score = np.array([0.1, 0.9, 0.5])
    assert list(rank_categories(score, n_categories=2)) == [1, 2]


def test_rank_categories_rejects_request_beyond_available():
    with pytest.raises(ValueError, match="only 3 categories"):
        rank_categories(np.zeros(3), n_categories=4)


def test_screen_categories_combines_both_criteria(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    result = screen_categories(
        cat_activations,
        MODELS,
        compactness,
        category_names,
        n_categories=3,
    )
    expected_score = combined_category_score(
        result.correlations, result.difference.difference
    )
    assert np.allclose(result.score, expected_score)
    assert list(result.selected) == list(
        rank_categories(expected_score, n_categories=3)
    )


def test_screen_categories_reports_selected_names(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    result = screen_categories(
        cat_activations,
        MODELS,
        compactness,
        category_names,
        n_categories=3,
    )
    expected = [category_names[i] for i in result.selected]
    assert list(result.selected_names) == expected


def test_screen_categories_selects_the_requested_count(
    cat_activations, category_names
):
    compactness = compute_compactness_pair(
        cat_activations, MODELS, category_names
    )
    result = screen_categories(
        cat_activations,
        MODELS,
        compactness,
        category_names,
        n_categories=4,
    )
    assert len(result.selected) == 4
