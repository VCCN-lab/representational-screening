"""Tests for repscreen.metrics.compactness."""

import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances

from repscreen.metrics.compactness import (
    COMPACTNESS_MEASURES,
    ch_index,
    ch_index_adapted,
    compute_compactness,
    compute_compactness_pair,
    davies_bouldin_index,
    fisher_discriminant,
    global_silhouette_score,
    r_squared,
    r_squared_adjusted,
    silhouette_score,
    simplified_silhouette_score,
)
from repscreen.config import VALID_COMPACTNESS_MEASURES


def _within_scatter(activations):
    """Mean squared distance of exemplars to their own centroid."""
    centroids = activations.mean(axis=1)
    return np.array(
        [
            np.mean(
                np.sum((activations[i] - centroids[i]) ** 2, axis=-1)
            )
            for i in range(len(activations))
        ]
    )


def test_measure_registry_covers_every_valid_measure():
    assert set(COMPACTNESS_MEASURES) == set(VALID_COMPACTNESS_MEASURES)


def test_r_squared_matches_explicit_formula(cat_activations):
    activations = cat_activations["model_a"]
    global_centroid = activations.mean(axis=(0, 1))
    global_radius = np.mean(
        np.sum((activations - global_centroid) ** 2, axis=2)
    )
    expected = 1.0 - _within_scatter(activations) / global_radius
    assert np.allclose(r_squared(activations), expected)


def test_r_squared_is_higher_for_more_compact_categories(
    cat_activations,
):
    compact = r_squared(cat_activations["model_a"]).mean()
    diffuse = r_squared(cat_activations["model_b"]).mean()
    assert compact > diffuse


def test_r_squared_adjusted_matches_explicit_formula(cat_activations):
    activations = cat_activations["model_a"]
    centroids = activations.mean(axis=1)
    centroid_distances = np.mean(
        euclidean_distances(centroids, centroids, squared=True), axis=0
    )
    expected = 1.0 - _within_scatter(activations) / centroid_distances
    assert np.allclose(r_squared_adjusted(activations), expected)


def test_fisher_discriminant_matches_explicit_formula(cat_activations):
    activations = cat_activations["model_a"]
    centroids = activations.mean(axis=1)
    n_categories = len(activations)
    within = _within_scatter(activations)
    between = np.array(
        [
            np.mean(
                [
                    np.mean(
                        np.sum(
                            (activations[j] - centroids[i]) ** 2,
                            axis=-1,
                        )
                    )
                    for j in range(n_categories)
                    if j != i
                ]
            )
            for i in range(n_categories)
        ]
    )
    expected = 1.0 - within / between
    assert np.allclose(fisher_discriminant(activations), expected)


def test_ch_index_matches_explicit_formula(cat_activations):
    activations = cat_activations["model_a"]
    global_centroid = activations.mean(axis=(0, 1))
    centroids = activations.mean(axis=1)
    to_centre = np.sum((centroids - global_centroid) ** 2, axis=-1)
    expected = to_centre / _within_scatter(activations)
    assert np.allclose(ch_index(activations), expected)


def test_ch_index_adapted_matches_explicit_formula(cat_activations):
    activations = cat_activations["model_a"]
    global_centroid = activations.mean(axis=(0, 1))
    centroids = activations.mean(axis=1)
    to_centre = np.sum((centroids - global_centroid) ** 2, axis=-1)
    expected = 1.0 - _within_scatter(activations) / to_centre
    assert np.allclose(ch_index_adapted(activations), expected)


def test_davies_bouldin_index_matches_explicit_formula(
    cat_activations,
):
    activations = cat_activations["model_a"]
    centroids = activations.mean(axis=1)
    within = _within_scatter(activations)
    centroid_distances = euclidean_distances(
        centroids, centroids, squared=True
    )
    n_categories = len(activations)
    expected = np.array(
        [
            max(
                (within[i] + within[j]) / centroid_distances[i, j]
                for j in range(n_categories)
                if j != i
            )
            for i in range(n_categories)
        ]
    )
    assert np.allclose(davies_bouldin_index(activations), expected)


def test_simplified_silhouette_matches_explicit_formula(
    cat_activations,
):
    activations = cat_activations["model_a"]
    centroids = activations.mean(axis=1)
    within = _within_scatter(activations)
    n_categories = len(activations)
    nearest = np.array(
        [
            min(
                np.sum((centroids[j] - centroids[i]) ** 2)
                for j in range(n_categories)
                if j != i
            )
            for i in range(n_categories)
        ]
    )
    expected = (nearest - within) / np.maximum(nearest, within)
    assert np.allclose(
        simplified_silhouette_score(activations), expected
    )


def test_global_silhouette_matches_explicit_formula(cat_activations):
    activations = cat_activations["model_a"]
    n_categories = len(activations)
    expected = np.zeros(n_categories)
    for i in range(n_categories):
        inside = euclidean_distances(
            activations[i], activations[i], squared=True
        )
        mask = np.triu(np.ones_like(inside, dtype=bool), k=1)
        within = np.mean(inside[mask])
        total, count = 0.0, 0
        for j in range(n_categories):
            if j == i:
                continue
            across = euclidean_distances(
                activations[i], activations[j], squared=True
            )
            total += np.sum(across)
            count += across.size
        between = total / count
        expected[i] = (between - within) / max(between, within)
    assert np.allclose(global_silhouette_score(activations), expected)


def test_silhouette_uses_the_nearest_other_category(cat_activations):
    activations = cat_activations["model_a"]
    n_categories = len(activations)
    expected = np.zeros(n_categories)
    for i in range(n_categories):
        inside = euclidean_distances(
            activations[i], activations[i], squared=True
        )
        mask = np.triu(np.ones_like(inside, dtype=bool), k=1)
        within = np.mean(inside[mask])
        between = min(
            np.mean(
                euclidean_distances(
                    activations[i], activations[j], squared=True
                )
            )
            for j in range(n_categories)
            if j != i
        )
        expected[i] = (between - within) / max(between, within)
    assert np.allclose(silhouette_score(activations), expected)


def test_silhouette_scores_are_bounded(cat_activations):
    for measure in (
        silhouette_score,
        global_silhouette_score,
        simplified_silhouette_score,
    ):
        values = measure(cat_activations["model_a"])
        assert np.all(values >= -1.0)
        assert np.all(values <= 1.0)


def test_compute_compactness_dispatches_to_named_measure(
    cat_activations,
):
    activations = cat_activations["model_a"]
    assert np.allclose(
        compute_compactness(activations, measure="R-squared"),
        r_squared(activations),
    )


def test_compute_compactness_runs_every_registered_measure(
    cat_activations,
):
    activations = cat_activations["model_a"]
    for measure in VALID_COMPACTNESS_MEASURES:
        values = compute_compactness(activations, measure=measure)
        assert values.shape == (activations.shape[0],)
        assert np.all(np.isfinite(values))


def test_compute_compactness_rejects_unknown_measure(cat_activations):
    with pytest.raises(ValueError, match="Unknown compactness measure"):
        compute_compactness(
            cat_activations["model_a"], measure="not-a-measure"
        )


def test_compute_compactness_rejects_non_3d_input():
    with pytest.raises(ValueError, match="three-dimensional"):
        compute_compactness(np.zeros((4, 3)))


def test_compute_compactness_pair_returns_one_entry_per_model(
    cat_activations, category_names
):
    models = ("model_a", "model_b")
    result = compute_compactness_pair(
        cat_activations, models, category_names
    )
    assert set(result.compactness) == set(models)
    assert set(result.sorted_compactness) == set(models)
    assert set(result.sorted_categories) == set(models)


def test_compute_compactness_pair_sorts_ascending(
    cat_activations, category_names
):
    result = compute_compactness_pair(
        cat_activations, ("model_a", "model_b"), category_names
    )
    values = result.sorted_compactness["model_a"]
    assert np.all(np.diff(values) >= 0)


def test_compute_compactness_pair_labels_follow_the_sort(
    cat_activations, category_names
):
    result = compute_compactness_pair(
        cat_activations, ("model_a", "model_b"), category_names
    )
    order = np.argsort(result.compactness["model_a"])
    expected = np.array(category_names)[order]
    assert list(result.sorted_categories["model_a"]) == list(expected)


def test_compute_compactness_pair_rejects_label_count_mismatch(
    cat_activations, category_names
):
    with pytest.raises(ValueError, match="category_names"):
        compute_compactness_pair(
            cat_activations,
            ("model_a", "model_b"),
            category_names[:-1],
        )
