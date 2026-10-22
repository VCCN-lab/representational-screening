"""Compactness of category clusters in an activation space.

Every measure takes activations grouped by category, of shape
``(n_categories, n_per_category, n_features)``, and returns one score
per category.  Higher scores mean a category whose exemplars sit
closer together relative to the rest of the space, except for
:func:`davies_bouldin_index`, where higher scores mean the opposite.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


@dataclass
class CompactnessResult:
    """Compactness of every category under each of two models.

    Parameters
    ----------
    compactness : dict
        Mapping from model name to compactness per category, in the
        order the categories were given.
    sorted_compactness : dict
        Mapping from model name to compactness sorted ascending.
    sorted_categories : dict
        Mapping from model name to the category labels in the same
        ascending order.
    measure : str
        Name of the compactness measure the scores come from.
    """

    compactness: Dict[str, np.ndarray]
    sorted_compactness: Dict[str, np.ndarray]
    sorted_categories: Dict[str, np.ndarray]
    measure: str


def _centroids(activations):
    """Return the centroid of every category."""
    return activations.mean(axis=1)


def _within_scatter(activations):
    """Mean squared distance of exemplars to their own centroid."""
    centroids = _centroids(activations)
    offsets = activations.transpose(1, 0, 2) - centroids
    return np.mean(np.sum(offsets**2, axis=2), axis=0)


def _squared_distances_to(activations, point):
    """Squared distances of every exemplar of a category to a point."""
    return np.sum((activations - point) ** 2, axis=-1)


def _pairwise_within(activations):
    """Mean squared distance between distinct exemplars of a category."""
    distances = euclidean_distances(
        activations, activations, squared=True
    )
    mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    return np.mean(distances[mask])


def r_squared(activations):
    """Within-category scatter relative to the scatter of the space.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category.
    """
    global_centroid = activations.mean(axis=(0, 1))
    global_radius = np.mean(
        np.sum((activations - global_centroid) ** 2, axis=2)
    )
    return 1.0 - _within_scatter(activations) / global_radius


def r_squared_adjusted(activations):
    """Within-category scatter relative to centroid separation.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category.
    """
    centroids = _centroids(activations)
    separation = np.mean(
        euclidean_distances(centroids, centroids, squared=True),
        axis=0,
    )
    return 1.0 - _within_scatter(activations) / separation


def fisher_discriminant(activations):
    """Within-category scatter relative to scatter about other centroids.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category.
    """
    centroids = _centroids(activations)
    n_categories = len(activations)
    between = np.array(
        [
            np.mean(
                [
                    np.mean(
                        _squared_distances_to(
                            activations[j], centroids[i]
                        )
                    )
                    for j in range(n_categories)
                    if j != i
                ]
            )
            for i in range(n_categories)
        ]
    )
    return 1.0 - _within_scatter(activations) / between


def _distance_to_global_centroid(activations):
    """Squared distance of every centroid to the global centroid."""
    global_centroid = activations.mean(axis=(0, 1))
    centroids = _centroids(activations)
    return np.sum((centroids - global_centroid) ** 2, axis=-1)


def ch_index(activations):
    """Centroid displacement relative to within-category scatter.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category.
    """
    return _distance_to_global_centroid(
        activations
    ) / _within_scatter(activations)


def ch_index_adapted(activations):
    """Within-category scatter relative to centroid displacement.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category, on the same scale as
        :func:`r_squared`.
    """
    return 1.0 - _within_scatter(
        activations
    ) / _distance_to_global_centroid(activations)


def davies_bouldin_index(activations):
    """Worst-case overlap of a category with any other category.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category.  Unlike the other measures, higher
        values mark a *less* compact category.
    """
    centroids = _centroids(activations)
    within = _within_scatter(activations)
    separation = euclidean_distances(
        centroids, centroids, squared=True
    )
    n_categories = len(activations)
    return np.array(
        [
            max(
                (within[i] + within[j]) / separation[i, j]
                for j in range(n_categories)
                if j != i
            )
            for i in range(n_categories)
        ]
    )


def _silhouette(within, between):
    """Combine within- and between-category distances."""
    return (between - within) / np.maximum(between, within)


def simplified_silhouette_score(activations):
    """Silhouette using centroid distances in place of exemplar pairs.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category, in ``[-1, 1]``.
    """
    centroids = _centroids(activations)
    n_categories = len(activations)
    nearest = np.array(
        [
            np.min(
                [
                    np.sum((centroids[j] - centroids[i]) ** 2)
                    for j in range(n_categories)
                    if j != i
                ]
            )
            for i in range(n_categories)
        ]
    )
    return _silhouette(_within_scatter(activations), nearest)


def global_silhouette_score(activations):
    """Silhouette against the pooled distance to all other categories.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category, in ``[-1, 1]``.
    """
    n_categories = len(activations)
    scores = np.zeros(n_categories)
    for i in range(n_categories):
        total, count = 0.0, 0
        for j in range(n_categories):
            if j == i:
                continue
            distances = euclidean_distances(
                activations[i], activations[j], squared=True
            )
            total += np.sum(distances)
            count += distances.size
        scores[i] = _silhouette(
            _pairwise_within(activations[i]), total / count
        )
    return scores


def silhouette_score(activations):
    """Silhouette against the nearest other category.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category.

    Returns
    -------
    numpy.ndarray
        One score per category, in ``[-1, 1]``.
    """
    n_categories = len(activations)
    scores = np.zeros(n_categories)
    for i in range(n_categories):
        nearest = np.min(
            [
                np.mean(
                    euclidean_distances(
                        activations[i], activations[j], squared=True
                    )
                )
                for j in range(n_categories)
                if j != i
            ]
        )
        scores[i] = _silhouette(
            _pairwise_within(activations[i]), nearest
        )
    return scores


COMPACTNESS_MEASURES = {
    "R-squared": r_squared,
    "R-squared_adjusted": r_squared_adjusted,
    "Fisher_discriminant": fisher_discriminant,
    "CH_Index": ch_index,
    "CH_Index_adapted": ch_index_adapted,
    "silhouette_score": silhouette_score,
    "global_silhouette_score": global_silhouette_score,
    "simplified_silhouette_score": simplified_silhouette_score,
    "Davies-Bouldin_Index": davies_bouldin_index,
}


def compute_compactness(activations, measure="R-squared"):
    """Score how compact each category is under one model.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations grouped by category, of shape
        ``(n_categories, n_per_category, n_features)``.
    measure : str
        Key of :data:`COMPACTNESS_MEASURES`.

    Returns
    -------
    numpy.ndarray
        One score per category.

    Raises
    ------
    ValueError
        If ``activations`` is not three-dimensional, or ``measure``
        is not a known measure.
    """
    if activations.ndim != 3:
        raise ValueError(
            f"activations must be three-dimensional (n_categories, "
            f"n_per_category, n_features), got shape "
            f"{activations.shape}."
        )
    if measure not in COMPACTNESS_MEASURES:
        raise ValueError(
            f"Unknown compactness measure: '{measure}'. Must be one "
            f"of {tuple(COMPACTNESS_MEASURES)}."
        )
    return COMPACTNESS_MEASURES[measure](activations)


def compute_compactness_pair(
    cat_activations: Dict[str, np.ndarray],
    models: Tuple[str, str],
    category_names: Sequence[str],
    measure: str = "R-squared",
) -> CompactnessResult:
    """Score category compactness under each of two models.

    Parameters
    ----------
    cat_activations : dict
        Mapping from model name to activations grouped by category.
    models : tuple of str
        The two model names to score.
    category_names : sequence of str
        Category labels, in the order they appear along the first
        axis of the activations.
    measure : str
        Key of :data:`COMPACTNESS_MEASURES`.

    Returns
    -------
    CompactnessResult
        Compactness per model, both unsorted and sorted ascending
        together with the matching labels.

    Raises
    ------
    ValueError
        If ``category_names`` does not have one entry per category.
    """
    compactness = {}
    sorted_compactness = {}
    sorted_categories = {}
    labels = np.array(list(category_names))

    for model in models:
        activations = cat_activations[model]
        if len(labels) != activations.shape[0]:
            raise ValueError(
                f"category_names has {len(labels)} entries but "
                f"model '{model}' has {activations.shape[0]} "
                f"categories."
            )
        scores = compute_compactness(activations, measure=measure)
        order = np.argsort(scores)
        compactness[model] = scores
        sorted_compactness[model] = scores[order]
        sorted_categories[model] = labels[order]

    return CompactnessResult(
        compactness=compactness,
        sorted_compactness=sorted_compactness,
        sorted_categories=sorted_categories,
        measure=measure,
    )
