"""Screening of exemplars within the selected categories."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from ..metrics.rdm import column_correlations, compute_rdm


@dataclass
class ExemplarSelection:
    """Outcome of the exemplar screening stage.

    Parameters
    ----------
    full_rdms : dict
        Mapping from model name to the dissimilarity matrix over
        every exemplar of the selected categories.
    selected_rdms : dict
        Mapping from model name to the dissimilarity matrix
        restricted to the kept exemplars.
    global_indices : numpy.ndarray
        Position of each kept exemplar in the full image list.
    local_indices : numpy.ndarray
        Position of each kept exemplar in ``full_rdms``.
    correlations : numpy.ndarray
        Per-exemplar correlation between the two models, over every
        exemplar of the selected categories.
    """

    full_rdms: Dict[str, np.ndarray]
    selected_rdms: Dict[str, np.ndarray]
    global_indices: np.ndarray
    local_indices: np.ndarray
    correlations: np.ndarray


def _validate_exemplar_request(
    category_indices, n_categories, n_exemplars, n_per_category
):
    """Raise when a curated set of this shape cannot be built."""
    if category_indices.size == 0:
        raise ValueError(
            "Exemplar screening needs at least one category."
        )
    if n_exemplars > n_per_category:
        raise ValueError(
            f"n_exemplars ({n_exemplars}) exceeds the "
            f"{n_per_category} exemplars each category holds."
        )
    if np.any(category_indices >= n_categories) or np.any(
        category_indices < 0
    ):
        raise IndexError(
            f"Category index out of range for {n_categories} "
            f"categories: {category_indices.tolist()}."
        )


def _least_correlated_exemplars(
    correlations, category_indices, n_exemplars, n_per_category
):
    """Pick the least correlated exemplars within each category.

    Returns
    -------
    local : numpy.ndarray
        Positions within the selected categories' own matrices.
    global_ : numpy.ndarray
        Positions in the full image list.
    """
    local, global_ = [], []
    for position, category in enumerate(category_indices):
        start = position * n_per_category
        within = correlations[start:start + n_per_category]
        kept = np.argsort(within)[:n_exemplars]
        local.append(start + kept)
        global_.append(category * n_per_category + kept)
    return np.concatenate(local), np.concatenate(global_)


def _category_rdms(
    cat_activations, models, category_indices, dissimilarity_metric
):
    """Dissimilarity matrix per model over the selected categories."""
    rdms = {}
    for model in models:
        subset = cat_activations[model][category_indices]
        rdms[model] = compute_rdm(
            subset.reshape(-1, subset.shape[-1]),
            metric=dissimilarity_metric,
        )
    return rdms


def screen_exemplars(
    cat_activations: Dict[str, np.ndarray],
    models: Tuple[str, str],
    category_indices: Sequence[int],
    n_exemplars: int = 4,
    n_per_category: int = 50,
    dissimilarity_metric: str = "L2squared",
) -> ExemplarSelection:
    """Keep the exemplars the two models relate most differently.

    The dissimilarity matrix over all exemplars of the selected
    categories is built per model and compared column by column.
    Within each category the exemplars with the lowest correlation
    are kept.

    Parameters
    ----------
    cat_activations : dict
        Mapping from model name to activations grouped by category.
    models : tuple of str
        The two model names.
    category_indices : sequence of int
        Indices of the selected categories, in the order they
        should appear in the curated set.
    n_exemplars : int
        How many exemplars to keep per category.
    n_per_category : int
        How many exemplars each category holds.
    dissimilarity_metric : str
        Metric passed to :func:`~repscreen.metrics.rdm.compute_rdm`.

    Returns
    -------
    ExemplarSelection
        Both dissimilarity matrix levels and the kept exemplars.

    Raises
    ------
    ValueError
        If no category is selected, or more exemplars are requested
        than a category holds.
    IndexError
        If a category index lies outside the activations.
    """
    category_indices = np.asarray(category_indices, dtype=int)
    _validate_exemplar_request(
        category_indices,
        cat_activations[models[0]].shape[0],
        n_exemplars,
        n_per_category,
    )

    full_rdms = _category_rdms(
        cat_activations, models, category_indices, dissimilarity_metric
    )
    correlations = column_correlations(
        full_rdms[models[0]], full_rdms[models[1]]
    )
    local_indices, global_indices = _least_correlated_exemplars(
        correlations, category_indices, n_exemplars, n_per_category
    )

    return ExemplarSelection(
        full_rdms=full_rdms,
        selected_rdms={
            model: full_rdms[model][
                np.ix_(local_indices, local_indices)
            ]
            for model in models
        },
        global_indices=global_indices,
        local_indices=local_indices,
        correlations=correlations,
    )
