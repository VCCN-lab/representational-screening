"""Screening of categories from a pair of model activation sets.

Two criteria are combined.  The relational criterion looks for
categories whose position among the other categories differs between
the models.  The compactness criterion looks for categories that one
model represents as a tight cluster and the other does not.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import VALID_DIFFERENCE_MEASURES
from ..metrics.compactness import CompactnessResult
from ..metrics.rdm import column_correlations, compute_rdm


@dataclass
class CompactnessDifference:
    """Contrast of two models' compactness, category by category.

    Parameters
    ----------
    difference : numpy.ndarray
        Difference per category, in category order.  A positive
        value means the second model scores higher.
    order : numpy.ndarray
        Category indices sorted by descending absolute difference.
    categories : numpy.ndarray
        Category labels in that order.
    measure : str
        Name of the difference measure used.
    """

    difference: np.ndarray
    order: np.ndarray
    categories: np.ndarray
    measure: str


@dataclass
class CategorySelection:
    """Outcome of the category screening stage.

    Parameters
    ----------
    correlations : numpy.ndarray
        Relational criterion per category: how similarly the two
        models place that category among the others.
    difference : CompactnessDifference
        Compactness criterion per category.
    score : numpy.ndarray
        Combined score per category.
    selected : numpy.ndarray
        Indices of the selected categories, best first.
    selected_names : List[str]
        Labels of the selected categories, in the same order.
    """

    correlations: np.ndarray
    difference: CompactnessDifference
    score: np.ndarray
    selected: np.ndarray
    selected_names: List[str]


def category_rdm_correlations(
    cat_activations: Dict[str, np.ndarray],
    models: Tuple[str, str],
    dissimilarity_metric: str = "L2squared",
) -> np.ndarray:
    """Compare how the two models relate each category to the others.

    Each category is summarised by the mean of its exemplars, the
    categories' dissimilarity matrix is built per model, and the two
    matrices are compared column by column.  A low value marks a
    category the models place differently.

    Parameters
    ----------
    cat_activations : dict
        Mapping from model name to activations grouped by category.
    models : tuple of str
        The two model names.
    dissimilarity_metric : str
        Metric passed to :func:`~repscreen.metrics.rdm.compute_rdm`.

    Returns
    -------
    numpy.ndarray
        One correlation per category.

    Raises
    ------
    ValueError
        If the two models do not share the same category structure.
    """
    first, second = models
    if (
        cat_activations[first].shape[:2]
        != cat_activations[second].shape[:2]
    ):
        raise ValueError(
            f"'{first}' and '{second}' must share the same category "
            f"structure, got {cat_activations[first].shape[:2]} and "
            f"{cat_activations[second].shape[:2]}."
        )

    rdms = [
        compute_rdm(
            cat_activations[model].mean(axis=1),
            metric=dissimilarity_metric,
        )
        for model in models
    ]
    return column_correlations(*rdms)


def _rank_difference(
    compactness: CompactnessResult,
    category_names: Sequence[str],
    models: Tuple[str, str],
) -> np.ndarray:
    """Difference of the two models' compactness ranks."""
    ranks = []
    for model in models:
        lookup = {
            category: rank
            for rank, category in enumerate(
                compactness.sorted_categories[model]
            )
        }
        ranks.append(
            np.array(
                [lookup[name] for name in category_names],
                dtype=float,
            )
        )
    return ranks[1] - ranks[0]


def _normalized_difference(
    compactness: CompactnessResult, models: Tuple[str, str]
) -> np.ndarray:
    """Difference of compactness centred and scaled per model."""
    scaled = []
    for model in models:
        values = compactness.compactness[model]
        centred = values - np.mean(values)
        scaled.append(centred / np.max(np.absolute(centred)))
    return scaled[1] - scaled[0]


def compactness_difference(
    compactness: CompactnessResult,
    category_names: Sequence[str],
    models: Tuple[str, str],
    difference_measure: str = "normalizedDiff",
) -> CompactnessDifference:
    """Rank categories by how differently the models compact them.

    Parameters
    ----------
    compactness : CompactnessResult
        Compactness of every category under both models.
    category_names : sequence of str
        Category labels in category order.
    models : tuple of str
        The two model names.
    difference_measure : str
        ``"normalizedDiff"`` subtracts compactness centred and
        scaled to unit maximum absolute value; ``"rank"`` subtracts
        compactness ranks.

    Returns
    -------
    CompactnessDifference
        Per-category difference together with its ranking.

    Raises
    ------
    ValueError
        If ``difference_measure`` is not recognised.
    """
    if difference_measure == "rank":
        difference = _rank_difference(
            compactness, category_names, models
        )
    elif difference_measure == "normalizedDiff":
        difference = _normalized_difference(compactness, models)
    else:
        raise ValueError(
            f"Unknown difference measure: '{difference_measure}'. "
            f"Must be one of {VALID_DIFFERENCE_MEASURES}."
        )

    order = np.argsort(-np.absolute(difference))
    return CompactnessDifference(
        difference=difference,
        order=order,
        categories=np.array(list(category_names))[order],
        measure=difference_measure,
    )


def combined_category_score(
    correlations: np.ndarray, difference: np.ndarray
) -> np.ndarray:
    """Combine the relational and compactness criteria.

    A category scores highly when the models compact it differently
    and place it differently among the other categories, so the
    absolute compactness difference is rewarded and the relational
    correlation penalised.

    Parameters
    ----------
    correlations : numpy.ndarray
        Relational criterion per category.
    difference : numpy.ndarray
        Compactness difference per category, in category order.

    Returns
    -------
    numpy.ndarray
        One score per category.

    Raises
    ------
    ValueError
        If the two inputs do not have the same length.
    """
    if len(correlations) != len(difference):
        raise ValueError(
            f"correlations and difference must have the same "
            f"length, got {len(correlations)} and "
            f"{len(difference)}."
        )
    return np.absolute(difference) - correlations


def rank_categories(score: np.ndarray, n_categories=None):
    """Order categories by descending combined score.

    Parameters
    ----------
    score : numpy.ndarray
        Combined score per category.
    n_categories : int or None
        How many indices to return.  All of them when None.

    Returns
    -------
    numpy.ndarray
        Category indices, best first.

    Raises
    ------
    ValueError
        If more categories are requested than there are scores.
    """
    if n_categories is not None and n_categories > len(score):
        raise ValueError(
            f"Requested {n_categories} categories but only "
            f"{len(score)} categories are available."
        )
    order = np.argsort(-score)
    if n_categories is None:
        return order
    return order[:n_categories]


def screen_categories(
    cat_activations: Dict[str, np.ndarray],
    models: Tuple[str, str],
    compactness: CompactnessResult,
    category_names: Sequence[str],
    n_categories: int = 12,
    dissimilarity_metric: str = "L2squared",
    difference_measure: str = "normalizedDiff",
) -> CategorySelection:
    """Select the categories that most divide the two models.

    Parameters
    ----------
    cat_activations : dict
        Mapping from model name to activations grouped by category.
    models : tuple of str
        The two model names.
    compactness : CompactnessResult
        Compactness of every category under both models.
    category_names : sequence of str
        Category labels in category order.
    n_categories : int
        How many categories to keep.
    dissimilarity_metric : str
        Metric used for the category-level dissimilarity matrices.
    difference_measure : str
        How the two models' compactness is contrasted.

    Returns
    -------
    CategorySelection
        Both criteria, the combined score, and the selection.
    """
    correlations = category_rdm_correlations(
        cat_activations,
        models,
        dissimilarity_metric=dissimilarity_metric,
    )
    difference = compactness_difference(
        compactness,
        category_names,
        models,
        difference_measure=difference_measure,
    )
    score = combined_category_score(
        correlations, difference.difference
    )
    selected = rank_categories(score, n_categories=n_categories)

    return CategorySelection(
        correlations=correlations,
        difference=difference,
        score=score,
        selected=selected,
        selected_names=[
            list(category_names)[index] for index in selected
        ],
    )
