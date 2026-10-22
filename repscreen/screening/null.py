"""Null distributions of model similarity over random subsets.

Drawing random subsets of the same size as a curated set shows how
much of the curated set's divergence between two models is specific
to the selection rather than to the subset size.
"""

from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from ..metrics.rdm import compare_rdms, compute_rdm


def _batch_bounds(n_samples: int, batch_size: int):
    """Yield the start and end index of every batch."""
    for start in range(0, n_samples, batch_size):
        yield start, min(start + batch_size, n_samples)


def _check_subset(subset_size: int, population: int, label: str):
    """Raise if a subset larger than the population is requested."""
    if subset_size > population:
        raise ValueError(
            f"{label} ({subset_size}) exceeds the {population} "
            f"available."
        )


def sample_rdm_similarity(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    n_samples: int = 1000,
    subset_size: int = 40,
    batch_size: int = 1000,
    seed=None,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Correlate two dissimilarity matrices over random subsets.

    Parameters
    ----------
    rdm1, rdm2 : numpy.ndarray
        Square dissimilarity matrices of equal shape.
    n_samples : int
        How many subsets to draw.
    subset_size : int
        How many items each subset holds.
    batch_size : int
        How many subsets to hold in memory at once.
    seed : int or None
        Seed for the random generator.
    progress : bool
        Whether to show a progress bar over the batches.

    Returns
    -------
    similarities : numpy.ndarray
        One correlation per subset, of shape ``(n_samples,)``.
    indices : numpy.ndarray
        The drawn indices, of shape ``(n_samples, subset_size)``.

    Raises
    ------
    ValueError
        If ``subset_size`` exceeds the size of the matrices.
    """
    _check_subset(subset_size, len(rdm1), "subset_size")
    rng = np.random.default_rng(seed)

    similarities = np.zeros(n_samples)
    indices = np.zeros((n_samples, subset_size), dtype=int)
    bounds = list(_batch_bounds(n_samples, batch_size))
    for start, end in tqdm(bounds, disable=not progress):
        for i in range(start, end):
            drawn = np.sort(
                rng.choice(len(rdm1), size=subset_size, replace=False)
            )
            similarities[i] = compare_rdms(
                rdm1[np.ix_(drawn, drawn)],
                rdm2[np.ix_(drawn, drawn)],
                metric="pearson",
            )
            indices[i] = drawn
    return similarities, indices


def sample_image_similarity(
    activations: Dict[str, np.ndarray],
    models: Tuple[str, str],
    n_samples: int = 1000,
    subset_size: int = 40,
    batch_size: int = 1000,
    seed=None,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compare two models over random subsets of images.

    Parameters
    ----------
    activations : dict
        Mapping from model name to activations of shape
        ``(n_images, n_features)``.
    models : tuple of str
        The two model names.
    n_samples : int
        How many subsets to draw.
    subset_size : int
        How many images each subset holds.
    batch_size : int
        How many subsets to hold in memory at once.
    seed : int or None
        Seed for the random generator.
    progress : bool
        Whether to show a progress bar over the batches.

    Returns
    -------
    similarities : numpy.ndarray
        One correlation per subset.
    indices : numpy.ndarray
        The drawn image indices.

    Raises
    ------
    ValueError
        If ``subset_size`` exceeds the number of images.
    """
    first, second = models
    n_images = activations[first].shape[0]
    _check_subset(subset_size, n_images, "subset_size")
    rng = np.random.default_rng(seed)

    similarities = np.zeros(n_samples)
    indices = np.zeros((n_samples, subset_size), dtype=int)
    bounds = list(_batch_bounds(n_samples, batch_size))
    for start, end in tqdm(bounds, disable=not progress):
        for i in range(start, end):
            drawn = np.sort(
                rng.choice(n_images, size=subset_size, replace=False)
            )
            similarities[i] = compare_rdms(
                compute_rdm(activations[first][drawn]),
                compute_rdm(activations[second][drawn]),
                metric="pearson",
            )
            indices[i] = drawn
    return similarities, indices


def sample_category_similarity(
    cat_activations: Dict[str, np.ndarray],
    models: Tuple[str, str],
    n_samples: int = 1000,
    n_categories: int = 12,
    batch_size: int = 1000,
    seed=None,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compare two models over random subsets of whole categories.

    Parameters
    ----------
    cat_activations : dict
        Mapping from model name to activations grouped by category.
    models : tuple of str
        The two model names.
    n_samples : int
        How many subsets to draw.
    n_categories : int
        How many categories each subset holds.
    batch_size : int
        How many subsets to hold in memory at once.
    seed : int or None
        Seed for the random generator.
    progress : bool
        Whether to show a progress bar over the batches.

    Returns
    -------
    similarities : numpy.ndarray
        One correlation per subset.
    indices : numpy.ndarray
        The drawn category indices.

    Raises
    ------
    ValueError
        If ``n_categories`` exceeds the number of categories.
    """
    first, second = models
    available = cat_activations[first].shape[0]
    _check_subset(n_categories, available, "n_categories")
    rng = np.random.default_rng(seed)

    similarities = np.zeros(n_samples)
    indices = np.zeros((n_samples, n_categories), dtype=int)
    bounds = list(_batch_bounds(n_samples, batch_size))
    for start, end in tqdm(bounds, disable=not progress):
        for i in range(start, end):
            drawn = rng.choice(
                available, size=n_categories, replace=False
            )
            rdms = []
            for model in (first, second):
                subset = cat_activations[model][drawn]
                rdms.append(
                    compute_rdm(subset.reshape(-1, subset.shape[-1]))
                )
            similarities[i] = compare_rdms(
                rdms[0], rdms[1], metric="pearson"
            )
            indices[i] = drawn
    return similarities, indices
