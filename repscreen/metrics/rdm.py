"""Representational dissimilarity matrices and their comparison."""

import numpy as np
from scipy.stats import spearmanr

from .cka import center_matrix
from .distances import (
    cosine_similarity,
    dotproduct_dissimilarity,
    l2_dissimilarity,
)


def compute_rdm(activations, metric="L2squared"):
    """Build a representational dissimilarity matrix.

    Parameters
    ----------
    activations : numpy.ndarray
        Array of shape ``(n_samples, n_features)``.
    metric : str
        Dissimilarity metric.  ``"L2squared"`` and ``"L2"`` give
        squared and plain Euclidean distances, ``"pearson"`` gives
        one minus the correlation, and ``"dotproduct"`` one minus
        the dot product.  Appending ``"_normalize"`` scales every
        activation vector to unit length first.

    Returns
    -------
    numpy.ndarray
        Dissimilarity matrix of shape ``(n_samples, n_samples)``.

    Raises
    ------
    ValueError
        If ``activations`` is not two-dimensional, or if ``metric``
        names no known dissimilarity.
    """
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be two-dimensional "
            f"(n_samples, n_features), got shape "
            f"{activations.shape}."
        )

    if "normalize" in metric:
        norms = np.linalg.norm(activations, axis=1, keepdims=True)
        activations = activations / norms

    if "pearson" in metric:
        return 1.0 - np.corrcoef(activations)
    if "L2squared" in metric:
        return l2_dissimilarity(activations, squared=True)
    if "L2" in metric:
        return l2_dissimilarity(activations)
    if "dotproduct" in metric:
        return dotproduct_dissimilarity(activations)

    raise ValueError(
        f"Unknown dissimilarity metric: '{metric}'. Must name one "
        f"of L2squared, L2, pearson, or dotproduct."
    )


def _upper_triangle(matrix):
    """Return the strict upper triangle of ``matrix`` as a vector."""
    indices = np.triu_indices(len(matrix), k=1)
    return matrix[indices]


def compare_rdms(rdm1, rdm2, metric="cosine", center=False):
    """Compare two dissimilarity matrices.

    Only the strict upper triangle is used, so the zero diagonal and
    the duplicated lower triangle do not inflate the similarity.

    Parameters
    ----------
    rdm1, rdm2 : numpy.ndarray
        Square dissimilarity matrices of equal shape.
    metric : str
        ``"cosine"``, ``"pearson"``, or ``"spearman"``.
    center : bool
        Whether to row- and column-centre both matrices first, as
        done in centred kernel alignment.

    Returns
    -------
    float
        Similarity between the two matrices.

    Raises
    ------
    ValueError
        If the matrices differ in shape, or if ``metric`` names no
        known similarity.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError(
            f"Dissimilarity matrices must have the same shape, got "
            f"{rdm1.shape} and {rdm2.shape}."
        )

    if center:
        rdm1 = center_matrix(rdm1)
        rdm2 = center_matrix(rdm2)

    upper1 = _upper_triangle(rdm1)
    upper2 = _upper_triangle(rdm2)

    if metric == "cosine":
        return cosine_similarity(upper1, upper2)
    if metric == "pearson":
        return float(np.corrcoef(upper1, upper2)[0, 1])
    if metric == "spearman":
        return float(spearmanr(upper1, upper2).statistic)

    raise ValueError(
        f"Unknown similarity metric: '{metric}'. Must be one of "
        f"('cosine', 'pearson', 'spearman')."
    )


def compare_rdms_global(rdm1, rdm2, mean1, mean2, norm):
    """Correlate two dissimilarity matrices against outside statistics.

    The means and the normalising term are supplied rather than
    estimated from the matrices, so the correlation of a subset of
    entries can be expressed on the scale of the larger set those
    statistics were taken from.

    Parameters
    ----------
    rdm1, rdm2 : numpy.ndarray
        Square dissimilarity matrices of equal shape.
    mean1, mean2 : float
        Means to subtract from the entries of ``rdm1`` and ``rdm2``.
    norm : float
        Normalising term the covariance is divided by, typically the
        product of the two standard deviations over the larger set.

    Returns
    -------
    float
        Correlation on the scale set by ``mean1``, ``mean2`` and
        ``norm``.

    Raises
    ------
    ValueError
        If the matrices differ in shape.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError(
            f"Dissimilarity matrices must have the same shape, got "
            f"{rdm1.shape} and {rdm2.shape}."
        )

    centred1 = _upper_triangle(rdm1) - mean1
    centred2 = _upper_triangle(rdm2) - mean2
    return float(np.mean(centred1 * centred2) / norm)


def _off_diagonal_columns(matrix):
    """Return the columns of ``matrix`` with the diagonal removed."""
    size = len(matrix)
    stacked = np.array(
        [np.delete(matrix[i], i) for i in range(size)]
    )
    return stacked.transpose()


def column_correlations(rdm1, rdm2):
    """Correlate two dissimilarity matrices column by column.

    Each column of a dissimilarity matrix describes how one item
    relates to all the others.  Correlating the two models' columns
    gives a per-item score: a low value marks an item whose
    relational position differs between the models.

    The diagonal is dropped before the columns are centred by the
    grand mean of the off-diagonal entries and scaled to unit norm.

    Parameters
    ----------
    rdm1, rdm2 : numpy.ndarray
        Square dissimilarity matrices of equal shape.

    Returns
    -------
    numpy.ndarray
        One correlation per item, of shape ``(n_items,)``.

    Raises
    ------
    ValueError
        If the matrices differ in shape.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError(
            f"Dissimilarity matrices must have the same shape, got "
            f"{rdm1.shape} and {rdm2.shape}."
        )

    columns1 = _off_diagonal_columns(rdm1)
    columns2 = _off_diagonal_columns(rdm2)

    centred1 = columns1 - np.mean(columns1)
    centred2 = columns2 - np.mean(columns2)

    centred1 = centred1 / np.sqrt(np.sum(centred1**2, axis=0))
    centred2 = centred2 / np.sqrt(np.sum(centred2**2, axis=0))

    return np.sum(centred1 * centred2, axis=0)
