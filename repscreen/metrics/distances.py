"""Pairwise dissimilarity measures over activation vectors."""

import warnings

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def _unit_rows(vectors):
    """Scale every row of ``vectors`` to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def cosine_similarity(vector1, vector2):
    """Compute the cosine similarity between two vectors.

    Parameters
    ----------
    vector1, vector2 : numpy.ndarray
        One-dimensional vectors of equal shape.  When comparing
        dissimilarity matrices these hold the flattened upper
        triangle, optionally centred.

    Returns
    -------
    float
        Cosine similarity, or ``0.0`` when either vector has zero
        norm.

    Raises
    ------
    ValueError
        If the two vectors do not have the same shape.

    Warns
    -----
    UserWarning
        If either vector has zero norm, leaving the similarity
        undefined.
    """
    if vector1.shape != vector2.shape:
        raise ValueError(
            f"Inputs must have the same shape, got {vector1.shape} "
            f"and {vector2.shape}."
        )

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        warnings.warn(
            "Cosine similarity is undefined for a vector of zero "
            "norm; returning 0.0.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0

    return float(np.dot(vector1, vector2) / (norm1 * norm2))


def dotproduct_dissimilarity(vectors, normalize=False):
    """Compute one minus the dot-product similarity matrix.

    Parameters
    ----------
    vectors : numpy.ndarray
        Array of shape ``(n_samples, n_features)``.
    normalize : bool
        Whether to scale every row to unit length first, which turns
        the dot product into a cosine similarity.

    Returns
    -------
    numpy.ndarray
        Dissimilarity matrix of shape ``(n_samples, n_samples)``.
    """
    if normalize:
        vectors = _unit_rows(vectors)
    return 1.0 - np.dot(vectors, vectors.T)


def l2_dissimilarity(vectors, squared=False, normalize=False):
    """Compute the matrix of pairwise Euclidean distances.

    Parameters
    ----------
    vectors : numpy.ndarray
        Array of shape ``(n_samples, n_features)``.
    squared : bool
        Whether to return squared distances.
    normalize : bool
        Whether to scale every row to unit length first.

    Returns
    -------
    numpy.ndarray
        Dissimilarity matrix of shape ``(n_samples, n_samples)``.
    """
    if normalize:
        vectors = _unit_rows(vectors)
    return euclidean_distances(vectors, squared=squared)
