"""Linear decompositions and shape alignment of activation spaces."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PrincipalComponents:
    """Principal component decomposition of a data matrix.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Array of shape ``(n_variables, n_variables)``, each column
        holding the coefficients of one component, ordered by
        descending eigenvalue.
    scores : numpy.ndarray
        Array of shape ``(n_variables, n_observations)``: the
        centred data projected onto the components, one row per
        component.
    eigenvalues : numpy.ndarray
        Eigenvalues of the covariance matrix, descending.
    explained : numpy.ndarray
        Percentage of total variance carried by each component.
    """

    coefficients: np.ndarray
    scores: np.ndarray
    eigenvalues: np.ndarray
    explained: np.ndarray


def principal_components(data: np.ndarray) -> PrincipalComponents:
    """Decompose a data matrix into its principal components.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_observations, n_variables)``, with
        observations in rows and variables in columns.

    Returns
    -------
    PrincipalComponents
        Coefficients, scores, eigenvalues and explained variance,
        ordered by descending eigenvalue.

    Raises
    ------
    ValueError
        If ``data`` is not two-dimensional.
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be two-dimensional (n_observations, "
            f"n_variables), got shape {data.shape}."
        )

    centred = (data - data.mean(axis=0)).T
    eigenvalues, coefficients = np.linalg.eigh(np.cov(centred))

    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    coefficients = coefficients[:, order]

    return PrincipalComponents(
        coefficients=coefficients,
        scores=np.dot(coefficients.T, centred),
        eigenvalues=eigenvalues,
        explained=100.0 * eigenvalues / np.sum(eigenvalues),
    )


def classical_mds(
    distances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed a distance matrix by classical multidimensional scaling.

    Only the dimensions with a positive eigenvalue are returned, so
    a matrix that is not exactly Euclidean yields the embedding of
    its Euclidean part.  Each dimension is determined up to an
    overall sign.

    Parameters
    ----------
    distances : numpy.ndarray
        Square, symmetric distance matrix.

    Returns
    -------
    embedding : numpy.ndarray
        Array of shape ``(n_items, n_positive)``, one column per
        retained dimension.
    eigenvalues : numpy.ndarray
        All eigenvalues of the doubly centred matrix, descending.

    Raises
    ------
    ValueError
        If ``distances`` is not square.
    """
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(
            f"A distance matrix must be square, got shape "
            f"{distances.shape}."
        )

    size = len(distances)
    projector = np.eye(size) - np.ones((size, size)) / size
    centred = -projector.dot(distances**2).dot(projector) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(centred)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = np.where(eigenvalues > 0)[0]
    embedding = eigenvectors[:, positive].dot(
        np.diag(np.sqrt(eigenvalues[positive]))
    )
    return embedding, eigenvalues


@dataclass
class _Centred:
    """A configuration moved to the origin and scaled to unit norm.

    Parameters
    ----------
    points : numpy.ndarray
        The centred, unit-norm points.
    mean : numpy.ndarray
        The centroid that was subtracted.
    norm : float
        The centred Frobenius norm that was divided out.
    scatter : float
        The sum of squared deviations from the centroid.
    """

    points: np.ndarray
    mean: np.ndarray
    norm: float
    scatter: float


def _pad_dimensions(points, n_dimensions):
    """Zero-pad ``points`` out to ``n_dimensions`` columns."""
    missing = n_dimensions - points.shape[1]
    if missing <= 0:
        return points
    return np.concatenate(
        (points, np.zeros((len(points), missing))), axis=1
    )


def _centre(points, n_dimensions=None):
    """Centre and scale a configuration, optionally zero-padding it."""
    mean = points.mean(0)
    centred = points - mean
    scatter = float((centred**2.0).sum())
    norm = np.sqrt(scatter)
    centred = centred / norm
    if n_dimensions is not None:
        centred = _pad_dimensions(centred, n_dimensions)
    return _Centred(centred, mean, norm, scatter)


def _optimal_rotation(target, moved, reflection):
    """Return the aligning rotation and its singular values.

    ``reflection`` of ``"best"`` accepts whichever handedness fits
    better; True or False flips the last axis when the better fit
    disagrees with what was asked for.
    """
    left, singular, right_t = np.linalg.svd(
        np.dot(target.T, moved), full_matrices=False
    )
    right = right_t.T
    rotation = np.dot(right, left.T)

    if reflection == "best":
        return rotation, singular
    if bool(reflection) == bool(np.linalg.det(rotation) < 0):
        return rotation, singular

    right = right.copy()
    singular = singular.copy()
    right[:, -1] *= -1
    singular[-1] *= -1
    return np.dot(right, left.T), singular


def _fit(target, moved, rotation, trace, scaling):
    """Return the scale, disparity and transformed points of a fit."""
    rotated = np.dot(moved.points, rotation)
    if scaling:
        return (
            trace * target.norm / moved.norm,
            1 - trace**2,
            target.norm * trace * rotated + target.mean,
        )
    return (
        1,
        1
        + moved.scatter / target.scatter
        - 2 * trace * moved.norm / target.norm,
        moved.norm * rotated + target.mean,
    )


def _validate_procrustes(target, moved):
    """Raise when two configurations cannot be aligned."""
    if target.shape[0] != moved.shape[0]:
        raise ValueError(
            f"Both configurations must hold the same number of "
            f"points, got {target.shape[0]} and {moved.shape[0]}."
        )
    if moved.shape[1] > target.shape[1]:
        raise ValueError(
            f"moved must not have more dimensions than target, got "
            f"{moved.shape[1]} and {target.shape[1]}."
        )


def procrustes(
    target: np.ndarray,
    moved: np.ndarray,
    scaling: bool = True,
    reflection="best",
):
    """Align one configuration of points to another.

    Finds the translation, rotation, optional reflection and
    optional uniform scaling of ``moved`` that minimises the sum of
    squared distances to ``target``.  Follows MATLAB's
    ``procrustes``.

    Parameters
    ----------
    target : numpy.ndarray
        Array of shape ``(n_points, n_dimensions)``.
    moved : numpy.ndarray
        Array of shape ``(n_points, n_dimensions)`` with the same
        number of points as ``target`` and no more dimensions.
        Fewer dimensions are zero-padded.
    scaling : bool
        Whether the transformation may scale ``moved``.  The scale
        is held at 1 when False.
    reflection : {'best'} or bool
        ``"best"`` lets the fit decide whether to reflect; True or
        False forces a solution with or without a reflection.

    Returns
    -------
    disparity : float
        Residual sum of squared errors, normalised by the scale of
        ``target``, so it does not depend on the units of either
        configuration.
    transformed : numpy.ndarray
        ``moved`` after the transformation, of the same shape as
        ``target``.
    transform : dict
        The ``rotation``, ``scale`` and ``translation`` taking
        ``moved`` to ``transformed``.

    Raises
    ------
    ValueError
        If the two configurations hold different numbers of points,
        or ``moved`` has more dimensions than ``target``.
    """
    _validate_procrustes(target, moved)
    n_dimensions = target.shape[1]
    n_moved_dimensions = moved.shape[1]

    centred_target = _centre(target)
    centred_moved = _centre(moved, n_dimensions)

    rotation, singular = _optimal_rotation(
        centred_target.points, centred_moved.points, reflection
    )
    trace = singular.sum()
    scale, disparity, transformed = _fit(
        centred_target, centred_moved, rotation, trace, scaling
    )

    if n_moved_dimensions < n_dimensions:
        rotation = rotation[:n_moved_dimensions, :]

    return (
        disparity,
        transformed,
        {
            "rotation": rotation,
            "scale": scale,
            "translation": centred_target.mean
            - scale * np.dot(centred_moved.mean, rotation),
        },
    )
