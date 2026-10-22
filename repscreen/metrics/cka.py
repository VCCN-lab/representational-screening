"""Centred kernel alignment between two sets of features.

Follows the formulation of Kornblith et al. (2019), "Similarity of
Neural Network Representations Revisited".
"""

import math

import numpy as np


def center_matrix(matrix):
    """Remove the row and column means of a square matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix.

    Returns
    -------
    numpy.ndarray
        Matrix with column means and then row means subtracted.
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    return centred - centred.mean(axis=1, keepdims=True)


def center_gram(matrix):
    """Double-centre a Gram matrix with the centring projector.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square Gram matrix.

    Returns
    -------
    numpy.ndarray
        ``H @ matrix @ H`` with ``H = I - 1 / n``.
    """
    size = matrix.shape[0]
    projector = np.eye(size) - np.ones((size, size)) / size
    return np.dot(np.dot(projector, matrix), projector)


def rbf_kernel(features, sigma=None):
    """Build a radial basis function kernel matrix.

    Parameters
    ----------
    features : numpy.ndarray
        Array of shape ``(n_samples, n_features)``.
    sigma : float or None
        Kernel width.  When None, the square root of the median
        non-zero squared distance is used.

    Returns
    -------
    numpy.ndarray
        Kernel matrix of shape ``(n_samples, n_samples)``.
    """
    gram = np.dot(features, features.T)
    squared = np.diag(gram) - gram
    squared = squared + squared.T
    if sigma is None:
        sigma = math.sqrt(np.median(squared[squared != 0]))
    return np.exp(squared * (-0.5 / (sigma * sigma)))


def linear_hsic(features1, features2):
    """Hilbert-Schmidt independence criterion with linear kernels.

    Parameters
    ----------
    features1, features2 : numpy.ndarray
        Arrays of shape ``(n_samples, n_features)`` sharing the same
        number of samples.

    Returns
    -------
    float
        Unnormalised dependence between the two feature sets.
    """
    gram1 = np.dot(features1, features1.T)
    gram2 = np.dot(features2, features2.T)
    return float(np.sum(center_matrix(gram1) * center_matrix(gram2)))


def kernel_hsic(features1, features2, sigma=None):
    """Hilbert-Schmidt independence criterion with RBF kernels.

    Parameters
    ----------
    features1, features2 : numpy.ndarray
        Arrays of shape ``(n_samples, n_features)`` sharing the same
        number of samples.
    sigma : float or None
        Kernel width passed to :func:`rbf_kernel`.

    Returns
    -------
    float
        Unnormalised dependence between the two feature sets.
    """
    kernel1 = center_matrix(rbf_kernel(features1, sigma))
    kernel2 = center_matrix(rbf_kernel(features2, sigma))
    return float(np.sum(kernel1 * kernel2))


def linear_cka(features1, features2):
    """Centred kernel alignment with linear kernels.

    Parameters
    ----------
    features1, features2 : numpy.ndarray
        Arrays of shape ``(n_samples, n_features)`` sharing the same
        number of samples.

    Returns
    -------
    float
        Alignment in ``[0, 1]``, invariant to isotropic scaling and
        orthogonal transformation of either feature set.
    """
    dependence = linear_hsic(features1, features2)
    scale1 = math.sqrt(linear_hsic(features1, features1))
    scale2 = math.sqrt(linear_hsic(features2, features2))
    return dependence / (scale1 * scale2)


def kernel_cka(features1, features2, sigma=None):
    """Centred kernel alignment with RBF kernels.

    Parameters
    ----------
    features1, features2 : numpy.ndarray
        Arrays of shape ``(n_samples, n_features)`` sharing the same
        number of samples.
    sigma : float or None
        Kernel width passed to :func:`rbf_kernel`.

    Returns
    -------
    float
        Alignment in ``[0, 1]``.
    """
    dependence = kernel_hsic(features1, features2, sigma)
    scale1 = math.sqrt(kernel_hsic(features1, features1, sigma))
    scale2 = math.sqrt(kernel_hsic(features2, features2, sigma))
    return dependence / (scale1 * scale2)
