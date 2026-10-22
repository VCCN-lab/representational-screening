"""Chromaticity content of an image in CIELab."""

from typing import Tuple

import numpy as np
from skimage import color


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    """Return an RGB image scaled to the unit interval."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Needs an image with three colour channels, got shape "
            f"{image.shape}."
        )
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    return image.astype(np.float64)


def lab_samples(
    image: np.ndarray, n_samples: int = 10000, seed=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an image's pixels in CIELab coordinates.

    Sampling keeps the size of a category's pooled distribution
    independent of the images' resolution.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image of shape ``(height, width, 3)``, either ``uint8``
        or already scaled to the unit interval.
    n_samples : int
        How many pixels to draw.  Every pixel is returned when the
        image holds fewer.
    seed : int or None
        Seed for the random generator.

    Returns
    -------
    lightness : numpy.ndarray
        ``L*`` of the sampled pixels.
    chroma_a : numpy.ndarray
        ``a*`` of the sampled pixels, green to red.
    chroma_b : numpy.ndarray
        ``b*`` of the sampled pixels, blue to yellow.

    Raises
    ------
    ValueError
        If ``image`` does not have three colour channels.
    """
    lab = color.rgb2lab(_as_float_rgb(image)).reshape(-1, 3)

    if lab.shape[0] > n_samples:
        rng = np.random.default_rng(seed)
        drawn = rng.choice(
            lab.shape[0], size=n_samples, replace=False
        )
        lab = lab[drawn]

    return lab[:, 0], lab[:, 1], lab[:, 2]


def lab_to_rgb(
    lightness: np.ndarray,
    chroma_a: np.ndarray,
    chroma_b: np.ndarray,
) -> np.ndarray:
    """Convert CIELab coordinates to displayable RGB colours.

    Colours outside the RGB gamut are clipped to its boundary, so
    every input yields a drawable colour.

    Parameters
    ----------
    lightness, chroma_a, chroma_b : numpy.ndarray
        One-dimensional arrays of equal length.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_samples, 3)`` with entries in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If the three arrays do not have the same length.
    """
    lengths = {
        len(lightness), len(chroma_a), len(chroma_b)
    }
    if len(lengths) != 1:
        raise ValueError(
            f"lightness, chroma_a and chroma_b must have the same "
            f"length, got {sorted(lengths)}."
        )

    lab = np.stack([lightness, chroma_a, chroma_b], axis=1)
    rgb = color.lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(rgb, 0.0, 1.0)
