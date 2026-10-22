"""Spatial-frequency content of an image."""

import cv2
import numpy as np


def fourier_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute the centred Fourier transform of an image.

    The image is reduced to luminance and transformed with the
    origin at the centre of the output, so low spatial frequencies
    sit in the middle and high frequencies at the edges.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(height, width)`` or
        ``(height, width, 3)``.

    Returns
    -------
    numpy.ndarray
        Complex spectrum of shape ``(height, width)``.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gray)))


def log_magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Take the logarithm of a spectrum's magnitude.

    The magnitude spans several orders of magnitude, so the
    logarithm is what makes the structure visible.  Zero entries are
    floored at the smallest positive double rather than mapped to
    negative infinity.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Complex or real spectrum.

    Returns
    -------
    numpy.ndarray
        Log magnitude, of the same shape.
    """
    magnitude = np.abs(spectrum)
    return np.log(
        np.maximum(magnitude, np.finfo(float).tiny)
    )


def downsample(values: np.ndarray, factor: int = 2) -> np.ndarray:
    """Shrink a two-dimensional array by area averaging.

    Parameters
    ----------
    values : numpy.ndarray
        Array of shape ``(height, width)``.
    factor : int
        Factor each side is divided by.

    Returns
    -------
    numpy.ndarray
        Array of shape
        ``(height // factor, width // factor)``.

    Raises
    ------
    ValueError
        If ``factor`` is smaller than one.
    """
    if factor < 1:
        raise ValueError(
            f"factor must be at least 1, got {factor}."
        )
    height, width = values.shape[:2]
    return cv2.resize(
        values.astype(np.float64),
        (width // factor, height // factor),
        interpolation=cv2.INTER_AREA,
    )
