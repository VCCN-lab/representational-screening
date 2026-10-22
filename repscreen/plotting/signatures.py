"""Figures for category image signatures."""

import matplotlib.pyplot as plt
import numpy as np

from ..signatures.chromaticity import lab_to_rgb
from ..signatures.spectra import downsample, log_magnitude


def _chromaticity_panel(ax, signature, n_plot, seed):
    """Scatter a category's sampled pixels in the a*-b* plane."""
    size = len(signature.chroma_a)
    indices = np.arange(size)
    if size > n_plot:
        rng = np.random.default_rng(seed)
        indices = rng.choice(size, size=n_plot, replace=False)

    chroma_a = signature.chroma_a[indices]
    chroma_b = signature.chroma_b[indices]
    colours = lab_to_rgb(
        signature.lightness[indices], chroma_a, chroma_b
    )

    ax.scatter(
        chroma_a,
        chroma_b,
        c=colours,
        alpha=0.6,
        s=10,
        edgecolors="none",
    )
    ax.set_xlabel("a* (green to red)")
    ax.set_ylabel("b* (blue to yellow)")
    ax.set_title("CIELab chromaticity")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect("equal")


def plot_category_signature(
    signature,
    figsize=(8, 4),
    n_plot: int = 50000,
    seed=None,
    spectrum_factor: int = 2,
):
    """Draw a category's spectrum and chromaticity side by side.

    Parameters
    ----------
    signature : CategorySignature
        Signature to draw, as returned by
        :func:`~repscreen.signatures.analysis.category_signature`.
    figsize : tuple of float
        Figure size in inches.
    n_plot : int
        How many sampled pixels to scatter.
    seed : int or None
        Seed for that subsampling.
    spectrum_factor : int
        Factor the spectrum is shrunk by before drawing.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    figure, axes = plt.subplots(1, 2, figsize=figsize)

    spectrum = downsample(
        log_magnitude(signature.spectrum), factor=spectrum_factor
    )
    axes[0].imshow(spectrum, cmap="grey")
    axes[0].set_title("Fourier transform")
    axes[0].axis("off")

    _chromaticity_panel(axes[1], signature, n_plot, seed)

    figure.suptitle(signature.category)
    figure.tight_layout()
    return figure
