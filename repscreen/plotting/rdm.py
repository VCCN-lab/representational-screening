"""Figures for dissimilarity matrices."""

import math
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_rdm(
    rdm: np.ndarray,
    title: str = "",
    ax=None,
    cmap: str = "Greys",
    vmin: float = 0.0,
    vmax=None,
    colorbar: bool = True,
):
    """Draw one dissimilarity matrix as a heatmap.

    Parameters
    ----------
    rdm : numpy.ndarray
        Square dissimilarity matrix.
    title : str
        Title of the panel.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  A new figure is created when None.
    cmap : str
        Matplotlib colormap name.
    vmin : float
        Lower end of the colour scale.
    vmax : float or None
        Upper end of the colour scale.  The matrix maximum when
        None.
    colorbar : bool
        Whether to draw a colorbar.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the heatmap was drawn on.

    Raises
    ------
    ValueError
        If ``rdm`` is not square.
    """
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError(
            f"A dissimilarity matrix must be square, got shape "
            f"{rdm.shape}."
        )
    if ax is None:
        _, ax = plt.subplots()
    sns.heatmap(
        rdm,
        annot=False,
        cmap=cmap,
        square=True,
        cbar=colorbar,
        cbar_kws={"label": "Dissimilarity"} if colorbar else None,
        linewidths=0,
        ax=ax,
        vmin=vmin,
        vmax=np.max(rdm) if vmax is None else vmax,
    )
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_rdm_pair(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    model_names: Sequence[str],
    similarity=None,
    cmap: str = "coolwarm",
    figsize=(8, 4),
):
    """Draw two models' dissimilarity matrices side by side.

    Parameters
    ----------
    rdm1, rdm2 : numpy.ndarray
        Square dissimilarity matrices of equal shape.
    model_names : sequence of str
        The two model names, used as panel titles.
    similarity : float or None
        Similarity between the two matrices, reported in the figure
        title when given.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple of float
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.

    Raises
    ------
    ValueError
        If the matrices do not have the same shape.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError(
            f"Both matrices must have the same shape, got "
            f"{rdm1.shape} and {rdm2.shape}."
        )

    figure, axes = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=figsize
    )
    for panel, (matrix, name) in enumerate(
        zip((rdm1, rdm2), model_names)
    ):
        axes[panel].imshow(matrix, cmap=cmap)
        axes[panel].set_title(name)
        axes[panel].axis("off")

    title = "Representational dissimilarity"
    if similarity is not None:
        title = f"{title}\nsimilarity = {np.round(similarity, 3)}"
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def _grid_shape(n_panels: int):
    """Choose a near-square subplot grid for ``n_panels`` panels."""
    columns = math.ceil(math.sqrt(n_panels))
    rows = math.ceil(n_panels / columns)
    return rows, columns


def plot_layer_similarities(
    similarities: Dict[str, Dict[str, Sequence[float]]],
    models: Sequence[str],
):
    """Plot model-to-model similarity as a function of layer depth.

    Parameters
    ----------
    similarities : dict
        ``similarities[model1][model2]`` holds one value per layer,
        as returned by
        :func:`~repscreen.metrics.layers.compare_rdms_per_layer`.
    models : sequence of str
        Model names, in the order they were compared.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per model pair.
    """
    pairs = [
        (model1, model2)
        for i, model1 in enumerate(models)
        for model2 in models[i + 1:]
    ]
    rows, columns = _grid_shape(max(len(pairs), 1))
    figure, axes = plt.subplots(
        rows,
        columns,
        sharex=True,
        sharey=True,
        figsize=(columns * 2 + 1, rows * 2 + 1),
        squeeze=False,
    )

    for panel, (model1, model2) in enumerate(pairs):
        ax = axes[panel // columns][panel % columns]
        ax.plot(similarities[model1][model2])
        ax.set_title(f"{model1}_{model2}")

    for panel in range(len(pairs), rows * columns):
        axes[panel // columns][panel % columns].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Layer")
    for row in axes:
        row[0].set_ylabel("Correlation")

    figure.tight_layout()
    return figure
