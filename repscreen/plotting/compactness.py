"""Figures for category compactness."""

import math
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_compactness(
    compactness: Dict[str, np.ndarray],
    models: Sequence[str],
    xlabel: str = "Categories",
    ylabel: str = "Compactness",
    ax=None,
    figsize=(6, 6),
):
    """Plot compactness per category, one line per model.

    Parameters
    ----------
    compactness : dict
        Mapping from model name to compactness per category.  Pass
        the sorted values to see each model's own ordering.
    models : sequence of str
        Model names to draw.
    xlabel, ylabel : str
        Axis labels.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  A new figure is created when None.
    figsize : tuple of float
        Figure size in inches, used when ``ax`` is None.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The figure the lines were drawn on.
    ax : matplotlib.axes.Axes
        The axes the lines were drawn on.
    """
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.get_figure()

    for model in models:
        ax.plot(compactness[model], label=model)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    figure.tight_layout()
    return figure, ax


def plot_compactness_per_model(
    compactness: Dict[str, np.ndarray],
    models: Sequence[str],
    xlabel: str = "Categories",
    ylabel: str = "Compactness",
):
    """Plot compactness per category in one panel per model.

    Parameters
    ----------
    compactness : dict
        Mapping from model name to compactness per category.
    models : sequence of str
        Model names to draw.
    xlabel, ylabel : str
        Axis labels.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per model.
    """
    columns = math.ceil(math.sqrt(len(models)))
    rows = math.ceil(len(models) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        sharex=True,
        sharey=True,
        figsize=(columns * 2 + 1, rows * 2 + 1),
        squeeze=False,
    )

    for panel, model in enumerate(models):
        ax = axes[panel // columns][panel % columns]
        ax.plot(compactness[model])
        ax.set_title(model)

    for panel in range(len(models), rows * columns):
        axes[panel // columns][panel % columns].remove()
    for ax in figure.axes:
        ax.set_xlabel(xlabel)
    for row in axes:
        if row[0] in figure.axes:
            row[0].set_ylabel(ylabel)

    figure.tight_layout()
    return figure


def plot_compactness_scatter(
    compactness: Dict[str, np.ndarray],
    models: Sequence[str],
    ax=None,
    figsize=(6, 5),
):
    """Plot one model's compactness against the other's.

    Parameters
    ----------
    compactness : dict
        Mapping from model name to compactness per category, in
        category order so the two models' values line up.
    models : sequence of str
        The two model names.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  A new figure is created when None.
    figsize : tuple of float
        Figure size in inches, used when ``ax`` is None.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The figure the scatter was drawn on.
    ax : matplotlib.axes.Axes
        The axes the scatter was drawn on, titled with the
        correlation between the two models.

    Raises
    ------
    ValueError
        If the two models do not report the same number of
        categories.
    """
    first, second = models
    if len(compactness[first]) != len(compactness[second]):
        raise ValueError(
            f"'{first}' and '{second}' must report the same length, "
            f"got {len(compactness[first])} and "
            f"{len(compactness[second])}."
        )

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.get_figure()

    ax.scatter(
        compactness[first], compactness[second], color="k", s=3
    )
    correlation = np.round(
        np.corrcoef(compactness[first], compactness[second])[0, 1], 2
    )
    ax.set_title(f"{first} vs {second}: {correlation}")
    ax.set_xlabel(f"Compactness {first}")
    ax.set_ylabel(f"Compactness {second}")
    figure.tight_layout()
    return figure, ax
