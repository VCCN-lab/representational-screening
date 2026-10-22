"""Figures for curated stimulus sets."""

import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_rgb_image(path):
    """Read one image from disk as an RGB array.

    Parameters
    ----------
    path : path-like
        Path of the image.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(height, width, 3)``, or None when the file
        is absent or cannot be decoded.

    Warns
    -----
    UserWarning
        If the file is absent or cannot be decoded.
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(
            f"Image not found: {path}", UserWarning, stacklevel=2
        )
        return None

    image = cv2.imread(str(path))
    if image is None:
        warnings.warn(
            f"Image could not be decoded: {path}",
            UserWarning,
            stacklevel=2,
        )
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_stimulus_images(
    paths: Sequence[str],
) -> Tuple[List[np.ndarray], List[str]]:
    """Read a list of images, skipping the ones that fail.

    Parameters
    ----------
    paths : sequence of str
        Image paths.

    Returns
    -------
    images : list of numpy.ndarray
        The images that could be read.
    valid_paths : list of str
        The paths those images came from, in the same order.
    """
    images = []
    valid_paths = []
    for path in paths:
        image = load_rgb_image(path)
        if image is None:
            continue
        images.append(image)
        valid_paths.append(str(path))
    return images, valid_paths


def plot_stimulus_grid(
    paths: Sequence[str],
    n_columns=None,
    title: str = "Curated stimulus set",
    figsize=(20, 7),
    label_first_row: bool = True,
):
    """Draw a curated stimulus set as a grid of images.

    Images are laid out column by column, so each column holds the
    exemplars of one category when the paths are grouped by
    category.

    Parameters
    ----------
    paths : sequence of str
        Image paths, grouped by category.
    n_columns : int or None
        Number of columns.  The number of distinct categories among
        the paths when None.
    title : str
        Figure title.
    figsize : tuple of float
        Figure size in inches.
    label_first_row : bool
        Whether to title the top panel of each column with its
        category.

    Returns
    -------
    matplotlib.figure.Figure
        The grid figure.

    Raises
    ------
    ValueError
        If ``paths`` is empty, or no image could be read.
    """
    if len(paths) == 0:
        raise ValueError(
            "Plotting a stimulus grid needs at least one image."
        )

    images, valid_paths = load_stimulus_images(paths)
    if len(images) == 0:
        raise ValueError(
            "None of the stimulus images could be read."
        )

    if n_columns is None:
        categories = []
        for path in valid_paths:
            category = path.split("/")[-2]
            if category not in categories:
                categories.append(category)
        n_columns = len(categories)
    n_rows = int(np.ceil(len(images) / n_columns))

    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=figsize, squeeze=False
    )
    figure.suptitle(title)

    for panel in range(n_rows * n_columns):
        row, column = panel % n_rows, panel // n_rows
        ax = axes[row][column]
        ax.axis("off")
        if panel >= len(images):
            continue
        ax.imshow(images[panel])
        if row == 0 and label_first_row:
            ax.set_title(valid_paths[panel].split("/")[-2])

    figure.tight_layout()
    return figure
