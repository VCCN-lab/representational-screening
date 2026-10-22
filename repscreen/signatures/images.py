"""Reading and framing the images of one category."""

import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPEG")


def resize_and_center_crop(image: np.ndarray, target_size: int = 448):
    """Scale an image to a square of ``target_size`` by centre crop.

    The shorter side is scaled to ``target_size`` and the longer side
    cropped symmetrically, so the aspect ratio of the retained
    content is preserved.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(height, width)`` or
        ``(height, width, channels)``.
    target_size : int
        Side length of the output square.

    Returns
    -------
    numpy.ndarray
        Image of shape ``(target_size, target_size)`` or
        ``(target_size, target_size, channels)``.

    Raises
    ------
    ValueError
        If ``target_size`` is not strictly positive.
    """
    if target_size <= 0:
        raise ValueError(
            f"target_size must be positive, got {target_size}."
        )

    height, width = image.shape[:2]
    scale = target_size / min(height, width)
    new_height = max(int(round(height * scale)), target_size)
    new_width = max(int(round(width * scale)), target_size)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )
    top = (new_height - target_size) // 2
    left = (new_width - target_size) // 2
    return resized[
        top:top + target_size, left:left + target_size
    ]


def _category_image_paths(
    category_dir: Path, extensions: Sequence[str]
) -> List[Path]:
    """List the image files of one category directory."""
    paths: List[Path] = []
    for pattern in extensions:
        paths.extend(category_dir.glob(pattern))
    return sorted(set(paths))


def load_category_images(
    category_dir,
    n_exemplars=None,
    target_size=None,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> Tuple[List[np.ndarray], List[str]]:
    """Read the images of one category as RGB arrays.

    Parameters
    ----------
    category_dir : path-like
        Directory holding the images of one category.
    n_exemplars : int or None
        How many images to read, in sorted path order.  All of them
        when None.
    target_size : int or None
        Side length every image is framed to with
        :func:`resize_and_center_crop`.  Images are left at their
        own size when None.
    extensions : sequence of str
        Glob patterns for the image files to pick up.

    Returns
    -------
    images : list of numpy.ndarray
        The images that could be read, in path order.
    paths : list of str
        The paths those images came from.

    Raises
    ------
    NotADirectoryError
        If ``category_dir`` is not a directory.

    Warns
    -----
    UserWarning
        If the directory holds no images, or an image cannot be
        decoded.
    """
    category_dir = Path(category_dir)
    if not category_dir.is_dir():
        raise NotADirectoryError(
            f"Category directory is absent: {category_dir}"
        )

    paths = _category_image_paths(category_dir, extensions)
    if n_exemplars is not None:
        paths = paths[:n_exemplars]
    if not paths:
        warnings.warn(
            f"Directory holds no images: {category_dir}",
            UserWarning,
            stacklevel=2,
        )
        return [], []

    images: List[np.ndarray] = []
    valid_paths: List[str] = []
    for path in paths:
        raw = cv2.imread(str(path))
        if raw is None:
            warnings.warn(
                f"Image could not be decoded: {path}",
                UserWarning,
                stacklevel=2,
            )
            continue
        if target_size is not None:
            raw = resize_and_center_crop(raw, target_size)
        images.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        valid_paths.append(str(path))
    return images, valid_paths
