"""Spatial-frequency and chromaticity signatures per category."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .chromaticity import lab_samples
from .images import IMAGE_EXTENSIONS, load_category_images
from .spectra import fourier_spectrum


@dataclass
class CategorySignature:
    """Image statistics pooled over the exemplars of one category.

    Parameters
    ----------
    category : str
        Name of the category.
    spectrum : numpy.ndarray
        Mean centred Fourier transform over the exemplars.
    lightness : numpy.ndarray
        ``L*`` of the pixels sampled across the exemplars.
    chroma_a : numpy.ndarray
        ``a*`` of those pixels.
    chroma_b : numpy.ndarray
        ``b*`` of those pixels.
    n_images : int
        How many exemplars contributed.
    """

    category: str
    spectrum: np.ndarray
    lightness: np.ndarray
    chroma_a: np.ndarray
    chroma_b: np.ndarray
    n_images: int


def category_signature(
    category_dir,
    n_exemplars=None,
    target_size: int = 448,
    n_samples: int = 10000,
    seed=None,
    extensions=IMAGE_EXTENSIONS,
) -> CategorySignature:
    """Pool the image statistics of one category.

    Every exemplar is framed to the same square so the spectra share
    a shape and can be averaged, and its pixels are sampled in
    CIELab and pooled across exemplars.

    Parameters
    ----------
    category_dir : path-like
        Directory holding the images of one category.
    n_exemplars : int or None
        How many exemplars to read.  All of them when None.
    target_size : int
        Side length every exemplar is framed to.
    n_samples : int
        How many pixels to sample per exemplar.
    seed : int or None
        Seed for the pixel sampling.
    extensions : sequence of str
        Glob patterns for the image files to pick up.

    Returns
    -------
    CategorySignature
        The pooled statistics of the category.

    Raises
    ------
    ValueError
        If no exemplar of the category could be read.
    """
    category_dir = Path(category_dir)
    images, _ = load_category_images(
        category_dir,
        n_exemplars=n_exemplars,
        target_size=target_size,
        extensions=extensions,
    )
    if not images:
        raise ValueError(
            f"No image of category '{category_dir.name}' could be "
            f"read."
        )

    spectra = []
    lightness: List[np.ndarray] = []
    chroma_a: List[np.ndarray] = []
    chroma_b: List[np.ndarray] = []
    for image in images:
        spectra.append(fourier_spectrum(image))
        sampled = lab_samples(
            image, n_samples=n_samples, seed=seed
        )
        lightness.append(sampled[0])
        chroma_a.append(sampled[1])
        chroma_b.append(sampled[2])

    return CategorySignature(
        category=category_dir.name,
        spectrum=np.mean(spectra, axis=0),
        lightness=np.concatenate(lightness),
        chroma_a=np.concatenate(chroma_a),
        chroma_b=np.concatenate(chroma_b),
        n_images=len(images),
    )


def dataset_signatures(
    dataset_path,
    n_categories=None,
    n_exemplars=None,
    target_size: int = 448,
    n_samples: int = 10000,
    seed=None,
    extensions=IMAGE_EXTENSIONS,
    progress: bool = False,
) -> Dict[str, CategorySignature]:
    """Pool the image statistics of every category of a dataset.

    Parameters
    ----------
    dataset_path : path-like
        Dataset root holding one subdirectory per category.
    n_categories : int or None
        How many categories to cover, in sorted name order.  All of
        them when None.
    n_exemplars : int or None
        How many exemplars to read per category.
    target_size : int
        Side length every exemplar is framed to.
    n_samples : int
        How many pixels to sample per exemplar.
    seed : int or None
        Seed for the pixel sampling.
    extensions : sequence of str
        Glob patterns for the image files to pick up.
    progress : bool
        Whether to show a progress bar over the categories.

    Returns
    -------
    dict
        Mapping from category name to its
        :class:`CategorySignature`.

    Raises
    ------
    NotADirectoryError
        If ``dataset_path`` is not a directory.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise NotADirectoryError(
            f"Dataset root is absent: {dataset_path}"
        )

    directories = sorted(
        entry for entry in dataset_path.iterdir() if entry.is_dir()
    )
    if n_categories is not None:
        directories = directories[:n_categories]

    signatures = {}
    for directory in tqdm(directories, disable=not progress):
        signatures[directory.name] = category_signature(
            directory,
            n_exemplars=n_exemplars,
            target_size=target_size,
            n_samples=n_samples,
            seed=seed,
            extensions=extensions,
        )
    return signatures


def save_signatures(signatures, output_path) -> Path:
    """Write computed signatures to disk.

    Parameters
    ----------
    signatures : dict
        Mapping from category name to :class:`CategorySignature`.
    output_path : path-like
        File the signatures are written to.

    Returns
    -------
    pathlib.Path
        Path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(signatures, handle)
    return output_path


def load_signatures(input_path) -> Dict[str, CategorySignature]:
    """Read signatures written by :func:`save_signatures`.

    Parameters
    ----------
    input_path : path-like
        File to read.

    Returns
    -------
    dict
        Mapping from category name to :class:`CategorySignature`.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Signature file is absent: {input_path}"
        )
    with open(input_path, "rb") as handle:
        return pickle.load(handle)
