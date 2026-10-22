"""Loading of paired model activations and their category structure."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ActivationConfig

ACTIVATION_FILE = "activations.npy"
IMAGE_PATH_FILE = "imagepaths.txt"


@dataclass
class ActivationSet:
    """Activations of two models over one category-structured dataset.

    Parameters
    ----------
    activations : dict
        Mapping from model name to activations grouped by category,
        of shape ``(n_categories, n_per_category, n_features)``.
    image_paths : list of str
        Paths of the images, in the order they were presented.
    category_names : list of str
        Category labels, in the order they appear in the activations.
    n_per_category : int
        Number of exemplars per category.
    models : tuple of str
        The two model names.
    """

    activations: Dict[str, np.ndarray]
    image_paths: List[str]
    category_names: List[str]
    n_per_category: int
    models: Tuple[str, ...]


def load_image_paths(model_dir) -> List[str]:
    """Read the image list recorded alongside a set of activations.

    Parameters
    ----------
    model_dir : path-like
        Directory holding ``imagepaths.txt``.

    Returns
    -------
    list of str
        One path per presented image, in presentation order.

    Raises
    ------
    FileNotFoundError
        If ``imagepaths.txt`` is absent.
    """
    path = Path(model_dir) / IMAGE_PATH_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"No {IMAGE_PATH_FILE} in {model_dir}."
        )
    with open(path, "r") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_activations(model_dir) -> np.ndarray:
    """Read activations and flatten them to one vector per image.

    Parameters
    ----------
    model_dir : path-like
        Directory holding ``activations.npy``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_images, n_features)``.

    Raises
    ------
    FileNotFoundError
        If ``activations.npy`` is absent.
    """
    path = Path(model_dir) / ACTIVATION_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"No {ACTIVATION_FILE} in {model_dir}."
        )
    activations = np.load(path)
    return activations.reshape(activations.shape[0], -1)


def normalize_activations(activations: np.ndarray) -> np.ndarray:
    """Scale every activation vector to unit length.

    Unit-length vectors make the dissimilarity between two images
    depend on the direction of their activation patterns rather than
    on their magnitude.

    Parameters
    ----------
    activations : numpy.ndarray
        Array of shape ``(n_images, n_features)``.

    Returns
    -------
    numpy.ndarray
        Array of the same shape with unit-length rows.
    """
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    return activations / norms


def derive_categories(
    image_paths: Sequence[str],
) -> Tuple[List[str], int]:
    """Read the category structure off the image paths.

    The directory holding an image names its category, and the
    images of one category are expected to form a contiguous block
    of equal length.

    Parameters
    ----------
    image_paths : sequence of str
        Image paths in presentation order.

    Returns
    -------
    names : list of str
        Category labels in order of first appearance.
    n_per_category : int
        Number of exemplars per category.

    Raises
    ------
    ValueError
        If ``image_paths`` is empty, if a category's images are not
        contiguous, or if the categories differ in size.
    """
    if len(image_paths) == 0:
        raise ValueError(
            "Deriving categories needs at least one image path."
        )

    names: List[str] = []
    counts: List[int] = []
    for path in image_paths:
        category = path.split("/")[-2]
        if names and category == names[-1]:
            counts[-1] += 1
            continue
        if category in names:
            raise ValueError(
                f"Images of category '{category}' are not "
                f"contiguous in the image list."
            )
        names.append(category)
        counts.append(1)

    if len(set(counts)) != 1:
        sizes = {
            name: count for name, count in zip(names, counts)
        }
        raise ValueError(
            f"Every category must hold the same number of "
            f"exemplars; got sizes {sizes}."
        )

    return names, counts[0]


def group_by_category(
    activations: np.ndarray, n_per_category: int
) -> np.ndarray:
    """Split activations into one block per category.

    Parameters
    ----------
    activations : numpy.ndarray
        Array of shape ``(n_images, n_features)``.
    n_per_category : int
        Number of exemplars per category.

    Returns
    -------
    numpy.ndarray
        Array of shape
        ``(n_categories, n_per_category, n_features)``.

    Raises
    ------
    ValueError
        If the number of images is not a multiple of
        ``n_per_category``.
    """
    n_images = activations.shape[0]
    if n_images % n_per_category != 0:
        raise ValueError(
            f"{n_images} images is not divisible by "
            f"{n_per_category} exemplars per category."
        )
    return activations.reshape(-1, n_per_category, activations.shape[-1])


def resolve_image_paths(
    image_paths: Sequence[str],
    dataset_path: Optional[Path],
    replace_path_prefix: Optional[str],
) -> List[str]:
    """Point the recorded image paths at the local dataset copy.

    Parameters
    ----------
    image_paths : sequence of str
        Paths as recorded when the activations were computed.
    dataset_path : path-like or None
        Local dataset root.  The paths are returned unchanged when
        None.
    replace_path_prefix : str or None
        Prefix to strip before joining to ``dataset_path``.  When
        None, the path is taken apart at its category directory.

    Returns
    -------
    list of str
        Paths under ``dataset_path``.
    """
    if dataset_path is None:
        return list(image_paths)

    root = str(dataset_path).rstrip("/")
    resolved = []
    for path in image_paths:
        if replace_path_prefix is None:
            remainder = "/".join(path.split("/")[-2:])
        else:
            remainder = path.split(replace_path_prefix, 1)[-1]
        resolved.append(f"{root}/{remainder.lstrip('/')}")
    return resolved


def load_activation_pair(
    config: ActivationConfig,
) -> ActivationSet:
    """Load the two activation sets a screening run compares.

    Parameters
    ----------
    config : ActivationConfig
        Which activation sets to load and how to resolve the image
        paths.

    Returns
    -------
    ActivationSet
        Activations grouped by category, together with the image
        paths and category labels they share.

    Raises
    ------
    ValueError
        If the two models did not see the same images in the same
        order, or if a model has a different number of activations
        than images.
    """
    image_lists = {}
    flat = {}
    for model in config.models:
        model_dir = config.activation_dir / model
        image_lists[model] = load_image_paths(model_dir)
        flat[model] = load_activations(model_dir)

    first, second = config.models
    if image_lists[first] != image_lists[second]:
        raise ValueError(
            f"'{first}' and '{second}' must have seen the same "
            f"images in the same order."
        )

    image_paths = image_lists[first]
    for model in config.models:
        if flat[model].shape[0] != len(image_paths):
            raise ValueError(
                f"Model '{model}' needs one activation per image; "
                f"got {flat[model].shape[0]} activations for "
                f"{len(image_paths)} images."
            )

    category_names, n_per_category = derive_categories(image_paths)

    activations = {}
    for model in config.models:
        values = flat[model]
        if config.normalize:
            values = normalize_activations(values)
        activations[model] = group_by_category(
            values, n_per_category
        )

    return ActivationSet(
        activations=activations,
        image_paths=resolve_image_paths(
            image_paths,
            config.dataset_path,
            config.replace_path_prefix,
        ),
        category_names=category_names,
        n_per_category=n_per_category,
        models=config.models,
    )
