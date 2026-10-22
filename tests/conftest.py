"""Shared test fixtures for repscreen."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

N_CATEGORIES = 6
N_PER_CATEGORY = 5
N_FEATURES = 8
MODELS = ("model_a", "model_b")


def _category_structured(seed, separation):
    """Build activations with per-category offsets.

    Parameters
    ----------
    seed : int
        Seed for the random generator.
    separation : float
        Scale of the per-category centroid offset relative to the
        within-category noise.  Larger values make categories more
        compact.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N_CATEGORIES, N_PER_CATEGORY, N_FEATURES)``.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(N_CATEGORIES, 1, N_FEATURES))
    noise = rng.normal(size=(N_CATEGORIES, N_PER_CATEGORY, N_FEATURES))
    return separation * centroids + noise


@pytest.fixture
def category_names():
    """Category labels matching the activation fixtures."""
    return [f"{i:04d}_cat{i}" for i in range(N_CATEGORIES)]


@pytest.fixture
def flat_activations():
    """Per-model activations of shape ``(n_images, n_features)``."""
    return {
        MODELS[0]: _category_structured(0, 3.0).reshape(-1, N_FEATURES),
        MODELS[1]: _category_structured(1, 0.5).reshape(-1, N_FEATURES),
    }


@pytest.fixture
def cat_activations():
    """Per-model activations grouped by category.

    Each value has shape
    ``(N_CATEGORIES, N_PER_CATEGORY, N_FEATURES)``.  The two models
    differ in how compact their categories are, so compactness
    differences between them are non-zero.
    """
    return {
        MODELS[0]: _category_structured(0, 3.0),
        MODELS[1]: _category_structured(1, 0.5),
    }


@pytest.fixture
def image_paths(category_names):
    """Image paths ordered to match the activation fixtures."""
    return [
        f"/dataset/{name}/{name}_{j:03d}.jpg"
        for name in category_names
        for j in range(N_PER_CATEGORY)
    ]


@pytest.fixture
def activation_tree(tmp_path, flat_activations, image_paths):
    """Write an on-disk activation directory per model.

    Each model directory holds ``activations.npy`` and
    ``imagepaths.txt``, the layout the screening pipeline reads.

    Returns
    -------
    pathlib.Path
        Root directory holding one subdirectory per model.
    """
    root = tmp_path / "activations"
    for model, array in flat_activations.items():
        model_dir = root / model
        model_dir.mkdir(parents=True)
        np.save(model_dir / "activations.npy", array)
        (model_dir / "imagepaths.txt").write_text(
            "\n".join(image_paths) + "\n"
        )
    return root


@pytest.fixture
def image_tree(tmp_path, category_names):
    """Write a category-structured directory of small RGB images.

    Returns
    -------
    pathlib.Path
        Dataset root with one subdirectory per category.
    """
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "dataset"
    rng = np.random.default_rng(2)
    for name in category_names:
        category_dir = root / name
        category_dir.mkdir(parents=True)
        for j in range(N_PER_CATEGORY):
            image = rng.integers(
                0, 256, size=(16, 16, 3), dtype=np.uint8
            )
            cv2.imwrite(str(category_dir / f"{name}_{j:03d}.jpg"), image)
    return root


@pytest.fixture
def rdm_pair():
    """A pair of symmetric dissimilarity matrices with zero diagonal."""
    rng = np.random.default_rng(3)
    first = rng.random((N_CATEGORIES, N_CATEGORIES))
    second = rng.random((N_CATEGORIES, N_CATEGORIES))
    out = []
    for matrix in (first, second):
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        out.append(matrix)
    return tuple(out)


@pytest.fixture
def screening_result(activation_tree):
    """Run the screening pipeline over the on-disk fixture tree."""
    from repscreen.config import (
        ActivationConfig,
        ScreeningConfig,
        SelectionConfig,
    )
    from repscreen.screening.pipeline import run_screening

    config = ScreeningConfig(
        activations=ActivationConfig(
            activation_dir=activation_tree, models=MODELS
        ),
        selection=SelectionConfig(n_categories=3, n_exemplars=2),
    )
    return run_screening(config)
