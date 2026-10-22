"""Tests for repscreen.data.loaders."""

import numpy as np
import pytest

from repscreen.config import ActivationConfig
from repscreen.data.loaders import (
    derive_categories,
    group_by_category,
    load_activation_pair,
    load_activations,
    load_image_paths,
    normalize_activations,
    resolve_image_paths,
)

from .conftest import N_CATEGORIES, N_FEATURES, N_PER_CATEGORY


def test_load_image_paths_strips_newlines(activation_tree):
    paths = load_image_paths(activation_tree / "model_a")
    assert len(paths) == N_CATEGORIES * N_PER_CATEGORY
    assert all(not path.endswith("\n") for path in paths)


def test_load_image_paths_requires_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="imagepaths.txt"):
        load_image_paths(tmp_path)


def test_load_activations_returns_two_dimensions(activation_tree):
    activations = load_activations(activation_tree / "model_a")
    assert activations.shape == (
        N_CATEGORIES * N_PER_CATEGORY,
        N_FEATURES,
    )


def test_load_activations_flattens_trailing_axes(tmp_path):
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()
    np.save(
        model_dir / "activations.npy", np.zeros((4, 3, 1, 1))
    )
    assert load_activations(model_dir).shape == (4, 3)


def test_load_activations_requires_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="activations.npy"):
        load_activations(tmp_path)


def test_normalize_activations_gives_unit_rows():
    activations = np.array([[3.0, 4.0], [0.0, 2.0]])
    result = normalize_activations(activations)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_normalize_activations_keeps_direction():
    activations = np.array([[3.0, 4.0]])
    result = normalize_activations(activations)
    assert np.allclose(result, [[0.6, 0.8]])


def test_derive_categories_reads_the_parent_directory(image_paths):
    names, n_per_category = derive_categories(image_paths)
    assert n_per_category == N_PER_CATEGORY
    assert len(names) == N_CATEGORIES
    assert names[0] == "0000_cat0"


def test_derive_categories_preserves_first_appearance_order():
    paths = [
        "/d/beta/b0.jpg",
        "/d/beta/b1.jpg",
        "/d/alpha/a0.jpg",
        "/d/alpha/a1.jpg",
    ]
    names, _ = derive_categories(paths)
    assert names == ["beta", "alpha"]


def test_derive_categories_rejects_unequal_category_sizes():
    paths = [
        "/d/alpha/a0.jpg",
        "/d/alpha/a1.jpg",
        "/d/beta/b0.jpg",
    ]
    with pytest.raises(ValueError, match="same number of exemplars"):
        derive_categories(paths)


def test_derive_categories_rejects_repeated_category_blocks():
    paths = [
        "/d/alpha/a0.jpg",
        "/d/beta/b0.jpg",
        "/d/alpha/a1.jpg",
        "/d/beta/b1.jpg",
    ]
    with pytest.raises(ValueError, match="contiguous"):
        derive_categories(paths)


def test_derive_categories_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one image"):
        derive_categories([])


def test_group_by_category_reshapes():
    activations = np.arange(24.0).reshape(12, 2)
    grouped = group_by_category(activations, 3)
    assert grouped.shape == (4, 3, 2)
    assert np.allclose(grouped[0], activations[:3])


def test_group_by_category_rejects_indivisible_counts():
    with pytest.raises(ValueError, match="not divisible"):
        group_by_category(np.zeros((10, 2)), 3)


def test_resolve_image_paths_without_dataset_path_is_identity(
    image_paths,
):
    assert resolve_image_paths(image_paths, None, None) == image_paths


def test_resolve_image_paths_replaces_a_prefix():
    paths = ["/old/root/cat/a.jpg"]
    result = resolve_image_paths(paths, "/new/root/", "/old/root/")
    assert result == ["/new/root/cat/a.jpg"]


def test_resolve_image_paths_joins_relative_remainder():
    paths = ["/old/root/cat/a.jpg"]
    result = resolve_image_paths(paths, "/new/root", "/old/root/")
    assert result == ["/new/root/cat/a.jpg"]


def test_load_activation_pair_groups_by_category(activation_tree):
    config = ActivationConfig(
        activation_dir=activation_tree,
        models=("model_a", "model_b"),
    )
    dataset = load_activation_pair(config)
    assert dataset.models == ("model_a", "model_b")
    assert dataset.n_per_category == N_PER_CATEGORY
    assert len(dataset.category_names) == N_CATEGORIES
    for model in dataset.models:
        assert dataset.activations[model].shape == (
            N_CATEGORIES,
            N_PER_CATEGORY,
            N_FEATURES,
        )


def test_load_activation_pair_normalises_by_default(activation_tree):
    config = ActivationConfig(
        activation_dir=activation_tree,
        models=("model_a", "model_b"),
    )
    dataset = load_activation_pair(config)
    flat = dataset.activations["model_a"].reshape(-1, N_FEATURES)
    assert np.allclose(np.linalg.norm(flat, axis=1), 1.0)


def test_load_activation_pair_can_skip_normalisation(activation_tree):
    config = ActivationConfig(
        activation_dir=activation_tree,
        models=("model_a", "model_b"),
        normalize=False,
    )
    dataset = load_activation_pair(config)
    flat = dataset.activations["model_a"].reshape(-1, N_FEATURES)
    assert not np.allclose(np.linalg.norm(flat, axis=1), 1.0)


def test_load_activation_pair_rejects_differing_image_order(
    activation_tree, image_paths
):
    reordered = list(reversed(image_paths))
    (activation_tree / "model_b" / "imagepaths.txt").write_text(
        "\n".join(reordered) + "\n"
    )
    config = ActivationConfig(
        activation_dir=activation_tree,
        models=("model_a", "model_b"),
    )
    with pytest.raises(ValueError, match="same order"):
        load_activation_pair(config)


def test_load_activation_pair_rejects_differing_image_counts(
    activation_tree, image_paths
):
    (activation_tree / "model_b" / "imagepaths.txt").write_text(
        "\n".join(image_paths[:-N_PER_CATEGORY]) + "\n"
    )
    config = ActivationConfig(
        activation_dir=activation_tree,
        models=("model_a", "model_b"),
    )
    with pytest.raises(ValueError, match="same order"):
        load_activation_pair(config)


def test_load_activation_pair_rejects_path_count_mismatch(
    activation_tree, flat_activations
):
    truncated = flat_activations["model_a"][:-1]
    np.save(
        activation_tree / "model_a" / "activations.npy", truncated
    )
    config = ActivationConfig(
        activation_dir=activation_tree,
        models=("model_a", "model_b"),
    )
    with pytest.raises(ValueError, match="one activation per image"):
        load_activation_pair(config)


def test_load_activation_pair_applies_path_resolution(
    activation_tree,
):
    config = ActivationConfig(
        activation_dir=activation_tree,
        models=("model_a", "model_b"),
        dataset_path="/local/images",
        replace_path_prefix="/dataset/",
    )
    dataset = load_activation_pair(config)
    assert dataset.image_paths[0].startswith("/local/images/")
