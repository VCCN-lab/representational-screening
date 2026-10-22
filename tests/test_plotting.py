"""Tests for the repscreen.plotting modules."""

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from repscreen.metrics.layers import (
    compare_rdms_per_layer,
    correlations_per_model,
)
from repscreen.plotting.compactness import (
    plot_compactness,
    plot_compactness_per_model,
    plot_compactness_scatter,
)
from repscreen.plotting.embedding import (
    cluster_quality,
    plot_tsne,
    plot_tsne_comparison,
    tsne_from_rdm,
    tsne_model_comparison,
)
from repscreen.plotting.rdm import (
    plot_layer_similarities,
    plot_rdm,
    plot_rdm_pair,
)
from repscreen.plotting.stimuli import (
    load_rgb_image,
    load_stimulus_images,
    plot_stimulus_grid,
)

from .conftest import MODELS, N_CATEGORIES, N_PER_CATEGORY


@pytest.fixture
def compactness_values():
    """Compactness per category for each of two models."""
    rng = np.random.default_rng(0)
    return {
        MODELS[0]: rng.random(N_CATEGORIES),
        MODELS[1]: rng.random(N_CATEGORIES),
    }


@pytest.fixture
def stimulus_paths(image_tree, category_names):
    """Paths of two images from each of three categories."""
    return [
        str(image_tree / name / f"{name}_{j:03d}.jpg")
        for name in category_names[:3]
        for j in range(2)
    ]


def test_plot_rdm_returns_axes(rdm_pair):
    first, _ = rdm_pair
    axes = plot_rdm(first, title="model_a")
    assert isinstance(axes, Axes)
    assert axes.get_title() == "model_a"


def test_plot_rdm_draws_on_a_supplied_axes(rdm_pair):
    import matplotlib.pyplot as plt

    first, _ = rdm_pair
    _, axes = plt.subplots()
    assert plot_rdm(first, ax=axes) is axes


def test_plot_rdm_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        plot_rdm(np.zeros((3, 4)))


def test_plot_rdm_pair_has_two_panels(rdm_pair):
    figure = plot_rdm_pair(*rdm_pair, model_names=MODELS)
    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 2


def test_plot_rdm_pair_titles_each_panel(rdm_pair):
    figure = plot_rdm_pair(*rdm_pair, model_names=MODELS)
    titles = [axes.get_title() for axes in figure.axes[:2]]
    assert titles == list(MODELS)


def test_plot_rdm_pair_reports_the_similarity(rdm_pair):
    figure = plot_rdm_pair(
        *rdm_pair, model_names=MODELS, similarity=0.25
    )
    assert "0.25" in figure._suptitle.get_text()


def test_plot_rdm_pair_rejects_differing_shapes():
    with pytest.raises(ValueError, match="same shape"):
        plot_rdm_pair(
            np.zeros((3, 3)), np.zeros((4, 4)), model_names=MODELS
        )


def test_plot_compactness_draws_one_line_per_model(
    compactness_values,
):
    figure, axes = plot_compactness(compactness_values, MODELS)
    assert isinstance(figure, Figure)
    assert len(axes.get_lines()) == len(MODELS)


def test_plot_compactness_labels_its_axes(compactness_values):
    _, axes = plot_compactness(
        compactness_values,
        MODELS,
        xlabel="Categories",
        ylabel="Compactness",
    )
    assert axes.get_xlabel() == "Categories"
    assert axes.get_ylabel() == "Compactness"


def test_plot_compactness_per_model_makes_one_panel_each(
    compactness_values,
):
    figure = plot_compactness_per_model(compactness_values, MODELS)
    assert len(figure.axes) == len(MODELS)


def test_plot_compactness_scatter_reports_the_correlation(
    compactness_values,
):
    figure, axes = plot_compactness_scatter(
        compactness_values, MODELS
    )
    expected = np.corrcoef(
        compactness_values[MODELS[0]], compactness_values[MODELS[1]]
    )[0, 1]
    assert isinstance(figure, Figure)
    assert f"{expected:.2f}" in axes.get_title()


def test_plot_compactness_scatter_rejects_length_mismatch(
    compactness_values,
):
    compactness_values[MODELS[1]] = compactness_values[MODELS[1]][:-1]
    with pytest.raises(ValueError, match="same length"):
        plot_compactness_scatter(compactness_values, MODELS)


def test_tsne_from_rdm_returns_two_dimensions(rdm_pair):
    first, _ = rdm_pair
    result = tsne_from_rdm(first, random_state=0)
    assert result.shape == (len(first), 2)


def test_tsne_from_rdm_is_reproducible(rdm_pair):
    first, _ = rdm_pair
    left = tsne_from_rdm(first, random_state=1)
    right = tsne_from_rdm(first, random_state=1)
    assert np.allclose(left, right)


def test_tsne_from_rdm_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        tsne_from_rdm(np.zeros((3, 4)))


def test_tsne_from_rdm_caps_perplexity_for_small_sets(rdm_pair):
    first, _ = rdm_pair
    assert tsne_from_rdm(first, perplexity=1000).shape == (
        len(first),
        2,
    )


def test_plot_tsne_returns_a_figure(rdm_pair, category_names):
    first, _ = rdm_pair
    embedding = tsne_from_rdm(first, random_state=0)
    figure, axes = plot_tsne(embedding, category_names)
    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)


def test_plot_tsne_comparison_has_two_panels(
    rdm_pair, category_names
):
    embeddings = [
        tsne_from_rdm(matrix, random_state=0) for matrix in rdm_pair
    ]
    figure = plot_tsne_comparison(
        *embeddings, labels=category_names, model_names=MODELS
    )
    assert len(figure.axes) >= 2


def test_plot_tsne_comparison_rejects_label_mismatch(rdm_pair):
    embeddings = [
        tsne_from_rdm(matrix, random_state=0) for matrix in rdm_pair
    ]
    with pytest.raises(ValueError, match="one label per item"):
        plot_tsne_comparison(
            *embeddings, labels=["a"], model_names=MODELS
        )


def test_tsne_model_comparison_returns_both_embeddings(
    rdm_pair, category_names
):
    embeddings, figure = tsne_model_comparison(
        *rdm_pair,
        labels=category_names,
        model_names=MODELS,
        random_state=0,
    )
    assert len(embeddings) == 2
    assert isinstance(figure, Figure)


@pytest.fixture
def grouped_labels():
    """Labels putting the six fixture items into three pairs.

    The silhouette is undefined when every item is its own
    cluster, so the labels have to group.
    """
    return [f"cat{i // 2}" for i in range(N_CATEGORIES)]


def test_cluster_quality_reports_both_scores(
    rdm_pair, grouped_labels
):
    first, _ = rdm_pair
    embedding = tsne_from_rdm(first, random_state=0)
    scores = cluster_quality(embedding, grouped_labels)
    assert set(scores) == {
        "silhouette",
        "adjusted_rand_index",
        "n_clusters",
    }
    assert scores["n_clusters"] == len(set(grouped_labels))


def test_cluster_quality_scores_are_bounded(
    rdm_pair, grouped_labels
):
    first, _ = rdm_pair
    embedding = tsne_from_rdm(first, random_state=0)
    scores = cluster_quality(embedding, grouped_labels)
    assert -1.0 <= scores["silhouette"] <= 1.0
    assert -1.0 <= scores["adjusted_rand_index"] <= 1.0


def test_cluster_quality_rejects_one_cluster_per_item(
    rdm_pair, category_names
):
    first, _ = rdm_pair
    embedding = tsne_from_rdm(first, random_state=0)
    with pytest.raises(ValueError, match="distinct labels"):
        cluster_quality(embedding, category_names)


def test_cluster_quality_rejects_a_single_cluster(rdm_pair):
    first, _ = rdm_pair
    embedding = tsne_from_rdm(first, random_state=0)
    with pytest.raises(ValueError, match="distinct labels"):
        cluster_quality(embedding, ["only"] * N_CATEGORIES)


def test_load_rgb_image_reads_an_image(stimulus_paths):
    image = load_rgb_image(stimulus_paths[0])
    assert image.shape == (16, 16, 3)


def test_load_rgb_image_warns_on_a_missing_file(tmp_path):
    with pytest.warns(UserWarning, match="not found"):
        assert load_rgb_image(tmp_path / "absent.jpg") is None


def test_load_stimulus_images_skips_missing_files(
    stimulus_paths, tmp_path
):
    paths = stimulus_paths + [str(tmp_path / "absent.jpg")]
    with pytest.warns(UserWarning, match="not found"):
        images, valid = load_stimulus_images(paths)
    assert len(images) == len(stimulus_paths)
    assert valid == stimulus_paths


def test_plot_stimulus_grid_has_one_panel_per_image(
    stimulus_paths,
):
    figure = plot_stimulus_grid(stimulus_paths, n_columns=3)
    assert len(figure.axes) == 6


def test_plot_stimulus_grid_titles_the_first_row(stimulus_paths):
    figure = plot_stimulus_grid(stimulus_paths, n_columns=3)
    titles = [axes.get_title() for axes in figure.axes]
    assert sum(1 for title in titles if title) == 3


def test_plot_stimulus_grid_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="at least one image"):
        plot_stimulus_grid([])


def test_compare_rdms_per_layer_pairs_every_model(rdm_pair):
    first, second = rdm_pair
    per_model = {
        MODELS[0]: [first, second],
        MODELS[1]: [second, first],
    }
    result = compare_rdms_per_layer(per_model, MODELS)
    assert set(result) == set(MODELS)
    assert len(result[MODELS[0]][MODELS[1]]) == 2


def test_compare_rdms_per_layer_self_comparison_is_one(rdm_pair):
    first, second = rdm_pair
    per_model = {MODELS[0]: [first, second]}
    result = compare_rdms_per_layer(per_model, (MODELS[0],))
    assert np.allclose(result[MODELS[0]][MODELS[0]], 1.0)


def test_compare_rdms_per_layer_rejects_differing_depths(rdm_pair):
    first, second = rdm_pair
    per_model = {
        MODELS[0]: [first, second],
        MODELS[1]: [second],
    }
    with pytest.raises(ValueError, match="same number of layers"):
        compare_rdms_per_layer(per_model, MODELS)


def test_plot_layer_similarities_has_one_panel_per_pair(rdm_pair):
    first, second = rdm_pair
    per_model = {
        MODELS[0]: [first, second],
        MODELS[1]: [second, first],
    }
    similarities = compare_rdms_per_layer(per_model, MODELS)
    figure = plot_layer_similarities(similarities, MODELS)
    assert len(figure.axes) >= 1


def test_image_tree_is_shaped_as_expected(image_tree):
    categories = sorted(p.name for p in image_tree.iterdir())
    assert len(categories) == N_CATEGORIES
    images = list((image_tree / categories[0]).glob("*.jpg"))
    assert len(images) == N_PER_CATEGORY


def test_correlations_per_model_shapes():
    rng = np.random.default_rng(20)
    activations = rng.normal(size=(3, 7, 5, 4))
    result = correlations_per_model(activations)
    assert result.shape == (3, 7, 7)


def test_correlations_per_model_matches_a_direct_computation():
    # mean_axis counts on the array with the instance axis already
    # moved to the front, so axis 2 of (instances, items, a, b)
    # leaves one pattern of length b per item.
    rng = np.random.default_rng(21)
    activations = rng.normal(size=(3, 7, 5, 4))
    result = correlations_per_model(activations)
    patterns = activations.mean(axis=2)
    for instance in range(3):
        expected = np.corrcoef(patterns[instance])
        assert np.allclose(result[instance], expected)


def test_correlations_per_model_honours_the_averaging_axis():
    rng = np.random.default_rng(22)
    activations = rng.normal(size=(2, 6, 5, 4))
    result = correlations_per_model(activations, mean_axis=3)
    expected = np.corrcoef(activations.mean(axis=3)[0])
    assert np.allclose(result[0], expected)


def test_correlations_per_model_honours_the_instance_axis():
    rng = np.random.default_rng(23)
    activations = rng.normal(size=(7, 3, 5, 4))
    result = correlations_per_model(activations, instance_axis=1)
    assert result.shape == (3, 7, 7)


def test_correlations_per_model_adds_a_missing_instance_axis():
    rng = np.random.default_rng(24)
    activations = rng.normal(size=(6, 5))
    result = correlations_per_model(activations, mean_axis=None)
    assert result.shape == (1, 6, 6)


def test_correlations_per_model_diagonal_is_one():
    rng = np.random.default_rng(25)
    activations = rng.normal(size=(2, 6, 5, 4))
    result = correlations_per_model(activations)
    for instance in range(2):
        assert np.allclose(np.diag(result[instance]), 1.0)
