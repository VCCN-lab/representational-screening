"""Tests for repscreen.plotting.report."""

import pytest

from repscreen.config import (
    ActivationConfig,
    ScreeningConfig,
    SelectionConfig,
)
from repscreen.plotting.report import (
    SCREENING_FIGURES,
    save_screening_figures,
)
from repscreen.screening.pipeline import run_screening

from .conftest import MODELS

IMAGE_FREE = ("RDMs.png", "catRDMs.png", "compactness.png")


@pytest.fixture
def result_with_images(activation_tree, image_tree):
    """A screening result whose stimulus paths resolve on disk."""
    config = ScreeningConfig(
        activations=ActivationConfig(
            activation_dir=activation_tree,
            models=MODELS,
            dataset_path=image_tree,
            replace_path_prefix="/dataset/",
        ),
        selection=SelectionConfig(n_categories=3, n_exemplars=2),
    )
    return run_screening(config)


def test_figure_names_are_declared():
    assert set(SCREENING_FIGURES) == set(IMAGE_FREE) | {"subset.png"}


def test_writes_every_figure_when_images_resolve(
    result_with_images, tmp_path
):
    written = save_screening_figures(result_with_images, tmp_path)
    run_dir = tmp_path / result_with_images.name
    for name in SCREENING_FIGURES:
        assert (run_dir / name).is_file()
    assert len(written) == len(SCREENING_FIGURES)


def test_returns_the_written_paths(result_with_images, tmp_path):
    written = save_screening_figures(result_with_images, tmp_path)
    run_dir = tmp_path / result_with_images.name
    assert sorted(written) == sorted(
        run_dir / name for name in SCREENING_FIGURES
    )


def test_writes_into_a_subdirectory_named_after_the_run(
    result_with_images, tmp_path
):
    save_screening_figures(result_with_images, tmp_path)
    assert (tmp_path / result_with_images.name).is_dir()


def test_skips_the_grid_when_no_stimulus_image_resolves(
    screening_result, tmp_path
):
    with pytest.warns(UserWarning, match="stimulus image"):
        written = save_screening_figures(screening_result, tmp_path)
    run_dir = tmp_path / screening_result.name
    for name in IMAGE_FREE:
        assert (run_dir / name).is_file()
    assert not (run_dir / "subset.png").exists()
    assert len(written) == len(IMAGE_FREE)


def test_writes_nothing_without_a_directory(screening_result):
    assert save_screening_figures(screening_result, None) == []


def test_figures_are_closed_after_writing(
    result_with_images, tmp_path
):
    import matplotlib.pyplot as plt

    plt.close("all")
    save_screening_figures(result_with_images, tmp_path)
    assert plt.get_fignums() == []


def test_produces_non_empty_images(result_with_images, tmp_path):
    save_screening_figures(result_with_images, tmp_path)
    run_dir = tmp_path / result_with_images.name
    for name in SCREENING_FIGURES:
        assert (run_dir / name).stat().st_size > 1000
