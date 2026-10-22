"""Tests for repscreen.screening.pipeline."""

import numpy as np
import pytest

from repscreen.config import (
    ActivationConfig,
    CompactnessConfig,
    OutputConfig,
    ScreeningConfig,
    SelectionConfig,
)
from repscreen.metrics.rdm import compare_rdms
from repscreen.screening.pipeline import run_screening

from .conftest import MODELS, N_CATEGORIES, N_PER_CATEGORY

N_SELECTED = 3
N_EXEMPLARS = 2


def _config(activation_tree, **overrides):
    """Build a screening configuration over the fixture tree."""
    return ScreeningConfig(
        activations=ActivationConfig(
            activation_dir=activation_tree, models=MODELS
        ),
        compactness=CompactnessConfig(
            measure=overrides.pop("measure", "R-squared")
        ),
        selection=SelectionConfig(
            n_categories=N_SELECTED, n_exemplars=N_EXEMPLARS
        ),
        output=OutputConfig(**overrides),
    )


def test_run_screening_reports_every_category(activation_tree):
    result = run_screening(_config(activation_tree))
    assert len(result.category_names) == N_CATEGORIES
    assert result.score.shape == (N_CATEGORIES,)
    assert result.correlations.shape == (N_CATEGORIES,)


def test_run_screening_selects_the_requested_set(activation_tree):
    result = run_screening(_config(activation_tree))
    assert len(result.selected_categories) == N_SELECTED
    assert len(result.selected_category_names) == N_SELECTED
    assert len(result.stimulus_paths) == N_SELECTED * N_EXEMPLARS


def test_run_screening_stimulus_paths_come_from_the_image_list(
    activation_tree, image_paths
):
    result = run_screening(_config(activation_tree))
    assert set(result.stimulus_paths).issubset(set(image_paths))


def test_run_screening_stimulus_paths_match_the_global_indices(
    activation_tree, image_paths
):
    result = run_screening(_config(activation_tree))
    expected = [image_paths[i] for i in result.global_indices]
    assert result.stimulus_paths == expected


def test_run_screening_keeps_the_requested_exemplars_per_category(
    activation_tree
):
    result = run_screening(_config(activation_tree))
    categories = [
        path.split("/")[-2] for path in result.stimulus_paths
    ]
    for name in result.selected_category_names:
        assert categories.count(name) == N_EXEMPLARS


def test_run_screening_similarity_matches_the_selected_rdms(
    activation_tree,
):
    result = run_screening(_config(activation_tree))
    expected = compare_rdms(
        result.selected_rdms[MODELS[0]],
        result.selected_rdms[MODELS[1]],
        metric="pearson",
    )
    assert result.similarity == pytest.approx(expected)


def test_run_screening_similarity_is_a_correlation(activation_tree):
    result = run_screening(_config(activation_tree))
    assert -1.0 <= result.similarity <= 1.0


def test_run_screening_exposes_the_full_category_rdms(
    activation_tree,
):
    result = run_screening(_config(activation_tree))
    size = N_SELECTED * N_PER_CATEGORY
    for model in MODELS:
        assert result.full_rdms[model].shape == (size, size)


def test_run_screening_exposes_compactness_per_model(
    activation_tree,
):
    result = run_screening(_config(activation_tree))
    for model in MODELS:
        assert result.compactness.compactness[model].shape == (
            N_CATEGORIES,
        )


def test_run_screening_records_its_configuration(activation_tree):
    config = _config(activation_tree)
    result = run_screening(config)
    assert result.config is config
    assert result.name == f"{MODELS[0]}_{MODELS[1]}"


def test_run_screening_reports_exemplars_per_category(
    activation_tree,
):
    result = run_screening(_config(activation_tree))
    assert result.n_per_category == N_PER_CATEGORY


def test_run_screening_honours_the_compactness_measure(
    activation_tree,
):
    result = run_screening(
        _config(activation_tree, measure="Fisher_discriminant")
    )
    assert result.compactness.measure == "Fisher_discriminant"


def test_run_screening_writes_nothing_without_an_output_dir(
    activation_tree, tmp_path
):
    before = set(tmp_path.rglob("*"))
    run_screening(_config(activation_tree))
    assert set(tmp_path.rglob("*")) == before


def test_run_screening_writes_to_the_output_dir(
    activation_tree, tmp_path
):
    output_dir = tmp_path / "results"
    result = run_screening(
        _config(activation_tree, output_dir=output_dir)
    )
    run_dir = output_dir / result.name
    assert (run_dir / "stimulipaths.json").is_file()
    assert (run_dir / "compactness.npy").is_file()


def test_run_screening_writes_figures_when_asked(
    activation_tree, image_tree, tmp_path
):
    config = ScreeningConfig(
        activations=ActivationConfig(
            activation_dir=activation_tree,
            models=MODELS,
            dataset_path=image_tree,
            replace_path_prefix="/dataset/",
        ),
        selection=SelectionConfig(
            n_categories=N_SELECTED, n_exemplars=N_EXEMPLARS
        ),
        output=OutputConfig(figure_dir=tmp_path / "figures"),
    )
    result = run_screening(config)
    run_dir = tmp_path / "figures" / result.name
    assert (run_dir / "RDMs.png").is_file()
    assert (run_dir / "compactness.png").is_file()
    assert (run_dir / "subset.png").is_file()


def test_run_screening_writes_no_figures_by_default(
    activation_tree, tmp_path
):
    run_screening(_config(activation_tree))
    assert not (tmp_path / "figures").exists()


def test_run_screening_rejects_more_categories_than_available(
    activation_tree,
):
    config = ScreeningConfig(
        activations=ActivationConfig(
            activation_dir=activation_tree, models=MODELS
        ),
        selection=SelectionConfig(
            n_categories=N_CATEGORIES + 1, n_exemplars=N_EXEMPLARS
        ),
    )
    with pytest.raises(ValueError, match="only 6 categories"):
        run_screening(config)


def test_run_screening_rejects_more_exemplars_than_available(
    activation_tree,
):
    config = ScreeningConfig(
        activations=ActivationConfig(
            activation_dir=activation_tree, models=MODELS
        ),
        selection=SelectionConfig(
            n_categories=N_SELECTED,
            n_exemplars=N_PER_CATEGORY + 1,
        ),
    )
    with pytest.raises(ValueError, match="n_exemplars"):
        run_screening(config)


def test_run_screening_is_deterministic(activation_tree):
    first = run_screening(_config(activation_tree))
    second = run_screening(_config(activation_tree))
    assert first.stimulus_paths == second.stimulus_paths
    assert np.allclose(first.score, second.score)
