"""Tests for repscreen.config."""

from pathlib import Path

import pytest

from repscreen.config import (
    ActivationConfig,
    CompactnessConfig,
    OutputConfig,
    ScreeningConfig,
    SelectionConfig,
    VALID_COMPACTNESS_MEASURES,
    VALID_DIFFERENCE_MEASURES,
    VALID_DISSIMILARITY_METRICS,
    VALID_OUTPUT_FORMATS,
    VALID_SIMILARITY_METRICS,
)


def test_activation_config_defaults():
    cfg = ActivationConfig(
        activation_dir="/acts", models=("model_a", "model_b")
    )
    assert cfg.activation_dir == Path("/acts")
    assert cfg.models == ("model_a", "model_b")
    assert cfg.normalize is True
    assert cfg.dataset_path is None
    assert cfg.replace_path_prefix is None


def test_activation_config_requires_two_models():
    with pytest.raises(ValueError, match="exactly two models"):
        ActivationConfig(
            activation_dir="/acts", models=("model_a",)
        )


def test_activation_config_rejects_duplicate_models():
    with pytest.raises(ValueError, match="distinct"):
        ActivationConfig(
            activation_dir="/acts", models=("model_a", "model_a")
        )


def test_activation_config_coerces_dataset_path():
    cfg = ActivationConfig(
        activation_dir="/acts",
        models=("model_a", "model_b"),
        dataset_path="/data/images",
    )
    assert cfg.dataset_path == Path("/data/images")


def test_compactness_config_defaults():
    cfg = CompactnessConfig()
    assert cfg.measure == "R-squared"
    assert cfg.difference_measure == "normalizedDiff"


def test_compactness_config_invalid_measure():
    with pytest.raises(ValueError, match="Unknown compactness measure"):
        CompactnessConfig(measure="not-a-measure")


def test_compactness_config_invalid_difference_measure():
    with pytest.raises(ValueError, match="Unknown difference measure"):
        CompactnessConfig(difference_measure="not-a-measure")


def test_compactness_config_accepts_every_valid_measure():
    for measure in VALID_COMPACTNESS_MEASURES:
        assert CompactnessConfig(measure=measure).measure == measure


def test_compactness_config_accepts_every_difference_measure():
    for measure in VALID_DIFFERENCE_MEASURES:
        cfg = CompactnessConfig(difference_measure=measure)
        assert cfg.difference_measure == measure


def test_selection_config_defaults():
    cfg = SelectionConfig()
    assert cfg.n_categories == 12
    assert cfg.n_exemplars == 4
    assert cfg.dissimilarity_metric == "L2squared"
    assert cfg.similarity_metric == "pearson"


def test_selection_config_rejects_non_positive_categories():
    with pytest.raises(ValueError, match="n_categories must be positive"):
        SelectionConfig(n_categories=0)


def test_selection_config_rejects_non_positive_exemplars():
    with pytest.raises(ValueError, match="n_exemplars must be positive"):
        SelectionConfig(n_exemplars=-1)


def test_selection_config_invalid_dissimilarity_metric():
    with pytest.raises(ValueError, match="Unknown dissimilarity metric"):
        SelectionConfig(dissimilarity_metric="manhattan")


def test_selection_config_invalid_similarity_metric():
    with pytest.raises(ValueError, match="Unknown similarity metric"):
        SelectionConfig(similarity_metric="kendall")


def test_selection_config_accepts_every_valid_metric():
    for metric in VALID_DISSIMILARITY_METRICS:
        cfg = SelectionConfig(dissimilarity_metric=metric)
        assert cfg.dissimilarity_metric == metric
    for metric in VALID_SIMILARITY_METRICS:
        cfg = SelectionConfig(similarity_metric=metric)
        assert cfg.similarity_metric == metric


def test_output_config_defaults():
    cfg = OutputConfig()
    assert cfg.output_dir is None
    assert cfg.formats == ("npy", "json")
    assert cfg.save_rdms is True


def test_output_config_coerces_output_dir():
    cfg = OutputConfig(output_dir="results/run")
    assert cfg.output_dir == Path("results/run")


def test_output_config_figure_dir_defaults_to_none():
    assert OutputConfig().figure_dir is None


def test_output_config_coerces_figure_dir():
    cfg = OutputConfig(figure_dir="figures/run")
    assert cfg.figure_dir == Path("figures/run")


def test_output_config_invalid_format():
    with pytest.raises(ValueError, match="Unknown output format"):
        OutputConfig(formats=("parquet",))


def test_output_config_accepts_every_valid_format():
    cfg = OutputConfig(formats=VALID_OUTPUT_FORMATS)
    assert cfg.formats == VALID_OUTPUT_FORMATS


def test_screening_config_composes_sub_configs():
    cfg = ScreeningConfig(
        activations=ActivationConfig(
            activation_dir="/acts", models=("model_a", "model_b")
        )
    )
    assert isinstance(cfg.compactness, CompactnessConfig)
    assert isinstance(cfg.selection, SelectionConfig)
    assert isinstance(cfg.output, OutputConfig)
    assert cfg.name == "model_a_model_b"


def test_screening_config_keeps_explicit_name():
    cfg = ScreeningConfig(
        activations=ActivationConfig(
            activation_dir="/acts", models=("model_a", "model_b")
        ),
        name="custom",
    )
    assert cfg.name == "custom"
