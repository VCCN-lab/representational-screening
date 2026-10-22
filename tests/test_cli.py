"""Tests for repscreen.cli.main."""

import json

import pytest

from repscreen.cli.main import build_parser, config_from_args, main

from .conftest import MODELS


def _argv(activation_tree, output_dir):
    """Minimal argument list for a screening run."""
    return [
        "--activation-dir",
        str(activation_tree),
        "--models",
        MODELS[0],
        MODELS[1],
        "--n-categories",
        "3",
        "--n-exemplars",
        "2",
        "--output-dir",
        str(output_dir),
    ]


def test_parser_requires_an_activation_dir():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--models", "a", "b"])


def test_parser_requires_models():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--activation-dir", "/acts"])


def test_parser_requires_exactly_two_models():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--activation-dir", "/acts", "--models", "a"]
        )


def test_parser_rejects_an_unknown_compactness_measure():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--activation-dir",
                "/acts",
                "--models",
                "a",
                "b",
                "--compactness-measure",
                "not-a-measure",
            ]
        )


def test_config_from_args_uses_the_documented_defaults():
    args = build_parser().parse_args(
        ["--activation-dir", "/acts", "--models", "a", "b"]
    )
    config = config_from_args(args)
    assert config.activations.models == ("a", "b")
    assert config.activations.normalize is True
    assert config.compactness.measure == "R-squared"
    assert config.compactness.difference_measure == "normalizedDiff"
    assert config.selection.n_categories == 12
    assert config.selection.n_exemplars == 4
    assert config.name == "a_b"


def test_config_from_args_passes_every_option_through(tmp_path):
    args = build_parser().parse_args(
        [
            "--activation-dir",
            "/acts",
            "--models",
            "a",
            "b",
            "--dataset-path",
            "/images",
            "--replace-path-prefix",
            "/old/",
            "--no-normalize",
            "--compactness-measure",
            "Fisher_discriminant",
            "--difference-measure",
            "rank",
            "--n-categories",
            "5",
            "--n-exemplars",
            "3",
            "--dissimilarity-metric",
            "pearson",
            "--similarity-metric",
            "spearman",
            "--output-dir",
            str(tmp_path),
            "--formats",
            "csv",
            "--no-rdms",
            "--name",
            "custom",
        ]
    )
    config = config_from_args(args)
    assert str(config.activations.dataset_path) == "/images"
    assert config.activations.replace_path_prefix == "/old/"
    assert config.activations.normalize is False
    assert config.compactness.measure == "Fisher_discriminant"
    assert config.compactness.difference_measure == "rank"
    assert config.selection.n_categories == 5
    assert config.selection.n_exemplars == 3
    assert config.selection.dissimilarity_metric == "pearson"
    assert config.selection.similarity_metric == "spearman"
    assert config.output.formats == ("csv",)
    assert config.output.save_rdms is False
    assert config.name == "custom"


def test_config_from_args_maps_the_figure_dir(tmp_path):
    args = build_parser().parse_args(
        [
            "--activation-dir",
            "/acts",
            "--models",
            "a",
            "b",
            "--figure-dir",
            str(tmp_path / "figures"),
        ]
    )
    config = config_from_args(args)
    assert config.output.figure_dir == tmp_path / "figures"


def test_config_from_args_leaves_figure_dir_unset_by_default():
    args = build_parser().parse_args(
        ["--activation-dir", "/acts", "--models", "a", "b"]
    )
    assert config_from_args(args).output.figure_dir is None


def test_main_returns_zero_and_writes_results(
    activation_tree, tmp_path
):
    output_dir = tmp_path / "results"
    assert main(_argv(activation_tree, output_dir)) == 0
    run_dir = output_dir / f"{MODELS[0]}_{MODELS[1]}"
    assert (run_dir / "stimulipaths.json").is_file()
    assert (run_dir / "screening_summary.json").is_file()


def test_main_writes_the_requested_number_of_stimuli(
    activation_tree, tmp_path
):
    output_dir = tmp_path / "results"
    main(_argv(activation_tree, output_dir))
    run_dir = output_dir / f"{MODELS[0]}_{MODELS[1]}"
    paths = json.loads((run_dir / "stimulipaths.json").read_text())
    assert len(paths) == 6


def test_main_reports_the_similarity(
    activation_tree, tmp_path, capsys
):
    main(_argv(activation_tree, tmp_path / "results"))
    assert "similarity" in capsys.readouterr().out.lower()


def test_main_writes_figures_when_asked(
    activation_tree, image_tree, tmp_path
):
    figure_dir = tmp_path / "figures"
    argv = _argv(activation_tree, tmp_path / "results") + [
        "--dataset-path",
        str(image_tree),
        "--replace-path-prefix",
        "/dataset/",
        "--figure-dir",
        str(figure_dir),
    ]
    assert main(argv) == 0
    run_dir = figure_dir / f"{MODELS[0]}_{MODELS[1]}"
    for name in (
        "RDMs.png",
        "catRDMs.png",
        "compactness.png",
        "subset.png",
    ):
        assert (run_dir / name).is_file()


def test_main_skips_the_stimulus_grid_without_the_images(
    activation_tree, tmp_path
):
    # The recorded image paths name a directory that does not
    # exist here, which is what --dataset-path is for.  The three
    # figures that need no images must still be written.
    figure_dir = tmp_path / "figures"
    argv = _argv(activation_tree, tmp_path / "results") + [
        "--figure-dir",
        str(figure_dir),
    ]
    with pytest.warns(UserWarning, match="stimulus image"):
        assert main(argv) == 0
    run_dir = figure_dir / f"{MODELS[0]}_{MODELS[1]}"
    assert (run_dir / "RDMs.png").is_file()
    assert (run_dir / "catRDMs.png").is_file()
    assert (run_dir / "compactness.png").is_file()
    assert not (run_dir / "subset.png").exists()


def test_main_writes_no_figures_by_default(
    activation_tree, tmp_path
):
    main(_argv(activation_tree, tmp_path / "results"))
    assert not (tmp_path / "figures").exists()


def test_main_propagates_a_configuration_error(
    activation_tree, tmp_path
):
    argv = _argv(activation_tree, tmp_path / "results")
    argv[argv.index("--n-categories") + 1] = "99"
    with pytest.raises(ValueError, match="only 6 categories"):
        main(argv)
