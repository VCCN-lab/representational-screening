"""Tests for repscreen.output.writers."""

import csv
import json

import numpy as np
import pytest

from repscreen.output.writers import (
    save_results,
    write_csv,
    write_json,
    write_npy,
    write_stimulus_paths,
)

from .conftest import MODELS, N_CATEGORIES


def test_write_npy_writes_the_compactness_arrays(
    screening_result, tmp_path
):
    written = write_npy(screening_result, tmp_path)
    for name in (
        "compactness.npy",
        "sorted_compactness.npy",
        "sorted_compact_categories.npy",
        "category_score.npy",
    ):
        assert (tmp_path / name).is_file()
        assert tmp_path / name in written


def test_write_npy_round_trips_the_compactness_dict(
    screening_result, tmp_path
):
    write_npy(screening_result, tmp_path)
    loaded = np.load(
        tmp_path / "compactness.npy", allow_pickle=True
    ).item()
    expected = screening_result.compactness.compactness
    for model in MODELS:
        assert np.allclose(loaded[model], expected[model])


def test_write_npy_writes_both_rdm_levels(screening_result, tmp_path):
    write_npy(screening_result, tmp_path)
    for model in MODELS:
        assert (tmp_path / f"catRDM_{model}.npy").is_file()
        assert (tmp_path / f"RDM_{model}.npy").is_file()


def test_write_npy_can_skip_the_rdms(screening_result, tmp_path):
    write_npy(screening_result, tmp_path, save_rdms=False)
    assert not (tmp_path / f"RDM_{MODELS[0]}.npy").exists()


def test_write_npy_rdms_round_trip(screening_result, tmp_path):
    write_npy(screening_result, tmp_path)
    loaded = np.load(tmp_path / f"RDM_{MODELS[0]}.npy")
    assert np.allclose(
        loaded, screening_result.selected_rdms[MODELS[0]]
    )


def test_write_stimulus_paths_writes_json(
    screening_result, tmp_path
):
    path = write_stimulus_paths(screening_result, tmp_path)
    assert path == tmp_path / "stimulipaths.json"
    assert json.loads(path.read_text()) == (
        screening_result.stimulus_paths
    )


def test_write_json_writes_a_summary(screening_result, tmp_path):
    written = write_json(screening_result, tmp_path)
    summary = json.loads(
        (tmp_path / "screening_summary.json").read_text()
    )
    assert tmp_path / "screening_summary.json" in written
    assert summary["models"] == list(MODELS)
    assert summary["name"] == screening_result.name
    assert summary["similarity"] == pytest.approx(
        screening_result.similarity
    )


def test_write_json_summary_lists_the_selection(
    screening_result, tmp_path
):
    write_json(screening_result, tmp_path)
    summary = json.loads(
        (tmp_path / "screening_summary.json").read_text()
    )
    assert summary["selected_categories"] == list(
        screening_result.selected_category_names
    )
    assert summary["n_exemplars"] == (
        screening_result.config.selection.n_exemplars
    )
    assert summary["compactness_measure"] == (
        screening_result.compactness.measure
    )


def test_write_json_also_writes_the_stimulus_paths(
    screening_result, tmp_path
):
    write_json(screening_result, tmp_path)
    assert (tmp_path / "stimulipaths.json").is_file()


def test_write_csv_writes_one_row_per_category(
    screening_result, tmp_path
):
    write_csv(screening_result, tmp_path)
    with open(tmp_path / "category_scores.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == N_CATEGORIES


def test_write_csv_category_columns(screening_result, tmp_path):
    write_csv(screening_result, tmp_path)
    with open(tmp_path / "category_scores.csv") as handle:
        header = next(csv.reader(handle))
    expected = [
        "category",
        f"compactness_{MODELS[0]}",
        f"compactness_{MODELS[1]}",
        "compactness_difference",
        "rdm_correlation",
        "score",
        "selected",
    ]
    assert header == expected


def test_write_csv_marks_the_selected_categories(
    screening_result, tmp_path
):
    write_csv(screening_result, tmp_path)
    with open(tmp_path / "category_scores.csv") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row["category"] for row in rows if row["selected"] == "1"
    ]
    assert sorted(selected) == sorted(
        screening_result.selected_category_names
    )


def test_write_csv_writes_one_row_per_stimulus(
    screening_result, tmp_path
):
    write_csv(screening_result, tmp_path)
    with open(tmp_path / "selected_stimuli.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(screening_result.stimulus_paths)
    assert rows[0]["path"] == screening_result.stimulus_paths[0]


def test_write_csv_stimulus_columns(screening_result, tmp_path):
    write_csv(screening_result, tmp_path)
    with open(tmp_path / "selected_stimuli.csv") as handle:
        header = next(csv.reader(handle))
    assert header == ["path", "category", "image_index"]


def test_save_results_creates_a_run_subdirectory(
    screening_result, tmp_path
):
    written = save_results(screening_result, tmp_path)
    run_dir = tmp_path / screening_result.name
    assert run_dir.is_dir()
    assert all(path.is_relative_to(run_dir) for path in written)


def test_save_results_honours_the_requested_formats(
    screening_result, tmp_path
):
    save_results(screening_result, tmp_path, formats=("csv",))
    run_dir = tmp_path / screening_result.name
    assert (run_dir / "category_scores.csv").is_file()
    assert not (run_dir / "compactness.npy").exists()


def test_save_results_rejects_an_unknown_format(
    screening_result, tmp_path
):
    with pytest.raises(ValueError, match="Unknown output format"):
        save_results(
            screening_result, tmp_path, formats=("parquet",)
        )


def test_save_results_returns_nothing_without_a_directory(
    screening_result,
):
    assert save_results(screening_result, None) == []
