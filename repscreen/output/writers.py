"""Writers for screening results and curated stimulus sets."""

import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..config import VALID_OUTPUT_FORMATS

STIMULUS_PATH_FILE = "stimulipaths.json"
SUMMARY_FILE = "screening_summary.json"
CATEGORY_SCORE_FILE = "category_scores.csv"
SELECTED_STIMULUS_FILE = "selected_stimuli.csv"


def write_npy(result, output_dir, save_rdms: bool = True):
    """Write the screening arrays as NumPy files.

    Parameters
    ----------
    result : ScreeningResult
        Result to write.
    output_dir : path-like
        Directory the files are written to.
    save_rdms : bool
        Whether to also write the category-level and curated-set
        dissimilarity matrices.

    Returns
    -------
    list of pathlib.Path
        Paths of the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "compactness.npy": result.compactness.compactness,
        "sorted_compactness.npy": (
            result.compactness.sorted_compactness
        ),
        "sorted_compact_categories.npy": (
            result.compactness.sorted_categories
        ),
        "category_score.npy": result.score,
        "category_correlations.npy": result.correlations,
        "compactness_difference.npy": result.difference,
    }
    if save_rdms:
        for model in result.models:
            arrays[f"catRDM_{model}.npy"] = result.full_rdms[model]
            arrays[f"RDM_{model}.npy"] = result.selected_rdms[model]

    written = []
    for name, array in arrays.items():
        path = output_dir / name
        np.save(path, array)
        written.append(path)
    return written


def write_stimulus_paths(result, output_dir) -> Path:
    """Write the curated stimulus paths as JSON.

    Parameters
    ----------
    result : ScreeningResult
        Result to write.
    output_dir : path-like
        Directory the file is written to.

    Returns
    -------
    pathlib.Path
        Path of the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / STIMULUS_PATH_FILE
    with open(path, "w") as handle:
        json.dump(list(result.stimulus_paths), handle, indent=4)
    return path


def _summary(result):
    """Build the JSON-serialisable summary of a screening run."""
    return {
        "name": result.name,
        "models": list(result.models),
        "compactness_measure": result.compactness.measure,
        "difference_measure": (
            result.categories.difference.measure
        ),
        "n_categories": len(result.selected_categories),
        "n_exemplars": result.config.selection.n_exemplars,
        "n_per_category": result.n_per_category,
        "dissimilarity_metric": (
            result.config.selection.dissimilarity_metric
        ),
        "similarity_metric": (
            result.config.selection.similarity_metric
        ),
        "similarity": float(result.similarity),
        "selected_categories": list(
            result.selected_category_names
        ),
        "n_stimuli": len(result.stimulus_paths),
    }


def write_json(result, output_dir) -> List[Path]:
    """Write the run summary and the curated stimulus paths.

    Parameters
    ----------
    result : ScreeningResult
        Result to write.
    output_dir : path-like
        Directory the files are written to.

    Returns
    -------
    list of pathlib.Path
        Paths of the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SUMMARY_FILE
    with open(path, "w") as handle:
        json.dump(_summary(result), handle, indent=4)
    return [path, write_stimulus_paths(result, output_dir)]


def _category_rows(result):
    """Yield one row per category for the category score table."""
    selected = set(int(index) for index in result.selected_categories)
    first, second = result.models
    for index, name in enumerate(result.category_names):
        yield [
            name,
            f"{result.compactness.compactness[first][index]:.6f}",
            f"{result.compactness.compactness[second][index]:.6f}",
            f"{result.difference[index]:.6f}",
            f"{result.correlations[index]:.6f}",
            f"{result.score[index]:.6f}",
            "1" if index in selected else "0",
        ]


def write_csv(result, output_dir) -> List[Path]:
    """Write per-category scores and the curated stimulus table.

    Parameters
    ----------
    result : ScreeningResult
        Result to write.
    output_dir : path-like
        Directory the files are written to.

    Returns
    -------
    list of pathlib.Path
        Paths of the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first, second = result.models

    category_path = output_dir / CATEGORY_SCORE_FILE
    with open(category_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "category",
                f"compactness_{first}",
                f"compactness_{second}",
                "compactness_difference",
                "rdm_correlation",
                "score",
                "selected",
            ]
        )
        writer.writerows(_category_rows(result))

    stimulus_path = output_dir / SELECTED_STIMULUS_FILE
    with open(stimulus_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "category", "image_index"])
        for path, index in zip(
            result.stimulus_paths, result.global_indices
        ):
            writer.writerow(
                [path, path.split("/")[-2], int(index)]
            )

    return [category_path, stimulus_path]


WRITERS = {
    "npy": write_npy,
    "json": write_json,
    "csv": write_csv,
}


def save_results(
    result,
    output_dir: Optional[Path],
    formats: Sequence[str] = ("npy", "json"),
    save_rdms: bool = True,
) -> List[Path]:
    """Write a screening result in each requested format.

    The files are placed in a subdirectory of ``output_dir`` named
    after the run, so several runs can share one output root.

    Parameters
    ----------
    result : ScreeningResult
        Result to write.
    output_dir : path-like or None
        Output root.  Nothing is written when None.
    formats : sequence of str
        Which formats to write; entries of
        ``VALID_OUTPUT_FORMATS``.
    save_rdms : bool
        Whether the ``npy`` writer also writes the dissimilarity
        matrices.

    Returns
    -------
    list of pathlib.Path
        Paths of every written file, empty when ``output_dir`` is
        None.

    Raises
    ------
    ValueError
        If a requested format is not recognised.
    """
    for fmt in formats:
        if fmt not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Unknown output format: '{fmt}'. Must be one of "
                f"{VALID_OUTPUT_FORMATS}."
            )
    if output_dir is None:
        return []

    run_dir = Path(output_dir) / result.name
    run_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for fmt in formats:
        if fmt == "npy":
            written.extend(
                write_npy(result, run_dir, save_rdms=save_rdms)
            )
        else:
            written.extend(WRITERS[fmt](result, run_dir))
    return written
