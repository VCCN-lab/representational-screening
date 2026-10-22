"""The standard set of figures for a screening run."""

import warnings
from pathlib import Path
from typing import List, Optional

from .compactness import plot_compactness
from .rdm import plot_rdm_pair
from .stimuli import plot_stimulus_grid

SCREENING_FIGURES = (
    "RDMs.png",
    "catRDMs.png",
    "compactness.png",
    "subset.png",
)


def _image_free_figures(result, models):
    """Build the figures that need no stimulus image on disk."""
    return {
        "RDMs.png": plot_rdm_pair(
            result.selected_rdms[models[0]],
            result.selected_rdms[models[1]],
            model_names=models,
            similarity=result.similarity,
        ),
        "catRDMs.png": plot_rdm_pair(
            result.full_rdms[models[0]],
            result.full_rdms[models[1]],
            model_names=models,
        ),
        "compactness.png": plot_compactness(
            result.compactness.sorted_compactness, models
        )[0],
    }


def _stimulus_grid(result):
    """Build the stimulus grid, or None when no image is readable.

    Warns
    -----
    UserWarning
        If no stimulus image is on disk, so the grid is skipped.
    """
    if any(Path(path).is_file() for path in result.stimulus_paths):
        return plot_stimulus_grid(
            result.stimulus_paths,
            n_columns=len(result.selected_categories),
        )

    warnings.warn(
        "No stimulus image was found on disk, so the stimulus grid "
        "is not drawn. Set ActivationConfig.dataset_path (or pass "
        "--dataset-path) to point the recorded image paths at a "
        "local copy of the dataset.",
        UserWarning,
        stacklevel=3,
    )
    return None


def save_screening_figures(
    result, figure_dir: Optional[Path]
) -> List[Path]:
    """Write the standard figures of a screening run.

    The files are placed in a subdirectory of ``figure_dir`` named
    after the run, so several runs can share one figure root.  The
    names are listed in :data:`SCREENING_FIGURES`.

    The recorded image paths may name the machine the activations
    were computed on, in which case the stimulus grid cannot be
    drawn.  The other three figures need no images, so they are
    written either way.

    Parameters
    ----------
    result : ScreeningResult
        Result to draw, as returned by
        :func:`~repscreen.screening.pipeline.run_screening`.
    figure_dir : path-like or None
        Figure root.  Nothing is written when None.

    Returns
    -------
    list of pathlib.Path
        Paths of every written figure, empty when ``figure_dir`` is
        None.

    Warns
    -----
    UserWarning
        If no stimulus image can be found on disk, so the stimulus
        grid is skipped.
    """
    import matplotlib.pyplot as plt

    if figure_dir is None:
        return []

    models = list(result.models)
    figures = _image_free_figures(result, models)

    grid = _stimulus_grid(result)
    if grid is not None:
        figures["subset.png"] = grid

    run_dir = Path(figure_dir) / result.name
    run_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, figure in figures.items():
        path = run_dir / name
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written
