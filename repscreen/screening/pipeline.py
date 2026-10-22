"""End-to-end representational screening run."""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..config import ScreeningConfig
from ..data.loaders import load_activation_pair
from ..metrics.compactness import (
    CompactnessResult,
    compute_compactness_pair,
)
from ..metrics.rdm import compare_rdms
from ..output.writers import save_results
from .categories import CategorySelection, screen_categories
from .exemplars import ExemplarSelection, screen_exemplars


@dataclass
class ScreeningResult:
    """Everything one screening run produces.

    Parameters
    ----------
    config : ScreeningConfig
        The configuration the run was started from.
    name : str
        Name of the run, used as the output subdirectory.
    models : tuple of str
        The two model names.
    category_names : list of str
        Every category label, in category order.
    n_per_category : int
        Number of exemplars each category holds.
    compactness : CompactnessResult
        Compactness of every category under both models.
    categories : CategorySelection
        Outcome of the category screening stage.
    exemplars : ExemplarSelection
        Outcome of the exemplar screening stage.
    similarity : float
        Similarity between the two models over the curated set.
    stimulus_paths : list of str
        Image paths of the curated set.
    """

    config: ScreeningConfig
    name: str
    models: Sequence[str]
    category_names: List[str]
    n_per_category: int
    compactness: CompactnessResult
    categories: CategorySelection
    exemplars: ExemplarSelection
    similarity: float
    stimulus_paths: List[str]

    @property
    def correlations(self) -> np.ndarray:
        """Relational criterion per category."""
        return self.categories.correlations

    @property
    def difference(self) -> np.ndarray:
        """Compactness difference per category."""
        return self.categories.difference.difference

    @property
    def score(self) -> np.ndarray:
        """Combined category score."""
        return self.categories.score

    @property
    def selected_categories(self) -> np.ndarray:
        """Indices of the selected categories, best first."""
        return self.categories.selected

    @property
    def selected_category_names(self) -> List[str]:
        """Labels of the selected categories, best first."""
        return self.categories.selected_names

    @property
    def full_rdms(self) -> Dict[str, np.ndarray]:
        """Dissimilarity matrices over all selected categories."""
        return self.exemplars.full_rdms

    @property
    def selected_rdms(self) -> Dict[str, np.ndarray]:
        """Dissimilarity matrices over the curated set."""
        return self.exemplars.selected_rdms

    @property
    def global_indices(self) -> np.ndarray:
        """Position of each curated stimulus in the image list."""
        return self.exemplars.global_indices


def run_screening(config: ScreeningConfig) -> ScreeningResult:
    """Run both screening stages and collect the curated set.

    Categories are scored on how differently the two models compact
    them and place them among the other categories, and within the
    selected categories the exemplars the models relate most
    differently are kept.

    Parameters
    ----------
    config : ScreeningConfig
        Activation sources, compactness measure, selection sizes
        and output settings.  Results are written when
        ``config.output.output_dir`` is set, and the standard
        figures when ``config.output.figure_dir`` is set.

    Returns
    -------
    ScreeningResult
        Both stages' outcomes, the curated stimulus paths, and the
        similarity between the models over the curated set.
    """
    dataset = load_activation_pair(config.activations)
    models = dataset.models

    compactness = compute_compactness_pair(
        dataset.activations,
        models,
        dataset.category_names,
        measure=config.compactness.measure,
    )

    categories = screen_categories(
        dataset.activations,
        models,
        compactness,
        dataset.category_names,
        n_categories=config.selection.n_categories,
        dissimilarity_metric=config.selection.dissimilarity_metric,
        difference_measure=config.compactness.difference_measure,
    )

    exemplars = screen_exemplars(
        dataset.activations,
        models,
        categories.selected,
        n_exemplars=config.selection.n_exemplars,
        n_per_category=dataset.n_per_category,
        dissimilarity_metric=config.selection.dissimilarity_metric,
    )

    similarity = compare_rdms(
        exemplars.selected_rdms[models[0]],
        exemplars.selected_rdms[models[1]],
        metric=config.selection.similarity_metric,
    )

    result = ScreeningResult(
        config=config,
        name=config.name,
        models=models,
        category_names=dataset.category_names,
        n_per_category=dataset.n_per_category,
        compactness=compactness,
        categories=categories,
        exemplars=exemplars,
        similarity=similarity,
        stimulus_paths=[
            dataset.image_paths[index]
            for index in exemplars.global_indices
        ],
    )

    save_results(
        result,
        config.output.output_dir,
        formats=config.output.formats,
        save_rdms=config.output.save_rdms,
    )

    # Imported here so that using the pipeline without figures does
    # not pull in matplotlib.
    if config.output.figure_dir is not None:
        from ..plotting.report import save_screening_figures

        save_screening_figures(result, config.output.figure_dir)

    return result
