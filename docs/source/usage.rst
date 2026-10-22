
===========
Usage Guide
===========

Preparing the activations
=========================

repscreen starts from activations that are already saved, so the
models themselves are never loaded.  For each of the two models,
record one activation vector per image, typically from the
penultimate layer, and write them next to the image list they came
from::

    activations/
      places365/
        activations.npy
        imagepaths.txt
      imagenet/
        activations.npy
        imagepaths.txt

Both models must have seen the same images in the same order; the
loader checks this and refuses to continue otherwise.  The directory
holding an image names its category, every category must contribute
the same number of exemplars, and a category's images must form one
contiguous block.

Screening
=========

.. code-block:: python

   from repscreen.config import (
       ActivationConfig, OutputConfig, ScreeningConfig,
       SelectionConfig,
   )
   from repscreen.screening.pipeline import run_screening

   config = ScreeningConfig(
       activations=ActivationConfig(
           activation_dir="data/activations",
           models=("places365", "imagenet"),
           dataset_path="data/dataset",
       ),
       selection=SelectionConfig(n_categories=12, n_exemplars=4),
       output=OutputConfig(
           output_dir="results", formats=("npy", "json", "csv"),
       ),
   )

   result = run_screening(config)

``result.stimulus_paths`` is the curated set and
``result.similarity`` the similarity between the two models over it.
See :doc:`modules/screening` for what each stage does and
:doc:`modules/output` for the files that are written.

Choosing a compactness measure
==============================

``R-squared`` is the default and the measure used in the published
analysis.  The other eight are listed in :doc:`modules/metrics` and
selected the same way:

.. code-block:: python

   from repscreen.config import CompactnessConfig

   config = ScreeningConfig(
       activations=...,
       compactness=CompactnessConfig(
           measure="Fisher_discriminant",
           difference_measure="rank",
       ),
   )

To compare measures without re-loading the activations, call
:func:`~repscreen.metrics.compactness.compute_compactness` on the
grouped activations directly:

.. code-block:: python

   from repscreen.data.loaders import load_activation_pair
   from repscreen.metrics.compactness import compute_compactness

   dataset = load_activation_pair(config.activations)
   scores = compute_compactness(
       dataset.activations["places365"], measure="CH_Index"
   )

Inspecting the outcome
======================

The quickest route is to let the run write the standard figures, by
setting ``figure_dir`` alongside ``output_dir``:

.. code-block:: python

   from repscreen.config import OutputConfig

   output=OutputConfig(output_dir="results", figure_dir="figures")

That writes ``RDMs.png``, ``catRDMs.png``, ``compactness.png`` and
``subset.png`` under ``figures/<run name>/``.  The same thing for a
result already in hand:

.. code-block:: python

   from repscreen.plotting.report import save_screening_figures

   save_screening_figures(result, "figures")

For anything beyond that set, build the figures yourself from the
result object:

.. code-block:: python

   from repscreen.plotting.compactness import (
       plot_compactness, plot_compactness_scatter,
   )
   from repscreen.plotting.embedding import tsne_model_comparison
   from repscreen.plotting.rdm import plot_rdm_pair
   from repscreen.plotting.stimuli import plot_stimulus_grid

   models = list(result.models)

   plot_compactness(
       result.compactness.sorted_compactness, models
   )
   plot_compactness_scatter(result.compactness.compactness, models)
   plot_rdm_pair(
       result.selected_rdms[models[0]],
       result.selected_rdms[models[1]],
       model_names=models,
       similarity=result.similarity,
   )
   plot_stimulus_grid(result.stimulus_paths)

   labels = [
       path.split("/")[-2] for path in result.stimulus_paths
   ]
   embeddings, figure = tsne_model_comparison(
       result.selected_rdms[models[0]],
       result.selected_rdms[models[1]],
       labels=labels,
       model_names=models,
   )

Every function returns its figure, so saving is up to you:

.. code-block:: python

   figure.savefig("figures/tsne.png", dpi=150, bbox_inches="tight")

Testing a selection against chance
==================================

A curated set of twelve categories is small, and a small set will
show some divergence between any two models.  Drawing random subsets
of the same size says how much of the curated set's divergence comes
from the selection:

.. code-block:: python

   import numpy as np
   from repscreen.screening.null import sample_category_similarity

   similarities, _ = sample_category_similarity(
       dataset.activations,
       tuple(result.models),
       n_samples=1000,
       n_categories=12,
       seed=0,
   )
   print(np.mean(similarities), result.similarity)

Checking the image statistics
=============================

A selection may also differ from the pool in plain image statistics.
:doc:`modules/signatures` measures per-category Fourier spectra and
CIELab chromaticity, which can be compared between the selected
categories and the rest.

From the command line
=====================

The same run without writing Python:

.. code-block:: bash

   repscreen --activation-dir data/activations \
             --models places365 imagenet \
             --dataset-path data/dataset \
             --n-categories 12 \
             --n-exemplars 4 \
             --output-dir results \
             --figure-dir figures

See :doc:`cli` for every argument.
