
========
Plotting
========

Every plotting function returns its figure or axes rather than
showing or saving it, so the same call works from a script, a
notebook and the documentation gallery.  Functions that draw a
single panel also accept an existing ``ax``, which makes them
composable into a larger layout.

The standard set
================

:func:`~repscreen.plotting.report.save_screening_figures` writes the
four figures a screening run is usually inspected through, into a
subdirectory of the given root named after the run:

``RDMs.png``
    The two models' dissimilarity matrices over the curated set,
    with their similarity in the title.
``catRDMs.png``
    The same, over every exemplar of the selected categories.
``compactness.png``
    Compactness per category, each model sorted on its own.
``subset.png``
    The curated stimuli, one column per category.

It is called for you when ``OutputConfig.figure_dir`` is set, so one
call produces results and figures together:

.. code-block:: python

   from repscreen.config import (
       ActivationConfig, OutputConfig, ScreeningConfig,
   )
   from repscreen.screening.pipeline import run_screening

   config = ScreeningConfig(
       activations=ActivationConfig(
           activation_dir="data/activations",
           models=("places365", "imagenet"),
           dataset_path="data/dataset",
       ),
       output=OutputConfig(
           output_dir="results", figure_dir="figures",
       ),
   )
   result = run_screening(config)

Or call it yourself, which is useful when the figures are wanted for
a result already in hand:

.. code-block:: python

   from repscreen.plotting.report import save_screening_figures

   save_screening_figures(result, "figures")

``subset.png`` needs the stimulus images on disk.  The recorded image
paths may name the machine the activations were computed on, so
``dataset_path`` is what lets the grid be drawn; without it that one
figure is skipped with a warning and the other three are still
written.

The functions below build the individual figures, for a layout of
your own.

Dissimilarity matrices
======================

:func:`~repscreen.plotting.rdm.plot_rdm` draws one matrix as a
heatmap.  :func:`~repscreen.plotting.rdm.plot_rdm_pair` draws two
side by side and reports their similarity in the figure title.

Compactness
===========

:func:`~repscreen.plotting.compactness.plot_compactness` draws
compactness per category, one line per model; pass
``sorted_compactness`` to see each model's own ordering and
``compactness`` to keep the categories aligned between the models.
:func:`~repscreen.plotting.compactness.plot_compactness_scatter`
plots one model's compactness against the other's and reports their
correlation.

Embeddings
==========

:func:`~repscreen.plotting.embedding.tsne_from_rdm` embeds a
dissimilarity matrix in two dimensions, passing it to t-SNE as
precomputed distances so the layout reflects the model's own
geometry.
:func:`~repscreen.plotting.embedding.plot_tsne_comparison` draws two
models' embeddings with the same colour per category, so a category
that clusters under one model and scatters under the other shows up
as a change in the layout of one colour.
:func:`~repscreen.plotting.embedding.cluster_quality` returns the
silhouette of the labelled grouping and the adjusted Rand index
against a k-means partition of the embedding.

Stimuli
=======

:func:`~repscreen.plotting.stimuli.plot_stimulus_grid` draws a
curated set column by column, so each column holds one category's
exemplars when the paths are grouped by category.  An image that
cannot be read is reported as a warning and skipped.

Signatures
==========

:func:`~repscreen.plotting.signatures.plot_category_signature` draws
a category's mean spectrum next to its pooled CIELab chromaticity,
with each sampled pixel drawn in its own colour.
