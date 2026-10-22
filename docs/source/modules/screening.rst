
=================
The Screening Run
=================

A screening run takes two models' activations over the same
category-structured image set and returns a small stimulus set whose
representational geometry differs as much as possible between them.
:func:`~repscreen.screening.pipeline.run_screening` drives the whole
sequence.

How the run works
=================

1. **Loading** --
   :func:`~repscreen.data.loaders.load_activation_pair` reads
   ``activations.npy`` and ``imagepaths.txt`` for each of the two
   models, checks that both saw the same images in the same order,
   scales every activation vector to unit length, and groups the
   activations into one block per category.

2. **Compactness** --
   :func:`~repscreen.metrics.compactness.compute_compactness_pair`
   scores, for each model separately, how tightly every category's
   exemplars cluster relative to the rest of the activation space.

3. **Category screening** --
   :func:`~repscreen.screening.categories.screen_categories`
   combines two criteria.  The relational criterion correlates the
   two models' category-level dissimilarity matrices column by
   column, so a low value marks a category the models place
   differently among the others.  The compactness criterion
   contrasts the two models' compactness per category.  The combined
   score rewards a large absolute compactness difference and
   penalises a high relational correlation; the highest-scoring
   categories are kept.

4. **Exemplar screening** --
   :func:`~repscreen.screening.exemplars.screen_exemplars` builds
   the dissimilarity matrix over every exemplar of the selected
   categories, correlates the two models' matrices column by column
   again, and within each category keeps the exemplars with the
   lowest correlation.

5. **Reporting** -- the two models' dissimilarity matrices over the
   curated set are compared, giving the similarity the curation was
   trying to minimise, and the result is written in the requested
   formats.

Input layout
============

The activations of each model live in their own subdirectory::

    activations/
      places365/
        activations.npy     # shape (n_images, n_features)
        imagepaths.txt      # one path per image, same order
      imagenet/
        activations.npy
        imagepaths.txt

``activations.npy`` holds one row per presented image; trailing axes
are flattened, so an array of shape ``(n_images, n_features, 1, 1)``
is accepted.  ``imagepaths.txt`` records the images in presentation
order, and the directory holding an image names its category.  Every
category must contribute the same number of exemplars, and its
images must form one contiguous block.

Running it
==========

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
       selection=SelectionConfig(
           n_categories=12, n_exemplars=4,
       ),
       output=OutputConfig(output_dir="results"),
   )

   result = run_screening(config)
   print(result.selected_category_names)
   print(result.similarity)

What comes back
===============

:class:`~repscreen.screening.pipeline.ScreeningResult` carries both
stages' output.  ``selected_category_names`` and
``stimulus_paths`` are the curated set; ``score``,
``correlations`` and ``difference`` give one value per category, so
a selection can be inspected or re-cut without re-running the
screening; ``full_rdms`` and ``selected_rdms`` hold the
dissimilarity matrices at the category and curated-set level.

Null distributions
==================

How much of a curated set's divergence comes from the selection
rather than from its size is answered by drawing random subsets of
the same size.  :mod:`repscreen.screening.null` samples at three
levels: random entries of a pair of dissimilarity matrices, random
images, or random whole categories.

.. code-block:: python

   from repscreen.screening.null import sample_category_similarity

   similarities, drawn = sample_category_similarity(
       dataset.activations,
       ("places365", "imagenet"),
       n_samples=1000,
       n_categories=12,
       seed=0,
   )
