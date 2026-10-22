
=============
Configuration
=============

A screening run is described by one
:class:`~repscreen.config.ScreeningConfig`, which composes four
sub-configurations.  Each validates its fields on construction, so
an unusable combination is reported where it is written rather than
part-way through a run.

:class:`~repscreen.config.ActivationConfig`
    Where the two activation sets are, whether to scale activation
    vectors to unit length, and how to resolve the recorded image
    paths against a local copy of the dataset.  Exactly two
    distinct model names are required.

:class:`~repscreen.config.CompactnessConfig`
    Which compactness measure to use and how the two models'
    compactness is contrasted.

:class:`~repscreen.config.SelectionConfig`
    How many categories and exemplars the curated set holds, and
    which metrics build and compare the dissimilarity matrices.

:class:`~repscreen.config.OutputConfig`
    Where results are written, in which formats, whether the
    dissimilarity matrices are written alongside them, and where the
    standard figures go.

.. code-block:: python

   from repscreen.config import (
       ActivationConfig, CompactnessConfig, OutputConfig,
       ScreeningConfig, SelectionConfig,
   )

   config = ScreeningConfig(
       activations=ActivationConfig(
           activation_dir="data/activations",
           models=("places365", "imagenet"),
       ),
       compactness=CompactnessConfig(measure="R-squared"),
       selection=SelectionConfig(n_categories=12, n_exemplars=4),
       output=OutputConfig(
           output_dir="results", formats=("npy", "json", "csv"),
       ),
   )

``name`` defaults to the two model names joined by an underscore and
becomes the subdirectory results are written to, so several runs can
share one output root.

Resolving image paths
=====================

The image paths in ``imagepaths.txt`` are the ones recorded when the
activations were computed, which may name a directory that no longer
exists.  Setting ``dataset_path`` points them at a local copy; when
``replace_path_prefix`` is also given, that prefix is stripped and
the remainder joined to ``dataset_path``, and otherwise only the
category directory and file name are kept.
