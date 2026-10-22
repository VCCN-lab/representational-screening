
======================
Command-line Interface
======================

``repscreen`` runs a screening without writing any Python.  The two
required arguments name where the activations are and which two
models to compare:

.. code-block:: bash

   repscreen --activation-dir data/activations \
             --models places365 imagenet \
             --output-dir results

Everything else has a default matching the published analysis:
twelve categories, four exemplars each, ``R-squared`` compactness
contrasted with ``normalizedDiff``, squared Euclidean
dissimilarity, and Pearson correlation between the two models'
dissimilarity matrices.

Adding ``--figure-dir`` writes the standard figures alongside the
results:

.. code-block:: bash

   repscreen --activation-dir data/activations \
             --models places365 imagenet \
             --dataset-path data/dataset \
             --output-dir results \
             --figure-dir figures

``--dataset-path`` is what lets the stimulus grid find the images,
since the recorded paths may name the machine the activations were
computed on.

``--figure-dir`` sets ``OutputConfig.figure_dir``, so the command and
a Python script produce figures through the same code path; see
:doc:`plotting`.

See :doc:`../cli` for the full argument reference, and
:func:`~repscreen.cli.main.config_from_args` for how the arguments
map onto a :class:`~repscreen.config.ScreeningConfig`.
