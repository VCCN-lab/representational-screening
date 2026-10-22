Examples Gallery
================

A collection of examples demonstrating how to screen a pair of vision
models down to an experiment-ready stimulus set with repscreen.

The activations used in the paper are not distributed with the
repository, so the examples build a small synthetic pair of
activation sets whose categories differ in compactness between the
two models.  Every example therefore runs unchanged, and swapping
``_synthetic_activations`` for
:func:`~repscreen.data.loaders.load_activation_pair` on your own
activation directory is the only change needed to use real data.
