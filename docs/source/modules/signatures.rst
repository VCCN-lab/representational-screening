
===================
Category Signatures
===================

A curated stimulus set that separates two models may also differ
from the pool it was drawn from in plain image statistics.
:mod:`repscreen.signatures` measures two of them per category, so a
selection can be checked against them.

What is measured
================

**Spatial frequency.**
:func:`~repscreen.signatures.spectra.fourier_spectrum` reduces each
exemplar to luminance and transforms it with the origin at the
centre of the output, so low spatial frequencies sit in the middle
and high frequencies at the edges.  The exemplars' spectra are
averaged per category.

**Chromaticity.**
:func:`~repscreen.signatures.chromaticity.lab_samples` converts each
exemplar to CIELab and samples its pixels, and the samples are
pooled across the category.  Sampling keeps the size of the pooled
distribution independent of the images' resolution.

Framing
=======

Averaging spectra requires every exemplar to have the same shape, so
:func:`~repscreen.signatures.images.resize_and_center_crop` scales
the shorter side to ``target_size`` and crops the longer side
symmetrically.  An image already square at ``target_size`` passes
through unchanged.

Running it
==========

.. code-block:: python

   from repscreen.plotting.signatures import plot_category_signature
   from repscreen.signatures.analysis import (
       dataset_signatures, save_signatures,
   )

   signatures = dataset_signatures(
       "data/dataset",
       n_exemplars=50,
       target_size=448,
       seed=0,
   )
   save_signatures(signatures, "results/signatures.pkl")

   figure = plot_category_signature(signatures["0001_bird"])

The dataset root is expected to hold one subdirectory per category,
the same layout the screening run's image paths describe.
