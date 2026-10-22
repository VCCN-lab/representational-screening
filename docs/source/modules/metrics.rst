
=======
Metrics
=======

Dissimilarity matrices
======================

A representational dissimilarity matrix records how far apart a
model holds every pair of items.
:func:`~repscreen.metrics.rdm.compute_rdm` builds one from an array
of activations with ``L2squared``, ``L2``, ``pearson`` or
``dotproduct`` dissimilarity; appending ``_normalize`` to the metric
name scales every activation vector to unit length first.

:func:`~repscreen.metrics.rdm.compare_rdms` compares two such
matrices over their strict upper triangle, so the zero diagonal and
the duplicated lower triangle do not inflate the result.

Column correlations
===================

:func:`~repscreen.metrics.rdm.column_correlations` is the measure
both screening stages are built on.  Each column of a dissimilarity
matrix describes how one item relates to all the others, so
correlating the two models' columns gives a per-item score: a low
value marks an item whose relational position differs between the
models.  The diagonal is dropped before the columns are centred by
the grand mean and scaled to unit norm.

Compactness
===========

Compactness asks how tightly a category's exemplars cluster relative
to the rest of the activation space.  Nine measures are available
through :func:`~repscreen.metrics.compactness.compute_compactness`
and registered in
:data:`~repscreen.metrics.compactness.COMPACTNESS_MEASURES`:

``R-squared``
    Within-category scatter relative to the scatter of the whole
    space.  The default.

``R-squared_adjusted``
    Within-category scatter relative to the mean separation of the
    category centroids.

``Fisher_discriminant``
    Within-category scatter relative to the scatter about the other
    categories' centroids.

``CH_Index``
    Displacement of the category centroid from the global centroid,
    relative to within-category scatter.

``CH_Index_adapted``
    The same quantities, expressed on the scale of ``R-squared``.

``silhouette_score``
    Within-category exemplar distances against the mean distance to
    the nearest other category.

``global_silhouette_score``
    The same, against the pooled distance to all other categories.

``simplified_silhouette_score``
    The same, using centroid distances in place of exemplar pairs.

``Davies-Bouldin_Index``
    Worst-case overlap with any other category.  Unlike the others,
    a higher value marks a *less* compact category.

Only eight of the nine give an independent ranking:
``CH_Index_adapted`` is ``1 - 1 / CH_Index``, a monotone increasing
transform, so those two always order the categories identically. They
differ only in scale, with ``CH_Index_adapted`` on the same scale as
``R-squared``.

How much the remaining measures differ depends on the data. On
activations scaled to unit length, which is what
:class:`~repscreen.config.ActivationConfig` does by default, the
measures rank categories almost identically, because removing the
magnitudes leaves the angular spread as the one thing they all
respond to. Keeping the magnitudes separates them. The
:ref:`sphx_glr_auto_examples_plot_compactness_measures.py` example
shows both regimes side by side.

Contrasting two models
======================

:func:`~repscreen.screening.categories.compactness_difference`
contrasts the two models' compactness per category in one of two
ways.  ``normalizedDiff`` centres each model's scores and scales
them to unit maximum absolute value before subtracting, which keeps
measures on different scales comparable.  ``rank`` subtracts the
categories' compactness ranks instead, which ignores the size of
the difference and keeps only its order.

Centred kernel alignment
========================

:mod:`repscreen.metrics.cka` implements linear and RBF centred
kernel alignment, an alternative to comparing dissimilarity matrices
that is invariant to isotropic scaling and orthogonal transformation
of either feature set.

Decompositions
==============

:mod:`repscreen.metrics.decomposition` holds three linear-algebra
helpers for looking at an activation space directly rather than
through a dissimilarity matrix.
:func:`~repscreen.metrics.decomposition.principal_components`
decomposes a data matrix into its principal components and reports
how much variance each carries.
:func:`~repscreen.metrics.decomposition.classical_mds` embeds a
distance matrix, keeping the dimensions with a positive eigenvalue,
which makes it a deterministic alternative to the t-SNE embeddings
in :mod:`repscreen.plotting.embedding`.
:func:`~repscreen.metrics.decomposition.procrustes` aligns one
configuration of points to another up to translation, rotation,
reflection and uniform scaling, and returns the residual as a
scale-free disparity.

Across model instances
======================

When a set of models trained from different seeds is held in one
array,
:func:`~repscreen.metrics.layers.correlations_per_model` averages
each instance's activations over one axis and correlates the
resulting patterns, giving one correlation matrix per instance.
