"""
A full screening run
====================

Screen a pair of models down to a small set of categories and
exemplars, and look at what the selection did to the agreement
between the two models.

This example runs the two screening stages directly on in-memory
activations.  To screen your own activations from disk, build a
:class:`~repscreen.config.ScreeningConfig` and call
:func:`~repscreen.screening.pipeline.run_screening` instead; see the
usage guide.
"""

# %%
# The activation sets
# -------------------
#
# Two models, 24 categories of 10 exemplars each.  The models agree
# about where the category centroids sit but disagree about which
# categories form tight clusters, which is what the screening looks
# for.

import matplotlib.pyplot as plt
import numpy as np

from _synthetic import MODELS, N_PER_CATEGORY, synthetic_activations
from repscreen.metrics.compactness import compute_compactness_pair
from repscreen.metrics.rdm import compare_rdms
from repscreen.plotting.compactness import plot_compactness
from repscreen.plotting.rdm import plot_rdm_pair
from repscreen.screening.categories import screen_categories
from repscreen.screening.exemplars import screen_exemplars

activations, category_names = synthetic_activations(seed=0)
print(f"{len(category_names)} categories of {N_PER_CATEGORY}")
print(f"activation shape: {activations[MODELS[0]].shape}")

# %%
# Compactness
# -----------
#
# How tightly each category's exemplars cluster, under each model
# separately.  Sorted on each model's own scores the two curves look
# alike, because neither model is globally more compact than the
# other: they cover the same range and disagree about *which*
# categories fall where in it.  That disagreement is what the
# screening is after, and it is visible in the scatter plot of the
# compactness comparison example rather than here.

compactness = compute_compactness_pair(
    activations, MODELS, category_names, measure="R-squared"
)
figure, _ = plot_compactness(
    compactness.sorted_compactness,
    MODELS,
    xlabel="Categories, sorted per model",
    ylabel="Compactness (R-squared)",
)
plt.show()

# %%
# Screening the categories
# ------------------------
#
# The combined score rewards a large compactness difference between
# the models and penalises agreement about where the category sits
# among the others.

selection = screen_categories(
    activations,
    MODELS,
    compactness,
    category_names,
    n_categories=12,
)
for name in selection.selected_names[:5]:
    index = category_names.index(name)
    print(
        f"{name}: score {selection.score[index]:+.3f} "
        f"(difference {selection.difference.difference[index]:+.3f}, "
        f"correlation {selection.correlations[index]:+.3f})"
    )

# %%
# Where the selected categories sit
# ---------------------------------
#
# Plotting the combined score against the category index shows the
# selection is not a contiguous block: it picks out individual
# categories across the pool.

figure, axes = plt.subplots(figsize=(8, 3))
axes.bar(
    np.arange(len(category_names)),
    selection.score,
    color=[
        "firebrick" if i in set(selection.selected) else "lightgrey"
        for i in range(len(category_names))
    ],
)
axes.set_xlabel("Category")
axes.set_ylabel("Combined score")
axes.set_title("Selected categories in red")
figure.tight_layout()
plt.show()

# %%
# Screening the exemplars
# -----------------------
#
# Within the selected categories, the exemplars the two models relate
# most differently are kept: four per category, for 48 stimuli.

exemplars = screen_exemplars(
    activations,
    MODELS,
    selection.selected,
    n_exemplars=4,
    n_per_category=N_PER_CATEGORY,
)
print(f"curated {len(exemplars.global_indices)} stimuli")

# %%
# What the curation achieved
# --------------------------
#
# Comparing the two models over all exemplars of the selected
# categories, and then over the curated set alone, shows what the
# exemplar stage contributed on top of the category stage.

over_categories = compare_rdms(
    exemplars.full_rdms[MODELS[0]],
    exemplars.full_rdms[MODELS[1]],
    metric="pearson",
)
over_curated = compare_rdms(
    exemplars.selected_rdms[MODELS[0]],
    exemplars.selected_rdms[MODELS[1]],
    metric="pearson",
)
print(f"selected categories, all exemplars: {over_categories:+.3f}")
print(f"curated set:                        {over_curated:+.3f}")

# %%
# The curated dissimilarity matrices
# ----------------------------------
#
# The two models' geometries over the 48 curated stimuli, side by
# side.

figure = plot_rdm_pair(
    exemplars.selected_rdms[MODELS[0]],
    exemplars.selected_rdms[MODELS[1]],
    model_names=MODELS,
    similarity=over_curated,
)
plt.show()
