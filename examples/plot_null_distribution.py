"""
Testing a selection against chance
==================================

A curated set of twelve categories is small, and a small set will
show some divergence between any two models.  Drawing random subsets
of the same size says how much of the curated set's divergence comes
from the selection itself.
"""

# %%
# Screening first
# ---------------

import matplotlib.pyplot as plt
import numpy as np

from _synthetic import MODELS, N_PER_CATEGORY, synthetic_activations
from repscreen.metrics.compactness import compute_compactness_pair
from repscreen.metrics.rdm import compare_rdms
from repscreen.screening.categories import screen_categories
from repscreen.screening.exemplars import screen_exemplars
from repscreen.screening.null import (
    sample_category_similarity,
    sample_rdm_similarity,
)

activations, category_names = synthetic_activations(seed=0)
compactness = compute_compactness_pair(
    activations, MODELS, category_names
)
selection = screen_categories(
    activations, MODELS, compactness, category_names, n_categories=12
)
exemplars = screen_exemplars(
    activations,
    MODELS,
    selection.selected,
    n_exemplars=4,
    n_per_category=N_PER_CATEGORY,
)
curated = compare_rdms(
    exemplars.selected_rdms[MODELS[0]],
    exemplars.selected_rdms[MODELS[1]],
    metric="pearson",
)
print(f"curated set similarity: {curated:+.3f}")

# %%
# Random categories
# -----------------
#
# Twelve categories drawn at random, 500 times, each time comparing
# the two models over all their exemplars.  This is the null for the
# category stage.

category_null, _ = sample_category_similarity(
    activations,
    MODELS,
    n_samples=500,
    n_categories=12,
    seed=0,
)
print(
    f"random categories: {category_null.mean():+.3f} "
    f"+/- {category_null.std():.3f}"
)

# %%
# Random exemplars
# ----------------
#
# Within the categories the screening selected, 48 exemplars drawn at
# random rather than chosen. This isolates what the exemplar stage
# contributed.

exemplar_null, _ = sample_rdm_similarity(
    exemplars.full_rdms[MODELS[0]],
    exemplars.full_rdms[MODELS[1]],
    n_samples=500,
    subset_size=48,
    seed=0,
)
print(
    f"random exemplars: {exemplar_null.mean():+.3f} "
    f"+/- {exemplar_null.std():.3f}"
)

# %%
# Both nulls against the curated set
# ----------------------------------
#
# The curated set sits below both distributions, and the gap to each
# is what that stage of the screening bought.

figure, axes = plt.subplots(figsize=(8, 4))
for values, label, colour in (
    (category_null, "random categories", "steelblue"),
    (exemplar_null, "random exemplars", "darkorange"),
):
    axes.hist(
        values,
        bins=30,
        alpha=0.6,
        label=label,
        color=colour,
        density=True,
    )
axes.axvline(
    curated,
    color="firebrick",
    linewidth=2,
    label="curated set",
)
axes.set_xlabel("Similarity between the two models")
axes.set_ylabel("Density")
axes.set_title("Curated set against two null distributions")
axes.legend()
figure.tight_layout()
plt.show()

# %%
# Reporting it
# ------------
#
# The proportion of random subsets at least as divergent as the
# curated set is the empirical p-value of the selection.

for values, label in (
    (category_null, "categories"),
    (exemplar_null, "exemplars"),
):
    tail = float(np.mean(values <= curated))
    print(f"{label:12s} proportion at or below curated: {tail:.4f}")
