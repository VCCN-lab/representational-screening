"""
Comparing compactness measures
==============================

The compactness measure decides which categories the screening
considers differently represented, so it is worth seeing how much the
nine available measures agree before settling on one.
"""

# %%
# Scoring every measure
# ---------------------

import matplotlib.pyplot as plt
import numpy as np

from _synthetic import MODELS, synthetic_activations
from repscreen.metrics.compactness import (
    COMPACTNESS_MEASURES,
    compute_compactness,
    compute_compactness_pair,
)
from repscreen.plotting.compactness import plot_compactness_scatter
from repscreen.screening.categories import screen_categories

activations, category_names = synthetic_activations(seed=0)

scores = {
    measure: compute_compactness(
        activations[MODELS[0]], measure=measure
    )
    for measure in COMPACTNESS_MEASURES
}
for measure, values in scores.items():
    print(
        f"{measure:30s} range "
        f"[{values.min():+.3f}, {values.max():+.3f}]"
    )

# %%
# How much do they agree?
# -----------------------
#
# The measures live on different scales, so they are compared by
# rank.  ``Davies-Bouldin_Index`` is negated first, because for that
# measure a higher value marks a *less* compact category.
#
# Two regimes are shown.  The left panel uses the activations as the
# screening sees them, scaled to unit length; the right keeps the
# magnitudes.  The measures nearly coincide once magnitude is
# removed, and separate when it is kept, so the near-uniform left
# panel is a consequence of normalisation rather than evidence that
# the measures are interchangeable.

names = list(COMPACTNESS_MEASURES)


def rank_agreement(acts):
    """Rank-correlate every pair of measures on one model."""
    scored = {
        name: compute_compactness(acts, measure=name)
        for name in names
    }
    ranked = np.array(
        [
            np.argsort(
                np.argsort(
                    -scored[name]
                    if name == "Davies-Bouldin_Index"
                    else scored[name]
                )
            )
            for name in names
        ]
    )
    return np.corrcoef(ranked)


unnormalised, _ = synthetic_activations(seed=0, normalize=False)
panels = (
    ("unit length\n(as screening runs)", activations[MODELS[0]]),
    ("magnitudes kept", unnormalised[MODELS[0]]),
)

figure, axes = plt.subplots(1, 2, figsize=(13, 6))
for axis, (title, acts) in zip(axes, panels):
    agreement = rank_agreement(acts)
    off = agreement[np.triu_indices(len(names), k=1)]
    image = axis.imshow(agreement, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=90)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names if title.startswith("unit") else [])
    axis.set_title(f"{title}\nmin {off.min():+.2f}, mean {off.mean():+.2f}")
figure.colorbar(image, ax=axes, shrink=0.7)
plt.show()

# %%
# One tie is unavoidable
# ----------------------
#
# ``CH_Index_adapted`` is ``1 - 1 / CH_Index``, and that transform is
# monotone increasing, so those two measures can never disagree about
# the order of the categories.  Of the nine measures only eight give
# an independent ranking.

reference = rank_agreement(activations[MODELS[0]])
pair = (
    names.index("CH_Index"),
    names.index("CH_Index_adapted"),
)
print(
    f"CH_Index vs CH_Index_adapted: "
    f"{reference[pair[0], pair[1]]:+.4f}"
)

# %%
# Do the models agree with each other?
# ------------------------------------
#
# Under the default measure, plotting one model's compactness against
# the other's shows how much room the screening has.  These two models
# are built to disagree, so the correlation is negative: the
# categories one model holds tightly are the ones the other spreads
# out.

compactness = compute_compactness_pair(
    activations, MODELS, category_names, measure="R-squared"
)
figure, _ = plot_compactness_scatter(
    compactness.compactness, MODELS
)
plt.show()

# %%
# Does the measure change the selection?
# --------------------------------------
#
# Running the category stage with each measure and comparing the
# selections shows how much the choice matters in practice.

reference = None
for measure in ("R-squared", "Fisher_discriminant", "CH_Index"):
    pair = compute_compactness_pair(
        activations, MODELS, category_names, measure=measure
    )
    selected = set(
        screen_categories(
            activations,
            MODELS,
            pair,
            category_names,
            n_categories=12,
        ).selected_names
    )
    if reference is None:
        reference = selected
        print(f"{measure:22s} reference selection")
    else:
        shared = len(reference & selected)
        print(f"{measure:22s} shares {shared}/12 categories")
