"""
Inspecting a curated set
========================

Once a set is curated, the question is whether the two models really
lay it out differently.  A t-SNE of each model's dissimilarity matrix
makes that visible, and clustering scores put a number on it.
"""

# %%
# Screening first
# ---------------

import matplotlib.pyplot as plt

from _synthetic import MODELS, N_PER_CATEGORY, synthetic_activations
from repscreen.metrics.compactness import compute_compactness_pair
from repscreen.metrics.rdm import compare_rdms
from repscreen.plotting.embedding import (
    cluster_quality,
    tsne_model_comparison,
)
from repscreen.plotting.rdm import plot_rdm_pair
from repscreen.screening.categories import screen_categories
from repscreen.screening.exemplars import screen_exemplars

activations, category_names = synthetic_activations(seed=0)
compactness = compute_compactness_pair(
    activations, MODELS, category_names
)
selection = screen_categories(
    activations, MODELS, compactness, category_names, n_categories=8
)
exemplars = screen_exemplars(
    activations,
    MODELS,
    selection.selected,
    n_exemplars=6,
    n_per_category=N_PER_CATEGORY,
)

# %%
# Labelling the curated stimuli
# -----------------------------
#
# The exemplar stage returns the stimuli grouped by category, in the
# order the categories were selected, so the labels follow from the
# selection.

labels = [
    name
    for name in selection.selected_names
    for _ in range(6)
]
print(f"{len(labels)} stimuli across {len(set(labels))} categories")

# %%
# The two geometries
# ------------------

curated = compare_rdms(
    exemplars.selected_rdms[MODELS[0]],
    exemplars.selected_rdms[MODELS[1]],
    metric="pearson",
)
figure = plot_rdm_pair(
    exemplars.selected_rdms[MODELS[0]],
    exemplars.selected_rdms[MODELS[1]],
    model_names=MODELS,
    similarity=curated,
)
plt.show()

# %%
# The same stimuli in two embeddings
# ----------------------------------
#
# Both panels colour a stimulus by its category, so a category that
# clusters under one model and scatters under the other appears as a
# change in the layout of one colour.

embeddings, figure = tsne_model_comparison(
    exemplars.selected_rdms[MODELS[0]],
    exemplars.selected_rdms[MODELS[1]],
    labels=labels,
    model_names=MODELS,
    title="Curated set under both models",
    random_state=0,
)
plt.show()

# %%
# Scoring the separation
# ----------------------
#
# The silhouette says how well each model's embedding separates the
# categories, and the adjusted Rand index how well an unsupervised
# partition of the embedding recovers them.  A model that holds the
# categories apart scores higher on both.

for model, embedding in zip(MODELS, embeddings):
    scores = cluster_quality(embedding, labels)
    print(
        f"{model:16s} silhouette {scores['silhouette']:+.3f}, "
        f"ARI {scores['adjusted_rand_index']:+.3f}"
    )
