"""Synthetic activation sets shared by the gallery examples.

The activations the paper screens are not distributed with the
repository, so the examples build a pair of activation sets with the
structure screening looks for: the two models agree about where the
categories sit, and disagree about which categories form tight
clusters.
"""

import numpy as np

MODELS = ("model_a", "model_b")
N_CATEGORIES = 24
N_PER_CATEGORY = 10
N_FEATURES = 32

#: Range the per-category angular spread is drawn from.  The second
#: model's spread is this range's reflection of the first model's, so
#: the two are anti-correlated by construction.
ANGLE_RANGE = (0.05, 1.2)


def _unit(vectors):
    """Scale the last axis of ``vectors`` to unit length."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def synthetic_activations(seed: int = 0, normalize: bool = True):
    """Build a pair of category-structured activation sets.

    Both models place the categories at the same centroids, so they
    agree about where the categories sit relative to one another.
    They differ in how tightly each category's exemplars cluster
    around its centroid: the second model's per-category spread is
    the reflection of the first's within :data:`ANGLE_RANGE`, so a
    category that is tight under one model is diffuse under the
    other. That disagreement is what the screening exploits.

    The spread is angular, applied before the vectors are scaled to
    unit length. Screening normalises activations, which removes
    magnitude differences, so a purely radial spread would be washed
    out and the two models would end up equally compact.

    Parameters
    ----------
    seed : int
        Seed for the random generator.
    normalize : bool
        Whether to scale every activation vector to unit length, as
        :class:`~repscreen.config.ActivationConfig` does by default.
        Pass False to keep the magnitude differences, which is how
        the compactness measures can be compared in both regimes.

    Returns
    -------
    activations : dict
        Mapping from model name to an array of shape
        ``(N_CATEGORIES, N_PER_CATEGORY, N_FEATURES)``.
    category_names : list of str
        Category labels in category order.
    """
    rng = np.random.default_rng(seed)
    centroids = _unit(
        rng.normal(size=(N_CATEGORIES, 1, N_FEATURES))
    )

    low, high = ANGLE_RANGE
    angle = rng.uniform(low, high, size=(N_CATEGORIES, 1, 1))

    activations = {}
    for index, model in enumerate(MODELS):
        spread = angle if index == 0 else low + high - angle
        noise = rng.normal(
            size=(N_CATEGORIES, N_PER_CATEGORY, N_FEATURES)
        )
        values = centroids + spread * noise
        activations[model] = _unit(values) if normalize else values

    category_names = [
        f"{i:04d}_category{i}" for i in range(N_CATEGORIES)
    ]
    return activations, category_names
