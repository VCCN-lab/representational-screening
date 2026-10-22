"""Comparison of representations across model layers."""

from typing import Dict, List, Sequence

import numpy as np

from .rdm import compare_rdms


def compare_rdms_per_layer(
    rdms: Dict[str, Sequence],
    models: Sequence[str],
    metric: str = "cosine",
) -> Dict[str, Dict[str, List[float]]]:
    """Compare every pair of models layer by layer.

    Parameters
    ----------
    rdms : dict
        Mapping from model name to its dissimilarity matrices, one
        per layer and in layer order.
    models : sequence of str
        Model names to compare.  The result is keyed so that each
        pair appears once, under the model that comes first here.
    metric : str
        Metric passed to
        :func:`~repscreen.metrics.rdm.compare_rdms`.

    Returns
    -------
    dict
        ``result[model1][model2]`` holds one similarity per layer.

    Raises
    ------
    ValueError
        If the models do not have the same number of layers.
    """
    depths = {model: len(rdms[model]) for model in models}
    if len(set(depths.values())) > 1:
        raise ValueError(
            f"Every model must have the same number of layers, got "
            f"{depths}."
        )

    similarities: Dict[str, Dict[str, List[float]]] = {
        model: {} for model in models
    }
    for i, model1 in enumerate(models):
        for model2 in models[i:]:
            similarities[model1][model2] = [
                compare_rdms(
                    rdms[model1][layer],
                    rdms[model2][layer],
                    metric=metric,
                )
                for layer in range(depths[model1])
            ]
    return similarities


def correlations_per_model(
    activations,
    instance_axis: int = 0,
    mean_axis=2,
):
    """Correlate activation patterns within each model instance.

    A set of models trained from different seeds is held in one
    array; each instance's activations are averaged over one axis and
    the resulting patterns correlated with one another.

    Parameters
    ----------
    activations : numpy.ndarray
        Array whose ``instance_axis`` indexes the model instances.
        An array with fewer than three axes is treated as a single
        instance.
    instance_axis : int
        Axis indexing the model instances, moved to the front before
        averaging.
    mean_axis : int or None
        Axis of the reordered array to average over, so that one
        pattern remains per item.  No averaging is done when None.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_instances, n_items, n_items)``, one
        correlation matrix per instance.
    """
    activations = np.asarray(activations)
    if activations.ndim < 3:
        activations = activations[np.newaxis, :]
    activations = np.moveaxis(activations, instance_axis, 0)

    if mean_axis is None:
        patterns = activations
    else:
        patterns = np.mean(activations, axis=mean_axis)

    n_instances = len(activations)
    n_items = activations.shape[1]
    correlations = np.zeros((n_instances, n_items, n_items))
    for instance in range(n_instances):
        correlations[instance] = np.corrcoef(patterns[instance])
    return correlations
