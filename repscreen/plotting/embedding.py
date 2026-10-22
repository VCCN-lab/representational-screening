"""Two-dimensional embeddings of dissimilarity matrices."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder


def tsne_from_rdm(
    rdm: np.ndarray,
    perplexity=None,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Embed a dissimilarity matrix in two dimensions with t-SNE.

    The matrix is passed to t-SNE as precomputed distances, so the
    embedding reflects the model's own dissimilarity structure.

    Parameters
    ----------
    rdm : numpy.ndarray
        Square dissimilarity matrix.
    perplexity : float or None
        t-SNE perplexity.  When None, 50 is used for more than 100
        items and 30 otherwise.  A value too large for the number of
        items is reduced to the largest admissible one.
    n_iter : int
        Number of optimisation iterations.
    random_state : int
        Seed for the embedding.

    Returns
    -------
    numpy.ndarray
        Embedding of shape ``(n_items, 2)``.

    Raises
    ------
    ValueError
        If ``rdm`` is not square.
    """
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError(
            f"A dissimilarity matrix must be square, got shape "
            f"{rdm.shape}."
        )

    size = len(rdm)
    if perplexity is None:
        perplexity = 50 if size > 100 else 30
    perplexity = min(perplexity, size - 1)

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init="random",
    )
    return tsne.fit_transform(rdm)


def _ordered_labels(labels: Sequence[str]):
    """Encode labels as integers in order of first appearance."""
    unique = []
    seen = set()
    for label in labels:
        if label not in seen:
            unique.append(label)
            seen.add(label)
    lookup = {label: i for i, label in enumerate(unique)}
    encoded = np.array([lookup[label] for label in labels])
    return unique, encoded


def _marker_style(n_items: int):
    """Choose a marker size and opacity for ``n_items`` points."""
    if n_items > 10000:
        return 2, 0.6
    if n_items > 100:
        return 50, 0.7
    return 100, 0.9


def plot_tsne(
    embedding: np.ndarray,
    labels: Sequence[str],
    ax=None,
    title: str = "t-SNE of representational dissimilarity",
    cmap: str = "tab20",
    figsize=(8, 5),
):
    """Scatter one t-SNE embedding, coloured by category.

    Parameters
    ----------
    embedding : numpy.ndarray
        Embedding of shape ``(n_items, 2)``.
    labels : sequence of str
        One category label per item.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  A new figure is created when None.
    title : str
        Title of the panel.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple of float
        Figure size in inches, used when ``ax`` is None.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The figure the scatter was drawn on.
    ax : matplotlib.axes.Axes
        The axes the scatter was drawn on.

    Raises
    ------
    ValueError
        If there is not one label per item.
    """
    if len(labels) != len(embedding):
        raise ValueError(
            f"Needs one label per item, got {len(labels)} labels "
            f"for {len(embedding)} items."
        )

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.get_figure()

    _, encoded = _ordered_labels(labels)
    size, alpha = _marker_style(len(embedding))
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=encoded,
        cmap=cmap,
        alpha=alpha,
        s=size,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    figure.tight_layout()
    return figure, ax


def plot_tsne_comparison(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    labels: Sequence[str],
    model_names: Sequence[str],
    title: str = "t-SNE of representational dissimilarity",
    cmap: str = "tab20",
    figsize=(9.2, 4),
    legend_limit: int = 100,
):
    """Scatter two models' t-SNE embeddings side by side.

    Both panels use the same colour per category, so a category that
    clusters under one model and scatters under the other is visible
    as a change in the layout of one colour.

    Parameters
    ----------
    embedding1, embedding2 : numpy.ndarray
        Embeddings of shape ``(n_items, 2)``.
    labels : sequence of str
        One category label per item.
    model_names : sequence of str
        The two model names, used as panel titles.
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple of float
        Figure size in inches.
    legend_limit : int
        A legend is drawn when there are fewer items than this.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.

    Raises
    ------
    ValueError
        If there is not one label per item.
    """
    if len(labels) != len(embedding1):
        raise ValueError(
            f"Needs one label per item, got {len(labels)} labels "
            f"for {len(embedding1)} items."
        )

    unique, encoded = _ordered_labels(labels)
    size, alpha = _marker_style(len(embedding1))

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    scatter = None
    for panel, (embedding, name) in enumerate(
        zip((embedding1, embedding2), model_names)
    ):
        scatter = axes[panel].scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=encoded,
            cmap=cmap,
            alpha=alpha,
            s=size,
        )
        axes[panel].set_title(name)
        axes[panel].set_xlabel("t-SNE component 1")
    axes[0].set_ylabel("t-SNE component 2")
    figure.suptitle(title)

    if len(embedding1) < legend_limit:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=scatter.cmap(scatter.norm(i)),
                markersize=8,
                label=label,
            )
            for i, label in enumerate(unique)
        ]
        axes[1].legend(
            handles=handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    figure.tight_layout()
    return figure


def tsne_model_comparison(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    labels: Sequence[str],
    model_names: Sequence[str],
    title: str = "t-SNE of representational dissimilarity",
    random_state: int = 42,
):
    """Embed both models' dissimilarity matrices and plot them.

    Parameters
    ----------
    rdm1, rdm2 : numpy.ndarray
        Square dissimilarity matrices of equal shape.
    labels : sequence of str
        One category label per item.
    model_names : sequence of str
        The two model names.
    title : str
        Figure title.
    random_state : int
        Seed for both embeddings.

    Returns
    -------
    embeddings : list of numpy.ndarray
        The two embeddings.
    figure : matplotlib.figure.Figure
        The two-panel figure.
    """
    embeddings = [
        tsne_from_rdm(matrix, random_state=random_state)
        for matrix in (rdm1, rdm2)
    ]
    figure = plot_tsne_comparison(
        embeddings[0],
        embeddings[1],
        labels,
        model_names,
        title=title,
    )
    return embeddings, figure


def cluster_quality(embedding: np.ndarray, labels: Sequence[str]):
    """Score how well an embedding separates the categories.

    Parameters
    ----------
    embedding : numpy.ndarray
        Embedding of shape ``(n_items, 2)``.
    labels : sequence of str
        One category label per item.

    Returns
    -------
    dict
        ``silhouette`` of the labelled grouping, the
        ``adjusted_rand_index`` between the labels and a k-means
        partition of the embedding, and the ``n_clusters`` the
        labels define.

    Raises
    ------
    ValueError
        If there is not one label per item, or if the labels do not
        define between 2 and ``n_items - 1`` clusters.  A single
        cluster, or one cluster per item, leaves the silhouette
        undefined.
    """
    if len(labels) != len(embedding):
        raise ValueError(
            f"Needs one label per item, got {len(labels)} labels "
            f"for {len(embedding)} items."
        )

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    n_clusters = len(np.unique(encoded))
    n_items = len(embedding)

    if not 2 <= n_clusters <= n_items - 1:
        raise ValueError(
            f"Scoring {n_items} items needs between 2 and "
            f"{n_items - 1} distinct labels, got {n_clusters}. "
            f"The silhouette is undefined for a single cluster or "
            f"for one cluster per item."
        )

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    )
    predicted = kmeans.fit_predict(embedding)

    return {
        "silhouette": float(silhouette_score(embedding, encoded)),
        "adjusted_rand_index": float(
            adjusted_rand_score(encoded, predicted)
        ),
        "n_clusters": n_clusters,
    }
