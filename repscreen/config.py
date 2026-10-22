"""Configuration dataclasses for repscreen."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

VALID_DISSIMILARITY_METRICS = (
    "L2squared",
    "L2",
    "pearson",
    "dotproduct",
)
VALID_SIMILARITY_METRICS = ("cosine", "pearson", "spearman")
VALID_COMPACTNESS_MEASURES = (
    "R-squared",
    "R-squared_adjusted",
    "Fisher_discriminant",
    "CH_Index",
    "CH_Index_adapted",
    "silhouette_score",
    "global_silhouette_score",
    "simplified_silhouette_score",
    "Davies-Bouldin_Index",
)
VALID_DIFFERENCE_MEASURES = ("rank", "normalizedDiff")
VALID_OUTPUT_FORMATS = ("npy", "json", "csv")


def validate_activation_config(config: "ActivationConfig"):
    """Validate ActivationConfig fields.

    Parameters
    ----------
    config : ActivationConfig
        Activation configuration to validate.

    Raises
    ------
    ValueError
        If ``models`` does not name exactly two models, or if the two
        names are not distinct.
    """
    if len(config.models) != 2:
        raise ValueError(
            f"Screening compares exactly two models, got "
            f"{len(config.models)}: {config.models}."
        )
    if config.models[0] == config.models[1]:
        raise ValueError(
            f"The two model names must be distinct, got "
            f"'{config.models[0]}' twice."
        )


def validate_compactness_config(config: "CompactnessConfig"):
    """Validate CompactnessConfig fields.

    Parameters
    ----------
    config : CompactnessConfig
        Compactness configuration to validate.

    Raises
    ------
    ValueError
        If ``measure`` is not in ``VALID_COMPACTNESS_MEASURES`` or
        ``difference_measure`` is not in
        ``VALID_DIFFERENCE_MEASURES``.
    """
    if config.measure not in VALID_COMPACTNESS_MEASURES:
        raise ValueError(
            f"Unknown compactness measure: '{config.measure}'. "
            f"Must be one of {VALID_COMPACTNESS_MEASURES}."
        )
    if config.difference_measure not in VALID_DIFFERENCE_MEASURES:
        raise ValueError(
            f"Unknown difference measure: "
            f"'{config.difference_measure}'. "
            f"Must be one of {VALID_DIFFERENCE_MEASURES}."
        )


def validate_selection_config(config: "SelectionConfig"):
    """Validate SelectionConfig fields.

    Parameters
    ----------
    config : SelectionConfig
        Selection configuration to validate.

    Raises
    ------
    ValueError
        If ``n_categories`` or ``n_exemplars`` is not strictly
        positive, or if either metric name is not recognised.
    """
    if config.n_categories <= 0:
        raise ValueError(
            f"n_categories must be positive, got "
            f"{config.n_categories}."
        )
    if config.n_exemplars <= 0:
        raise ValueError(
            f"n_exemplars must be positive, got {config.n_exemplars}."
        )
    if config.dissimilarity_metric not in VALID_DISSIMILARITY_METRICS:
        raise ValueError(
            f"Unknown dissimilarity metric: "
            f"'{config.dissimilarity_metric}'. "
            f"Must be one of {VALID_DISSIMILARITY_METRICS}."
        )
    if config.similarity_metric not in VALID_SIMILARITY_METRICS:
        raise ValueError(
            f"Unknown similarity metric: "
            f"'{config.similarity_metric}'. "
            f"Must be one of {VALID_SIMILARITY_METRICS}."
        )


def validate_output_config(config: "OutputConfig"):
    """Validate OutputConfig fields.

    Parameters
    ----------
    config : OutputConfig
        Output configuration to validate.

    Raises
    ------
    ValueError
        If any entry in ``formats`` is not in
        ``VALID_OUTPUT_FORMATS``.
    """
    for fmt in config.formats:
        if fmt not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Unknown output format: '{fmt}'. "
                f"Must be one of {VALID_OUTPUT_FORMATS}."
            )


@dataclass
class ActivationConfig:
    """Configuration for the pair of activation sets to compare.

    Parameters
    ----------
    activation_dir : path-like
        Directory holding one subdirectory per model.  Each
        subdirectory is expected to contain ``activations.npy`` and
        ``imagepaths.txt``.
    models : tuple of str
        The two model names, matching the subdirectory names.
    normalize : bool
        Whether to scale every activation vector to unit length
        before screening.
    dataset_path : path-like or None
        Root of the image dataset the activations were computed from.
        Used to resolve the recorded image paths.
    replace_path_prefix : str or None
        Prefix to strip from the recorded image paths before joining
        them to ``dataset_path``.  Use it when the activations were
        computed on a machine with a different dataset location.
    """

    activation_dir: Path
    models: Tuple[str, ...]
    normalize: bool = True
    dataset_path: Optional[Path] = None
    replace_path_prefix: Optional[str] = None

    def __post_init__(self):
        self.activation_dir = Path(self.activation_dir)
        self.models = tuple(self.models)
        if self.dataset_path is not None:
            self.dataset_path = Path(self.dataset_path)
        validate_activation_config(self)


@dataclass
class CompactnessConfig:
    """Configuration for the compactness stage.

    Parameters
    ----------
    measure : str
        Compactness measure, one of
        ``VALID_COMPACTNESS_MEASURES``.
    difference_measure : str
        How the compactness of the two models is contrasted.
        ``"rank"`` subtracts compactness ranks, ``"normalizedDiff"``
        subtracts centred compactness scaled to unit maximum
        absolute value.
    """

    measure: str = "R-squared"
    difference_measure: str = "normalizedDiff"

    def __post_init__(self):
        validate_compactness_config(self)


@dataclass
class SelectionConfig:
    """Configuration for how many categories and exemplars to keep.

    Parameters
    ----------
    n_categories : int
        Number of categories in the curated stimulus set.
    n_exemplars : int
        Number of exemplars kept per selected category.
    dissimilarity_metric : str
        Metric used to build representational dissimilarity
        matrices, one of ``VALID_DISSIMILARITY_METRICS``.
    similarity_metric : str
        Metric used to compare two dissimilarity matrices, one of
        ``VALID_SIMILARITY_METRICS``.
    """

    n_categories: int = 12
    n_exemplars: int = 4
    dissimilarity_metric: str = "L2squared"
    similarity_metric: str = "pearson"

    def __post_init__(self):
        validate_selection_config(self)


@dataclass
class OutputConfig:
    """Configuration for what the screening run writes to disk.

    Parameters
    ----------
    output_dir : path-like or None
        Directory results are written to.  Nothing is written when
        None.
    formats : tuple of str
        Which output formats to produce.  Valid entries: ``"npy"``,
        ``"json"``, ``"csv"``.
    save_rdms : bool
        Whether to write the category-level and exemplar-level
        dissimilarity matrices alongside the selection.
    figure_dir : path-like or None
        Directory the standard figures are written to, under a
        subdirectory named after the run.  No figures are produced
        when None.
    """

    output_dir: Optional[Path] = None
    formats: Tuple[str, ...] = ("npy", "json")
    save_rdms: bool = True
    figure_dir: Optional[Path] = None

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.figure_dir is not None:
            self.figure_dir = Path(self.figure_dir)
        self.formats = tuple(self.formats)
        validate_output_config(self)


@dataclass
class ScreeningConfig:
    """Top-level configuration combining all sub-configs.

    Parameters
    ----------
    activations : ActivationConfig
        Which activation sets to compare.
    compactness : CompactnessConfig
        Compactness stage configuration.
    selection : SelectionConfig
        Size and metrics of the curated set.
    output : OutputConfig
        Output configuration.
    name : str or None
        Name of the screening run, used as the output subdirectory.
        Defaults to the two model names joined by an underscore.
    """

    activations: ActivationConfig
    compactness: CompactnessConfig = field(
        default_factory=CompactnessConfig
    )
    selection: SelectionConfig = field(
        default_factory=SelectionConfig
    )
    output: OutputConfig = field(default_factory=OutputConfig)
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = "_".join(self.activations.models)
