"""Command-line interface for repscreen."""

import argparse
from pathlib import Path

from ..config import (
    ActivationConfig,
    CompactnessConfig,
    OutputConfig,
    ScreeningConfig,
    SelectionConfig,
    VALID_COMPACTNESS_MEASURES,
    VALID_DIFFERENCE_MEASURES,
    VALID_DISSIMILARITY_METRICS,
    VALID_OUTPUT_FORMATS,
    VALID_SIMILARITY_METRICS,
)
from ..screening.pipeline import run_screening


def _add_activation_arguments(parser):
    """Register the arguments naming the activation sets."""
    group = parser.add_argument_group("activations")
    group.add_argument(
        "--activation-dir",
        type=Path,
        required=True,
        help=(
            "Directory with one subdirectory per model, each "
            "holding activations.npy and imagepaths.txt."
        ),
    )
    group.add_argument(
        "--models",
        nargs=2,
        metavar=("MODEL1", "MODEL2"),
        required=True,
        help="Names of the two models to compare.",
    )
    group.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Root of the image dataset, used to resolve the "
            "recorded image paths."
        ),
    )
    group.add_argument(
        "--replace-path-prefix",
        type=str,
        default=None,
        help=(
            "Prefix to strip from the recorded image paths before "
            "joining them to --dataset-path."
        ),
    )
    group.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep activation magnitudes instead of scaling every "
        "vector to unit length.",
    )


def _add_compactness_arguments(parser):
    """Register the arguments of the compactness stage."""
    group = parser.add_argument_group("compactness")
    group.add_argument(
        "--compactness-measure",
        choices=VALID_COMPACTNESS_MEASURES,
        default="R-squared",
        help="Compactness measure (default: R-squared).",
    )
    group.add_argument(
        "--difference-measure",
        choices=VALID_DIFFERENCE_MEASURES,
        default="normalizedDiff",
        help=(
            "How the two models' compactness is contrasted "
            "(default: normalizedDiff)."
        ),
    )


def _add_selection_arguments(parser):
    """Register the arguments sizing the curated set."""
    group = parser.add_argument_group("selection")
    group.add_argument(
        "--n-categories",
        type=int,
        default=12,
        help="Categories in the curated set (default: 12).",
    )
    group.add_argument(
        "--n-exemplars",
        type=int,
        default=4,
        help="Exemplars per category (default: 4).",
    )
    group.add_argument(
        "--dissimilarity-metric",
        choices=VALID_DISSIMILARITY_METRICS,
        default="L2squared",
        help="Metric for the dissimilarity matrices "
        "(default: L2squared).",
    )
    group.add_argument(
        "--similarity-metric",
        choices=VALID_SIMILARITY_METRICS,
        default="pearson",
        help="Metric comparing two dissimilarity matrices "
        "(default: pearson).",
    )


def _add_output_arguments(parser):
    """Register the arguments controlling what is written."""
    group = parser.add_argument_group("output")
    group.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory results are written to, under a "
            "subdirectory named after the run."
        ),
    )
    group.add_argument(
        "--formats",
        nargs="+",
        choices=VALID_OUTPUT_FORMATS,
        default=["npy", "json"],
        help="Output formats (default: npy json).",
    )
    group.add_argument(
        "--no-rdms",
        action="store_true",
        help="Skip writing the dissimilarity matrices.",
    )
    group.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help=(
            "Directory the standard figures are written to. No "
            "figures are produced when this is left out."
        ),
    )
    group.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Name of the run, used as the output subdirectory "
            "(default: the two model names joined by '_')."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for repscreen.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all activation, compactness, selection and
        output arguments registered.  See :doc:`../cli` for the full
        argument reference.
    """
    parser = argparse.ArgumentParser(
        prog="repscreen",
        description=(
            "Curate a stimulus set that exposes the "
            "representational differences between two vision "
            "models."
        ),
    )
    _add_activation_arguments(parser)
    _add_compactness_arguments(parser)
    _add_selection_arguments(parser)
    _add_output_arguments(parser)
    return parser


def config_from_args(args) -> ScreeningConfig:
    """Build a screening configuration from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments as returned by :func:`build_parser`.

    Returns
    -------
    ScreeningConfig
        Configuration ready for
        :func:`~repscreen.screening.pipeline.run_screening`.
    """
    return ScreeningConfig(
        activations=ActivationConfig(
            activation_dir=args.activation_dir,
            models=tuple(args.models),
            normalize=not args.no_normalize,
            dataset_path=args.dataset_path,
            replace_path_prefix=args.replace_path_prefix,
        ),
        compactness=CompactnessConfig(
            measure=args.compactness_measure,
            difference_measure=args.difference_measure,
        ),
        selection=SelectionConfig(
            n_categories=args.n_categories,
            n_exemplars=args.n_exemplars,
            dissimilarity_metric=args.dissimilarity_metric,
            similarity_metric=args.similarity_metric,
        ),
        output=OutputConfig(
            output_dir=args.output_dir,
            formats=tuple(args.formats),
            save_rdms=not args.no_rdms,
            figure_dir=args.figure_dir,
        ),
        name=args.name,
    )


def main(argv=None) -> int:
    """Run a screening from the command line.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        ``0`` once the run has completed.
    """
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    result = run_screening(config)

    print(
        f"Screened {len(result.category_names)} categories down to "
        f"{len(result.selected_categories)}:"
    )
    for name in result.selected_category_names:
        print(f"  {name}")
    print(
        f"Curated {len(result.stimulus_paths)} stimuli with a "
        f"{config.selection.similarity_metric} similarity of "
        f"{result.similarity:.4f} between {result.models[0]} and "
        f"{result.models[1]}."
    )

    if args.figure_dir is not None:
        print(f"Wrote figures to {args.figure_dir / config.name}")

    return 0
