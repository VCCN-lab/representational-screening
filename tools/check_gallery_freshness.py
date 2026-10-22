"""Compare committed gallery output against the example sources.

sphinx-gallery writes a ``<example>.py.md5`` sidecar next to each
generated page, holding the md5 of the example it was built from.
The documentation build reuses committed output instead of running
the examples, so a source edited without a rebuild would render the
previous figures.  Comparing the sidecars against the sources
detects that.
"""

import argparse
import hashlib
import sys
from pathlib import Path

EXAMPLE_GLOB = "plot_*.py"
MD5_SUFFIX = ".md5"


def _digest(path):
    """Return the md5 hex digest of ``path``."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _sidecar(gallery_dir, example):
    """Return the md5 sidecar path for ``example``."""
    return Path(gallery_dir) / f"{example.name}{MD5_SUFFIX}"


def _stale_or_missing(examples_dir, gallery_dir):
    """Report examples whose committed output is absent or outdated."""
    problems = []
    for example in sorted(Path(examples_dir).glob(EXAMPLE_GLOB)):
        sidecar = _sidecar(gallery_dir, example)
        if not sidecar.is_file():
            problems.append(
                f"{example.name}: no committed gallery output"
            )
        elif sidecar.read_text().strip() != _digest(example):
            problems.append(
                f"{example.name}: example changed since the gallery "
                f"was built"
            )
    return problems


def _orphaned(examples_dir, gallery_dir):
    """Report committed output whose example no longer exists."""
    problems = []
    pattern = f"{EXAMPLE_GLOB}{MD5_SUFFIX}"
    for sidecar in sorted(Path(gallery_dir).glob(pattern)):
        name = sidecar.name[: -len(MD5_SUFFIX)]
        if not (Path(examples_dir) / name).is_file():
            problems.append(
                f"{name}: gallery output for a removed example"
            )
    return problems


def check_gallery(examples_dir, gallery_dir):
    """Return a message per gallery output that is out of date.

    Parameters
    ----------
    examples_dir : path-like
        Directory holding the ``plot_*.py`` example sources.
    gallery_dir : path-like
        Directory holding the committed sphinx-gallery output.

    Returns
    -------
    list of str
        One message per problem found; empty when the committed
        gallery matches every example.
    """
    return _stale_or_missing(examples_dir, gallery_dir) + _orphaned(
        examples_dir, gallery_dir
    )


def main(argv=None):
    """Report stale gallery output and return a process exit code.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        ``0`` when the committed gallery is current, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Check committed gallery output against examples."
    )
    parser.add_argument(
        "examples_dir",
        nargs="?",
        default="examples",
        help="Directory of plot_*.py examples (default: examples).",
    )
    parser.add_argument(
        "gallery_dir",
        nargs="?",
        default="docs/source/auto_examples",
        help="Directory of committed gallery output.",
    )
    args = parser.parse_args(argv)

    problems = check_gallery(args.examples_dir, args.gallery_dir)
    if not problems:
        print("Committed gallery output is current.")
        return 0

    print("Committed gallery output is out of date:")
    for problem in problems:
        print(f"  {problem}")
    print(
        "\nRebuild it with REPSCREEN_BUILD_GALLERY=1 and commit the "
        "result; see docs/source/installation.rst."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
