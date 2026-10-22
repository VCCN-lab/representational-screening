"""Tests the container recipes this repository ships.

No image is published. The recipes are what users build from, so the
way they go wrong is in a user's hands rather than in a build log here.

``%runscript`` is the one that matters for the documented invocation.
neurodocker's startup script runs ``"$@"`` when it is given arguments
and drops to a shell when it is not, so a ``%runscript`` that names
the entrypoint without forwarding arguments turns
``apptainer run IMAGE repscreen --help`` into an interactive shell with the
command discarded.
"""

import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_singularity_runscript_forwards_arguments():
    """``apptainer run IMAGE repscreen ...`` must reach the CLI.

    Docker needs no equivalent: the runtime appends arguments to an
    exec-form ``ENTRYPOINT``.
    """
    definition = _ROOT / "Singularity.def"
    if not definition.is_file():
        pytest.skip("no Singularity.def in this checkout")

    _, _, runscript = definition.read_text().partition("%runscript")
    assert runscript, "Singularity.def declares no %runscript"

    body = runscript.split("%")[0]

    assert '"$@"' in body, (
        "%runscript names the entrypoint without forwarding arguments, "
        "so `apptainer run IMAGE repscreen ...` drops the command: "
        f"{body.strip()!r}"
    )


def test_the_recipes_a_user_builds_from_are_present():
    """The repository ships what it documents building."""
    missing = [
        name
        for name in ("Dockerfile", "Singularity.def", "generate_images.sh")
        if not (_ROOT / name).is_file()
    ]

    assert not missing, f"documented as buildable but absent: {missing}"


def test_no_workflow_publishes_an_image():
    """Publishing made this project the redistributor of everything
    inside an image. The recipes are shared instead, so what goes in is
    the builder's choice."""
    workflows = _ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        pytest.skip("no workflows in this checkout")

    publishing = sorted(
        path.name
        for path in workflows.glob("*.yml")
        if "docker/build-push-action" in path.read_text()
        or "docker push" in path.read_text()
    )

    assert not publishing, f"these push an image: {publishing}"


# Ported from dnndatasets, where each of these pinned a defect that
# cost a failed build. They matter more here than there: nothing is
# published, so a user builds, and a generator that resolves fresh
# dependencies or fetches whichever uv is current makes "building from
# a checkout pins the version" untrue.

_UV_PIN = re.compile(r"astral-sh/uv/releases/download/(\d+\.\d+\.\d+)/")


@pytest.fixture(scope="module")
def generator():
    path = _ROOT / "generate_images.sh"
    if not path.is_file():
        pytest.skip("generate_images.sh is not in this checkout")
    return path.read_text()


@pytest.fixture(scope="module")
def definitions():
    present = {
        path.name: path.read_text()
        for path in sorted(
            [
                p
                for p in (
                    *_ROOT.glob("Dockerfile"),
                    *_ROOT.glob("Singularity*.def"),
                )
            ]
        )
    }
    if not present:
        pytest.skip("no container definition is in this checkout")
    return present


def test_the_generator_installs_the_locked_environment(generator):
    assert "uv sync --locked" in generator, (
        "generate_images.sh must install the environment with "
        "`uv sync --locked`, so the image gets what uv.lock names "
        "rather than whatever resolves on the day it is built"
    )


def test_every_definition_installs_the_locked_environment(definitions):
    missing = sorted(
        name
        for name, text in definitions.items()
        if "uv sync --locked" not in text
    )

    assert not missing, (
        f"{missing} do not install from the lock file; regenerate them "
        f"from generate_images.sh"
    )


def test_the_generator_pins_the_uv_version(generator):
    assert _UV_PIN.search(generator), (
        "generate_images.sh must fetch a pinned uv release, not "
        "whichever one is current when the image happens to be built"
    )


def test_every_definition_pins_the_same_uv_version(definitions, generator):
    wanted = _UV_PIN.search(generator)
    if wanted is None:
        pytest.skip("the generator pins no uv version to compare against")

    wrong = {
        name: [found.group(1) for found in _UV_PIN.finditer(text)]
        for name, text in definitions.items()
        if [found.group(1) for found in _UV_PIN.finditer(text)]
        != [wanted.group(1)] * len(_UV_PIN.findall(text))
    }

    assert not wrong, f"these pin a uv version the generator does not: {wrong}"
