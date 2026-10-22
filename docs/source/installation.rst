
============
Installation
============

Requirements
============

repscreen needs Python 3.11 or newer.  Dependencies are declared in
``pyproject.toml`` and resolved with `uv <https://docs.astral.sh/uv/>`_.

From source
===========

.. code-block:: bash

   git clone https://github.com/PeerHerholz/representational-screening.git
   cd representational-screening
   uv pip install -e .

This installs the core dependencies: ``numpy``, ``scikit-learn``,
``scipy``, ``scikit-image``, ``opencv-python-headless``,
``matplotlib``, ``seaborn`` and ``tqdm``.

Optional extras
===============

.. code-block:: bash

   uv pip install -e ".[dev]"    # pytest, flake8, black, codespell
   uv pip install -e ".[docs]"   # sphinx and the gallery

Checking the installation
=========================

.. code-block:: bash

   repscreen --help

Running the tests
=================

.. code-block:: bash

   uv run pytest

Code style is checked with flake8 at a 79-character line length, and
spelling with codespell:

.. code-block:: bash

   uv run flake8 repscreen/ tests/ tools/ examples/
   uv run codespell

Building the documentation
==========================

.. code-block:: bash

   uv pip install -e ".[docs]"
   cd docs && make clean html

The examples gallery under ``docs/source/auto_examples`` is committed
and reused by the build, because the examples read model activations
that are not distributed with the repository.  To re-run them and
regenerate that output:

.. code-block:: bash

   REPSCREEN_BUILD_GALLERY=1 make -C docs clean html

Commit the regenerated gallery along with the example you changed;
``tools/check_gallery_freshness.py`` compares the committed output
against the example sources and the documentation workflow fails if
they have drifted apart.

Containers
==========

``Dockerfile`` and ``Singularity.def`` are committed, so an image can
be built without regenerating them.  ``generate_images.sh`` rewrites
them with neurodocker, run through ``uvx`` at a pinned version:

.. code-block:: bash

   bash generate_images.sh both

Adding ``local`` also builds the Apptainer image:

.. code-block:: bash

   bash generate_images.sh both local

which is the same as

.. code-block:: bash

   apptainer build --fakeroot repscreen.sif Singularity.def

``--fakeroot`` needs an ``/etc/subuid`` and ``/etc/subgid`` entry for
your user; ``apptainer build --help`` lists the alternatives when
that entry is absent.

No image is published, so an image is built from a checkout, which
pins the version as a side effect.  Generating the definitions and
building the Apptainer image need no Docker daemon; building the
Docker image does:

.. code-block:: bash

   docker build -t repscreen:local .

Running it
----------

Both runtimes pass arguments through to the ``repscreen`` command:

.. code-block:: bash

   docker run -ti --rm repscreen:local repscreen --help
   apptainer run repscreen.sif repscreen --help

Mounting a directory lets results outlive the container:

.. code-block:: bash

   docker run -ti --rm \
     -v "$(pwd)":/workspace \
     -w /workspace \
     repscreen:local \
     repscreen --help
