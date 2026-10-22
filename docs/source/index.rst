
===============================================================
repscreen: model-guided stimulus curation
===============================================================

.. image:: https://img.shields.io/github/actions/workflow/status/PeerHerholz/representational-screening/.github%2Fworkflows%2Ftests.yml?branch=main&style=plastic
        :target: https://github.com/PeerHerholz/representational-screening

.. image:: https://img.shields.io/github/repo-size/PeerHerholz/representational-screening.svg
        :target: https://github.com/PeerHerholz/representational-screening

.. image:: https://img.shields.io/github/issues/PeerHerholz/representational-screening.svg
        :target: https://github.com/PeerHerholz/representational-screening/issues

.. image:: https://img.shields.io/github/issues-pr/PeerHerholz/representational-screening.svg
        :target: https://github.com/PeerHerholz/representational-screening/pulls

.. image:: https://img.shields.io/github/license/PeerHerholz/representational-screening.svg
        :target: https://github.com/PeerHerholz/representational-screening


.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Getting Started
        :link: installation
        :link-type: doc
        :class-card: sd-border-0
        :shadow: md

        New to repscreen? Start here to install and configure the package.

        **Quick install:** ``uv pip install -e .``

    .. grid-item-card:: Usage Guide
        :link: usage
        :link-type: doc
        :class-card: sd-border-0
        :shadow: md

        Learn how to screen two models down to an experiment-ready stimulus set.

    .. grid-item-card:: Examples
        :link: auto_examples/index
        :link-type: doc
        :class-card: sd-border-0
        :shadow: md

        Runnable tutorials covering the screening run and its analyses.

    .. grid-item-card:: API Reference
        :link: api_ref
        :link-type: doc
        :class-card: sd-border-0
        :shadow: md

        Detailed documentation for all modules, classes, and functions.

|

What is repscreen?
==================

**repscreen** curates stimuli that expose where two deep vision models
disagree.  Given each model's activations over a large,
category-structured image set, it selects a small set of categories the
two models represent differently, then selects the exemplars within
those categories that maximise the divergence, quantified from the
models' representational geometries.

The resulting stimuli are few enough to run in a behavioural or
neuroimaging experiment while still separating the models, which makes
the comparison interpretable and theory-driven rather than a single
aggregate similarity score.

.. image:: _static/Methods_schematic.png
   :width: 750
   :alt: The two screening stages, from a category-structured image
         pool to a curated stimulus set.

The method and its validation across model pairs are described in
van Dyck, Flachot and Dobs (in preparation), *Model-guided stimulus
curation for comparing artificial and biological vision*.

Key Features
------------

.. grid:: 1 1 3 3
    :gutter: 2

    .. grid-item-card:: Two-stage Screening
        :link: modules/screening
        :link-type: doc
        :class-card: sd-border-0
        :shadow: sm

        Categories first, then exemplars within them.

        +++
        One call from activations to stimulus set

    .. grid-item-card:: Model-Agnostic
        :link: modules/config
        :link-type: doc
        :class-card: sd-border-0
        :shadow: sm

        Any pair of models, any dataset that divides into equal categories.

        +++
        Only saved activations are required

    .. grid-item-card:: Nine Compactness Measures
        :link: modules/metrics
        :link-type: doc
        :class-card: sd-border-0
        :shadow: sm

        R-squared, Fisher, Calinski-Harabasz, silhouette, Davies-Bouldin.

        +++
        Swappable through one argument

    .. grid-item-card:: Null Distributions
        :link: modules/screening
        :link-type: doc
        :class-card: sd-border-0
        :shadow: sm

        Random subsets at the matrix, image or category level.

        +++
        Shows what the selection itself contributes

    .. grid-item-card:: Figures Included
        :link: modules/plotting
        :link-type: doc
        :class-card: sd-border-0
        :shadow: sm

        RDM pairs, compactness curves, t-SNE comparisons, stimulus grids.

        +++
        Returned as figures, not written or shown

    .. grid-item-card:: Image Signatures
        :link: modules/signatures
        :link-type: doc
        :class-card: sd-border-0
        :shadow: sm

        Per-category Fourier spectra and CIELab chromaticity.

        +++
        Check a selection against plain image statistics

|

Quick Example
=============

.. code-block:: python

   from repscreen.config import (
       ActivationConfig, OutputConfig, ScreeningConfig,
       SelectionConfig,
   )
   from repscreen.screening.pipeline import run_screening

   config = ScreeningConfig(
       activations=ActivationConfig(
           activation_dir="data/activations",
           models=("places365", "imagenet"),
           dataset_path="data/dataset",
       ),
       selection=SelectionConfig(n_categories=12, n_exemplars=4),
       output=OutputConfig(output_dir="results"),
   )

   result = run_screening(config)
   print(result.selected_category_names)
   print(result.similarity)

Or from the command line:

.. code-block:: bash

   repscreen --activation-dir data/activations \
             --models places365 imagenet \
             --output-dir results

|

Results
=======

Screening a ResNet trained on Places365 against a ResNet trained on
ImageNet, the comparison the paper reports:

.. image:: _static/Results_1.png
   :width: 750
   :alt: Curated stimuli and the resulting representational
         dissimilarity matrices for both models.

|

Installation
============

.. code-block:: bash

   uv pip install -e .

See the :doc:`installation guide <installation>` for detailed
instructions.

|

Support & Contributing
======================

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Report Issues
        :class-card: sd-border-0
        :shadow: sm

        Found a bug or have a feature request?

        +++
        `Open an issue on GitHub <https://github.com/PeerHerholz/representational-screening/issues>`_

    .. grid-item-card:: Get Help
        :class-card: sd-border-0
        :shadow: sm

        Questions about usage?

        +++
        `Ask on GitHub <https://github.com/PeerHerholz/representational-screening/issues>`_

|

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   installation
   usage
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Core Concepts

   modules/config
   modules/screening
   modules/metrics
   modules/output
   modules/plotting
   modules/signatures
   modules/cli

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   cli
   api_ref
   release-history
