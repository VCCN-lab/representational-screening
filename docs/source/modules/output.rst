
======
Output
======

:func:`~repscreen.output.writers.save_results` writes a screening
result into a subdirectory of the output root named after the run.
Three formats are available.

``npy``
    ``compactness.npy``, ``sorted_compactness.npy`` and
    ``sorted_compact_categories.npy`` hold one entry per model;
    ``category_score.npy``, ``category_correlations.npy`` and
    ``compactness_difference.npy`` hold one value per category.
    With ``save_rdms``, ``catRDM_<model>.npy`` and
    ``RDM_<model>.npy`` hold the dissimilarity matrices at the
    category and curated-set level.  The per-model files are
    dictionaries, so they are read back with
    ``np.load(..., allow_pickle=True).item()``.

``json``
    ``screening_summary.json`` records the run's parameters, the
    selected categories and the resulting similarity.
    ``stimulipaths.json`` holds the curated stimulus paths.

``csv``
    ``category_scores.csv`` gives one row per category with both
    models' compactness, their difference, the relational
    correlation, the combined score and whether the category was
    selected.  ``selected_stimuli.csv`` gives one row per curated
    stimulus.

The ``npy`` and ``stimulipaths.json`` file names match those written
by the original screening script, so code that read those outputs
keeps working.
