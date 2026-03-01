import numpy as np
import lib.utils_RSA as rsa
import tqdm

def sample_catrdm_pairs(cat_activations, submodels, n_samples=1000, nb_subcategories=12, nb_per_category = 50,
                                    batch_size=10, seed=None):
    """
    Memory-efficient version that processes in batches and optionally saves to disk.

    Parameters:
    -----------
    batch_size : int
        Number of samples to process at once (default: 1000)
    output_file : str, optional
        If provided, saves results to this file using pickle
    """

    if seed is not None:
        np.random.seed(seed)

    dissimilarity_metric = 'L2squared'

    nb_categories = len(cat_activations[submodels[0]])
    n_batches = (n_samples + batch_size - 1) // batch_size

    all_sims_samples = []
    all_indices = []
    print(f"Processing {n_samples} samples in {n_batches} batches of {batch_size}...")

    batch_rdms = {}
    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        current_batch_size = end_idx - start_idx

        subset_size = nb_subcategories
        # Allocate batch arrays
        batch_sim = np.zeros((current_batch_size))
        batch_indices = np.zeros((current_batch_size, subset_size), dtype=int)
        for model in submodels:
            batch_rdms[model] = np.zeros((current_batch_size, nb_subcategories*nb_per_category, nb_subcategories*nb_per_category))
        for i in range(current_batch_size):
            # Randomly select images
            cat_indices = np.random.choice(nb_categories, size=nb_subcategories, replace=False)

            # Compute subrdms
            for model in submodels:
                batch_rdms[model][i] = rsa.compute_RDMs(cat_activations[model][cat_indices].reshape(nb_subcategories*nb_per_category, -1),
                            metric=dissimilarity_metric, display=False)
            # Extract submatrices
            batch_sim[i] = rsa.Compute_sim_RDMs(batch_rdms[submodels[0]][i], batch_rdms[submodels[1]][i], center = False, metric = 'pearson' )
            batch_indices[i] = cat_indices

        all_sims_samples.append(batch_sim)
        all_indices.append(batch_indices)

    # Concatenate all batches
    sim_samples = np.concatenate(all_sims_samples, axis=0)
    indices_used = np.concatenate(all_indices, axis=0)


    return sim_samples, indices_used


def subsimilar_categories(cat_activations, submodels, dissimilarity_metric = 'L2squared', similarity_metric = 'pearson', nb_subcategories = 12):
    assert len(submodels)== 2
    assert cat_activations[submodels[0]].shape[:2] == cat_activations[submodels[1]].shape[:2]

    shape = cat_activations[submodels[0]].shape

    nb_categories = shape[0]
    nb_per_categories = shape[1]

    mean_cat_activations1 = cat_activations[submodels[0]].mean(axis = 1)
    mean_cat_activations2 = cat_activations[submodels[1]].mean(axis = 1)

    RDM1 = rsa.compute_RDMs(mean_cat_activations1,
                            metric=dissimilarity_metric, display=False)
    RDM2 = rsa.compute_RDMs(mean_cat_activations2,
                            metric=dissimilarity_metric, display=False)

    # exclude diagonal
    RDM1_short = np.array([np.delete(RDM1[i], i) for i in range(len(RDM1))]).transpose()
    RDM2_short = np.array([np.delete(RDM2[i], i) for i in range(len(RDM2))]).transpose()
    #center
    RDM1_centered = RDM1_short - np.mean(RDM1_short)
    RDM2_centered = RDM2_short - np.mean(RDM2_short)

    #RDM1_centered = RDM1_short
    #RDM2_centered = RDM2_short

    RDM1_centered = RDM1_centered / np.sqrt(np.sum(RDM1_centered ** 2, axis = 0))
    RDM2_centered = RDM2_centered / np.sqrt(np.sum(RDM2_centered ** 2, axis = 0))

    correlations = np.sum(RDM1_centered * RDM2_centered, axis=0)
    subsimiliar_categories = np.argsort(correlations)[:nb_subcategories]


    return correlations, subsimiliar_categories


