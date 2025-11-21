import numpy as np
import sklearn.metrics
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import lib.utils_RSA as rsa
import seaborn as sns

#SCORING_FUNCTIONS_CLUSTERING = {"score1": fun1, "score2": fun2}



def compute_compactness(cat_activations, models, listcat, measure = 'Fisher_discriminant'):
    """
    Memory-efficient version that processes one model at a time and uses generators.
    Good for very large datasets.
    """
    compactness = {}
    sorted_compactness = {}
    sorted_compact_categories = {}

    for model in models:
        print(model)
        n_cats = len(cat_activations[model])
        #scoring_fun_choisie = SCORING_FUNCTIONS_CLUSTERING[measure]
        #scoring_result = scoring_fun_choisie(x, y, nom_modele, liscat)

        if measure == 'Fisher_discriminant':
            # Pre-compute centroids
            centroids = np.array([np.mean(cat_act, axis=0) for cat_act in cat_activations[model]])

            # Compute metrics using generator expressions to save memory
            inter_vars = np.array([
                np.mean(np.sum((cat_activations[model][i] - centroids[i])**2, axis =-1))
                for i in range(n_cats)
            ])

            intra_vars = np.array([
                np.mean([
                    np.mean(np.sum((cat_activations[model][j] - centroids[i])**2, axis =-1))
                    for j in range(n_cats) if j != i
                ])
                for i in range(n_cats)
            ])

            # Compute normalized variances
            compact = 1-inter_vars / intra_vars

        elif measure == 'CH_Index':
            # Pre-compute centroids
            center = np.mean(cat_activations[model], axis=(0,1))
            centroids = np.array([np.mean(cat_act, axis=0) for cat_act in cat_activations[model]])

            # Compute metrics using generator expressions to save memory
            inter_vars = np.array([
                np.mean(np.sum((cat_activations[model][i] - centroids[i])**2, axis =-1))
                for i in range(n_cats)
            ])

            distance2center = np.sum((centroids - center)**2, axis =-1)

            # Compute normalized variances
            compact =  distance2center / inter_vars

        elif measure == 'CH_Index_adapted':
            # Pre-compute centroids
            center = np.mean(cat_activations[model], axis=(0,1))
            centroids = np.array([np.mean(cat_act, axis=0) for cat_act in cat_activations[model]])

            # Compute metrics using generator expressions to save memory
            inter_vars = np.array([
                np.mean(np.sum((cat_activations[model][i] - centroids[i])**2, axis =-1))
                for i in range(n_cats)
            ])

            distance2center = np.sum((centroids - center)**2, axis =-1)

            # Compute normalized variances
            compact =  1 - inter_vars / distance2center

        elif measure == 'global_silhouette_score':
            compact = []

            for i in range(n_cats):
                current_cat = cat_activations[model][i]

                # Compute within-category distances
                if len(current_cat) > 1:
                    # Use sklearn's optimized euclidean_distances with squared option
                    intra_dist_matrix = euclidean_distances(current_cat, current_cat, squared=True)
                    # Get upper triangle (excluding diagonal) to avoid counting pairs twice
                    mask = np.triu(np.ones_like(intra_dist_matrix, dtype=bool), k=1)
                    intra_distances = intra_dist_matrix[mask]
                    mean_within_distance = np.mean(intra_distances)

                # Compute between-category distances in batches to save memory
                inter_distances_sum = 0.0
                inter_count = 0

                for j in range(n_cats):
                    if i != j:
                        other_cat = cat_activations[model][j]
                        # Use sklearn's optimized euclidean_distances with squared option
                        distances = euclidean_distances(current_cat, other_cat, squared=True)

                        inter_distances_sum += np.sum(distances)
                        inter_count += distances.size

                mean_between_distance = inter_distances_sum / inter_count if inter_count > 0 else 1.0

                # Compactness ratio: lower is better (tight within, far between)
                silhouette = (mean_between_distance - mean_within_distance)/np.amax(mean_between_distance, mean_within_distance)
                compact.append(silhouette)

            compact = np.array(compact)
        elif measure == 'silhouette_score':
            compact = []

            for i in range(n_cats):
                current_cat = cat_activations[model][i]

                # Compute within-category distances
                if len(current_cat) > 1:
                    # Use sklearn's optimized euclidean_distances with squared option
                    intra_dist_matrix = euclidean_distances(current_cat, current_cat, squared=True)
                    # Get upper triangle (excluding diagonal) to avoid counting pairs twice
                    mask = np.triu(np.ones_like(intra_dist_matrix, dtype=bool), k=1)
                    intra_distances = intra_dist_matrix[mask]
                    mean_within_distance = np.mean(intra_distances)

                # Compute between-category distances in batches to save memory
                inter_distances_sum = []
                inter_count = 0

                for j in range(n_cats):
                    if i != j:
                        other_cat = cat_activations[model][j]
                        # Use sklearn's optimized euclidean_distances with squared option
                        distances = euclidean_distances(current_cat, other_cat, squared=True)

                        inter_distances_sum.append(np.mean(distances)) # save average distances for min

                mean_between_distance = np.amin(np.array(inter_distances_sum)) # avereage distance of nearby cluster

                # Compactness ratio: lower is better (tight within, far between)
                silhouette = (mean_between_distance - mean_within_distance)/np.amax(mean_between_distance, mean_within_distance)
                compact.append(silhouette)

            compact = np.array(compact)

        elif measure == 'simplified_silhouette_score':
            compact = []
            # Pre-compute centroids
            centroids = np.array([np.mean(cat_act, axis=0) for cat_act in cat_activations[model]])

            # Compute metrics using generator expressions to save memory
            inter_vars = np.array([
                np.mean(np.sum((cat_activations[model][i] - centroids[i]) ** 2, axis=-1))
                for i in range(n_cats)
            ])

            intra_vars = np.array([
                np.min([ np.sum((centroids[j] - centroids[i]) ** 2, axis=-1)
                    for j in range(n_cats) if j != i
                ])
                for i in range(n_cats)
            ])

            compact = (intra_vars - inter_vars) / np.amax((intra_vars, inter_vars), axis = 0)

        elif measure == "Davies-Bouldin_Index":
            centroids = np.array([np.mean(cat_act, axis=0) for cat_act in cat_activations[model]])

            # Distance within categoreis to centroid
            inter_vars = np.array([
                np.mean(np.sum((cat_activations[model][i] - centroids[i]) ** 2, axis=-1))
                for i in range(n_cats)
            ])

            # Distance between centroids
            intra_dist_matrix = euclidean_distances(centroids, centroids, squared=True)

            compact = []
            for i in range(n_cats):
                DB = []
                for j in range(n_cats):
                    if i != j:
                        DB.append((inter_vars[i] + inter_vars[j])/intra_dist_matrix[i,j])
                compact.append(max(DB)) # we take the maximum
            compact = np.array(compact)

        elif measure == 'R-squared':
            gen_centroid = np.mean(cat_activations[model], axis = (0,1)) # shape (nb_features)
            cat_centroids = np.mean(cat_activations[model], axis = (1)) # shape (nb_categories, nb_features)
            #dist_centroids2center = np.sum((cat_centroids - gen_centroid)**2, axis = -1) # square radius
            #gen_radius = np.amax(dist_centroids2center)
            gen_radius = np.mean(np.sum((cat_activations[model] - gen_centroid)**2, axis = 2)) # square radius
            cat_radius = np.mean(np.sum((cat_activations[model].transpose(1, 0, 2) - cat_centroids)**2, axis = 2),0) # square radius
            compact = 1-cat_radius/gen_radius

        elif measure == 'R-squared_adjusted':
            cat_centroids = np.mean(cat_activations[model], axis = (1)) # shape (nb_categories, nb_features)

            intra_dist_matrix = np.mean(euclidean_distances(cat_centroids, cat_centroids, squared=True), axis = 0)

            cat_radius = np.mean(np.sum((cat_activations[model].transpose(1, 0, 2) - cat_centroids)**2, axis = 2),0) # square radius
            compact = 1-cat_radius/intra_dist_matrix

        # Sort and store results
        sort_indices = np.argsort(compact)
        compactness[model] = compact
        sorted_compactness[model] = compact[sort_indices]
        sorted_compact_categories[model] = np.array(listcat)[sort_indices]

    return sorted_compactness, sorted_compact_categories, compactness


def alternate_pos_neg(arr):
    """
    Rearrange array by alternating positive and negative values
    while preserving descending absolute value order as much as possible.

    Args:
        arr: List sorted by absolute values in descending order

    Returns:
        List with alternating positive/negative values
    """
    # Separate into positive and negative values (maintaining their order)
    positives = [x for x in arr if x > 0]
    negatives = [x for x in arr if x < 0]

    result = []
    i, j = 0, 0

    # Alternate between positive and negative
    # Start with whichever has the larger absolute value
    start_with_positive = True
    if positives and negatives:
        start_with_positive = abs(positives[0]) >= abs(negatives[0])
    elif negatives:
        start_with_positive = False

    while i < len(positives) or j < len(negatives):
        if start_with_positive:
            if i < len(positives):
                result.append(positives[i])
                i += 1
            elif j < len(negatives):
                result.append(negatives[j])
                j += 1
            start_with_positive = False
        else:
            if j < len(negatives):
                result.append(negatives[j])
                j += 1
            elif i < len(positives):
                result.append(positives[i])
                i += 1
            start_with_positive = True

    return result



def max_compactness_difference(compact_categories, compactness, listcat, models = ['saycam', 'ego'], nb_considered_categories = 12, compactness_diff_measure = 'rank'):
    '''
    Function that sorts categories following the maximum difference in compactness given 2 models
    '''

    if compactness_diff_measure == 'rank':
        diff = np.zeros(len(listcat))

        model1_rank_lookup = {cat: rank for rank, cat in enumerate(compact_categories[models[0]])}
        model2_rank_lookup = {cat: rank for rank, cat in enumerate(compact_categories[models[1]])}
        for c, cat in enumerate(listcat):
            diff[c] = model2_rank_lookup[cat] - model1_rank_lookup[cat]
    elif compactness_diff_measure == 'normalizedDiff':
        compactness0 = compactness[models[0]] - np.mean(compactness[models[0]]) # center compactness
        compactness1 = compactness[models[1]] - np.mean(compactness[models[1]])
        compactness0 = compactness0 / np.max(np.absolute(compactness0))
        compactness1 = compactness1 / np.max(np.absolute(compactness1))

        diff = compactness1 - compactness0 # distance in terms of compactness

    indexes = np.argsort(-np.absolute(diff))
    sortedmaxdiffcats = np.array(listcat)[indexes]
    maxdiffs = diff[indexes] # if diff is positive, compactness measure is higher in model2,so model1 is actually more compact
    print(nb_considered_categories)
    print(f'The {nb_considered_categories} categories leading to the max differences between {models[0]} and {models[1]} are {sortedmaxdiffcats[:nb_considered_categories]}')
    print(f'Category numbers are {indexes[:nb_considered_categories]}')
    print(f'With differences in compactness of  {maxdiffs[:nb_considered_categories]}')

    return indexes, sortedmaxdiffcats, maxdiffs

from itertools import combinations
def find_max_dissimilarity_images(cat_activations, models, categories, nb_per_cat,
                                  images_per_subset=4, dissimilarity_metric = 'L2squared', similarity_metric = 'pearson', diff = np.array([0])):
    """
    Find the subset of images per category that maximizes RDM dissimilarity between two models.

    Parameters:
    -----------
    cat_activations : dict
        Dictionary with structure: cat_activations[model][category] = array of activations (n_images, n_features)
    models : list
        List of two model names, e.g., ['model1', 'model2']
    categories : list
        List of category names/indices
    compute_RDM : function
        Function that takes activations and returns RDM: RDM = compute_RDM(activations)
    images_per_subset : int
        Number of images to select per category (default: 4)
    method : str
        'exhaustive' or 'random' sampling of combinations

    Returns:
    --------
    results : dict
        Dictionary with results for each category:
        {
            category: {
                'best_indices': array of selected image indices,
                'max_dissimilarity': maximum dissimilarity value,
                'model1_rdm': RDM for model1 with selected images,
                'model2_rdm': RDM for model2 with selected images,
                'similarity': similarity between the two RDMs
            }
        }
    """

    if len(models) != 2:
        raise ValueError("This function requires exactly 2 models")

    results = {}

    #### First build the RDMs using all images of the chosen categories to get the general stats
    cat_activations_subset1 = cat_activations[models[0]][categories]
    cat_activations_subset2 = cat_activations[models[1]][categories]

    cat_shape = cat_activations_subset1.shape

    RDM1 = rsa.compute_RDMs(cat_activations_subset1.reshape(cat_shape[0]*cat_shape[1], -1), metric = dissimilarity_metric, display = False)
    RDM2 = rsa.compute_RDMs(cat_activations_subset2.reshape(cat_shape[0] * cat_shape[1], -1),
                            metric=dissimilarity_metric, display=False)
    means = {}
    n = len(RDM1)
    upper_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
    means['x'] = np.mean(RDM1[upper_indices])
    means['y'] = np.mean(RDM2[upper_indices])
    means['norm'] = np.std(RDM1[upper_indices]) * np.std(RDM2[upper_indices])
    print(means)
    for c, category in enumerate(tqdm(categories, desc="Processing categories")):
        print(f"\nProcessing category: {category}")
        # Get activations for both models for this category
        cat_RDM1 = RDM1[c*nb_per_cat:(c+1)*nb_per_cat, c*nb_per_cat:(c+1)*nb_per_cat]  # Shape: (50, 50)
        cat_RDM2 = RDM2[c*nb_per_cat:(c+1)*nb_per_cat, c*nb_per_cat:(c+1)*nb_per_cat]  # Shape: (50, 50)

        # Generate combinations of image indices
        all_combinations = list(combinations(range(nb_per_cat), images_per_subset))

        print(f"Testing {len(all_combinations)} combinations of {images_per_subset} images")

        best_indices = None
        best_model1_rdm = None
        best_model2_rdm = None
        best_similarity = np.inf

        # Test each combination
        for combination in tqdm(all_combinations, desc="Testing combinations", leave=False, position=1):
            indices = np.array(combination)
            # Get subset of activations
            rdm1 = cat_RDM1[np.ix_(indices, indices)]  # Shape: (4, 4)
            rdm2 = cat_RDM2[np.ix_(indices, indices)]  # Shape: (4, 4)

            # Compute similarity between RDMs
            if similarity_metric == 'pearson':
                similarity = rsa.Compute_sim_RDMs(rdm1, rdm2, center = False, metric = 'pearson_global', means= means)

            elif similarity_metric == 'contrast': # We want 4 images perceived very similar in one case but very dissimilar in the other
                n = len(rdm1)
                upper_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
                if diff[c] <0:
                    similarity = -np.mean(rdm1[upper_indices])/means['x'] + np.mean(rdm2[upper_indices])/means['y']
                else:
                    similarity = np.mean(rdm1[upper_indices])/means['x'] - np.mean(rdm2[upper_indices])/means['y']

            # Update best if this is better
            if similarity < best_similarity:
                best_indices = indices
                best_model1_rdm = rdm1
                best_model2_rdm = rdm2
                best_similarity = similarity

        # Store results for this category
        results[category] = {
            'best_indices': best_indices,
            'model1_rdm': best_model1_rdm,
            'model2_rdm': best_model2_rdm,
            'similarity': best_similarity
        }

        print(f"Best indices for {category}: {best_indices}")
        print(f"Similarity: {best_similarity:.4f}")

    return results

def compute_sub_rdm_similarity(results, cat_activations, models, categories, dissimilarity_metric = 'L2squared', similarity_metric = 'pearson', savename = ''):
    """
    Compute sub-RDMs using the 40 selected images (4 per category × 10 categories)
    that maximize dissimilarity between models, then compute their similarity.

    Parameters:
    -----------
    results : dict
        Output from find_max_dissimilarity_images() with structure:
        results[category]['best_indices'] = array of 4 selected image indices
    full_rdms : dict
        Dictionary: full_rdms[model] = full RDM array (25000, 25000)
    models : list
        List of two model names, e.g., ['model1', 'model2']
    categories : list
        List of category names/indices (should have 10 categories)

    Returns:
    --------
    result : dict
        Dictionary containing:
        {
            'similarity': similarity between the two 40×40 RDMs,
            'model1_rdm': 40×40 RDM for model1,
            'model2_rdm': 40×40 RDM for model2,
            'image_info': list of (category, original_index) for each of the 40 images
        }
    """

    if len(models) != 2:
        raise ValueError("This function requires exactly 2 models")

    print(f"Collecting 40 selected images from {len(categories)} categories...")

    #### First build the RDMs using all images of the chosen categories to get the general stats
    cat_activations_subset1 = cat_activations[models[0]][categories]
    cat_activations_subset2 = cat_activations[models[1]][categories]

    cat_shape = cat_activations_subset1.shape

    RDM1 = rsa.compute_RDMs(cat_activations_subset1.reshape(cat_shape[0] * cat_shape[1], -1),
                            metric=dissimilarity_metric, display=False)
    RDM2 = rsa.compute_RDMs(cat_activations_subset2.reshape(cat_shape[0] * cat_shape[1], -1),
                            metric=dissimilarity_metric, display=False)
    means = {}
    n = len(RDM1)
    upper_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
    means['x'] = np.mean(RDM1[upper_indices])
    means['y'] = np.mean(RDM2[upper_indices])
    means['norm'] = np.std(RDM1[upper_indices]) * np.std(RDM2[upper_indices])
    print(means)

    # Collect selected image indices from all categories
    selected_indices = []
    selected_indices_2display = []
    image_info = []  # Track which category and original index each image comes from

    total_selected = 0

    for c, category in enumerate(categories):
        if category not in results:
            raise ValueError(f"Category {category} not found in results")

        # Get the 4 selected indices for this category
        cat_selected_indices = results[category]['best_indices']

        # Add to combined list
        selected_indices.extend(cat_selected_indices + 50*c)
        selected_indices_2display.extend(cat_selected_indices + 50 * category)

        # Track image information
        for idx in cat_selected_indices:
            image_info.append((category, idx))

        total_selected += len(cat_selected_indices)
        #print(f"Category {category}: selected indices {cat_selected_indices}")

    print(f"\nTotal selected images: {total_selected}")

    # Verify we have 40 images

    # Extract sub-RDMs for both models using the 40 selected images
    print("Extracting sub-RDMs...")
    rdm_model1 = RDM1[np.ix_(selected_indices , selected_indices )]  # Shape: (40, 40)
    rdm_model2 = RDM2[np.ix_(selected_indices , selected_indices  )]  # Shape: (40, 40)

    print(f"RDM shapes: {rdm_model1.shape}, {rdm_model2.shape}")

    # Compute similarity between the two RDMs
    print("Computing similarity between RDMs...")
    similarity = rsa.Compute_sim_RDMs(rdm_model1, rdm_model2, center = False, metric = similarity_metric)

    print(f"\nRDM similarity using 40 maximally dissimilar images: {similarity:.6f}")

    # Package results
    result = {
        'similarity': similarity,
        'model1_rdm': rdm_model1,
        'model2_rdm': rdm_model2,
        'image_info': image_info,
        'selected_indices': selected_indices_2display
    }

    fig, subs = plt.subplots(1,2)
    sns.heatmap(rdm_model1,
                annot=False,
                cmap='Greys',      # Blue to red colormap
                square=True,
                cbar=True,
                #cbar_kws={'label': 'Dissimilarity'},
                #fmt='.2f',
                linewidths=0,
                ax = subs[0],
                vmin=0,               # Set minimum value for color scale
                vmax=np.max(rdm_model1))               # Set maximum value for color scale
    sns.heatmap(rdm_model2,
                annot=False,
                cmap='Greys',      # Blue to red colormap
                square=True,
                cbar=True,
                #cbar_kws={'label': 'Dissimilarity'},
                #fmt='.2f',
                linewidths=0,
                ax = subs[1],
                vmin=0,               # Set minimum value for color scale
                vmax=np.max(rdm_model2))

    subs[0].axis('off')
    subs[1].axis('off')
    fig.tight_layout()
    plt.title(f'RDM similarity = {similarity:.6f}')
    if len(savename)>1:
        plt.savefig(savename)
    #plt.show()
    return result

def check_list_similarity(list1, list2):
    '''Checks if two lists contain the same elements, regardless of order,
    and calculates the proportion of common elements.'''
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1 & set2  # Intersection of sets
    proportion = (len(common_elements) / max(len(set1), len(set2))) * 100 if max(len(set1), len(set2)) > 0 else 0
    return proportion


def find_subsimilar_subset(cat_activations, submodels, categories, images_per_subset=4, nb_per_category=50):
    nb_categories = len(categories)
    dissimilarity_metric = 'L2squared'
    #### First build the RDMs using all images of the chosen categories to get the general stats
    cat_activations_subset1 = cat_activations[submodels[0]][categories]
    cat_activations_subset2 = cat_activations[submodels[1]][categories]

    cat_shape = cat_activations_subset1.shape

    RDM1 = rsa.compute_RDMs(cat_activations_subset1.reshape(cat_shape[0] * cat_shape[1], -1),
                            metric=dissimilarity_metric, display=False)
    RDM2 = rsa.compute_RDMs(cat_activations_subset2.reshape(cat_shape[0] * cat_shape[1], -1),
                            metric=dissimilarity_metric, display=False)

    # exclude diagonal
    RDM1_short = np.array([np.delete(RDM1[i], i) for i in range(len(RDM1))]).transpose()
    RDM2_short = np.array([np.delete(RDM2[i], i) for i in range(len(RDM2))]).transpose()
    #center
    RDM1_centered = RDM1_short - np.mean(RDM1_short)
    RDM2_centered = RDM2_short - np.mean(RDM2_short)

    RDM1_centered = RDM1_centered / np.sqrt(np.sum(RDM1_centered ** 2, axis = 0))
    RDM2_centered = RDM2_centered / np.sqrt(np.sum(RDM2_centered ** 2, axis = 0))

    correlations = np.sum(RDM1_centered * RDM2_centered, axis=0)
    #category_correlations = correlations.reshape(nb_categories, -1)

    # Initialize arrays for sorted results
    correlations_sorted = np.zeros(len(RDM1))
    sort_indices = list()
    sort_indices_global = list()
    category_boundaries = []

    # Sort within each category (12 categories, 50 images each)
    for cat, category in enumerate(categories):
        start_idx = cat * nb_per_category
        end_idx = (cat + 1) * nb_per_category

        # Get correlations for this category
        cat_correlations = correlations[start_idx:end_idx]

        # Get sorting indices (lowest to highest correlation)
        cat_sort_indices = np.argsort(cat_correlations)

        # Store sorted correlations
        correlations_sorted[start_idx:end_idx] = cat_correlations[cat_sort_indices]

        # Store original indices (adjusted for global position)
        sort_indices.append(start_idx + cat_sort_indices[:images_per_subset])
        sort_indices_global.append( category * nb_per_category + cat_sort_indices[:images_per_subset])

        # Store category boundaries
        category_boundaries.append((start_idx, end_idx))

    sort_indices = np.array(sort_indices).flatten()
    # Reorder RDM columns according to the sorting
    RDM1_sorted = RDM1[np.ix_(sort_indices, sort_indices)]
    RDM2_sorted = RDM2[np.ix_(sort_indices, sort_indices)]

    return RDM1, RDM2, RDM1_sorted, RDM2_sorted, np.array(sort_indices_global).flatten()

def analyze_selected_images(results, categories):
    """
    Analyze the results to understand patterns in selected images.
    """
    print("\n=== Analysis of Selected Images ===")

    for category in categories:
        result = results[int(category)]
        indices = result['best_indices']
        dissimilarity = result['max_dissimilarity']
        similarity = result['similarity']

        print(f"\nCategory: {int(category)}")
        print(f"Selected images: {indices}")
        print(f"Dissimilarity: {dissimilarity:.4f}")
        print(f"Similarity: {similarity:.4f}")

        # Could add more analysis here:
        # - Distribution of selected indices
        # - Patterns across categories
        # - Statistics on dissimilarity values

import math
def plot_stats(SIMs, submodels, labels = ['label1', 'label2']):
    '''plot the compactness as a function of sorted image category.
    Plot is a subplot of adaptative size, depending on the length of the list submodel given.
    '''
    nb_subs = len(submodels) # Number of subs
    sqrt = np.sqrt(nb_subs)
    cols = math.ceil(sqrt)
    rows = math.ceil(nb_subs // sqrt)
    while (cols*rows)<(nb_subs):
        rows =rows + 1 ## compute the number of columns and rows
    fig, subs = plt.subplots(rows,cols, sharex=True, sharey=True, figsize=(cols*2+1, rows*2+1)) # adaptative size
    count = 0
    minval = 1
    maxval = 0
    for i, model in enumerate(submodels):
        minval = min(minval, np.amin(SIMs[model]))
        maxval = max(maxval, np.amax(SIMs[model]))
        if rows ==1:
            subs[count%cols].plot(SIMs[model])
            subs[count%cols].set_title(f'{model}')
        else:
            subs[count//cols, count%cols].plot(SIMs[model])
            subs[count//cols, count%cols].set_title(f'{model}')
            count+=1
    maxval = min(maxval, 1.1)
    plt.ylim(np.round(minval,1)-0.1, np.round(maxval,1)+0.1)
    if rows == 1:
        subs[0].set_ylabel(labels[1])
        for sub in subs:
            sub.set_xlabel(labels[0])
    else:
        for sub in subs[-1]:
            sub.set_xlabel(labels[0])
        for sub in subs[:,0]:
            sub.set_ylabel(labels[1])

    fig.tight_layout()
    plt.show()
    plt.close()


def plot_stats_one(SIMs, submodels, labels=['label1', 'label2'], savename = None):
    '''Plot the compactness as a function of sorted image category.
    All model curves appear in one plot with the model names as labels.
    '''
    fig, ax = plt.subplots(figsize=(6, 6))

    minval = 1
    maxval = 0

    for model in submodels:
        ax.plot(SIMs[model], label=model)
        minval = min(minval, np.amin(SIMs[model]))
        maxval = max(maxval, np.amax(SIMs[model]))

    maxval = max(maxval, 1.1)
    ax.set_ylim(np.round(minval, 1) - 0.1, np.round(maxval, 1) + 0.1)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.legend()

    fig.tight_layout()
    # Save or show
    if savename:
        plt.savefig('figures/' + savename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return fig, ax


import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def display_low_similarity_images(image_paths, indices_vectorized, compactness_values, n_images=40, figsize=(20, 7), save_path=None):
    """
    Load and display the first n images corresponding to lowest similarity indices.

    Parameters:
    -----------
    image_paths : list
        List of image file paths matching the RDM column indices
    indices_vectorized : numpy.ndarray
        Column indices sorted from lowest to highest similarity
    n_images : int
        Number of images to display (default: 40)
    grid_cols : int
        Number of columns in the display grid (default: 8)
    figsize : tuple
        Figure size for matplotlib (default: (20, 10))
    save_path : str, optional
        Path to save the figure (if None, just display)

    Returns:
    --------
    loaded_images : list
        List of loaded images (as numpy arrays)
    valid_paths : list
        List of valid image paths that were successfully loaded
    """

    n_categories = len(compactness_values)
    nb_images_per_category = n_images // n_categories

    # Get the indices for the first n_images with lowest similarity
    low_similarity_indices = indices_vectorized[:n_images]

    # Get corresponding image paths
    selected_paths = [image_paths[idx] for idx in low_similarity_indices]

    loaded_images = []
    valid_paths = []
    valid_indices = []

    print(f"Loading {n_images} images with lowest RDM column similarity...")

    # Load images
    for i, (path, orig_idx) in enumerate(zip(selected_paths, low_similarity_indices)):
        try:
            # Check if file exists
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue

            # Load image with cv2
            img = cv2.imread(path)

            if img is None:
                print(f"Warning: Could not load image: {path}")
                continue

            # Convert BGR to RGB for matplotlib display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            loaded_images.append(img_rgb)
            valid_paths.append(path)
            valid_indices.append(orig_idx)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    print(f"Successfully loaded {len(loaded_images)} out of {n_images} requested images")

    if len(loaded_images) == 0:
        print("No images could be loaded!")
        return [], []

    # Calculate grid dimensions
    n_loaded = len(loaded_images)
    grid_cols = n_categories
    grid_rows = (n_loaded + grid_cols - 1) // grid_cols

    # Create figure and display images
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    fig.suptitle(f'Images leading to lowest similarity',
                 fontsize=16, y=0.98)

    # Handle case where we have only one row
    if grid_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(grid_rows * grid_cols):
        col = i // grid_rows
        row = i % grid_rows
        ax = axes[row, col]

        if i < len(loaded_images):
            # Display image
            ax.imshow(loaded_images[i])

            category_value = compactness_values[i//nb_images_per_category] # add borders to categories with low compactness for model 1
            '''if category_value is not None and category_value < 0:
                #print('Border added')
                # Turn on axis to show border
                ax.axis('on')
                # Set all spines to black with thick width
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(4)
                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')'''

        if row == 0:
            label = valid_paths[i].split('/')[-2].split('_')[1]
            ax.set_title(f'{label}', fontsize=19)

        ax.axis('off')

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    return loaded_images, valid_paths



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


def sample_rdm_pairs(RDM1, RDM2, n_samples=100000, subset_size=40,
                                    batch_size=10000, seed=None):
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

    n_images = RDM1.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    all_sims_samples = []
    all_indices = []
    print(f"Processing {n_samples} samples in {n_batches} batches of {batch_size}...")

    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        current_batch_size = end_idx - start_idx

        # Allocate batch arrays
        batch_sim = np.zeros((current_batch_size))
        batch_indices = np.zeros((current_batch_size, subset_size), dtype=int)

        for i in range(current_batch_size):
            # Randomly select images
            indices = np.random.choice(n_images, size=subset_size, replace=False)
            indices = np.sort(indices)

            # Extract submatrices
            batch_sim[i] = rsa.Compute_sim_RDMs(RDM1[np.ix_(indices, indices)], RDM2[np.ix_(indices, indices)], center = False, metric = 'pearson' )
            batch_indices[i] = indices

        all_sims_samples.append(batch_sim)
        all_indices.append(batch_indices)

    # Concatenate all batches
    sim_samples = np.concatenate(all_sims_samples, axis=0)
    indices_used = np.concatenate(all_indices, axis=0)


    return sim_samples, indices_used