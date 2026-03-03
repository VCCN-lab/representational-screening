from os.path import join
import os

#### Custom libraries
import lib.algos_maxRSA as max_rsa
import lib.utils_RSA as rsa

import lib.utils_screening as screen
from lib.algos import *
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--path2activations', help='path to activations of the models we want to compare')
parser.add_argument('--model1', help='name of model 1')
parser.add_argument('--model2', help='name of model 2')
parser.add_argument('--path2dataset', help='path to dataset the activations were computed from.'
                                           'Expected to have subdirectories with the model names given above, each with a imagelist and a npy activation')
parser.add_argument('--xpname', help='name of xp to save')
parser.add_argument('--metric', default = 'R-squared', help='name compactness metric to use')
parser.add_argument('--nb_subcategories', type = int, default = '12', help='final number of categories in the stimulus set')
parser.add_argument('--nb_exemplars', type = int, default = '4', help='final number of exemplar per category')

# Image Picking
opt = parser.parse_args()
print(opt)


rootsavedir = f'figures/{opt.xpname}/'
if not os.path.exists(rootsavedir):
    os.makedirs(rootsavedir)
if not os.path.exists(join(rootsavedir, 'subRDMs')):
    os.makedirs(join(rootsavedir, 'subRDMs'))
if not os.path.exists(join(rootsavedir, 'tSNE')):
    os.makedirs(join(rootsavedir, 'tSNE'))
if not os.path.exists(join(rootsavedir, 'subsets')):
    os.makedirs(join(rootsavedir, 'subsets'))


###-----------------------------------------------------------------------------------------------------------------
#   SCREENING ALGORITHM
###---------------------------------------------------------------------------------------------------------###

#### Load activations

path2activations = opt.path2activations + '/%s'
print(path2activations)
models = [opt.model1, opt.model2]

imagelists = {}
activations = {}
for model in models:
    with open(join(path2activations%model, 'imagepaths.txt'), 'r') as f:
        imagelists[model] = [line.strip() for line in f.readlines()]
    activations[model] = np.load(join(path2activations % model, 'activations.npy'))

### check if images were shown in the same order
assert imagelists[models[0]] == imagelists[models[1]]
imagelist = imagelists[models[0]] # since they are the same, only consider one list


#### Normalize vectors to bias towards colinearity measuer
print('Normalizing activations - make sure the activation arrays have the shape (nb_images, nb_activations)')
for model in models:
    activations[model] = activations[model].reshape(activations[model].shape[0], activations[model].shape[1])
    norms = np.linalg.norm(activations[model], axis=1, keepdims=True)
    activations[model] = activations[model]/norms # normalization


#### check if each category has the same number of exemplars and list all categories in listcats
print('checking if each category has the same number of exemplars and saving category names')
count = 0
cat = ''
listcat = list()
for i, imgp in enumerate(imagelist):
    current_cat = imgp.split('/')[-2] # assumes the dataset as subdirectories for each category
    if i == 0:
        cat = current_cat
        listcat.append(current_cat)
    if cat != current_cat:
        cat = current_cat
        listcat.append(current_cat)
        count = 1
    else:
        count += 1

nb_per_cat = count # in val, 50 images per cate

### reshape activations
cat_activations = activations.copy()
for model in models:
    shape = activations[model].shape
    cat_activations[model] = activations[model].reshape(-1, nb_per_cat, shape[-1])

#### Compute compactness
print('Computing compactness')
metric = opt.metric
savedir = f'results/{opt.xpname}'

if not os.path.exists(savedir):
    os.makedirs(savedir)
sorted_compactness, sorted_compact_categories, compactness = max_rsa.compute_compactness(cat_activations, models, listcat, measure = metric)
np.save(join(savedir, 'compactness.npy'), compactness)
np.save(join(savedir, 'sorted_compactness.npy'), sorted_compactness)
np.save(join(savedir, 'sorted_compact_categories.npy'), sorted_compact_categories)

#### Run screening of categories
print('screening categories')

nb_subcategories = opt.nb_subcategories

#### Screen categories accoring to the categorical relational representations
cat_similarities = {}
correlations = {}
similarities = {}

correlations, subsimilar_cats = screen.subsimilar_categories(cat_activations, models, nb_subcategories = nb_subcategories)

#### Screen categories according to compactness

nb_seleted_categories = nb_subcategories
cat_similarities_compact = {}
similarities_compact = {}
labels_maxcompact, sortedmaxdiffcats, maxdiffs = max_rsa.max_compactness_difference(
        sorted_compact_categories, compactness, listcat, models = models,
        nb_considered_categories = nb_seleted_categories, compactness_diff_measure = 'normalizedDiff'
    )

### And combine both
x = np.absolute(maxdiffs)[np.argsort(labels_maxcompact)]
y = correlations
score = x-y # maximizing compactness difference an minimizing correlation
indexes_corr_compact = np.argsort(-score)
cats_correlations_compact = np.array(listcat)[np.argsort(-score)]

#### screen examplars
print('Screening exemplars')
catRDM1, catRDM2, RDM1, RDM2, sorted_indices_corr_compact = max_rsa.find_subsimilar_subset(cat_activations, models, indexes_corr_compact[:nb_subcategories],  images_per_subset = opt.nb_exemplars, nb_per_category = nb_per_cat)
sim = rsa.Compute_sim_RDMs(RDM1, RDM2, metric = 'pearson')
print(f'Final similarity between the two models is {sim}')
np.save(f'results/{opt.xpname}/catRDM_{opt.model1}.npy', catRDM1)
np.save(f'results/{opt.xpname}/catRDM_{opt.model2}.npy', catRDM2)
np.save(f'results/{opt.xpname}/RDM_{opt.model1}.npy', RDM1)
np.save(f'results/{opt.xpname}/RDM_{opt.model2}.npy', RDM2)

#### Looking at selections
imagelist = [img.replace('/raid/leonard_vandyck/datasets/genloc/raw_v1/', opt.path2dataset) for img in imagelist]
print(imagelist[:10])
images, imagepaths = max_rsa.display_low_similarity_images(imagelist, sorted_indices_corr_compact, maxdiffs[:nb_seleted_categories], n_images=nb_subcategories*opt.nb_exemplars, save_path=f'figures/{opt.xpname}/subsets/')
