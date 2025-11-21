import time
import sklearn
import argparse
from matplotlib import pyplot as plt
from os.path import join
import os
import seaborn as sns
from tqdm.notebook import tqdm

#### Custum libraries
import lib.algos_maxRSA as max_rsa
import lib.utils_RSA as rsa
import lib.utils_CKA as cka
from lib.algos import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'genLOC2', help='dataset name')
parser.add_argument('--arch', default = 'vgg', help='name of model set')
parser.add_argument('--metric', default = 'R-squared', help='compactness metric used')
opt = parser.parse_args()


dataset = opt.dataset
arch = opt.arch

if arch == 'vgg':
    models = ['faces', 'dual', 'objects', 'random']
    submodels = ['faces', 'dual', 'objects']
elif arch == 'resnet':
    models  = ['faces', 'places', 'imagenet', 'random0']
    submodels = ['faces', 'places', 'imagenet']
elif arch == 'ohran':
    models = ['ego', 'saycam', 'imagenet', 'supervised', 'resnet', 'random']
    submodels = ['ego', 'saycam', 'imagenet', 'supervised', 'resnet']

compactness_metric = opt.metric

result2save = {}

rootsavedir = f'figures/compactness_search/'
if not os.path.exists(rootsavedir):
    os.makedirs(rootsavedir)


#models  = ['ego', 'saycam']
path2activations = f'/home/alban/Documents/activations_datadriven/{arch}%s_{dataset}/'

imagelists = {}
activations = {}
for model in models:
    with open(join(path2activations%model, 'imagepaths.txt'), 'r') as f:
        imagelists[model] = [line.strip() for line in f.readlines()]
    if arch == 'resnet':
        activations[model] = np.load(join(path2activations % model, 'fc_outputs.npy'))
    elif arch == 'vgg':
        activations[model] = np.load(join(path2activations % model, 'fc1_outputs.npy'))
    elif arch == 'ohran':
        activations[model] = np.load(join(path2activations % model, 'cls_tokens.npy'))

imagelist = imagelists[model]


#### Normalize vectors
for model in models:
    if arch == 'resnet':
        activations[model] = activations[model].reshape(activations[model].shape[0], activations[model].shape[1])
    norms = np.linalg.norm(activations[model], axis=1, keepdims=True)
    activations[model] = activations[model]/norms # normalization

### check if images were shown in the same order
assert imagelists[models[0]] == imagelists[models[1]]
imagelist = imagelists[models[0]] # since they are the same, only consider one list

#### check if each category has the same number of images and list all categories in listcats
count = 0
cat = ''
listcat = list()
for i, imgp in enumerate(imagelist):
    current_cat = imgp.split('/')[-2]
    if i == 0:
        cat = current_cat
        listcat.append(current_cat)
    if cat != current_cat:
        cat = current_cat
        listcat.append(current_cat)
        count = 1
    else:
        count += 1

nb_per_cat = count # in val and test, 50 images per cate

### reshape activations according to include categories
cat_activations = activations.copy()

for model in models:
    shape = activations[model].shape
    cat_activations[model] = activations[model].reshape(-1, nb_per_cat, shape[-1])

savedir = f'figures/compactness_search/{arch}_{compactness_metric}_{dataset}/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

#if not os.path.exists(join(savedir, 'compactness.npy')):
start_time = time.time()
sorted_compactness, sorted_compact_categories, compactness = max_rsa.compute_compactness(cat_activations, models, listcat, measure = compactness_metric)
end_time = time.time()
result2save['execution_time'] = end_time - start_time
np.save(join(savedir, 'compactness.npy'), compactness, )

fig_compactness, ax_compactness = max_rsa.plot_stats_one(sorted_compactness,models,  ['Categories', 'Compactness'])
fig_compactness.savefig(join(savedir, 'compactness.png'))

#### Compute correlations between the model's compactness
if len(submodels) == 3:
    nb_cols = 3
elif len(submodels) == 4:
    nb_cols = 6
elif len(submodels) == 5:
    nb_cols = 10

fig, subs = plt.subplots(1,nb_cols, sharex=True, sharey=True, figsize = (1+3*nb_cols, 4))
count = 0
for i, model1 in enumerate(submodels[:-1]):
    for j, model2 in enumerate(submodels[i+1:]):
        #print(f'{model1} vs {model2}')
        subs[count].scatter(compactness[model1], compactness[model2], color = 'k', s = 3)
        corr = np.round(np.corrcoef(compactness[model1], compactness[model2])[0,1], 2)
        subs[count].set_title(f'{model1} vs {model2}: {corr}')
        subs[count].set_xlabel('Compactness model1')
        subs[0].set_ylabel('Compactness model2')
        count+=1
fig.tight_layout()
fig.savefig(join(savedir,'Correlations_compactness.png'))


model_corr_matrix= np.zeros((len(models), len(models)))


for m1, model1 in enumerate(models):
    for m2, model2 in enumerate(models):
        model_corr_matrix[m1,m2] = np.round(np.corrcoef(compactness[model1], compactness[model2])[0,1], 2)
fig = plt.figure(figsize=(6, 6))
plt.rcParams['axes.grid'] = False
# Replace the plt.imshow() section with:
sns.heatmap(1-model_corr_matrix,
        annot=True, fmt='.2f', cmap='grey',vmin=0, vmax=1,
        xticklabels=models, yticklabels=models, cbar=False)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
fig.savefig(join(savedir,'RDM_compactness.png'))

nb_subcategories = 12
cat_similarities = {}
correlations = {}
similarities = {}


for i, model1 in enumerate(models[:-1]):
    for j, model2 in enumerate(models[i+1:]):
        print(f'{model1} vs {model2}')
        correlations[f'{model1}_{model2}'], subsimilar_cats = max_rsa.subsimilar_categories(cat_activations, [model1, model2], nb_subcategories = nb_subcategories)
        RDM1, RDM2, RDM1_sorted, RDM2_sorted, sorted_indices = max_rsa.find_subsimilar_subset(cat_activations, [model1, model2], subsimilar_cats,  images_per_subset = 4, nb_per_category = 50)
        cat_sim = rsa.Compute_sim_RDMs(RDM1, RDM2, metric = 'pearson')
        sim = rsa.Compute_sim_RDMs(RDM1_sorted, RDM2_sorted, metric = 'pearson')

        cat_similarities[f'{model1}_{model2}'] = cat_sim


        similarities[f'{model1}_{model2}'] = sim


nb_seleted_categories = nb_subcategories
cat_similarities_compact = {}
similarities_compact = {}
sorted_indices = {}
maxdiffs = {}
labels_maxcompact = {}
selected_categories_comp = {}
for i, model1 in enumerate(models[:-1]):
    for j, model2 in enumerate(models[i+1:]):
        print(f'{model1} vs {model2}')
        labels_maxcompact[f'{model1}_{model2}'], sortedmaxdiffcats, maxdiffs[f'{model1}_{model2}'] = max_rsa.max_compactness_difference(
                sorted_compact_categories, compactness, listcat, models = [model1, model2],
                nb_considered_categories = nb_seleted_categories, compactness_diff_measure = 'normalizedDiff'
            )
        RDM1, RDM2, RDM1_sorted, RDM2_sorted, sorted_indices[f'{model1}_{model2}'] = max_rsa.find_subsimilar_subset(cat_activations, [model1, model2], labels_maxcompact[f'{model1}_{model2}'][:nb_seleted_categories],  images_per_subset = 4, nb_per_category = 50)
        cat_sim = rsa.Compute_sim_RDMs(RDM1, RDM2, metric = 'pearson')
        sim = rsa.Compute_sim_RDMs(RDM1_sorted, RDM2_sorted, metric = 'pearson')

        selected_categories_comp[f'{model1}_{model2}'] = sortedmaxdiffcats[:nb_subcategories]
        cat_similarities_compact[f'{model1}_{model2}'] = cat_sim

        similarities_compact[f'{model1}_{model2}'] = sim

result2save['selected_categories_comp'] = selected_categories_comp

#### Compute correlations between the model's

fig, subs = plt.subplots(1,nb_cols, sharex=True, sharey=True, figsize = (1+3*nb_cols, 4))
count = 0
### let's try to combine both for a look!
cats_correlations_compact = {}
indexes_corr_compact = {}

for i, model1 in enumerate(models[:-1]):
    for j, model2 in enumerate(models[i+1:]):
        name = f'{model1}_{model2}'
        x = np.absolute(maxdiffs[f'{model1}_{model2}'])[np.argsort(labels_maxcompact[name])]
        y = correlations[f'{model1}_{model2}']
        score = x-y # maximizing compactness difference an minimizing correlation
        indexes_corr_compact[f'{model1}_{model2}'] = np.argsort(-score) # sort categories according to new score

        cats_correlations_compact[f'{model1}_{model2}'] = np.array(listcat)[np.argsort(-score)]
        if 'random' not in name:
            subs[count].scatter(x, y, color = 'k', s = 4)
            subs[count].scatter(x[np.argsort(-score)[:12]], y[np.argsort(-score)[:12]], color = 'green', s = 5)
            corr = np.round(np.corrcoef(x, y)[0,1], 2)
            subs[count].set_title(f'{model1} vs {model2}: {corr}')
            subs[count].set_xlabel('Compactness diff')
            subs[0].set_ylabel('Categorical cosine similarity')
            count+=1
fig.tight_layout()
fig.savefig(join(savedir,'Compactness Vs Intra-distance pattern.png'))

### compute new similiarity scores using hte newly selected categories

cat_similarities_corr_compact = {}
similarities_corr_compact = {}
sorted_indices_corr_compact = {}
controversial_RDMs_corr_compact = {}
images_per_subset = 4

selected_categories = {}

for i, model1 in enumerate(models[:-1]):
    for j, model2 in enumerate(models[i+1:]):
        print(f'{model1}_{model2}')
        RDM1, RDM2, RDM1_sorted, RDM2_sorted, sorted_indices_corr_compact[f'{model1}_{model2}'] = max_rsa.find_subsimilar_subset(cat_activations, [model1, model2], indexes_corr_compact[f'{model1}_{model2}'][:nb_subcategories],  images_per_subset = images_per_subset, nb_per_category = 50)
        controversial_RDMs_corr_compact[f'{model1}_{model2}'] = []
        controversial_RDMs_corr_compact[f'{model1}_{model2}'].append(RDM1_sorted)
        controversial_RDMs_corr_compact[f'{model1}_{model2}'].append(RDM2_sorted)
        cat_sim = rsa.Compute_sim_RDMs(RDM1, RDM2, metric = 'pearson')
        sim = rsa.Compute_sim_RDMs(RDM1_sorted, RDM2_sorted, metric = 'pearson')

        cat_similarities_corr_compact[f'{model1}_{model2}'] = cat_sim
        selected_categories[f'{model1}_{model2}'] = cats_correlations_compact[f'{model1}_{model2}'][:nb_subcategories]
        savename = f'corr_compact_{model1}_{model2}'
        similarities_corr_compact[f'{model1}_{model2}'] = sim

result2save['selected_categories'] = selected_categories

import glob
dirsamples = f'results/sim_samples_{arch}_%s_{dataset}.npy'
fig, subs = plt.subplots(nrows = 1, ncols = nb_cols, figsize = (1 + 4*nb_cols,6), sharex = True, sharey = True)
count = 0
for i, model1 in enumerate(submodels[:-1]):
    for j, model2 in enumerate(submodels[i+1:]):
        name = f'{model1}_{model2}'
        sample = np.load(dirsamples%name)
        hist, bin_edges = np.histogram(sample, 100)
        subs[count].bar(bin_edges[:-1],hist/max(hist), width = bin_edges[1] - bin_edges[0], linewidth = 0, align = 'edge')
        #subs[f//5, f%5].legend()
        subs[count].set_xlabel('Similarity')
        subs[count].set_ylabel('Density')
        subs[count].set_title(name)
        #subs[count].vlines(similarities[name],0,1, 'g')
        subs[count].vlines(similarities_compact[name],0,1, 'r')
        subs[count].vlines(similarities_corr_compact[name],0,1, 'orange')
        count+=1


plt.tight_layout()
fig.savefig(join(savedir,'sim_samples.png'))
plt.close()

result2save['similarities_compact'] = similarities_compact
result2save['similarities_corr_compact'] = similarities_corr_compact

import pickle
geeky_file = open(join(savedir,'results.pkl'), 'wb')
pickle.dump(result2save, geeky_file)
geeky_file.close()