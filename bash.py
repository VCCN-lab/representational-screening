import subprocess
import shlex
import os


### Run screening
path2activations = 'data/activations/'
model1 = 'places365'
model2 = 'imagenet'
metric = 'R-squared'
path2dataset = 'data/dataset/'
xpname = f'{model1}_{model2}'
nb_subcategories = 12
nb_exemplars = 4

command = (f"python3 representational_screening.py --path2activations {path2activations}\
 --model1 {model1} --model2 {model2} --metric {metric} --path2dataset {path2dataset}\
  --xpname {xpname} --nb_subcategories {nb_subcategories} --nb_exemplars {nb_exemplars}")
args = shlex.split(command)
subprocess.call(args)