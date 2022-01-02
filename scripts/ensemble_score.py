import numpy as np
from itertools import chain, combinations

from datetime import datetime
from os.path import join
import os
from numpy import genfromtxt

scores = []
names = []

score_dir = "scripts/Ensemble_score"
t = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
output_file = join(score_dir, "ensemble_scores_" + t + ".txt")

with open(output_file, 'w') as f:
    f.write("Evaluating ensemble at time " + t + "\n")

# Collect all csv files (model scores)
for path, subdirs, files in os.walk(join(os.getcwd(), score_dir)):
    for name in files:
        if '.csv' in name:
            names.append(name[7:-4])
            scores.append(genfromtxt(join(os.getcwd(), score_dir, name), delimiter=','))

nm = len(names)

scr = np.nan_to_num(np.stack(scores, axis=0), nan=0)
# Calculate the harmonic mean of ENS components
def hm(scores):
    if np.min(scores) == 0:
        return 0
    else:
        return 4 / np.sum(1/scores)

# Evaluate each model individually
for i in range(nm):
    components = np.mean(scr[i], axis=0)
    with open(output_file, 'a') as f:
        f.write(names[i] + "\n")
        f.write("    Old ENS " + str(components[-1]) + "\n")
        f.write("    Lazy ENS " + str(hm(components[:4])) + "\n")
        f.write("    MAD: {0} OLS: {1} EMD: {2} SSIM: {3}".format(components[0], components[2], components[3], components[1]) + "\n")

# Compute the ENS (+ component) for best prediction for each sample
def en_score(s):
    best_score = np.zeros((s.shape[1], 6))
    for i in range(s.shape[1]):
        best = np.argmax(s[:,i,-1])
        best_score[i, :5] = s[best,i,:]
        best_score[i, -1] = best
    means = np.mean(best_score[:,:-1], axis=0)
    return means[-1], hm(means[:4]), means[[0,2,3,1]]

with open(output_file, 'a') as f:
    f.write("Score of ensemble of ALL models" + "\n")
    f.write(str(en_score(scr)) + "\n")

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# Compute scores of each subset ensemble of models
lst = list(powerset(list(range(nm))))[1+nm:]
e_scores = []
for l in lst:
    l=list(l)
    e_scores.append(en_score(scr[l])[1])
    with open(output_file, 'a') as f:
        f.write(str(l) + "\n")
        f.write(str(e_scores[-1]) + "\n")

e_scores, lst = zip(*sorted(zip(e_scores, lst)))
for i in range(len(e_scores)):
    with open(output_file, 'a') as f:
        f.write(str(lst[i]) + ": " + str(e_scores[i]) + "\n")
    #print(str(lst[i]) + ": " + str(e_scores[i]))
