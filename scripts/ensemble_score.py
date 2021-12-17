import numpy as np
from itertools import product, chain, combinations


from os import listdir
import os
import pickle
from numpy import genfromtxt
from numpy.core.fromnumeric import mean, repeat

scores = []
names = []

for path, subdirs, files in os.walk(os.path.join(os.getcwd(), "scripts", "Ensemble_score")):
    for name in files:
        names.append(name[7:-4])
        scores.append(genfromtxt(os.path.join(os.getcwd(), "scripts", "Ensemble_score", name), delimiter=','))

nm = len(names)

scr = np.nan_to_num(np.stack(scores, axis=0), nan=0)

def hm(scores):
    if np.min(scores) == 0:
        return 0
    else:
        return 4 / np.sum(1/scores)

for i in range(nm):
    components = np.mean(scr[i], axis=0)
    print(names[i])
    print("    Old ENS " + str(components[-1]))
    print("    Lazy ENS " + str(hm(components[:4])))

def en_score(s):
    best_score = np.zeros((s.shape[1], 6))
    for i in range(s.shape[1]):
        best = np.argmax(s[:,i,-1])
        best_score[i, :5] = s[best,i,:]
        best_score[i, -1] = best
    means = np.mean(best_score[:,:-1], axis=0)
    return means[-1], hm(means[:4]), means[:4]
    
print("Score of ensemble of ALL models")
print(en_score(scr))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

lst = list(powerset(list(range(nm))))[1+nm:]
e_scores = []
for l in lst:
    l=list(l)
    print(l)
    e_scores.append(en_score(scr[l])[1])
    print(e_scores[-1])

e_scores, lst = zip(*sorted(zip(e_scores, lst)))
for i in range(len(e_scores)):
    print(str(lst[i]) + ": " + str(e_scores[i]))

#for i in range(2,nm+1):


#for i in ensemble:
