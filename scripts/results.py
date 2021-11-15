import os
import sys
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

file  = '/model_instances/model_12_11_2021_15_49_51/scores.csv'
#file = sys.argv[1]
print(file)

data = genfromtxt(os.getcwd() + file, delimiter=',')
means = data.mean(axis=0)
print(means)

plt.figure()
plt.hist(data[:,0], bins=10, alpha = 0.3, label = 'mean')
plt.hist(data[:,1], bins=10, alpha = 0.3, label = 'last')
plt.hist(data[:,2], bins=10, alpha = 0.3, label = 'model')

plt.show()