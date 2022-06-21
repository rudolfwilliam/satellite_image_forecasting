"""Compare the first 35 training epochs of a run of SGDConvLSTM with zero baseline and 
   one with last frame baseline. Store pdf images of the plots for ENS and the different score components."""

import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt

file = 'Data/lastframe_vs_zero.csv'
data = pd.read_csv(os.path.join(os.getcwd(), file), delimiter=',', index_col=0)
final_epoch = 35
data = data.iloc[:final_epoch]

# we want epochs to start from 1 in graph
data.index += 1

# Set up figure
fig = plt.figure(figsize=(6, 4)) 

plt.title("ENS convergence comparison for models" + "\n" + "with and without a baseline")
plt.ylabel("ENS")
plt.xlabel("Epoch")
plt.grid(True)

axes = plt.gca()
axes.set_xlim([0, final_epoch+1])
axes.set_ylim([0.251, 0.315])

plt.plot(data['zero'], label='no baseline', color='r')
plt.plot(data['last_frame'], label='last frame', color='b')
plt.plot(35, 0.2902, 'o', color='lime', label='U-Net')
plt.plot(35, 0.2803, 'o', color='c', label='Arcon')

plt.plot(list(range(1, final_epoch+1)), (final_epoch) * [0.31], '--', color='gray')

plt.legend(loc='lower right', title="Model")
fig.tight_layout()
outputFile = os.path.join(os.getcwd(), 'visualizations', 'lastframe_vs_none.pdf')
plt.savefig(outputFile)

plt.show()
