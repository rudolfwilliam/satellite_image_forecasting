import os, sys
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

sys.path.append(os.getcwd())

score = 'ENS'
model = 'SGConv_LSTM'
SGConvLSTM_score_file = 'wandb/run-20231025_001918-3175p8bn/files/scores_SGConvLSTM_-01_sea.csv'
SGEDConvLSTM_score_file = 'wandb/run-20231029_SGED_seas_by_section/files/scores_SGEDConvLSTM_-01_sea.csv' ##### EDIT!!!!!!!!!!!!!!!!!!!!!
out_file = 'seas_by_section'

sections = 7
num_score_components = 5 # ENS + the 4 components
# TODO: Rearrage these in the order that we have in the rest of the paper
x_labels = [20, 40, 60, 80, 100, 120, 140]
y_labels = ['ENS', 'MAD', 'SSIM', 'OLS', 'EMD']
legend_labels = ['SGConvLSTM', 'SGEDConvLSTM']
subplot_titles = ['SGConvLSTM', 'SGEDConvLSTM']
y_tick_step = 0.1
#subplot_titles = ['Frames 0-20', 'Frames 20-40', 'Frames 40-60', 'Frames 60-80', 'Frames 80-100', 'Frames 100-120', 'Frames 120-140']

# Parse in data
#SG = pd.read_csv(SGConvLSTM_score_file)

# All average components
average_scores = 35*[0]

all_data = np.genfromtxt(join(os.getcwd(), SGConvLSTM_score_file), delimiter=',')
all_data_SGED = np.genfromtxt(join(os.getcwd(), SGEDConvLSTM_score_file), delimiter=',')
all_data_SGED[np.isnan(all_data_SGED)] = 0
# Add extra dimension for time sections
############## THIS IS NOT RESHAPED CORRECTLY!!!!!!!!!!!!!!!!!!!!!!!!!!
#all_data_reshaped = all_data.reshape(all_data.shape[0], 5, sections)

avg_comp_by_section = np.zeros((sections, 5))
avg_comp_by_section_SGED = np.zeros((sections, 5))
for i in range(sections):
    for j in range(num_score_components):
        index = i*num_score_components + j
        avg_comp_by_section[i, j] = np.mean(all_data[:, index], axis=0)
        avg_comp_by_section_SGED[i, j] = np.mean(all_data_SGED[:, index], axis=0)
    #avg_score_components_by_section[i] = np.mean(all_data_reshaped[:, :, i], axis=0)

'''
# Create a figure with 5 subplots arranged in a 1x5 grid
fig, axs = plt.subplots(5, 1, figsize=(6, 10), sharex=True)

# Plot each column of the arrays in separate subplots
for i in range(5):
    x = x_labels[:len(avg_comp_by_section)]
    axs[i].plot(x, avg_comp_by_section[:, i], label=legend_labels[0], color='red')
    axs[i].plot(x, avg_comp_by_section_SGED[:, i], label=legend_labels[1], color='blue')
    axs[i].set_ylabel(y_labels[i])
    max_value = max(np.max(avg_comp_by_section[:, i]), np.max(avg_comp_by_section_SGED[:, i]))
    axs[i].set_ylim(0, 1.2 * max_value)  # Set y-limits

    y_max = 1.1 * max_value
    y_ticks = np.arange(0, y_max + y_tick_step, y_tick_step)
    axs[i].set_yticks(y_ticks)

    axs[i].legend(loc='lower left')  # Set legend to bottom left

axs[-1].set_xticks(x_labels)
axs[-1].set_xticklabels(x_labels)
axs[-1].set_xlabel('Prediction frame')

print("DONE")

plt.tight_layout()
plt.savefig('score_evolution.pdf', format='pdf')

#plt.show()

print('Showed')
'''



# Condensing all scores into 2 subplots (1 for each model)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=False)

# Plot all 5 columns of Array 1 in the first subplot
for i in range(5):
    x = x_labels
    axs[0].plot(x, avg_comp_by_section[:, i], label=y_labels[i])

# Plot all 5 columns of Array 2 in the second subplot
for i in range(5):
    x = x_labels
    axs[1].plot(x, avg_comp_by_section_SGED[:, i], label=y_labels[i])

# Set x-axis labels, a legend, and a common title for both subplots
for i, a in enumerate(axs):
    a.set_xticks(x_labels)
    a.set_xticklabels(x_labels)
    a.set_xlabel('Prediction frame')
    a.set_ylim(0, 0.6)  # Start the y-axis from 0
    a.margins(x=0)  # Remove extra whitespace on x-axis limits
    a.set_ylabel('Score')
    a.set_title(subplot_titles[i])  # Set subplot titles

axs[0].legend(ncol=5, loc='lower center')

plt.tight_layout()

# Save the figure as a PDF
plt.savefig('score_evolution_grouped.pdf', format='pdf')
plt.show()

print("DONE")
