import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

sys.path.append(os.getcwd())

score='ENS'
score='MAD'
score='OLS'
score='EMD'
score='SSIM'

x_labels = ['PBL', 'Unet', 'Arcon', 'Dia', 'SG', 'SGED']
colors = ['tab:gray', 'tab:gray', 'tab:gray', 'tab:gray', 'tab:orange', 'tab:orange']

if score == 'ENS':
    scores = {'iid': [0.2625, 0.2902, 0.2803, 0.3266, 0.3176 , 0.3164],
            'ood': [0.2587, 0.2854, 0.2655, 0.3204, 0.3146, 0.3121],
            'extreme': [0.1939, 0.2364, 0.2215, 0.2140, 0.2740, 0.2595],
            'seasonal': [0.2676, 0.1955, 0.1587, 0.2193, 0.2162, 0.1790]}
    y_max = 0.36
elif score == 'MAD':
    scores = {'iid': [0.2315, 0.2482, 0.2414, 0.2638, 0.2589, 0.2580],
            'ood': [0.2248, 0.2402, 0.2314, 0.2541, 0.2512, 0.2497],
            'extreme': [0.2158, 0.2286, 0.2243, 0.2137, 0.2366, 0.2304],
            'seasonal': [0.2329, 0.2169, 0.2014, 0.2146, 0.2207, 0.2056]}
    y_max = 0.31
elif score == 'OLS':
    scores = {'iid': [0.3239, 0.3381, 0.3216, 0.3513, 0.3456 , 0.3440],
            'ood': [0.3236, 0.3390, 0.3088, 0.3522, 0.3481, 0.3450],
            'extreme': [0.2806, 0.2973, 0.2753, 0.2906, 0.3199, 0.3164],
            'seasonal': [0.3848, 0.3811, 0.3788, 0.3778, 0.3756, 0.3585]}
    y_max = 0.43
elif score == 'EMD':
    scores = {'iid': [0.2099, 0.2336, 0.2258, 0.2623, 0.2533, 0.2532],
            'ood': [0.2123, 0.2371, 0.2177, 0.2660, 0.2597, 0.2587],
            'extreme': [0.1614, 0.2065, 0.1975, 0.1879, 0.2279, 0.2186],
            'seasonal': [0.2034, 0.1903, 0.1787, 0.2003, 0.1723, 0.1543]}
    y_max = 0.3
if score == 'SSIM':
    scores = {'iid': [0.3265, 0.3973, 0.3863, 0.5565, 0.5292, 0.5237],
            'ood': [0.3112, 0.3721, 0.3432, 0.5125, 0.4977, 0.4887],
            'extreme': [0.1605, 0.2306, 0.2084, 0.1904, 0.3497, 0.2993],
            'seasonal': [0.3184, 0.1255, 0.0834, 0.1685, 0.1817, 0.1218]}
    y_max = 0.63

socres_df = pd.DataFrame(data=scores).round(3)

index = np.arange(len(x_labels)) + 0.3

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,5))

loc = ticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals

axes[0,0].bar(x=index, height=socres_df['iid'], color=colors, edgecolor='#000000')
axes[0,0].set_xticks(index, x_labels)
axes[0,0].bar_label(axes[0,0].containers[0], labels=[f'{x:,.3f}' for x in axes[0,0].containers[0].datavalues])
axes[0,0].set_ylim(0, y_max)
axes[0,0].yaxis.set_major_locator(loc)
axes[0,0].title.set_text('iid')
axes[0,1].bar(x=index, height=socres_df['ood'], color=colors, edgecolor='#000000')
axes[0,1].set_xticks(index, x_labels)
axes[0,1].bar_label(axes[0,1].containers[0], labels=[f'{x:,.3f}' for x in axes[0,1].containers[0].datavalues])
axes[0,1].set_ylim(0, y_max)
axes[0,1].yaxis.set_major_locator(loc)
axes[0,1].title.set_text('ood')
axes[1,0].bar(x=index, height=socres_df['extreme'], color=colors, edgecolor='#000000')
axes[1,0].set_xticks(index, x_labels)
axes[1,0].bar_label(axes[1,0].containers[0], labels=[f'{x:,.3f}' for x in axes[1,0].containers[0].datavalues])
axes[1,0].set_ylim(0, y_max)
axes[1,0].yaxis.set_major_locator(loc)
axes[1,0].title.set_text('extreme')
axes[1,1].bar(x=index, height=socres_df['seasonal'], color=colors, edgecolor='#000000')
axes[1,1].set_xticks(index, x_labels)
axes[1,1].bar_label(axes[1,1].containers[0], labels=[f'{x:,.3f}' for x in axes[1,1].containers[0].datavalues])
axes[1,1].set_ylim(0, y_max)
axes[1,1].yaxis.set_major_locator(loc)
axes[1,1].title.set_text('seasonal')

fig.supxlabel('Model')
fig.supylabel(score)

plt.tight_layout()

plt.show()

plt.savefig(score+'.pdf')

print("DONE")
