import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.patches as mpatches

os.chdir('C://Users//Oto//Documents//GitHub//drought_impact_forecasting')

MAD = pd.read_csv('MAD.csv')
OLS = pd.read_csv('OLS.csv')
EMD = pd.read_csv('EMD.csv')
SSIM = pd.read_csv('SSIM.csv')

ALL = MAD.merge(OLS, on='epoch').merge(EMD, on='epoch').merge(SSIM, on='epoch')
ALL = ALL[:35]

ALL['epoch'] = ALL['epoch']+1

ALL.columns = ['epoch', 'zero mad', 'last-frame mad', 'zero ols', 'last-frame ols',
               'zero emd', 'last-frame emd', 'zero ssim', 'last-frame ssim']

#fig = plt.figure(figsize=(6, 3))
#fig, ax = plt.subplots()
#fig, ax = plt.subplots(nrows=3, ncols=2)

fig = plt.figure(figsize=(7, 7), constrained_layout=True)
gs = gridspec.GridSpec(3, 4)

# MAD
ax1 = plt.subplot(gs[0, 0:2])
ax1.plot(ALL['epoch'], ALL['zero mad'], color='r', linestyle='-', label='no baseline')
ax1.plot(ALL['epoch'], ALL['last-frame mad'], color='b', linestyle='-', label='baseline')
ax1.legend(loc='lower right')
ax1.grid()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('MAD')

ax1.set_xlim(0, 35)

# OLS
ax2 = plt.subplot(gs[0, 2:])
ax2.plot(ALL['epoch'], ALL['zero ols'], color='r', linestyle='-', label='no baseline')
ax2.plot(ALL['epoch'], ALL['last-frame ols'], color='b', linestyle='-', label='baseline')
ax2.legend(loc='lower right')
ax2.grid()

ax2.set_xlabel('Epoch')
ax2.set_ylabel('OLS')

ax2.set_xlim(0, 35)

# EMD
ax3 = plt.subplot(gs[1, 0:2])
ax3.plot(ALL['epoch'], ALL['zero emd'], color='r', linestyle='-', label='no baseline')
ax3.plot(ALL['epoch'], ALL['last-frame emd'], color='b', linestyle='-', label='baseline')
ax3.legend(loc='lower right')
ax3.grid()

ax3.set_xlabel('Epoch')
ax3.set_ylabel('EMD')

ax3.set_xlim(0, 35)

# SSIM
ax4 = plt.subplot(gs[1,2:])
ax4.plot(ALL['epoch'], ALL['zero ssim'], color='r', linestyle='-', label='no baseline')
ax4.plot(ALL['epoch'], ALL['last-frame ssim'], color='b', linestyle='-', label='baseline')
ax4.legend(loc='lower right')
ax4.grid()

ax4.set_xlabel('Epoch')
ax4.set_ylabel('SSIM')

ax4.set_xlim(0, 35)

# ENS
ax5 = plt.subplot(gs[2,1:3])
file = 'Data/lastframe_vs_zero.csv'
data = pd.read_csv(os.path.join(os.getcwd(), file), delimiter=',', index_col=0)
final_epoch = 35
data = data.iloc[:final_epoch]

# we want epochs to start from 1 in graph
data.index += 1

#ax_last = plt.subplot2grid((4,8), (4//2, 2), colspan=4)

ax5.plot(data['zero'], label='no baseline', color='r')
ax5.plot(data['last_frame'], label='baseline', color='b')
ax5.plot(35, 0.2902, 'o', color='lime', label='U-Net')
ax5.plot(35, 0.2803, 'o', color='c', label='Arcon')

ax5.grid(True)
ax5.set_ylabel("ENS")
ax5.set_xlabel("Epoch")

ax5.set_xlim(0, 36)
ax5.legend(loc='lower center')

ax5.plot(list(range(1, final_epoch+1)), (final_epoch) * [0.31], '--', color='gray')

plt.show()

gs.tight_layout(fig)

outputFile = 'ndvi_separated_components'
plt.savefig(outputFile+'.jpg', dpi=2000)
plt.savefig(outputFile+'.pdf')

print("DONE")

