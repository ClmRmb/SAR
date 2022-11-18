#%%

from matplotlib import pyplot as plt
import numpy as np

im = np.load('testlast2.npy')
plt.imshow(im)
# %%
from scipy.io import loadmat
sar = loadmat('../data/im_blue_calib.mat')
sar = sar['im_blue_calib']
# %%
plt.imshow(np.log(np.abs(sar[:,:,0])),cmap='gray')
plt.imshow(im, cmap='jet', alpha=0.1)
# %%
