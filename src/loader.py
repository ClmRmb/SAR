#%%
import struct
import math
import numpy as np

import matplotlib.pyplot as plt

####################################################################################

from sartools import fftlab
from sartools import filtrage
from sartools import phaselab_V41 as phaselab
from sartools import affichage_V41 as affichage

from sardecoupe import pydecoupe_V41 as pydec
from sardecoupe import pygeom_V41 as pygeom
from sardecoupe import ecriture_V41  as ecriture
from sardecoupe import lecture_V41  as lecture

from satellites import pydecoupe_TSX_V41 as TSX





nomthr = '/path/to/image/20120723/TSX1_SAR__SSC______HS_S_SRA_20120723T173457_20120723T173457.xml'


maspydec = TSX.charger(nomthr)

#%%

grille = maspydec.grille((5.449621,43.527140),1000,1000)

np.save('Paris/test_G',grille[0])
longlat = pygeom.tabXYZ2longlat(grille[0])
np.save('Paris/test_longlat',longlat)
grilleS = maspydec.recalagesurgrille_satellite(grille)
np.save('Paris/test_S',grilleS[1])
#%%
I = maspydec.lirecoordcrop((5.449621,43.527140,0.0),10000,10000)
im = np.abs(I[0][::5,::5])#[::-1,:])
sigma = np.std(im)
mean = np.mean(im)
imv = np.minimum(np.maximum(im,mean-3*sigma),mean+3*sigma)
plt.figure(figsize=(10,10))
plt.imshow(imv,cmap='gray')
np.save('Paris/test_visu',imv)