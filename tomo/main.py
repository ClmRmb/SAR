#%%

from scipy.io import loadmat
from tomogeo import Geo, Point
from matplotlib import pyplot as plt
import numpy as np

#data = loadmat('/cal/homes/crambour/data.mat')
#print(f'data = {data["data"][0][0][0].shape}')
#%%

c=299810000
fc=9.65e9
lambda_=c/fc

data = loadmat('traj_blue.mat')
longlat = loadmat('../geometry/longlat_blue.mat')['longlat_blue']
traj = data["traj_blue"]
#longlat = data["data"][0][0][1]
geo = Geo(lambda_,np.stack((traj[:,:,0],traj[:,:,1],traj[:,:,2]),axis=2),traj[:,:,3:],longlat = longlat)
point = Point(geo, pradar = [50,50,10])
point.print()
point = Point(geo, pground = [50,50,10])
point.print()
print(f'theta from geo = {geo.theta[50,50,0]}')

geo.get_theta_map()



plt.figure()
plt.imshow(geo.theta[:,:,geo.master])
plt.savefig('test.png')


# %%
