import numpy as np
from abc import ABC,abstractmethod
from matplotlib import pyplot as plt

class Animal(ABC):
    @abstractmethod
    def move(self):
        pass
class Imager(ABC):
    def __init__(self,geo,z_ax) -> None:
        self.Kz = geo.Kz
        self.z_ax = z_ax.reshape((1,len(z_ax)))
        self.rg_ax = geo.rg_ax
        self.Nr = geo.Nr
        self.N = geo.N
        self.Nz = len(z_ax)
        self.tomogram = np.zeros((self.Nz,self.Nr))
    
    @property
    def show(self,norm=True):
        t = self.tomogram
        if norm:
            t /= np.max(t,axis=0)
        plt.figure(figsize=(10,10))
        plt.imshow(t[::-1],aspect='auto',extent=[self.rg_ax[0,0],self.rg_ax[-1,0],self.z_ax[0,0],self.z_ax[0,-1]])

    @abstractmethod
    def inv(self):
        pass

    @abstractmethod
    def get_tomogram(self,im):
        pass
    

class FFT(Imager):
    def __init__(self, geo, z_ax) -> None:
        super().__init__(geo, z_ax)

    def inv(self,y,kz):
        A = np.asmatrix(np.exp(1j*kz*self.z_ax))
        return np.abs(A.H @ y )

    def get_tomogram(self,im):
        for r in range(self.Nr):
            kz = self.Kz[:,r].reshape((self.N,1))
            self.tomogram[:,r] = self.inv(im.Ii[:,r],kz)

    def show(self,norm=True):
        t = self.tomogram
        if norm:
            t /= np.max(t,axis=0)
        plt.figure(figsize=(10,10))
        plt.imshow(t[::-1],aspect='auto',extent=[self.rg_ax[0,0],self.rg_ax[-1,0],self.z_ax[0,0],self.z_ax[0,-1]])

class Beamforming(Imager):
    def __init__(self, geo, z_ax) -> None:
        super().__init__(geo, z_ax)

    def inv(self,C,kz):
        A = np.asmatrix(np.exp(1j*kz*self.z_ax))
        return np.abs(np.diag(A.H @ C @ A ))

    def get_tomogram(self,im):
        for r in range(self.Nr):
            kz = self.Kz[:,r].reshape((self.N,1))
            self.tomogram[:,r] = self.inv(im.Cov[:,:,r],kz)

    def show(self,norm=True):
        t = self.tomogram
        if norm:
            t /= np.max(t,axis=0)
        plt.figure(figsize=(10,10))
        plt.imshow(t[::-1],aspect='auto',extent=[self.rg_ax[0,0],self.rg_ax[-1,0],self.z_ax[0,0],self.z_ax[0,-1]])


