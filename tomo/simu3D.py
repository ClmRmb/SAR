
'''
SAR TOMOGRAPHY DATA SIMULATION AND FOCUSING
THIS SCRIPT IS INTENDED TO BE USED FOR THE TRAINING COURSE SAR TOMOGRAPHY
TRAINING, HELD IN BEIJING IN FEBRUARY 2015 BY LAURENT FERRO-FAMIL AND
STEFANO TEBALDINI.
THIS SCRIPT WAS DEVELOPED BY STEFANO TEBALDINI.
YOU ARE WELCOME TO ADDRESS ME QUESTIONS/COMMENTS/CORRECTIONS AT
stefano.tebaldini@polimi.it
'''
#%%
from enum import auto
import numpy as np
import math
from matplotlib import pyplot as plt
from imagers import Beamforming, FFT

class Scene:
    def __init__(self,topo,nb_acquisitions=20,master=5,
    sensors=None,B=40e6,f0=1e9,c=3e8,mean_height=1000,dy=None,flat_ref=False) -> None:
        
        # RADAR PARAMETERS
        self.B = B   # pulse bandwdith [Hz]
        self.f0 = f0  # carrier frequency [Hz]
        self.c = c    # light velocity [m/s]
        self.lambda_ = c/f0 # wavelength [m]
        self.pho_r = c/2/B # range resolution [m]
        self.master = master

        # ACQUISITION GEOMETRY
        self.N = nb_acquisitions     # number of flights
        self.H = mean_height   # mean flight altitude

        # Sensors matrix : [azimut,range,height]
        if sensors:
            self.S = sensors
        else:
            self.S = np.zeros((3,self.N))
            self.S[1] = np.linspace(0,self.N-1,self.N)*50
            self.S[2] = self.H + np.linspace(0,self.N-1,self.N)*15 

        if dy:
            self.dy = dy
        else:
            self.dy = self.pho_r/3   # ground range bin spacing

        self.y_ax = np.arange(topo.y_min,topo.y_max,self.dy) # ground range axis
        self.z_true = np.interp(self.y_ax,topo.y,topo.z)
        

        if flat_ref:
            self.z_ref = 0*self.z_true
        else:
            self.z_ref = self.z_true

        # SAR GEOMETRY

        self.R_ref = np.zeros((self.N,len(self.y_ax)))
        for n in range(self.N):
            self.R_ref[n] = np.sqrt((self.S[0,n] - self.y_ax)**2 + (self.S[1,n] - self.z_ref)**2)

        self.r_min = np.min(self.R_ref[self.master])
        self.r_max = np.max(self.R_ref[self.master])
        
        self.dr = self.pho_r
        self.rg_ax = np.arange(self.r_min,self.r_max,self.dr)
        self.rg_ax = np.reshape(self.rg_ax,(len(self.rg_ax),1))


        self.Nr = len(self.rg_ax)

        self.z_ref_of_r = np.interp(self.rg_ax,self.R_ref[self.master,:],self.z_ref)
        self.y_ref_of_r = np.interp(self.rg_ax,self.R_ref[self.master,:],self.y_ax)

        # Distances to reference topography (ground range)
        
        self.R_ref = np.zeros((self.N,len(self.rg_ax)))
        self.Teta = np.zeros((self.N,len(self.rg_ax)))
        for n in range(self.N):
            self.R_ref[n,:] = np.squeeze(np.sqrt((self.S[0,n] - self.y_ref_of_r)**2 + (self.S[1,n] - self.z_ref_of_r)**2))
            self.Teta[n,:] = np.squeeze(np.arccos( (self.S[1,n] - self.z_ref_of_r.T)/self.R_ref[n]  ))

        # Phase to height conversion factors

        self.Kz = np.zeros((self.N,len(self.rg_ax)))
        for n in range(self.N):
            dteta = self.Teta[n] - self.Teta[self.master]
            self.Kz[n] = -4*math.pi/self.lambda_*dteta/np.sin(self.Teta[n])

        # vertical resolution
        self.dKz_max = np.max(self.Kz) - np.min(self.Kz)
        self.min_pho_z = 2*math.pi/self.dKz_max

        # height of ambiguity
        self.dKz = self.Kz[2,:]-self.Kz[1,:]
        self.z_amb = 2*math.pi/np.max(np.abs(self.dKz))

    def show(self):
        plt.subplots(figsize = (10,10))
        plt.subplot(311)
        plt.plot(self.S[0],self.S[1],'r+')
        plt.plot(self.y_ax,self.z_true)
        plt.plot([self.S[0,self.master],self.y_ax[0]],[self.S[1,self.master],self.z_true[0]],
                    [self.S[0,self.master],self.y_ax[-1]],[self.S[1,self.master],self.z_true[-1]])
        
        plt.subplot(312)
        plt.plot(self.y_ax,self.z_ref,'b')
        plt.title('Topography - ground coordinates [m]')

        plt.subplot(313)
        plt.plot(self.rg_ax,self.z_ref_of_r)
        plt.title('Topography - SAR coordinates [m]')

class SARImage:
    def __init__(self,topo,geo) -> None:
       self.Np = len(topo.y)
       self.s = topo.a*np.exp(1j*np.random.rand(1,self.Np))
       self.rg_ax = geo.rg_ax
       self.pho_r = geo.pho_r
       self.I = np.zeros((geo.N,geo.Nr),dtype='complex64')
       self.Ic = np.zeros_like(self.I,dtype='complex64')
       self.Ii = np.zeros_like(self.I,dtype='complex64')
       self.Cov = np.zeros((geo.N,geo.N,geo.Nr),dtype='complex64')
       self.N = geo.N
       self.Nr = geo.Nr
       self.master = geo.master
       self.lambda_ = geo.lambda_

    def get_image(self,topo,geo):
        Rm = np.sqrt( (geo.S[0,geo.master] - topo.y)**2 + (geo.S[1,geo.master] - topo.z)**2 )
        for n in range(geo.N):
            Rn = np.sqrt( (geo.S[0,n] - topo.y)**2 + (geo.S[1,n] - topo.z)**2 )
            phi = self.phase(Rn)
            rho = self.psf(Rn).T
            self.I[n] = np.sum(self.s*np.exp(-1j*phi)*rho,1)
            rho = self.psf(Rm).T
            self.Ic[n] =  np.sum(self.s*np.exp(-1j*phi)*rho,1)
        phi = self.phase(geo.R_ref)
        self.Ii = self.Ic*np.exp(1j*phi)
        self.Ii = self.Ii*np.exp(-1j*np.angle(self.Ic[self.master]))

    def get_cov(self):
        for n in range(self.N):
            for m in range(n,self.N):
                self.Cov[n,m,:] = np.squeeze(self.Ii[n,:].reshape((self.Nr,1))*np.conjugate(self.Ii[m,:].reshape((self.Nr,1))))
                self.Cov[m,n,:] = np.conjugate(self.Cov[n,m,:])

        

    def psf(self,r,width=None):
        r = np.reshape(r,(len(r),1))
        if not width:
            width = 1/self.pho_r
        
        Delta = r - self.rg_ax.T
        idx = np.argmin(abs(Delta),1)

        r0 = self.rg_ax[idx]
        a = 2/width
        b = 1 - 2/width*r0
        spread = np.zeros_like(Delta)
        mask = Delta > 0
        Ranges = np.tile(self.rg_ax.T,(self.Np,1))
        spread += (Ranges*a + b)*mask
        spread += (-Ranges*a + b -2)*(mask==0)
        spread[spread<0] = 0
        return spread

    def phase(self,R):
        return 4*np.math.pi*R/self.lambda_



    def show(self):
        plt.subplots(figsize = (10,10))
        plt.subplot(321)
        plt.imshow(np.abs(self.I),aspect='auto')
        plt.subplot(322)
        plt.imshow(np.angle(self.I*np.exp(-1j*np.angle(self.I[self.master]))),aspect='auto')
        plt.subplot(323)
        plt.imshow(np.abs(self.Ic),aspect='auto')
        plt.subplot(324)
        plt.imshow(np.angle(self.Ic*np.exp(-1j*np.angle(self.Ic[self.master]))),aspect='auto')
        plt.subplot(325)
        plt.imshow(np.abs(self.Ii),aspect='auto')
        plt.subplot(326)
        plt.imshow(np.angle(self.Ii),aspect='auto')

class Topography:
    def __init__(self,points=None,amplitude=None,dy=1,nb_scatter=1,dense=True) -> None:
        
        if not points:
            dense = True
            y_min = 4500
            y_max = 6500
            y_ax = np.arange(y_min,y_max,1)
            points = [y_ax, 10*np.hamming(len(y_ax))]
            

        if dense:
            self.y = points[0]
            self.z = points[1]
            self.y_min = np.min(self.y)
            self.y_max = np.max(self.y)
            if amplitude:
                self.a = amplitude
            else:
                self.a = np.ones_like(self.y)


p = [np.concatenate([np.arange(7000,8000,10),np.arange(7000,8000,10)]),
    np.concatenate([20*np.hamming(100),30+20*np.hamming(100)])]   
p = [[5700,6200],[20,20]]         
topo_ref = Topography()
scene = Scene(topo_ref,mean_height=800,flat_ref=True)
topo = Topography(p)
#scene.show()
im = SARImage(topo_ref,scene)
#r = np.array((7509)).reshape((1,1))
#im.psf(r,width=2)
im.get_image(topo_ref,scene)
im.show()
im.get_cov()
z_ax = np.linspace(-30,30,50)
est = Beamforming(scene,z_ax)
est.get_tomogram(im)
est.show()
# %%

