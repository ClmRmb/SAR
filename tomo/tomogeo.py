import numpy as np
import math
import sys

class Geo:
    def __init__(self, lambda_, G, S, longlat = None, bperp=None, master=0, z_min=-5, z_max=100, Nbz=106):
        self.lambda_ = lambda_
        self.master = master
        self.z_min = z_min
        self.z_max = z_max
        self.Nbz = Nbz
        self.z_ax = np.linspace(self.z_min, self.z_max, self.Nbz)
        self.Nbx = G.shape[0]
        self.Nby = G.shape[1]
        self.G = G
        self.S = S
        self.master = master
        self.normal = self.get_normal_map()
        self.N = int(self.S.shape[2]/3)
        self.R0 = np.sqrt(G[0,0,0]**2 + G[0,0,1]**2 + G[0,0,2]**2)
        self.H0 = np.sqrt(S[0,0,0]**2 + S[0,0,1]**2 + S[0,0,2]**2) - self.R0

        self.rg = self.get_range_map()
        self.theta = self.get_theta_map()
        if bperp:
            self.bperp = bperp
        else:
            self.bperp = self.get_bperp_map()
        self.kz = self.get_kz_map()
        self.longlat = longlat
        self.res_z = self.lambda_*self.rg[0,0,0]/2/np.abs(np.max(self.bperp) - np.min(self.bperp))

    def get_range_map(self):
        return np.transpose(np.array([np.sqrt((self.G[:,:,0] - self.S[:,:,i])**2 
            + (self.G[:,:,1] - self.S[:,:,i+1])**2 + (self.G[:,:,2] - self.S[:,:,i+2])**2)
                for i in range(0,self.N)]),[1, 2, 0])

    def get_theta_map(self):
        
        return np.transpose(np.array([np.pi-np.arccos( (
            (self.G[:,:,0]-self.S[:,:,i  ])*self.G[:,:,0] 
            + (self.G[:,:,1]-self.S[:,:,i+1])*self.G[:,:,1] 
            + (self.G[:,:,2]-self.S[:,:,i+2])*self.G[:,:,2]
            )/np.sqrt((self.G[:,:,0]-self.S[:,:,i  ])**2+(self.G[:,:,1]-self.S[:,:,i+1])**2+(self.G[:,:,2]-self.S[:,:,i+2])**2)
             /np.sqrt(self.G[:,:,0]**2+self.G[:,:,1  ]**2+self.G[:,:,2  ]**2))
                for i in range(0,3*self.N,3)]),[1, 2, 0])
        '''
        return np.transpose(np.array([np.abs(np.arccos( (np.sqrt(self.S[:,:,i]**2 
            + self.S[:,:,i+1]**2 + self.S[:,:,i+2]**2) - self.R0)/self.rg[:,:,i] ))
                for i in range(0,self.N)]),[1, 2, 0])
        '''

    def get_bperp_map(self):
        dtheta =  np.transpose(np.array([self.theta[:,:,i] - self.theta[:,:,self.master] 
            for i in range(0,self.N)]),[1, 2, 0])
        return self.rg*dtheta
    
    def get_kz_map(self):
        return np.transpose(np.array([4*math.pi*self.bperp[:,:,i]
            /(self.lambda_*self.rg[:,:,self.master]*np.sin(self.theta[:,:,self.master]))
                for i in range(0,self.N)]),[1,2,0])

    def get_normal_map(self):
        n = np.sqrt(self.G[:,:,0]**2 + self.G[:,:,1]**2 + self.G[:,:,2]**2)
        return self.G/np.stack((n,n,n),axis=2)

    def ind2x(self,ix,iy,zrel):
        return self.G[ix,iy,0] + zrel*self.normal[ix,iy,0]
    def ind2y(self,ix,iy,zrel):
        return self.G[ix,iy,1] + zrel*self.normal[ix,iy,1]
    def ind2z(self,ix,iy,zrel):
        return self.G[ix,iy,2] + zrel*self.normal[ix,iy,2]

    def ground_coords2ind(self,x,y):
        return np.unravel_index(np.argmin( np.sqrt( (self.G[:,:,0] - x)**2 
            + (self.G[:,:,1] - y)**2  ), axis=None),(self.Nbx,self.Nby))

    def ind2r(self,ix,iy,zrel):
        Px = self.ind2x(ix,iy,zrel)
        Py = self.ind2y(ix,iy,zrel)
        Pz = self.ind2z(ix,iy,zrel)
        return np.sqrt( (self.S[ix,iy,self.master] - Px)**2 
                + (self.S[ix,iy,self.master+1] - Py)**2 
                + (self.S[ix,iy,self.master+2] - Pz)**2 )

    def ind2theta(self,ix,iy,zrel):
        P = np.array([self.ind2x(ix,iy,zrel), self.ind2y(ix,iy,zrel), self.ind2z(ix,iy,zrel)])
        S = np.array([self.S[ix,iy,self.master], self.S[ix,iy,self.master+1], self.S[ix,iy,self.master+2]])
        SP = P-S 
        Nsp = np.linalg.norm(SP)
        Np  = np.linalg.norm(P)
        return np.pi-np.arccos(np.dot(P,SP)/Np/Nsp)

    def ground_ind2radar_ind(self,ix,iy,zrel):
        r = self.ind2r(ix,iy,zrel)
        return np.argmin(abs( r - self.rg[ix,:,self.master] ))
    

    def radar_ind2ground_ind(self,ix,ir,zrel):
        if ir<self.Nby-1:
            uy = -np.array([ self.G[ix,ir,0] - self.G[ix,ir+1,0], 
                            self.G[ix,ir,1] - self.G[ix,ir+1,1], 
                            self.G[ix,ir,2] - self.G[ix,ir+1,2]])
        else:
            uy = np.array([ self.G[ix,ir,0] - self.G[ix,ir-1,0], 
                            self.G[ix,ir,1] - self.G[ix,ir-1,1], 
                            self.G[ix,ir,2] - self.G[ix,ir-1,2]])
        uy = uy/np.linalg.norm(uy)
        dy = zrel/np.tan(self.theta[ix,ir,self.master]) 
        P  = np.array([ self.G[ix,ir,0] + dy*uy[0],
                        self.G[ix,ir,1] + dy*uy[1],
                        self.G[ix,ir,2] + dy*uy[2]])
        idx = self.ground_coords2ind(P[0],P[1])
        return idx[1]
    
    def ground_ind2ray_ind(self,ix,iy,zrel,zq):
        if iy<self.Nby-1:
            uy = -np.array([ self.G[ix,iy,0] - self.G[ix,iy+1,0], 
                            self.G[ix,iy,1] - self.G[ix,iy+1,1], 
                            self.G[ix,iy,2] - self.G[ix,iy+1,2]])
        else:
            uy = np.array([ self.G[ix,iy,0] - self.G[ix,iy-1,0], 
                            self.G[ix,iy,1] - self.G[ix,iy-1,1], 
                            self.G[ix,iy,2] - self.G[ix,iy-1,2]])
        uy = uy/np.linalg.norm(uy)
        theta = self.ind2theta(ix,iy,zrel)
        dy = (zrel-zq)*np.tan(theta) if zrel-zq > 0 else 0
        P  = np.array([ self.G[ix,iy,0] + dy*uy[0],
                        self.G[ix,iy,1] + dy*uy[1],
                        self.G[ix,iy,2] + dy*uy[2]])
        idx = self.ground_coords2ind(P[0],P[1])
        return idx[1]
    

    def ind2t(self,ix,iy,zrel):
        P = np.array([self.ind2x(ix,iy,zrel), self.ind2y(ix,iy,zrel), self.ind2z(ix,iy,zrel)])
        S = np.array([self.S[ix,iy,self.master], self.S[ix,iy,self.master+1], self.S[ix,iy,self.master+2]])
        SP = P-S 
        return SP/np.linalg.norm(SP)
    
    def ind2v(self,ix,iy):
        if ix<self.Nbx-1:
            v = np.array([ self.G[ix+1,iy,0] - self.G[ix,iy,0], 
                            self.G[ix+1,iy,1] - self.G[ix,iy,1], 
                            self.G[ix+1,iy,2] - self.G[ix,iy,2]])
        else:
            v = -np.array([ self.G[ix-1,iy,0] - self.G[ix,iy,0], 
                            self.G[ix-1,iy,1] - self.G[ix,iy,1], 
                            self.G[ix-1,iy,2] - self.G[ix,iy,2]])
        v = v/np.linalg.norm(v)
        return v
    
    def ind2n(self,ix,iy,zrel):
        t = self.ind2t(ix,iy,zrel)
        v = self.ind2v(ix,iy)
        n = np.array([ v[1]*t[2] - v[2]*t[1],
                      -v[0]*t[2] + v[2]*t[0],
                      v[0]*t[1] - v[1]*t[0] ])
        return n
    
class Point:
    def __init__(self, geo=None, pground=None, pradar=None, longlat=None):
        self.geo = geo
        self.ix = None
        self.iy = None
        self.iz = None
        self.ir = None
        if pground:
            self.ground_point(pground)
        if pradar:
            self.radar_point(pradar)
        self.longlat = longlat

    def set_geo(self,geo):
        self.geo = geo

    def ground2radar(self):
        if not self.iy:
            sys.exit('No ground range coordinate')
        self.ir = self.geo.ground_ind2radar_ind(self.ix,self.iy,self.zrel)
 
    def radar2ground(self):
        if not self.ir:
            sys.exit('No slant range coordinate')
        self.iy = self.geo.radar_ind2ground_ind(self.ix,self.ir,self.zrel)

    def ground_point(self,pground):
        if not self.geo:
            sys.exit('This point has no geometry')
        self.ix = pground[0]
        self.iy = pground[1]
        self.iz = np.argmin(abs(self.geo.z_ax - pground[2]))
        self.zrel = self.geo.z_ax[self.iz]
        self.ir = self.geo.ground_ind2radar_ind(self.ix,self.iy,self.zrel)
        self.x = self.geo.ind2x(self.ix,self.iy,self.zrel)
        self.y = self.geo.ind2y(self.ix,self.iy,self.zrel)
        self.z = self.geo.ind2z(self.ix,self.iy,self.zrel)
        self.r = self.geo.ind2r(self.ix,self.iy,self.zrel)
        self.theta = self.geo.ind2theta(self.ix,self.iy,self.zrel)

    def radar_point(self,pradar):
        if not self.geo:
            sys.exit('This point has no geometry')
        self.ix = pradar[0]
        self.ir = pradar[1]
        self.iz = np.argmin(abs(self.geo.z_ax - pradar[2]))
        self.zrel = self.geo.z_ax[self.iz]
        self.iy = self.geo.radar_ind2ground_ind(self.ix,self.ir,self.zrel)
        self.x = self.geo.ind2x(self.ix,self.iy,self.zrel)
        self.y = self.geo.ind2y(self.ix,self.iy,self.zrel)
        self.z = self.geo.ind2z(self.ix,self.iy,self.zrel)
        self.r = self.geo.ind2r(self.ix,self.iy,self.zrel)
        self.theta = self.geo.ind2theta(self.ix,self.iy,self.zrel)

    def is_occluded(self,point=None,coords=None):
        if (not point and not coords) or (point and coords):
            sys.exit('The method needs either a point or coordinates in arguments')
        if point:
            ix = point.ix
            iy = point.iy
            x = point.x
            y = point.y
            z = point.z
        if coords:
            ix = coords[0]
            iy = coords[1]
            zrel = coords[2]
            x = self.geo.ind2x(ix,iy,zrel)
            y = self.geo.ind2y(ix,iy,zrel)
            z = self.geo.ind2z(ix,iy,zrel)
        S = np.array([self.geo.S[ix,iy,self.geo.master], 
                      self.geo.S[ix,iy,self.geo.master+1], 
                      self.geo.S[ix,iy,self.geo.master+2]])
        P = np.array([x,y,z])
        SP = P-S
        n = self.geo.ind2n(ix,iy,zrel)
        return np.dot(SP,n)<0
        
    def print(self):
            print(f'ground (indice, value) : x = {(self.ix, self.x)} , y = {(self.iy, self.y)}, z = {(self.iz,self.z)}, height = {self.zrel}')
            print(f'radar (indice, value): x = {(self.ix, self.x)} , r = {(self.ir, self.r)}, theta = {self.theta}') 


