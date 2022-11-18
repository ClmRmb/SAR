#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:05:37 2020

@author: rambo
"""

import argparse
from functools import partial

import fiona
import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import rasterio.features
import rasterio.plot
from descartes.patch import PolygonPatch
from matplotlib import pyplot as plt
from matplotlib.path import Path
from scipy.io import loadmat
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union
from skimage.draw import polygon_perimeter, polygon2mask, polygon

from tomogeo import Geo


def crop_shapefile_to_raster(shapefile, bounds):
    """
        Intersects of a GeoDataFrame with a raster

        This creates the intersection between a Geopandas shapefile and a
        rasterio raster. It returns a new shapefile containing only the polygons
        resulting from this intersection.

        :param shapefile: reference GeoDataFrame
        :param raster: RasterIO raster object
        :return: a GeoDataFrame containing the intersected polygons
    """
       
    xmin, ymin, xmax, ymax = bounds
    ymin_s, xmin_s, ymax_s, xmax_s = shapefile.total_bounds

    bounds_shp = Polygon( [(ymin_s,xmin_s), (ymin_s, xmax_s), (ymax_s, xmax_s), (ymax_s, xmin_s)] )
    bounds_raster = Polygon( [(ymin,xmin), (ymin, xmax), (ymax, xmax), (ymax, xmin)] )
    
    if not bounds_shp.intersects(bounds_raster):
        return None

    count=0
    shapes = []
    for k in range(len(shapefile)):
        coords = []
        polys = shapefile['geometry'][k]
        polys = polys.intersection(bounds_raster)
        if 'multipolygon' in str(type(polys)): 
            for poly in polys.geoms:
                count+=1
                if poly.area > 0:
                    shapes.append((poly,shapefile['hauteur'][k]))
        else:
            if polys.area > 0:
                shapes.append((polys,shapefile['hauteur'][k]))
    return shapes

    # if bounds_shp.intersects(bounds_raster):
    #     # Compute the intersection of the bounds with the shapes
    #     shapefile['geometry'] = shapefile['geometry'].intersection(bounds_raster)
    #     # Filter empty shapes
    #     shapes_cropped = shapefile[shapefile.geometry.area>0]
    #     return shapes_cropped
    # else:
    #     return None

def multi2list(multi):
    shapes = []
    if 'multipolygon' in str(type(multi)):
        polys = list(multi)
    else:
        polys = [multi]
    for poly in polys:
        coords = poly.exterior.coords
    shapes.append((Polygon(coords)))
    return shapes

def reproject(shapefile, crs):
    return shapefile.to_crs(crs) if shapefile.crs != crs else shapefile

def project_bbox(crs_in, crs_out, bounds):
    """
        Project a bounding box from a CRS to another

        :param crs_in: an input CoordinateReferenceSystem
        :param crs_out: the target CoordinateReferenceSystem
        :param bounds: a tuple of bounds (xmin, ymin, xmax, ymax)
        :param return: the tuple of projected bounds
    """
    xmin, ymin, xmax, ymax = bounds
    bbox = [(xmin,ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    #pyproj.Proj(crs_out)
    transform = partial(pyproj.transform, pyproj.Proj(crs_in), pyproj.Proj(crs_out))
    new_coords = []
    for x1, y1 in bbox:
        x2, y2 = transform(x1, y1)
        new_coords.append((x2, y2))
    return Polygon(new_coords).bounds

def coord2ind(coord_grid,shapes):
    shapes_ind = []
    for poly, height in shapes:
        inds = []
        # multi = shapes['geometry'][k]
        # if 'multipolygon' in str(type(multi)):
        #     polys = list(multi)
        # else:
        #     polys = [multi]
        # for poly in polys:
        coords = poly.exterior.coords
        for long, lat, z in coords:
            dist = np.sqrt((coord_grid[:,:,0] - long)**2
                    + (coord_grid[:,:,1] - lat)**2)
            ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            inds.append((ind[1], ind[0]))
        shapes_ind.append((Polygon(inds), height))
    return shapes_ind



def burn2grid(geo,shapes,Zq,slice_mode=False):
    nl, nc = (geo.Nbx, geo.Nby)
    x, y = np.meshgrid(np.arange(nl), np.arange(nc))
    x, y = x.flatten(), y.flatten()
    grid = np.vstack((x,y)).T
    raster = np.zeros((nl,nc,len(Zq)))
    shapes.sort(key=lambda tup: tup[1])
    #occlusions = get_occlusions(geo,shapes,Zq)

    if slice_mode:
        for l in range(len(Zq)):
            zq = Zq[l]
            #occlusion = occlusions[zq]
            for k in range(len(shapes)):
                
                #pol = occlusion.difference(occlusion.difference(shapes[k][0].buffer(0)))
                #pol = occlusion
                pol = shapes[k][0].buffer(0)
                if pol.is_empty:
                    continue
                z = shapes[k][1]
                if pol.geom_type == 'MultiPolygon':
                    y = []
                    x = []
                    for ip in pol:
                        y = y+(list(ip.exterior.coords.xy[0]))
                        x = x+(list(ip.exterior.coords.xy[1]))
                else:
                    y, x = pol.exterior.coords.xy
                

                rg = [geo.ground_ind2radar_ind(int(ix),int(iy),zrel) 
                        for ix, iy, zrel in zip(x,y,np.full(len(x),zq))]
                
                if zq >= z - geo.res_z/2 and zq <= z + geo.res_z/2: 
                    vertices = np.array((rg,x)).T
                    path = Path(vertices)
                    tmp = path.contains_points(grid).reshape((nc,nl)).T
                    raster[:,:,l] += tmp*zq    
                elif zq < z - geo.res_z/2:
                    
                    #rr, cc = polygon_perimeter(x, rg, shape=(nl,nc), clip=True)
                    #raster[rr,cc,l] = zq
                    mask = z*polygon2mask((nl,nc), list(zip(x, rg)))
                    raster[:,:,l]+=mask
    else:
        for k in range(len(shapes)):

            pol = shapes[k][0].buffer(0)
            if pol.is_empty:
                continue
            z = shapes[k][1]
            if pol.geom_type == 'MultiPolygon':
                y = []
                x = []
                for ip in pol:
                    y = y+(list(ip.exterior.coords.xy[0]))
                    x = x+(list(ip.exterior.coords.xy[1]))
            else:
                y, x = pol.exterior.coords.xy
            
            rg_ground = [geo.ground_ind2radar_ind(int(ix),int(iy),zrel) 
                    for ix, iy, zrel in zip(x,y,np.full(len(x),0))]
            rg_roof = [geo.ground_ind2radar_ind(int(ix),int(iy),zrel) 
                    for ix, iy, zrel in zip(x,y,np.full(len(x),z))]
            
            pol_ground =  sorted(list(zip(x,rg_ground)),key=lambda tup:tup[0])
            pol_roof = sorted(list(zip(x,rg_roof)),key=lambda tup:tup[0])
            pol_c = [pol_ground[-1],pol_ground[0],pol_roof[0],pol_roof[-1]]
            pol_ground =  list(zip(x,rg_ground))
            pol_roof = list(zip(x,rg_roof))
            mask = polygon2mask((nl,nc), pol_ground).astype('int')+polygon2mask((nl,nc), pol_roof).astype('int')+polygon2mask((nl,nc), pol_c).astype('int')
            raster[:,:,0]+=z*np.minimum(1,mask)

    return raster


def get_occlusions(geo,shapes,Zq):
    shapes_ind = {}
    for l in range(len(Zq)):
        zq = Zq[l]
        tmp = []
        for k in range(6,7):#:range(len(shapes)):
            poly = shapes[k][0]
            z = shapes[k][1]
            poly_occ = occlusion_from_poly(geo,poly,z,zq)
            tmp.append(poly_occ)
        global_occ = unary_union(tmp)
        shapes_ind[zq] = global_occ
    return shapes_ind
        

def occlusion_from_poly(geo,poly,z,zq):
    Nbx, Nby = (geo.Nbx, geo.Nby)
    xx, yy = poly.exterior.coords.xy
    xx = np.array(xx)
    yy = np.array(yy)
    xx_new = np.copy(xx)
    yy_new = np.copy(yy)
    idx_x = xx.argsort()[::-1]
    for idx in idx_x:
        x = xx[idx]
        y = yy[idx]
        x_proj = min(Nby-1,geo.ground_ind2ray_ind(int(y),int(x),z,zq))
        if poly.contains(Point(x_proj,y)):
            continue
        for i in range(len(xx_new)):
            if xx_new[i] == x and yy_new[i] == y:
                idx_new = i
                break
        xx_tmp = np.copy(xx_new)
        xx_tmp[idx_new] = x_proj
        coords = [(x,y) for x,y in zip(xx_tmp,yy_new)]
        poly_new = Polygon(coords)
        if poly_new.contains(Point(x,y)):
            xx_new = np.copy(xx_tmp)
            poly = poly_new
        else:
            idx_insert = np.argmin((xx_new - x_proj)**2)
            xx_new = np.insert(xx_new,idx_new,x_proj)
            yy_new = np.insert(yy_new,idx_new,y)
            coords = [(x,y) for x,y in zip(xx_new,yy_new)]
            poly = Polygon(coords)
            # if poly.is_valid:
            #     xx_new = np.copy(xx_tmp)
            #     yy_new = np.copy(yy_tmp)
            # else:
            #     xx_new = np.insert(xx_new,idx_new-1,x_proj)
            #     yy_new = np.insert(yy_new,idx_new-1,y)
            #     coords = [(x,y) for x,y in zip(xx_tmp,yy_tmp)]
            #     poly = Polygon(coords)
                # if poly.is_valid:
                #     xx_new = np.copy(xx_tmp)
                #     yy_new = np.copy(yy_tmp)
                # else:
                #     xx_new = np.insert(xx_new,idx_new+1,x_proj)
                #     yy_new = np.insert(yy_new,idx_new+1,y)
                #     coords = [(x,y) for x,y in zip(xx_new,yy_new)]
                #     poly = Polygon(coords)
                
    return poly
                
        

def main():
    
    parser = argparse.ArgumentParser(description='Burn GT into rasters correponding\
                                     to given slices')
    parser.add_argument('--shapefile',type=str, default=1, metavar='N',
                        help='Path of the shapefile.')
    parser.add_argument('--longlat',type=str, default=1,
                        help='Path of the coordinates grid on which\
                        the shapefile will be burned.')
    parser.add_argument('--Z',type=float, default=1, nargs='+',
                        help='Height of the slices')
    parser.add_argument('--traj',type=str, default=1,
                        help='Path of the sensors trajectories.')
    parser.add_argument('--out',type=str, default=1, metavar='N',
                        help='Output raster files')
    args = parser.parse_args()
    with fiona.open(args.shapefile,'r') as shp:
        traj = loadmat(args.traj)['traj_blue']
        coord_grid = np.array(loadmat(args.longlat)['longlat_blue'])
        bounds = np.amin(coord_grid[:,:,1]), np.amin(coord_grid[:,:,0]),\
                np.amax(coord_grid[:,:,1]), np.amax(coord_grid[:,:,0])
        crs = 'epsg:4326'
        c=299810000
        fc=9.65e9
        lambda_=c/fc
        geo = Geo(lambda_,np.stack((traj[:,:,0],traj[:,:,1],traj[:,:,2]),axis=2),
          traj[:,:,3:],longlat = coord_grid)
        destination = args.out
        bbox = project_bbox(crs, shp.crs, bounds)
        
        s = (s[1] for s in shp.items(bbox=bbox))
        shp = gpd.GeoDataFrame.from_features(s, crs=shp.crs)
        if shp.crs != crs:
            shp = reproject(shp, crs)
        clipped_shapes = crop_shapefile_to_raster(shp, bounds)
        clipped_shapes = coord2ind(coord_grid,clipped_shapes)
        out = burn2grid(geo,clipped_shapes,args.Z)
        #out_no_occ = remove_occlusions(geo,clipped_shapes,out,args.Z)
        plt.figure()
        plt.imshow(out[:,:,0])
        np.save(destination,out)
        
    
if __name__ == "__main__":
    main()

'''
Minimal example 

traj = 'traj_blue.mat'
longlat = '../geometry/longlat_blue.mat'
shapefile = '../data/bat_idf.gpkg'
Z = [0]
out = 'testlast2'


with fiona.open(shapefile,'r') as shp:
    traj = loadmat(traj)['traj_blue']
    coord_grid = np.array(loadmat(longlat)['longlat_blue'])
    bounds = np.amin(coord_grid[:,:,1]), np.amin(coord_grid[:,:,0]),\
            np.amax(coord_grid[:,:,1]), np.amax(coord_grid[:,:,0])
    crs = 'epsg:4326'
    c=299810000
    fc=9.65e9
    lambda_=c/fc
    geo = Geo(lambda_,np.stack((traj[:,:,0],traj[:,:,1],traj[:,:,2]),axis=2),
        traj[:,:,3:],longlat = coord_grid)
    destination = out
    bbox = project_bbox(crs, shp.crs, bounds)
    
    s = (s[1] for s in shp.items(bbox=bbox))
    shp = gpd.GeoDataFrame.from_features(s, crs=shp.crs)
    if shp.crs != crs:
        shp = reproject(shp, crs)
    clipped_shapes = crop_shapefile_to_raster(shp, bounds)
    clipped_shapes = coord2ind(coord_grid,clipped_shapes)
    out = burn2grid(geo,clipped_shapes,Z,slice_mode=False)
    #out_no_occ = remove_occlusions(geo,clipped_shapes,out,args.Z)
    plt.figure()
    plt.imshow(out[:,:,0])
    np.save(destination,out)
'''