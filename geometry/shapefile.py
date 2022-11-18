#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:57:02 2019

@author: crambour
"""

import fiona
import rasterio
import rasterio.features
import rasterio.plot
import pyproj
import geopandas as gpd
from matplotlib import pyplot
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from functools import partial
import argparse
from affine import Affine
import numpy as np
from scipy.io import loadmat
from matplotlib.path import Path
from skimage.draw import polygon_perimeter

bdhydro_codes = {'Acqueduc': 1,
 'Canal': 2,
 'Conduit buse': 3,
 'Conduit forcé': 4,
 'Delta': 5,
 'Ecoulement canalisé': 6,
 'Ecoulement endoréique': 7,
 'Ecoulement naturel': 8,
 'Ecoulement phréatique': 9,
 'Estuaire': 10,
 'Glacier, névé': 11,
 'Inconnue': 12,
 'Lac': 13,
 'Lagune': 14,
 'Mangrove': 15,
 'Marais': 16,
 'Mare': 17,
 'Plan d\'eau de gravière': 18,
 'Plan d\'eau de mine': 19,
 'Réservoir-bassin': 20,
 'Réservoir-bassin d\'orage': 21,
 'Réservoir-bassin piscicole': 22,
 'Retenue': 23,
 'Retenue-barrage': 24,
 'Retenue-bassin portuaire': 25,
 'Retenue-digue': 26}

def get_shapes(clipped_shapes, mode=None):
    """
        Extract a list of (polygon, value) tuples using specific dataset rules
        from a GeoDataFrame.

        :param clipped_shapes: GeoDataFrame to be processed
        :param mode: dataset name ('UA2012', 'cadastre')
        :return: a list of (polygon, value) tuples
    """
    if mode == 'UA2012':
        shapes = [(geometry, UA2012_codes[item]) for geometry, item in zip(clipped_shapes.geometry, clipped_shapes['CODE2012'])]
    elif mode == 'bdhydro':
        shapes = [(geometry, bdhydro_codes[item]) for geometry, item in zip(clipped_shapes.geometry, clipped_shapes['NATURE'])]
    elif mode == 'bdtopo':
        shapes = [(geometry, item) for geometry, item in zip(clipped_shapes.geometry, clipped_shapes['HAUTEUR'])]
    elif mode == 'cadastre':
        shapes = [(geometry, 255) for geometry in clipped_shapes.geometry]
    else:
        raise ValueError('Not implemented: {}.'.format(mode))
    return shapes

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
    
    if shapefile.crs == 'epsg:4326':
        ymin_s, xmin_s, ymax_s, xmax_s = shapefile.total_bounds
    else:
        xmin_s, ymin_s, xmax_s, ymax_s = shapefile.total_bounds
    bounds_shp = Polygon( [(xmin_s,ymin_s), (xmin_s, ymax_s), (xmax_s, ymax_s), (xmax_s, ymin_s)] )
    bounds_raster = Polygon( [(xmin,ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)] )
    
    if shapefile.crs == 'epsg:4326':
        for k in range(len(shapefile)):
            coords = []
            for long, lat, z in shapefile['geometry'][k].exterior.coords:
                coords.append((lat, long, z))
            shapefile['geometry'][k] = Polygon(coords)
            
    if bounds_shp.intersects(bounds_raster):
        # Compute the intersection of the bounds with the shapes
        shapefile['geometry'] = shapefile['geometry'].intersection(bounds_raster)
        # Filter empty shapes
        shapes_cropped = shapefile[shapefile.geometry.area>0]
        return shapes_cropped
    else:
        return None

def reproject(shapefile, crs):
    return shapefile.to_crs(crs) if shapefile.crs != crs else shapefile

def burn_shapes(shapes, destination, meta):
    with rasterio.open(destination, 'w+', **meta) as out:
        out_arr = out.read(1)
        burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr)
        #rasterio.plot.show(burned)
        out.write_band(1, burned)
        
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
    l=0
    for k in range(len(shapes)):
        inds = []
        multi = shapes['geometry'][k]
        if 'multipolygon' in str(type(multi)):
            polys = list(multi)
        else:
            polys = [multi]
        for poly in polys:
            coords = poly.exterior.coords
            for long, lat, z in coords:
               dist = np.sqrt((coord_grid[:,:,0] - lat)**2
                        + (coord_grid[:,:,1] - long)**2)
               ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
               inds.append((ind[1], ind[0], z))
            shapes_ind.append((Polygon(inds), shapes['HAUTEUR'][k]))
    return shapes_ind

def burn2grid(coord_grid,shapes):
    nl, nc, nb = coord_grid.shape
    x, y = np.meshgrid(np.arange(nl), np.arange(nc))
    x, y = x.flatten(), y.flatten()
    grid = np.vstack((x,y)).T
    raster = np.zeros((nl,nc))
    borders = np.zeros((nl,nc))
    shapes.sort(key=lambda tup: tup[1])
    for k in range(len(shapes)):
        z = shapes[k][1]
        x, y = shapes[k][0].exterior.coords.xy
        vertices = np.array((y,x)).T
        path = Path(vertices)
        tmp = path.contains_points(grid).reshape((nc,nl)).T
        raster += tmp*z
        rr, cc = polygon_perimeter(y, x, shape=borders.shape, clip=True)
        borders[rr,cc] = z
    return raster, borders
        

def main():
    def parse_bounds(s):
        try:
            xmin, ymin, xmax, zmax, crs = s.split(',')
            return tuple(map(float,(xmin, ymin, xmax, zmax))), crs.strip()
        except:
            raise argparse.ArgumentTypeError("bounds must be xmin, ymin, xmax,\
                                             ymax, crs")
    parser = argparse.ArgumentParser(description='Burn shapefile to raster')
    parser.add_argument('--raster', type=str, nargs='?', metavar='N',
                        help='Path of the raster file. If\
                        empty and no bounds are provided,\
                        the metadata are obtained from the shapefile.')
    parser.add_argument('--shapefile',type=str, default=1, metavar='N',
                        help='Path of the shapefile.')
    parser.add_argument('--coord_grid',type=str, default=1,
                        help='[OPTIONAL] Path of the coordinates grid on which\
                        the shapefile will be burned.')
    parser.add_argument('--bounds',type=parse_bounds, nargs=1,
                        help='[OPTIONAL] Bounds of the area.')
    parser.add_argument('--size',type=tuple, nargs=1,
                        help='[OPTIONAL] Size of the output file (width x height).')
    parser.add_argument('--out',type=str, default=1, metavar='N',
                        help='Output raster file')
    args = parser.parse_args()
    with fiona.open(args.shapefile,'r') as shp:
        if args.raster:
            with rasterio.open(args.raster) as raster:
                crs = raster.crs
                bounds = raster.bounds
                meta = raster.meta.copy()
        elif args.coord_grid:
            coord_grid = np.array(loadmat(args.coord_grid)['longlat_blue'])
            bounds = np.amin(coord_grid[:,:,1]), np.amin(coord_grid[:,:,0]),\
                    np.amax(coord_grid[:,:,1]), np.amax(coord_grid[:,:,0])
            crs = 'epsg:4326'
            meta = {'crs': rasterio.crs.CRS.from_epsg(4326),'height':coord_grid.shape[0], 'width':coord_grid.shape[1]}
        elif args.bounds:
            bounds, crs = args.bounds[0]
           # crs = 'epsg:4326'
            meta = {'crs': rasterio.crs.CRS.from_epsg(4326),'height':512, 'width':512, 'transform': Affine(8.983152841195215e-05, 0.0, 48.689854816226784,\
       0.0, -8.983152841195215e-05, 31.53525543741338)}
        else:
            bounds = shp.bounds
            crs = shp.crs
            meta = shp.meta.copy()
        destination = args.out
        bbox = project_bbox(crs, shp.crs, bounds)
        
        s = (s[1] for s in shp.items(bbox=bbox))
        shp = gpd.GeoDataFrame.from_features(s, crs=shp.crs)
        if shp.crs != crs:
            shp = reproject(shp, crs)
        clipped_shapes = crop_shapefile_to_raster(shp, bounds)
        if args.coord_grid:
            clipped_shapes = coord2ind(coord_grid,clipped_shapes)
            out, borders = burn2grid(coord_grid,clipped_shapes)
            np.save('shapes.npy',clipped_shapes)
            np.save('bdtopo_raster.npy',out)
            np.save('bdtopo_borders.npy',borders)
        else:
            shapes=[]
            if clipped_shapes is not None:
                shapes += get_shapes(clipped_shapes, mode='bdtopo')
            meta.update(count=1, driver='GTiff', compress='lzw', dtype=rasterio.float32)
            burn_shapes(shapes, destination, meta)
        
if __name__ == "__main__":
    main()