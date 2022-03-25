"""Maget data structure
Contains Funtions and classes to define magnets of various shape.
V2- uses shapes as basis to define points. Magnet itslef does not have to be shape specific anymore"""

import Shape_data_struct_V2 as shds
import Coord_data_struct as cods
import Function_lib as flib
import numpy as np

class Magnet:
    def __init__(self,
                 shape=shds.Cube(),
                 mag_vec=np.array((1,1,1)),
                 br_max=1,
                 ref=cods.Origin,
                 dtype=np.float32
                 ):
        self.shape = shape
        self.mag_vec = mag_vec
        self.br_max =br_max
        self.resolution = resolution
        self.ref = ref
        self.dtype =dtype
        self.ch_mag = self.calc_mag

    @property
    def calc_mag(self):#calculates magnetic magnitude for all point chages in shpe object depending on mag_vec and Br_max
        return np.nansum(self.mag_vec*self.shape.get_normals)*self.br_max*self.shape.get_area / (np.pi*(4e-7))

    def magnetize(self,
                  br_max=np.nan,
                  mag_vec=np.nan,
                  resolution=np.nan):
        #calculates new magnetic charge magnitudes according to new magnetization vector
        if not np.isnan(br_max):
            self.br_max = br_max
        if not np.isnan(mag_vec):
            self.mag_vec = mag_vec
        if not np.isnan(resolution):
            self.resolution = resolution
        self.ch_mag = self.calc_mag

    def copy(self,
                 shape=np.nan,
                 mag_vec=np.nan,
                 br_max=np.nan,
                 ref=np.nan,
                 dtype=np.float32
                 ):
        if not np.isnan(mag_vec):
            shape = self.shape
        if not np.isnan(mag_vec):
            mag_vec = self.mag_vec
        if not np.isnan(mag_vec):
            br_max = self.br_max
        if not np.isnan(mag_vec):
            resolution = self.resolution
        if not np.isnan(mag_vec):
            ref = self.ref
        return Magnet(shape=shape,
                      mag_vec=mag_vec,
                      br_max=br_max,
                      ref=ref,
                      dtype=dtype)

    @property
    def get_ch_pos(self):
        return self.shape.get_points(self.ref)

    @property
    def get_ch_mag(self):
        return self.ch_mag

    def plot(self,
             ax,
             col1='b',
             col2='r',
             col0='grey',
             alpha=1):
        if col1!=col2:
            colors=[]
            for normal in self.shape.get_normals:
                factor = np.dot(self.magnet.charge_vec, normal)
                if factor < -0.0001:
                    colors.append(col1)
                elif factor > 0.0001:
                    colors.append(col2)
                else:
                    colors.append(col0)
        else:
            colors=col1
        self.plot_collection=self.shape.plot(
             ref = self.ref,#refference system in which the shape is
             col = colors,
             alpha = alpha,
             ax = ax)