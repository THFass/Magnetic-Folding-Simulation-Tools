"""----------------------------------------------------------------------------------------------------
Component data structure
Offers classes and methods to handle components of a 3D kinematic Chain
Version 0.1
    -alpha
----------------------------------------------------------------------------------------------------"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Magnet_data_struct_V2 as mgds
import Shape_data_struct_V2 as shds
import Coord_data_struct as cods


class Component:
    def __init__(self,
                 shape,
                 hinge_p,
                 hinge_n,
                 magnet,
                 dtype=np.float32):
        self.shape = shape
        self.hinge_p = hinge_p
        self.hinge_n = hinge_n
        self.trans_abs = hinge_p.get_abs
        #self.magnetize(charge_vec=charge_vec,
        #               resolution=resolution,
        #               dtype=dtype)
        self.magnet=magnet
        self.force=np.array((0,0,0))#force acting on the center of component
        self.torque=np.array((0,0,0))#torque acting around center of component
        self.angel=np.array((0,0,0))#angular accelation of component
        self.dtype=dtype

    def make_next(self,
               comp_type='A',
               phi_0=0,
               magnet=3,
               charge_vec=np.array((1,1,1)),
               dtype=np.float32):
        #generates another component of same shape after current component and returns it
        if comp_type in ('A','a',1):
            comp_type=CubeA
        elif comp_type in ('B','b',2):
            comp_type=CubeB
        elif comp_type in ('C','c',3):
            comp_type=CubeC
        elif comp_type in ('D','d',4):
            comp_type=CubeD
        if type(magnet)==int:
            magnet=self.magnet.copy(resolution=magnet,
                                    charge_vec=charge_vec)
        elif isinstance(magnet,mgds.Magnet):
            print('todo')
        else:
            print('wrong magnet input')

        component=comp_type(dimension=self.shape,
                            phi_0=phi_0,
                            magnet=magnet,
                            reference=self.hinge_n,
                            dtype=dtype)
        component.magnet.ref=component.hinge_p
        return component

    def update_pos(self):
        #updates internal transformation matrice giving absolute coordinates
        self.trans_abs=self.hinge_p.get_abs()

    def update_from_last(self):
        #updates internal transformation matrice giving absolute coordinates
        #todo make more efficent by only updating from last component
        #todo update positions of charges etc
        self.trans_abs=self.hinge_p.get_abs()

    def rotate(self,phi):
        #rotates components degree of freedom to given value. update_pos needs to be called to update absolute position
        self.hinge_p.rotate_to(phi)

    @property
    def get_angle(self):
        return self.hinge_p.angle

    def set_angle(self,angle):
        self.rotate(angle)

    # readout functions---------------------------------------------------------#
    @property
    def get_prev_hinge(self):
        #returns sbaolute position of hinge to prievious component
        return self.hinge_p.abs(-self.hinge_p.trans)

    @property
    def get_hinge_p(self):
        #returns sbaolute position of hinge to prievious component
        return self.hinge_p.abs(-self.hinge_p.trans)

    @property
    def get_next_hinge(self):
        #returns sbaolute position of hinge to next component
        return self.hinge_n.abs((0,0,0))

    @property
    def get_hinge_p_axis(self):
        #returns the absolute axis of hinge_p
        return self.trans_abs.rotate(self.hinge_p.axis)

    @property
    def get_charge_mag(self):
        return self.magnet.charge_mag

    @property
    def get_charge_pos(self):
        return self.magnet.charge_pos

    @property
    def get_charge_pos_abs(self):
        #returns absolute charge position
        return self.magnet.get_charge_pos_abs

    @property
    def get_facets_abs(self):
        #returns the absolute coordinates of the factes
        return self.shape.get_facets_abs(self.hinge_p)

    @property
    def get_center(self):
        #returns cnter coordinates in abolute reference frame
        return self.hinge_p.abs((0,0,0))

    @property
    def get_mag_abs(self):
        #returns magnetisation vector in absolute reference frame
        return self.trans_abs.rotate(self.magnet.charge_vec)

    def get_torque_x(self,point):
        #calculates torque around a point
        #todo: doesnt work as force is updated only in global array. global array needs to be linked to local array
        return np.nansum(np.cross(self.get_charge_pos_abs-point,self.force),1)

    def get_torque_center(self,point):
        #calculates torque around a point
        #todo: doesnt work as force is updated only in global array. global array needs to be linked to local array
        return np.nansum(np.cross(self.get_charge_pos_abs-self.get_center,self.force),1)

    def plt_facets(self,
             ax,
             col,
             alpha=1):
        if col1!=col2:
            colors=[]
            for normal in self.shape.get_normals:
                factor = np.dot(self.magnet.charge_vec, normal)
                if factor < -0.0001:
                    colors.append(col[0])
                elif factor > 0.0001:
                    colors.append(col[1])
                else:
                    colors.append(col[2])
        else:
            colors=col1
        self.plot_collection=self.shape.plot(
             ref = self.ref,#refference system in which the shape is
             col = colors,
             alpha = alpha,
             ax = ax)


    # Plotting functions-------------------------------------------------------------#
    def plot(self,
             ax,
             plt_hinges=True,
             plt_mag=False,
             plt_force=False,
             plt_charges=False,
             plt_color=('b','r','grey'),
             alpha=0.5):
        facets = self.get_facets_abs
        self.plot_casing(plt_color,alpha)
        if plt_hinges:
            self.plot_hinges(ax)
        if plt_charges:
            self.plot_charges(ax)
        if plt_mag:
            self.plot_mag(ax)
        if plt_force:
            factor = facets[0][0] - facets[0][1]
            self.plot_force(ax, factor)

    def plot_hinges(self, ax):
        #plots the hinges of the component in given figure
        target = 0.5*self.shape.dimension*self.get_hinge_p_axis
        #print(target)
        base = self.get_prev_hinge - target  #
        #print(base)
        ax.quiver(base[0], base[1], base[2],
                  target[0], target[1], target[2],
                  length=2, color='black', linewidths=10, arrow_length_ratio=0)

    def plot_mag(self,
                 ax):
        target=self.get_mag_abs
        orig=self.get_center+self.get_mag_abs/2
        ax.quiver(orig[0],orig[1],orig[2],target[0],target[1],target[2] ,color='r')

    def plot_force(self, ax, factor):
        orig = self.get_center
        target = self.force
        ax.quiver(orig[0], orig[1], orig[2], target[0], target[1], target[2],
                  length=np.linalg.norm(self.force) * factor, color='b')

    def plot_charges(self, ax):
        self.magnet.plot_carges(ax)

    def magnetise(self,
                  charge_vec=np.array((1,1,1)),
                  br_max=np.nan,
                  resolution=np.nan):
        #remagnetises the component in current orientation
        charge_vec=np.array(self.hinge_p.get_abs_rot.inv().apply(charge_vec))
        if np.isnan(br_max):
            br_max=np.linalg.norm(charge_vec)
        self.magnet= self.magnet.magnetize(
            resolution=resolution,
            charge_vec=charge_vec,
            br_rmax=br_max)


#todo: Ready made cube parts as subclasses
class CubeA(Component):
    def __init__(self,
                 dimension=1,
                 phi_0=0, #inotal turning angle
                 charge_vec=np.array((1,0,0)), #vetor of magnetisation. Norm=magnitude
                 magnet=3, #resolutio**2 is number of point charges per surface
                 reference=cods.Origin(), #reference frame for positioning component
                 dtype=np.float32): #data type to save arrays
        """if isinstance(dimension,shds.Cube):
            shape=dimension
        elif type(dimension)==type(1) or type(dimension)==type(1.1):
            shape=shds.Cube(dimension)
        elif type(dimension) in (np.ndarray,list,tuple):
            shape=shds.Cube(np.max(dimension))
        else:
            print('wrong dimension input to generate Cube')"""
        shape = shds.Cube(np.max(dimension))
        print(shape)
        #initioating hinge connecting to privious component
        hinge_p=cods.RefFrame(translation=-shape.dimension*np.array((-0.5,-0.5,0)), #relative position of hinge
                                         axis=-np.array((0,0,1)),#turning axis of hinge
                                         angle=phi_0,#inital turning angle of hinge
                                         reference=reference,#reference to prievious components hinge
                                         dtype=dtype)
        #initioating hinge connecting to next component
        hinge_n=cods.RefFrame.from_abs_translation(translation=shape.dimension*np.array((0.5,0.5,0)),#relative position of hinge
                                         axis=np.array((1,0,0)),#axis next component gets rotatet by
                                         angle=np.pi,#amount of rotation
                                         reference=hinge_p,#own hinge to privous component as reference frame
                                         dtype=dtype)
        if type(magnet)==int:
            magnet= mgds.from_shape(shape=shape,
                                    charge_vec=charge_vec,
                                    resolution=magnet,
                                    reference=hinge_p,
                                    dtype=dtype)
        #creating component
        super().__init__(shape=shape,
                         hinge_p=hinge_p,
                         hinge_n=hinge_n,
                         magnet=magnet)
        self.type='A' #setting component type

class CubeB(Component):
    def __init__(self,
                 dimension=1,
                 phi_0=0, #inotal turning angle
                 charge_vec=np.array((1,1,1)), #vetor of magnetisation. Norm=magnitude
                 magnet=3, #resolutio**2 is number of point charges per surface
                 reference=cods.Origin(), #reference frame for positioning component
                 dtype=np.float32): #data type to save arrays
        if isinstance(dimension,shds.Cube):
            shape=dimension
        elif type(dimension) in (int,float):
            shape=shds.Cube(dimension)
        else:
            print('wrong dimension input to generate Cube')
        #initioating hinge connecting to privious component
        hinge_p=cods.RefFrame(translation=-shape.dimension*np.array((-0.5,-0.5,0)), #relative position of hinge
                                         axis=-np.array((0,0,1)),#turning axis of hinge
                                         angle=phi_0,#inital turning angle of hinge
                                         reference=reference,#reference to prievious components hinge
                                         dtype=dtype)
        #initioating hinge connecting to next component
        hinge_n=cods.RefFrame.from_abs_translation(translation=shape.dimension*np.array((0.5,0,-0.5)),#relative position of hinge
                                         axis=np.array((1,0,0)),#axis next component gets rotatet by
                                         angle=np.pi/2,#amount of rotation
                                         reference=hinge_p,#own hinge to privous component as reference frame
                                         dtype=dtype)
        if type(magnet)==int:
            magnet= mgds.from_shape(shape=shape,
                                    charge_vec=charge_vec,
                                    resolution=magnet,
                                    reference=hinge_p,
                                    dtype=dtype)
        #creating component
        super().__init__(shape=shape,
                         hinge_p=hinge_p,
                         hinge_n=hinge_n,
                         magnet=magnet)
        self.type='B' #setting component type

class CubeC(Component):
    def __init__(self,
                 dimension=1,
                 phi_0=0, #inotal turning angle
                 charge_vec=np.array((1,1,1)), #vetor of magnetisation. Norm=magnitude
                 magnet=3, #resolutio**2 is number of point charges per surface
                 reference=cods.Origin(), #reference frame for positioning component
                 dtype=np.float32): #data type to save arrays
        if isinstance(dimension,shds.Cube):
            shape=dimension
        elif type(dimension)==type(1) or type(dimension)==type(1.1):
            shape=shds.Cube(dimension)
        else:
            print('wrong dimension input to generate Cube')
        #initioating hinge connecting to privious component
        hinge_p=cods.RefFrame(translation=-shape.dimension*np.array((-0.5,-0.5,0)), #relative position of hinge
                                         axis=-np.array((0,0,1)),#turning axis of hinge
                                         angle=phi_0,#inital turning angle of hinge
                                         reference=reference,#reference to prievious components hinge
                                         dtype=dtype)
        #initioating hinge connecting to next component
        hinge_n=cods.RefFrame.from_abs_translation(translation=shape.dimension*np.array((0.5,-0.5,0)),#relative position of hinge
                                         axis=np.array((1,0,0)),#axis next component gets rotatet by
                                         angle=0,#amount of rotation
                                         reference=hinge_p,#own hinge to privous component as reference frame
                                         dtype=dtype)
        if type(magnet)==int:
            magnet= mgds.from_shape(shape=shape,
                                    charge_vec=charge_vec,
                                    resolution=magnet,
                                    reference=hinge_p,
                                    dtype=dtype)
        #creating component
        super().__init__(shape=shape,
                         hinge_p=hinge_p,
                         hinge_n=hinge_n,
                         magnet=magnet)
        self.type='C' #setting component type

class CubeD(Component):
    def __init__(self,
                 dimension=1,
                 phi_0=0, #inotal turning angle
                 charge_vec=np.array((1,0,0)), #vetor of magnetisation. Norm=magnitude
                 magnet=3, #resolutio**2 is number of point charges per surface
                 reference=cods.Origin(), #reference frame for positioning component
                 dtype=np.float32): #data type to save arrays
        if isinstance(dimension,shds.Cube):
            shape=dimension
        elif type(dimension)==type(1) or type(dimension)==type(1.1):
            shape=shds.Cube(dimension)
        else:
            print('wrong dimension input to generate Cube')
        #initioating hinge connecting to privious component
        hinge_p=cods.RefFrame(translation=-shape.dimension*np.array((-0.5,-0.5,0)), #relative position of hinge
                                         axis=-np.array((0,0,1)),#turning axis of hinge
                                         angle=phi_0,#inital turning angle of hinge
                                         reference=reference,#reference to prievious components hinge
                                         dtype=dtype)
        #initioating hinge connecting to next component
        hinge_n=cods.RefFrame.from_abs_translation(translation=shape.dimension*np.array((0.5,0,0.5)),#relative position of hinge
                                         axis=np.array((1,0,0)),#axis next component gets rotatet by
                                         angle=-np.pi/2,#amount of rotation
                                         reference=hinge_p,#own hinge to privous component as reference frame
                                         dtype=dtype)
        if type(magnet)==int:
            magnet= mgds.from_shape(shape=shape,
                                    charge_vec=charge_vec,
                                    resolution=magnet,
                                    reference=hinge_p,
                                    dtype=dtype)
        #creating component
        super().__init__(shape=shape,
                         hinge_p=hinge_p,
                         hinge_n=hinge_n,
                         magnet=magnet)
        self.type='D' #setting component type
