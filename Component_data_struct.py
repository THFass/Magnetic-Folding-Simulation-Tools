"""----------------------------------------------------------------------------------------------------
Component data structure
Offers classes and methods to handle components of a 3D kinematic Chain
Version 0.1
    -alpha
----------------------------------------------------------------------------------------------------"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Magnet_data_struct as mgds
import Shape_data_struct_V2 as shds
import Coord_data_struct as cods
import trimesh as tm

from Function_lib import mm, inch, myz,const


class Component:
    def __init__(self,
                 shape,
                 hinge_p,
                 hinge_n,
                 magnet,
                 mass=None,
                 inertia=None,
                 dtype=np.float32):
        self.shape = shape
        self.I=inertia or shape.get_I
        self.m=mass or shape.get_m
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

    @property
    def get_Q(self):
        return self.hinge_p.get_abs_dcm

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

    def plt_rb_factes(self,ax,alpha):
        facets = self.get_facets_abs
        for (face,normal) in zip(facets,self.shape.normals):
            factor = np.dot(self.magnet.charge_vec,normal)
            if factor < -0.0001:
                color = 'r'
            elif factor > 0.0001:
                color = 'b'
            else:
                color = 'grey'
            ax.add_collection3d(Poly3DCollection([[tuple(face[0]),tuple(face[1]),tuple(face[2]),tuple(face[3])]], alpha=alpha, facecolors=color))


    def plt_grey_factes(self,ax,alpha):
        facets = self.get_facets_abs
        for (face,normal) in zip(facets,self.shape.normals):
            factor = np.dot(self.magnet.charge_vec,normal)
            ax.add_collection3d(Poly3DCollection([[tuple(face[0]),tuple(face[1]),tuple(face[2]),tuple(face[3])]], alpha=alpha, facecolors='grey'))


    # Plotting functions-------------------------------------------------------------#
    def plot(self,
             ax,
             plt_hinges=True,
             plt_mag=False,
             plt_force=False,
             plt_charges=True,
             plt_color='rb',
             alpha=0.5):
        facets = self.get_facets_abs
        if plt_color=='rb':
            self.plt_rb_factes(ax,alpha)
        else:
            self.plt_grey_factes(ax,alpha)
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
                  Brmax=np.NAN,
                  resolution=3):
        #remagnetises the component in current orientatio
        charge_vec=np.array(self.hinge_p.get_abs_rot.inv().apply(charge_vec))
        if np.isnan(Brmax):
            Brmax=np.linalg.norm(charge_vec)
        charge_vec=charge_vec/np.linalg.norm(charge_vec)
        self.magnet= self.magnet.copy(
            resolution=resolution,
            charge_vec=charge_vec,
            Brmax=Brmax)


#todo: Ready made cube parts as subclasses
class CubeA(Component):
    def __init__(self,
                 dimension=1,
                 phi_0=0, #inotal turning angle
                 charge_vec=np.array((1,1,1)), #vetor of magnetisation. Norm=magnitude
                 magnet=3, #resolutio**2 is number of point charges per surface
                 reference=cods.Origin(), #reference frame for positioning component
                 dtype=np.float32): #data type to save arrays
        if type(dimension)==type(1) or type(dimension)==type(1.1):
            shape=shds.Cube(dimension)
        else:
            shape=dimension
            dimension=dimension.dimension
        #initioating hinge connecting to privious component
        print(shape)
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
        magnet= mgds.MSphere(charge_vec=charge_vec,
                             dimension=dimension,
                             resolution=2,
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
        if type(dimension)==type(1) or type(dimension)==type(1.1):
            shape=shds.Cube(dimension)
        else:
            shape=dimension
            dimension=dimension.dimension
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
        magnet= mgds.MSphere(charge_vec=charge_vec,
                             dimension=dimension,
                             resolution=2,
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
        if type(dimension)==type(1) or type(dimension)==type(1.1):
            shape=shds.Cube(dimension)
        else:
            shape=dimension
            dimension=dimension.dimension
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
        magnet= mgds.MSphere(charge_vec=charge_vec,
                             dimension=dimension,
                             resolution=2,
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
        if type(dimension)==type(1) or type(dimension)==type(1.1):
            shape=shds.Cube(dimension)
        else:
            shape=dimension
            dimension=dimension.dimension
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
        magnet= mgds.MSphere(charge_vec=charge_vec,
                             dimension=dimension,
                             resolution=2,
                             reference=hinge_p,
                             dtype=dtype)
        #creating component
        super().__init__(shape=shape,
                         hinge_p=hinge_p,
                         hinge_n=hinge_n,
                         magnet=magnet)
        self.type='D' #setting component type

class MCylinder_Ax(Component):
    def __init__(self,
                 diameter=1,
                 height=1,
                 reference=cods.Origin(),
                 resolution=30,
                 charge_vec=np.array((1, 1, 1)),
                 Brmax=1,
                 dtype=np.float32):
        shape=shds.Cylinder(radius=diameter/2,
                            height=height)

class MselfringFrag(Component):
    def __init__(self,
                 increment,
                 r,
                 layers=3,
                 dimension=6*mm,
                 mode='up',
                 reference=cods.Origin(),
                 charge_vec=np.array((0,0,1)),
                 Brmax=1/myz,
                 dtype=np.float32):
        self.increment=increment
        self.r=r
        self.layers=layers
        self.dimension=dimension
        self.mode=mode
        self.charge_vec=charge_vec
        self.type='sfRing'
        #make hinge
        if mode=='up':
            hinge_p=cods.RefFrame(translation=-np.array((r*np.cos(increment/2),r*np.sin(increment/2),layers*1.1*dimension/2)), #relative position of hinge
                                             axis=-np.array((1,0,0)),#turning axis of hinge
                                             angle=0,#inital turning angle of hinge
                                             reference=reference,#reference to prievious components hinge
                                             dtype=dtype)
            hinge_n=cods.RefFrame.from_abs_translation(translation=np.array((r*np.cos(increment/2),r*np.sin(increment/2),layers*1.1*dimension/2)),#relative position of hinge
                                             axis=np.array((np.cos(increment),np.sin(increment),0)),#axis next component gets rotatet by
                                             angle=0,#amount of rotation
                                             reference=hinge_p,#own hinge to privous component as reference frame
                                             dtype=dtype)
        else:
            hinge_p=cods.RefFrame(translation=np.array((r*np.cos(increment/2),r*np.sin(increment/2),-layers*1.1*dimension/2)), #relative position of hinge
                                             axis=-np.array((1,0,0)),#turning axis of hinge
                                             angle=0,#inital turning angle of hinge
                                             reference=reference,#reference to prievious components hinge
                                             dtype=dtype)
            hinge_n=cods.RefFrame.from_abs_translation(translation=np.array((-r*np.cos(increment/2),-r*np.sin(increment/2),layers*1.1*dimension/2)),#relative position of hinge
                                             axis=np.array((np.cos(increment),np.sin(increment),0)),#axis next component gets rotatet by
                                             angle=0,#amount of rotation
                                             reference=hinge_p,#own hinge to privous component as reference frame
                                             dtype=dtype)

        #make magnet
        ch_mag=[]
        ch_pos=[]
        for l in range(layers):
            f=(l+1)/layers
            if mode=='up':
                center=np.array((r*np.cos(increment/f),r*np.sin(increment/f),l*1.1*dimension+1.1*dimension/2))-np.array((r*np.cos(increment/2),r*np.sin(increment/2),layers*1.1*dimension/2))
            else:
                center=np.array((r*np.cos(increment/f),r*np.sin(increment/f),-l*1.1*dimension-1.1*dimension/2))-np.array((r*np.cos(increment/2),r*np.sin(increment/2),-layers*1.1*dimension/2))

            pos, mag=self.generate_point_charges(charge_vec=charge_vec,
                                                center=center,
                                                Brmax=Brmax,
                                                resolution=2,
                                                factor=1)
            ch_pos+=pos
            ch_mag+=mag
        magnet= mgds.Magnet(ch_pos,
                            ch_mag,
                            charge_vec=np.array((0,0,1)),
                            Brmax=1/myz,
                            reference=hinge_p,
                            dtype=np.float32)
        super().__init__(shds.Cube(dimension),
                              hinge_p,
                              hinge_n,
                              magnet,
                              mass=None,
                              inertia=None,
                              dtype=np.float32)
    @property
    def make_next(self):
        if self.mode=='up':
            mode='down'
        else:
            mode='up'
        return MselfringFrag(increment=self.increment,
                 r=self.r,
                 layers=self.layers,
                 dimension=self.dimension,
                 mode=mode,
                 reference=self.hinge_n,
                 charge_vec=self.charge_vec)

    def generate_point_charges(self,
                               charge_vec=np.array((0,0,1)),
                               center=np.array((0,0,0)),
                               Brmax=1/myz,
                               resolution=2,
                               factor=0.9,
                               dtype=np.float32):
        mesh = tm.creation.icosphere(subdivisions=resolution, radius=1.0, color=None)
        normals = mesh.vertices
        points = center+self.dimension*factor*normals/2
        l=len(points)
        area=np.array(mesh.area*np.average(self.dimension)**2,dtype=dtype)
        #print("area")
        #print(area)
        out=[]
        #print(Brmax)
        for nrm in normals:
            out.append(np.dot(charge_vec,nrm) * Brmax * area / l)
        return list(points), out

    # Plotting functions-------------------------------------------------------------#
    def plot(self,
             ax,
             plt_hinges=True,
             plt_mag=False,
             plt_force=False,
             plt_charges=True,
             plt_color='rb',
             alpha=0.5):
        if plt_hinges:
            self.plot_hinges(ax)
        if plt_charges:
            self.plot_charges(ax)
        if plt_mag:
            self.plot_mag(ax)
        if plt_force:
            factor = facets[0][0] - facets[0][1]
            self.plot_force(ax, factor)
