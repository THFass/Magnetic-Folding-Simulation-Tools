"""----------------------------------------------------------------------------------------------------
Magnet_data_struct
handles coorinate transformations
Version 1.0
    -funtional
----------------------------------------------------------------------------------------------------"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import Coord_data_struct as cods
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Shape_data_struct_V2 as shs
from Function_lib import mm, inch, myz,const
import trimesh as tm


class Magnet:
    def __init__(self,
                 ch_pos,
                 ch_mag,
                 charge_vec,
                 Brmax,
                 reference=cods.Origin(),
                 dtype=np.float32):
        self.Brmax=Brmax
        self.charge_vec=charge_vec
        self.force=[]
        self.torque_m=[]
        self.torque_p=[]
        self.charge_pos=dtype(ch_pos)
        self.charge_mag=dtype(ch_mag)
        self.dtype=dtype
        self.ref=reference
        self.const=9**(-7)
        if not(hasattr(self,'type')):
            self.type='genereic'

    @property
    def get_charge_pos_abs(self):
        #returns absolute charge position
        return self.ref.abs(point=self.charge_pos)

    @property
    def get_charge_mag(self):
        return self.magnet.charge_mag

    @property
    def get_charge_pos(self):
        return self.magnet.charge_pos

    def get_charge_pos(self,
                       ref):
        #returns absolute charge position
        return ref.abs(self.charge_pos)

    def plot_carges(self,ax):
        """x=[]
        y=[]
        z=[]
        c=[]
        #print(type(self.charge_pos))
        for (pos,mag) in zip(self.get_charge_pos_abs,self.charge_mag):
            if mag < -0.00000001:
                col = 'r'
            elif mag > 0.00000001:
                col = 'b'
            else:
                col = 'grey'
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
            c.append(col)"""
            #ax.scatter(pos[0], pos[1], pos[2], c=col)
        pos=self.get_charge_pos_abs
        ax.scatter(pos[0],pos[1],pos[2], c=self.charge_mag,cmap='RdBl')

    def calc_kin_f(self,
                   pos_other,
                   charges_other,
                   hinge_point=np.array((0,0,0))):
        len_other=len(pos_other)
        pos_own=self.get_charge_pos_abs
        len_own=len(pos_own)
        diffmat=np.tile(pos_other,(len_own,1,1))-np.transpose(np.tile(pos_own,(len_other,1,1)),(1,0,2))
        distmat=np.tile(np.sqrt(np.nansum(diffmat**2,2)),(3,1,1)).transpose((1,2,0))
        charges=np.tile(self.charge_mag,(len_other,3,1)).transpose((2,0,1))*np.tile(charges_other,(len_own,3,1)).transpose((0,2,1))
        self.force=np.nansum(self.const*charges*diffmat/distmat**3,1)#*10**6 #diff 10**-3m/ dist**3 10**-9m => 10**6
        self.torque_m=np.cross(self.charge_pos,self.force)
        self.torque_p=np.cross(hinge_point-pos_own,self.force)

    def br2charge(self,Brmax,area):
        #return Brmax*area / (np.pi*(4e-7)*points)
        return Brmax*area / (np.pi*(4e-7))


class MCube(Magnet):
    def __init__(self,
                 dimension=np.array((1,1,1)),
                 charge_vec=np.array((1,1,1)),
                 Brmax=1,
                 resolution=10,
                 reference=cods.Origin(),
                 factor=0.9,
                 dtype=np.float32):
        if type(dimension)==float or type(dimension)==int:
            dimension=dimension*np.array((1,1,1))
        self.type='cube'
        self.points=dtype(np.array(dimension) * np.array(((1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1),(1, 1,-1), (1, -1,-1), (-1, -1,-1), (-1, 1,-1))) / 2)
        self.facets=[[0, 1, 2, 3], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0], [4, 5, 6, 7]]
        self.normals=np.array(((0, 0, 1), (1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0), (0, 0, -1)))
        self.edges=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]
        self.dimension=dimension
        self.minmax=(-dimension[0]/2,dimension[0]/2,-dimension[1]/2,dimension[1]/2,-dimension[2]/2,dimension[2]/2)
        self.resolution=resolution
        [charge_pos,charge_mag]=self.generate_point_charges(charge_vec=charge_vec,
                                                            Brmax=Brmax,
                                                            resolution=resolution,
                                                            factor=factor,
                                                            dtype=dtype)
        super().__init__(ch_pos=charge_pos,
                         ch_mag=charge_mag,
                         reference=reference,
                         charge_vec=charge_vec,
                         Brmax=Brmax,
                         dtype=dtype)

    def copy(self,
             resolution=0,
             charge_vec=0,
             reference=0,
             Brmax=np.NAN,
             factor=0.9,):
        if type(charge_vec)!=np.ndarray:
            charge_vec=self.charge_vec
        if resolution==0:
            resolution=self.resolution
        if np.isnan(Brmax):
            Brmax=self.Brmax
        if reference==0:
            reference=self.ref
        return MCube(dimension=self.dimension,
                     reference=reference,
                     charge_vec=charge_vec,
                     Brmax=Brmax,
                     resolution=resolution,
                     factor=factor,
                     dtype=self.dtype)


    @property
    def get_facets(self):
        output = []
        for face in self.facets:
            #print(face)
            output.append(self.points[face])
        return np.array(output)


    @property
    def get_facets_abs(self):
        output = []
        for face in self.facets:
            #print(face)
            output.append(self.ref.abs(self.points[face]))
        return np.array(output)

    def get_facet(self, index):
        return self.points[self.facets[index]]

    def generate_point_charges(self,
                               charge_vec=np.array((1,1,1)),
                               Brmax=1,
                               resolution=3,
                               factor=0.9,
                               dtype=np.float32):
        #creates resolution**2 point charges for each surface of the polygon and saves them in a list
        charge_pos = []                                                    #initiates position list
        charge_mag = []                                                    #initiates magnitude list
        axis = charge_vec/np.linalg.norm(charge_vec)
        #area=np.prod(self.dimension*abs(abs(axis)-1))
        area=self.dimension[0]*self.dimension[1]
        abs_charge = self.br2charge(Brmax,area)      #computes max8imum magnitude per point charge                                              #normalises charge vector
        #loop through factes
        #print(self.get_facets[0])
        for index in range(len(self.facets)):
            magnitude_factor = abs_charge * np.dot(axis, self.normals[index])       #scalar product of facet normal and charge vector gives magnitude factor
            if np.abs(np.dot(axis, self.normals[index]))>0.01:
                corners = self.get_facet(index)                         #obtains corners of factes
                #print(corners[0])
                #creates list of points along two neighboring edges of facets. Only works with rectengular facets!
                A = np.linspace(np.array((0,0,0)), corners[1] - corners[0],resolution+2)
                B = np.linspace(np.array((0,0,0)), corners[2] - corners[1],resolution+2)
                #creates Grid from edge points
                for veca in A[1:-1]:
                    for vecb in B[1:-1]:
                        charge_pos.append(factor*np.array(corners[0] + veca + vecb,dtype))
                        charge_mag.append(dtype(magnitude_factor))
        return [charge_pos,charge_mag]

class MCylinder_Ax(Magnet):
    def __init__(self,
                 diameter=1,
                 height=1,
                 reference=cods.Origin(),
                 resolution=30,
                 charge_vec=np.array((1,1,1)),
                 Brmax=1,
                 dtype=np.float32):
        self.type='cylinder'
        self.charge_vec=charge_vec
        self.resolution=resolution
        self.minmax=(-radius,radius,-radius,radius,-height/2,height/2)
        area=np.pi*radius**2
        charge_pos=[]
        radius=diameter/2
        for x in np.linspace(-radius,radius,num=resolution):
            for y in np.linspace(-radius,radius,num=resoltution):
                if np.linalg.norm((x,y))<=radius:
                    charge_pos.append((x,y,height/2))
                    charge_pos.append((x,y,-height/2))
        charge_pos=np.array(charge_pos)
        abs_charge = Brmax*area / (np.pi*(4e-7)*len(charge_pos))      #computes max8imum magnitude per point charge
        charge_mag=[]
        for pos in charge_pos:
            if pos[2]>0:
                charge_mag.append(abs_charge)
            else:
                charge_mag.append(-abs_charge)

        #abs_charge = np.linalg.norm(charge_vec) / (resolution * resolution)      #computes magnitude per point charge
        #axis = charge_vec/np.linalg.norm(charge_vec)
        super().__init__(ch_pos=np.array(charge_pos),
                       ch_mag=np.array(charge_mag),
                       reference=reference,
                       dtype=dtype)

    def copy(self,
             resolution=0,
             charge_vec=0,
             reference=cods.Origin()):
        if type(charge_vec)!=np.ndarray:
            charge_vec=self.charge_vec
        if resolution==0:
            resolution=self.resolution
        [charge_pos,charge_mag]=self.generate_point_charges(charge_vec=charge_vec,
                                                            Brmax=self.Brmax,
                                                            resolution=resolution,
                                                            dtype=dtype)
        return Magnet(ch_pos=charge_pos,
                      ch_mag=charge_mag,
                      reference=reference,
                      dtype=self.dtype)
class MSphere(Magnet):
    def __init__(self,
                 dimension=np.array((1,1,1)),
                 charge_vec=np.array((1,0,0)),
                 Brmax=0,
                 resolution=3,
                 reference=cods.Origin(),
                 factor=0.9,
                 dtype=np.float32):
        if type(dimension)==float or type(dimension)==int:
            dimension=dimension*np.array((1,1,1))
        self.type='sphere'

        self.dimension=dimension
        self.minmax=(-dimension[0]/2,dimension[0]/2,-dimension[1]/2,dimension[1]/2,-dimension[2]/2,dimension[2]/2)
        self.resolution=resolution

        #self.mesh = tm.creation.icosphere(subdivisions=resolution, radius=1.0, color=None)
        #self.normals = np.array(mesh.vertices)
        #self.points = dimensions*self.normals
        [charge_pos,charge_mag]=self.generate_point_charges(charge_vec=charge_vec,
                                                            Brmax=Brmax,
                                                            resolution=resolution,
                                                            factor=factor,
                                                            dtype=dtype)
        super().__init__(ch_pos=charge_pos,
                         ch_mag=charge_mag,
                         reference=reference,
                         charge_vec=charge_vec,
                         Brmax=Brmax,
                         dtype=dtype)

    def generate_point_charges(self,
                 charge_vec=np.array((1,0,0)),
                 Brmax=1/myz,
                 resolution=2,
                 factor=0.9,
                 dtype=np.float32):
        mesh = tm.creation.icosphere(subdivisions=2, radius=1.0, color=None)
        normals = mesh.vertices
        points = self.dimension*factor*normals/2
        l=len(points)
        area=np.array(mesh.area*np.average(self.dimension)**2,dtype=dtype)
        #print("area")
        #print(area)
        out=[]
        #print(Brmax)
        for nrm in normals:
            out.append(np.dot(charge_vec,nrm) * Brmax * area / l)
        return points, out

    def copy(self,
             resolution=0,
             charge_vec=0,
             reference=0,
             Brmax=np.NAN,
             factor=0.9,):
        if type(charge_vec)!=np.ndarray:
            charge_vec=self.charge_vec
        if resolution==0:
            resolution=self.resolution
        if np.isnan(Brmax):
            Brmax=self.Brmax
        if reference==0:
            reference=self.ref
        return MSphere(dimension=self.dimension,
                     reference=reference,
                     charge_vec=charge_vec,
                     Brmax=Brmax,
                     resolution=resolution,
                     factor=factor,
                     dtype=self.dtype)
"""class MSpere(Magnet):
    def __init__(self,
                 dimension=1,
                 charge_vec=np.array((1,0,0)),
                 Brmax=0,
                 resolution=5,
                 reference=cods.Origin(),
                 dtype=np.float32):
        if type(dimension)!=float or type(dimension)!=int:
            dimension=dimension[0]
        self.type = 'sphere'
        self.points=shs.spherical_polyedron(resolution,radius=dimension[0],dtype=dtype)

        self.dimension=dimension
        self.minmax=(-dimension[0]/2,dimension[0]/2,-dimension[1]/2,dimension[1]/2,-dimension[2]/2,dimension[2]/2)
        self.resolution=resolution

        if Brmax>0:
            charge_vec = self.br2charge(Brmax,np.pi*dimension**2)*charge_vec/np.linalg.norm(charge_vec)

        A = np.tile(charge_vec,(len(self.points),1))
        charge_mag=np.nansum(A*self.points,axis=1)
        super().__init__(ch_pos=self.points,
                         ch_mag=charge_mag,
                         reference=reference,
                         charge_vec=charge_vec,
                         Brmax=Brmax,
                         dtype=dtype)

    def copy(self,
             resolution=np.nan,
             Brmax=np.nan,
             charge_vec=np.nan,
             reference=np.nan):
        if np.isnan(charge_vec):
            charge_vec=self.charge_vec
        if np.isnan(Brmax):
            Brmax=self.Brmax
        if np.isnan(resolution):
            resolution=self.resolution
        if np.isnan(reference):
            reference=self.ref

        return MSpere(dimension=self.dimension,
                 charge_vec=charge_vec,
                 Brmax=Brmax,
                 resolution=resolution,
                 reference=reference,
                 dtype=self.dtype)"""

def from_shape(shape,
               resolution=3,
               charge_vec=np.array((1,1,1)),
               Br_max=0,
               reference=cods.Origin(),
               factor=0.9,
               dtype=np.float32):
    if type(shape)==str:
        name=shape
    else:
        name=shape.name
    if name=='cube':
        return MCube(dimension=shape.dimension,
                     charge_vec=charge_vec,
                     Brmax=Br_max,
                     resolution=resolution,
                     reference=reference,
                     factor=factor,
                     dtype=dtype)
    elif name=='cylinder':
        #todo
        return MCube(dimension=shape.dimension,
                 charge_vec=charge_vec,
                 Brmax=Br_max,
                 resolution=resolution,
                 reference=reference,
                 dtype=dtype)
    elif name=='sphere':
        #todo
        return MSphere(dimension=shape.dimension,
                 charge_vec=charge_vec,
                 Brmax=Br_max,
                 resolution=resolution,
                 reference=reference,
                 dtype=dtype)



