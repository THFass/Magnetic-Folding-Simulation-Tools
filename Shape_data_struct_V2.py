"""----------------------------------------------------------------------------------------------------
Shape data structure
Offers classes and methods defining geometric shapes, thier points, factes and normals
Version 0.5
    -cube is implemented
----------------------------------------------------------------------------------------------------"""

import numpy as np
import Coord_data_struct as cods
import Function_lib as flib
import trimesh as tm


class Shape:
    # collection of points making up a shape
    def __init__(self,
                 points):
        self.points = points #points discribing the edges of the shape in local coordinate system
        self.max_dim= np.max(flib.norm(points)) #bounding box of shape

    def get_points(self,
                   ref=cods.Origin()):#refference system in which the shape is
        return ref.abs(self.points)

    def plot(self,
             ref = cods.Origin(),#refference system in which the shape is
             col = 'black',#color of points
             alpha = 1,
             ax = np.nan):#axis to plot to
        points=self.get_points(ref)
        return ax.scatter(points[:,0],points[:,1],points[:,2], c=col, alpha=alpha)

    def copy(self,
             points=np.nan):
        if np.isnan(points):
            points=self.points
        return Shape(points)

    @property
    def get_normals(self):
        #warning only true for spherical shapes
        return points/flib.norm(self.points)

    def get_facet_factor(self,vec):
        return np.nansum(vec*self.get_normals)

    @property
    def get_I(self):
        #assumes inertia of a spehre with same max dimention
        return np.eye(3)*self.get_m*(2/5)*np.power(self.max_dim,2)

    @property
    def get_V(self):
        #calculates volume under the assumtion of a a spherical shape
        return 4*np.pi*np.power(self.max_dim,3)/3

    @property
    def get_m(self):
        return self.get_V*7*1000


class Polyeder(Shape):
    #collection of points, factes, and edges making up a polyeder
    def __init__(self,
                 points,
                 facets,
                 edges,
                 normals):
        super().__init__(points=points)
        self.facets=facets
        self.normals=normals
        self.edges=edges

    def get_facet(self, index):
        return self.points[self.facets[index]]

    def get_facet_in(self, index, ref=cods.Origin()):
        return ref.rel(self.points[self.facets[index]])

    @property
    def get_facets(self):
        output = []
        for face in self.facets:
            output.append(self.points[face])
        return np.array(output)

    def get_facets_in(self,ref=cods.Origin()):
        output = []
        for face in self.facets:
            output.append(ref.rel(self.points[face]))
        return np.array(output)

    def get_facets_abs(self,ref=cods.Origin()):
        output = []
        for face in self.facets:
            output.append(ref.abs(self.points[face]))
        return np.array(output)


    @property
    def get_normals(self):
        return self.normals

    def plot(self,
             ref = cods.Origin(),#refference system in which the shape is
             col = 'b',
             alpha = 1,
             ax = np.nan):
        facets = self.get_facets_abs(ref=ref)
        out=[]
        if type(col)==str:
            col = [col]*len(factes)
        for face,c in zip(facets,col):
            out.append(ax.add_collection3d(Poly3DCollection([[tuple(face[0]),tuple(face[1]),tuple(face[2]),tuple(face[3])]], alpha=alpha, facecolors=c)))
        return out


class Cube_F(Polyeder):
    def __init__(self,
                 dimension=1):#size of cube. can be a 3dim vector to create
        super().__init__(
            points=dimension * np.array(((1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1),(1, 1, -1), (1, -1, -1), (-1, -1, -1), (-1, 1, -1))) / 2,
            facets=[[0, 1, 2, 3], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0], [4, 5, 6, 7]],
            normals=np.array(((0, 0, 1), (1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0), (0, 0, -1))),
            edges=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7],[7, 4]])
        self.dimension = dimension
        self.name = 'cube'

    def copy(self,
             dimension=np.nan):
        if np.isnan(dimension):
            dimension=self.dimension
        return Cube(dimension)


class Cube_P(Shape):
    def __init__(self,
                 dimension=1,
                 resolution=3,
                 dtype=np.float32):#size of cube. can be a 3dim vector to create
        poly=Cube_F(dimension)
        #print(self.get_facets[0])
        for index in range(len(poly.facets)):
                corners = poly.get_facet(index)                         #obtains corners of factes
                #print(corners[0])
                #creates list of points along two neighboring edges of facets. Only works with rectengular facets!
                A = np.linspace(np.array((0,0,0)), corners[1] - corners[0],resolution+2)
                B = np.linspace(np.array((0,0,0)), corners[2] - corners[1],resolution+2)
                #creates Grid from edge points
                for veca in A[1:-1]:
                    for vecb in B[1:-1]:
                        pos.append(factor*np.array(corners[0] + veca + vecb,dtype))
                        normal.append(dtype(poly.normals[index]))
        self.dimension = dimension
        self.name = 'cube'
        self.resolution = resolution
        self.points = np.array(pos,dtype)
        self.normals = np.array(normal,dtype)

    def copy(self,
             dimension=np.nan,
             resolution=np.nan,
             dtype=np.float32):
        if np.isnan(dimension):
            dimension=self.dimension
        if np.isnan(resolution):
            resolution=self.resolution
        return Cube(dimension=dimension,
                    resolution=resolution,
                    dtype=dtype)

    @property
    def get_normals(self):
        return self.normals

def Cube(dimesnion=1,
         resolution=0,
         dtype=np.float32):
    if resolution>0:
        return Cube_P(dimension=dimesnion,
                      resolution=resolution,
                      dtype=dtype)
    else:
        return Cube_F(dimension=dimesnion)



class RombicDodecaedron(Polyeder):
    def __init__(self, dimension=1):
        super().__init__(
            points=dimension * (np.array(((-2, 0, 0), (-1, 1, 1), (-1, -1, 1), (-1, -1, -1),
                                             (-1, 1, -1), (0, 2, 0), (0, 0, 2), (0, -2, 0),
                                             (0, 0, -2), (1, 1, 1), (1, -1, 1), (1, -1, -1),
                                             (1, 1, -1), (2, 0, 0)))) / 4,
            facets=[[0, 4, 5, 1], [0, 1, 6, 2], [0, 2, 7, 3], [0, 3, 8, 4], [1, 5, 9, 6], [2, 6, 10, 7],
                           [3, 7, 11, 8], [4, 8, 12, 5], [5, 12, 13, 9], [6, 9, 13, 10], [7, 10, 13, 11],
                           [8, 11, 13, 12]],
            edges=[],
            normals=[])
        self.dimension=dimension
        self.name='rombic dodecaeder'


"""class Cylinder(shape):
    def __init__(self,
                 radius=1,
                 height=1):
        phi= np.linspace(0,2*pi,num=10)
        points=[]
        facets=[]
        for i in range(len(phi)):
            points.append((radius*np.sin(phi[i]),radius*np.cos(phi[i]),height/2))
            points.append((radius*np.sin(phi[i]),radius*np.cos(phi[i]),-height/2))
            facets.append((2*i,2*i+1,2*i+2,2*i+3))
        facets[-1][2]=0
        facets[-1][3]=1
        self.facets=facets
        self.radius=radius
        self.height=height
        super().__init__(
            points=np.array(points))"""
"""class Sphere(Shape):
    def __init__(self,
                 resolution=1,
                 dimension=1,
                 dtype=np.float32):
        from numpy import sin, cos, pi
        self.name='spoly'
        self.dimension=dimension
        self.resolution=resolution
        self.dtype=dtype
        self.mesh = tm.creation.icosphere(subdivisions=resolution, radius=1.0, color=None)
        self.normals = np.array(self.mesh.vertices)
        self.points = dimension * self.normals/2
        super().__init__(radius*np.array(self.points,dtypr=dtype))

    def copy(self,
             resolution=np.nan,
             dimension=np.nan,
             dtype=np.nan):
        if np.isnan(resolution):
            resolution=self.resolution
        if np.isnan(dimension):
            dimension=self.dimension
        if np.isnan(dtype):
            dtype=self.dtype
        return Sphere(resolution=resolution,
                                   dimension=dimension,
                                   dtype=dtype)

    @property
    def get_normals(self):
        return self.points/flib.norm(self.points)"""