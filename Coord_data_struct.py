"""----------------------------------------------------------------------------------------------------
Coord_data_struct
handles coorinate transformations
Version 1.0
    -funtional
----------------------------------------------------------------------------------------------------"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot


class Origin:
    #class for origin. needs same callabiles as RefFrame class but with zero roatation and translation
    def __init__(self, dtype=np.float32):
        self.rot = Rot.from_dcm(np.identity(3))
        self.trans = np.array((0,0,0),dtype=dtype)
        self.axis = np.array((1,0,0),dtype=dtype)
        self.angle = 0
        self.ref = self
        self.dtype = dtype

    def abs(self,
            point=np.array((0,0,0))):
        return self.dtype(point)

    def rel(self, point=np.array((0,0,0))):
        return self.dtype(point)

    @staticmethod
    def express_in(ref):
        return ref

    @property
    def get_abs(self):
        return self

    @property
    def get_abs_dcm(self):
        return self.rot.as_matrix()

    @property
    def get_number(self):
        return 0

    def append(self, axis=np.array((1,0,0)),angle=0, translation=np.array((0,0,0))):
        return RefFrame.from_axis_angle(translation=translation,axis=axis,angle=angle,reference=self,dtype=self.dtype)

    @property
    def get_abs_axis(self):
        return self.axis

"""
    #todo returns absolute transformation due to recursive algo
    @staticmethod
    def abs_ref():
        return Origin()
"""


class RefFrame:
    def __init__(self,
                 translation=np.array((0,0,0)),
                 axis=np.array((1,0,0)),
                 angle=0,
                 reference=Origin(),
                 dtype=np.float32):
        self.axis = axis.astype(dtype)
        self.angle = dtype(angle)
        self.rot = axis_angle(axis,angle)
        self.trans = translation.astype(dtype)
        self.ref = reference
        self.dtype = dtype

    @staticmethod
    def from_abs_translation(translation=np.array((0,0,0)),
                        axis=np.array((1,0,0)),
                        angle=0,
                        reference=Origin(),
                        dtype=np.float32):
        return RefFrame(translation=axis_angle(-axis,angle).apply(translation),
                        axis=axis,
                        angle=angle,
                        reference=reference,
                        dtype=dtype)

    @staticmethod
    def from_rot(translation=np.array((0,0,0)),
                 rotation=Rot.from_rotvec(np.array((np.pi/2,0,0))),
                 reference=Origin(),
                 dtype=np.float32):
        rotvec = rotation.as_rotvec()
        if np.linalg.norm(rotvec)>0:
            axis = rotvec/np.linalg.norm(rotvec)
        else:
            axis = rotvec
        return RefFrame(translation=translation,
                        axis=axis,
                        angle=np.linalg.norm(rotvec),
                        reference=reference,
                        dtype=dtype)

    def abs(self,
            point=np.array((0,0,0))):
        return self.dtype(self.ref.abs(self.rot.apply(point+self.trans)))

    def rel(self,
            point=np.array((0,0,0))):
        return self.dtype(self.rot.apply(point+self.trans))

    def rotate(self,
               point):
        return self.dtype(self.rot.apply(point))

    def rotate_to(self,
                  angle):
        self.rot = axis_angle(self.axis, angle)
        self.angle = self.dtype(angle)

    def rotate_by(self, angle):
        self.rot = axis_angle(self.axis, self.angle+angle)
        self.angle += angle

    def express_in(self,ref):
        return RefFrame.from_rot(translation=self.trans+self.rot.apply(ref.trans),
                                 rotation=ref.rot*self.rot,
                                 reference=ref.ref,
                                 dtype=self.dtype)

    @property
    def get_abs_dcm(self):
        return np.dot(self.ref.get_abs_dcm,self.rot.as_dcm())

    @property
    def get_abs_rot(self):
        return Rot.from_dcm(self.get_abs_dcm)

    @property
    def get_abs(self):
        rot = self.get_abs_rot
        return RefFrame.from_rot(translation=self.abs(np.array((0,0,0))),
                 rotation=rot,
                 reference=Origin(),
                 dtype=self.dtype)
    def __add__(self, other):
        return self.express_in(other)

    def inv_rot(self,point):
        #applies inverse rotation to vector
        return self.dtype(self.rot.inv().apply(point))


    @property
    def get_abs_axis(self):
        return self.abs(self.axis)

    @property
    def get_number(self):
        return self.ref.get_number+1

    def append(self, axis=np.array((1,0,0)),
               angle=0,
               translation=np.array((0,0,0)),
               dtype=np.float32):
        return RefFrame.from_axis_angle(translation=translation,axis=axis,angle=angle,reference=self,dtype=dtype)


"""
    #todo returns absolute transformation due to recursive algo
    def abs_ref(self):
        absref=self.ref.abs_ref()
        newrot=Rot.from_dcm(np.dot(self.rot,absref.rot.as_dcm()))
        newtrans=absref.abs(self.trans)
        return(RefFrame(rotation=newrot,translation=newtrans))
"""


def axis_angle(axis, angle):
    if angle == 0:
        return Rot.from_dcm(np.identity(3))
    else:
        return Rot.from_rotvec(angle*axis/np.linalg.norm(axis))
