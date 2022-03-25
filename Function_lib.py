
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import trimesh as tr

mm=1/1000
inch=0.0254
myz=np.pi*4e-7
const=myz/(4*np.pi)
def get_fix_mins_maxs(mins, maxs):
    deltas = (maxs - mins) / 12.
    mins = mins + deltas / 4.
    maxs = maxs - deltas / 4.

    return [mins, maxs]

def make3D(inp,depth=3):
    #takes any 2D matrix and cpies it depth amount of layers
    return np.tile(inp,(depth,1,1)).transpose((1,2,0))

def repeat(inp,num):
    #repeats input array num times
    return np.tile(inp,(num,1))

def test_sanity(inp):
    return np.nansum(np.logical_not(np.isfinite(inp)))

def difmat(a,b):
    #calculates the distance between every point in a and b
    alen=len(a)
    blen=len(b)
    return np.tile(a,(blen,1,1))-np.transpose(np.tile(b,(alen,1,1)),(1,0,2))

def dist(a,b):
    #calculates the distance between every point in a and b
    diffmat= difmat(a,b)
    return np.sqrt(np.nansum(diffmat**2,2))

def bool(s):
    if s in [True,'True','TRUE','true','1',1]:
        return True
    elif s in [False,'False','FALSE','false','0',0]:
        return False
    else:
        print('error:no truth value detected')
        return s


def bool_or_type(s,dtype):
    if type(s) in [list,tuple,np.ndarray]:
        out=[]
        for item in s:
            out.append(bool_or_type(item,dtype))
    else:
        if s in [True,'True','TRUE','true','1',1]:
            return True
        elif s in [False,'False','FALSE','false','0',0]:
            return False
        else:
            try:
                return dtype(s)
            except:
                return s


def sanatise(s):
    if type(s) in [list,tuple,np.ndarray]:
        out=[]
        for item in s:
            out.append(sanatise(item))
        return out
    else:
        try:
            s_int=int(s)
            s_float=float(s)
            if s_int != s_float:
                return s_float
            else:
                return s_int
        except:
            if s in [True,'True','TRUE','true']:
                return True
            elif s in [False,'False','FALSE','false']:
                return False
            else:
                return s

class array_edit_widget:
        def __init__(self,master,array,title=''):
            self.outvars=[]
            for index,item in enumerate(array):
                if type(item) in [list,np.ndarray,tuple]:
                    self.outvars.append(array_edit_widget(master,item,title=str(index)))
                else:
                    l = tk.Label(master, text=title+str(index))
                    l.pack()
                    self.outvars.append(tk.StringVar(master,value=str(item)))
                    tk.Entry(master,textvariable=self.outvars[-1]).pack()

        def get(self,dtype=np.float32):
            out=[]
            for item in self.outvars:
                out.append(sanatise(item.get()))
            return np.array(out)

        def destroy(self):
            for var in self.outvars:
                var.destroy()

def calc_kin_f(ch_pos,ch_mag):
    #energy calculation test function
    const=1
    len_own=len(ch_pos)
    diffmat = np.tile(ch_pos, (len_own, 1, 1)) - np.transpose(np.tile(ch_pos, (len_own, 1, 1)), (1, 0, 2))
    distmat = np.sqrt(np.nansum(diffmat ** 2, 2))  # calculates the distance from every point to every other point
    np.fill_diagonal(distmat, 1)  # fills diagonal with ones to prevent zero divide for same charge distance
    ch_mat = np.tile(ch_mag,(len_own,1))*(np.ones((len_own, len_own)) - np.identity(len_own))
    charges = ch_mat*ch_mat.transpose()
    energy = np.nansum(np.triu(const * charges / distmat,1))
    force = np.nansum(const * np.tile(charges,(3,1,1)).transpose() * diffmat / np.tile(distmat,(3,1,1)).transpose() ** 3 , 1)
    torque_m = np.cross(ch_pos, force)
    return energy


def calc_energy(ch_pos,ch_mag):
    #energy calculation test function
    const=1
    len_own = len(ch_pos)
    diffmat = np.tile(ch_pos, (len_own, 1, 1)) - np.transpose(np.tile(ch_pos, (len_own, 1, 1)), (1, 0, 2))
    distmat = np.sqrt(np.nansum(diffmat ** 2, 2))  # calculates the distance from every point to every other point
    np.fill_diagonal(distmat, 1)  # fills diagonal with ones to prevent zero divide for same charge distance
    removediag = np.ones((len_own, len_own)) - np.identity(len_own)  # matrix filled with ones and zeros along diagonal to remove same charge multiplication
    ch_mat = np.tile(ch_mag,(len_own,1))*(np.ones((len_own, len_own)) *removediag)
    ##charges = np.tile(ch_mag, (len_own, 3, 1)).transpose((2, 0, 1)) * np.tile(ch_mag,(len_own, 3, 1)).transpose((0, 2, 1)) * removediag
    charges = ch_mat*ch_mat.transpose()
    energy = np.nansum(np.triu(const *charges / distmat,1))  # potential energy due to magnetic interaction of point charges
    return energy


def calc_energy_safe(ch_pos,ch_mag,const=1):
    #energy calculation test function
    # simple ineficent algo to calculate coulomb energy to compare validity of other algo
    len_own = len(ch_pos)
    energy = 0
    for l in range(len_own):
        for k in range(len_own):
            if l < k:
                energy += const * ch_mag[l] * ch_mag[k] / np.linalg.norm(ch_pos[k] - ch_pos[l])
    return energy

def norm(inp):
    #creates norm over last axes of input
    return np.sqrt(np.nansum(inp**2,np.ndim(inp)-1))

def plot_and_save_motion_study(energy,torque,angle,name):
    fig, ax1 = plt.subplots()
    angle=180*angle/np.pi
    color = 'tab:red'
    ax1.set_xlabel('angle [deg]')
    ax1.set_ylabel('Coulomb Energy [J]', color=color)
    en=ax1.plot(angle, energy, marker='o', color=color,label='pot energy')
    ax1.plot(merker='v',color='tab:blue')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Torque [Nm]', color=color)  # we already handled the x-label with ax1
    to=ax2.plot(angle, torque,marker='v', color=color,label='torque')
    ax2.tick_params(axis='y',  labelcolor=color)
    fig.legend(['pot energy','torque'])
    plt.xlim((0,180))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    np.savetxt(name+".csv",np.column_stack([energy,torque,angle]),header='Coulomb energy, torque and corresponding anble from a formation of '+name,delimiter=",")
    plt.savefig(name+".svg")

def icosahedron(level):
    from math import sin, cos, acos, sqrt, pi
    from mpl_toolkits.mplot3d import Axes3D
    s, c = 2 / sqrt(level), 1 / sqrt(level)
    topPoints = [(0, 0, 1)] + [(s * cos(i * 2 * pi / 5.), s * sin(i * 2 * pi / 5.), c) for i in range(5)]
    bottomPoints = [(-x, y, -z) for (x, y, z) in topPoints]
    icoPoints = topPoints + bottomPoints
    icoTriangs = [(0, i + 1, (i + 1) % 5 + 1) for i in range(5)] + \
                 [(6, i + 7, (i + 1) % 5 + 7) for i in range(5)] + \
                 [(i + 1, (i + 1) % 5 + 1, (7 - i) % 5 + 7) for i in range(5)] + \
                 [(i + 1, (7 - i) % 5 + 7, (8 - i) % 5 + 7) for i in range(5)]
    icoPoints=np.array(icoPoints)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    ax.scatter(icoPoints[:,0],icoPoints[:,1],icoPoints[:,2])
    return icoPoints


def D_scatter(points):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    ax.scatter(points[:,0],points[:,1],points[:,2])

def polygon(n=8):
    phi = np.arange(0, 2 * pi, num=n)
    points = []
    for p in phi:
        points.append((np.sin(p), np.cos(p)))

def sfr_centers(segments=8,
                layers=3,
                d=6*mm):
    out=[]
    r=segments*d/(2*np.pi)
    for l in range(layers):
        alpha=l*2*np.pi/(np.sqrt(2)*segments)
        out.append(np.array((r*np.sin(alpha),r*(1-np.cos(alpha)),l*d/np.sqrt(2))))
    return out


def sphere_points(r=6*mm,
                           res=2,
                           charge_vec=np.array((1, 0, 0)),
                           mag=1/myz,
                           center=np.array((0,0,0))):
    mesh = tr.creation.icosphere(subdivisions=res, radius=1.0, color=None)
    normals = mesh.vertices
    ch_v = charge_vec / np.linalg.norm(charge_vec)
    points = []
    l = len(normals)
    area = np.array(mesh.area * (r ** 2) / 4)
    out = []
    for nrm in normals:
        points.append(center+r * nrm)
        out.append(np.dot(ch_v, nrm) * mag * area / (l / 4))
    return np.array(points), np.array(out)

#points=spherical_polyedron(3)
#D_scatter(points)