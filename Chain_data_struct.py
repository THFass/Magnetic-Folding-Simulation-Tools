"""----------------------------------------------------------------------------------------------------
Chain data structure
Main data structure that handles classesa nd methods discribing a kinematic chaon
Version 0.0
    -empty
----------------------------------------------------------------------------------------------------"""
import numpy as np
import Component_data_struct as cds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Function_lib as fl
from scipy.spatial import distance_matrix
from Function_lib import mm, inch, myz,const
from progress.bar import Bar
import trimesh as tr
import Chain_Dynamics_lib_CPU as cdl
import Funklib_CPU as fnl
from scipy.spatial.transform.rotation import Rotation as rt

class KineticChain:
    def __init__(self,
                 sequence,      #list of components
                 resolution,    #resolution of point charge approxiamtion
                 upper_boundary=2*np.pi,
                 lower_boundary=0,
                 fig=0,
                 dtype=np.float32):
        self.comp=sequence
        self.resolution=resolution
        self.fig=fig
        self.dtype=dtype
        self.energy=0
        self.const=const
        self.optim_result=[]
        self.upper_boundary=upper_boundary
        self.lower_boundary=lower_boundary
        self.comp_type=sequence[0].type
        self.mag_vec=sequence[0].magnet.type
        self.results=[]
        self.ftsres=[]
        self.ax=False

        self.link_lists()

    def link_lists(self):
        #links component lists to chain lists, so that entries in both lists point to the same memory space
        self.ch_pos = []            #links to all charge positions
        self.ch_mag = []            #links to all charge magnetudes
        self.ch_connect=[]
        self.comp_pos = []          #links to all component absolute positions
        self.comp_force = []        #links to all forces acting on all components
        self.comp_torque = []       #links to all torque acting on all components
        self.comp_r_l = []          #links to vectors from last hinge to center with absolute rotation
        self.comp_r_n = []          #links to vectors from next hinge to center with absolute rotation
        self.comp_R_l = []          #links to r_l as cross product matrix for all components
        self.comp_R_l = []          #links to r_n as cross product matrix for all components
        self.force=[]
        self.torque=[]
        for index, comp in enumerate(self.comp):
            for ch_pos in comp.get_charge_pos:
                self.ch_pos.append(ch_pos)            #links to all charge positions
                self.ch_connect.append(index)
            for ch_mag in comp.get_charge_mag:
                self.ch_mag.append(ch_mag)            #links to all charge magnetudes
            self.comp_pos.append(comp.trans_abs.trans)          #links to all component absolute positions
            self.comp_force.append(comp.force)        #links to all forces acting on all components
            self.comp_torque.append(comp.torque)      #links to all torque acting on all components
            #self.comp_r_l.append(comp.trans_abs)          #links to vectors from last hinge to center with absolute rotation
            #self.comp_r_n.append(comp.magnet.charge_pos)          #links to vectors from next hinge to center with absolute rotation
            #self.comp_R_l.append(comp.magnet.charge_pos)          #links to r_l as cross product matrix for all components
            #self.comp_R_l.append(comp.magnet.charge_pos)         #links to r_n as cross product matrix for all components


    @property
    def update_coord(self):
        for comp in self.comp:
            comp.update_from_last()

    @property
    def update_lists(self):
        #updates force torque and internal energy of the chain
        self.ch_pos=[]
        self.ch_mag=[]
        self.ch_connect=[]
        for index, comp in enumerate(self.comp):
            for ch_pos in comp.get_charge_pos_abs:
                self.ch_pos.append(ch_pos)            #links to all charge positions
                self.ch_connect.append(index)
            for ch_mag in comp.get_charge_mag:
                self.ch_mag.append(ch_mag)            #links to all charge magnetudes
        self.ch_pos=np.array(self.ch_pos,dtype=self.dtype)
        self.ch_mag=np.array(self.ch_mag,dtype=self.dtype)

    @property
    def update_kin(self):
        #updates forces and torque in the chain
        self.update_lists
        self.calc_kin_f
        return self.energy

    @property
    def calc_kin_f(self):
        #calculates forces, and torque acting on all monopol charges in the system as well as coulomb enegy
        len_own = len(self.ch_pos)
        diffmat = np.tile(self.ch_pos, (len_own, 1, 1)) - np.transpose(np.tile(self.ch_pos, (len_own, 1, 1)), (1, 0, 2))
        distmat = np.sqrt(np.nansum(diffmat ** 2, 2))  # calculates the distance from every point to every other point
        np.fill_diagonal(distmat, 1)  # fills diagonal with ones to prevent zero divide for same charge distance
        removediag = np.ones((len_own, len_own)) - np.identity(len_own)  # matrix filled with ones and zeros along diagonal to remove same charge multiplication
        charges = np.tile(self.ch_mag, (len_own, 1)) * (np.ones((len_own, len_own)) *removediag)
        charges = charges * charges.transpose()
        self.energy = np.nansum(np.triu(self.const * charges / distmat, 1))  # potential energy due to magnetic interaction of point charges
        self.force = np.nansum(self.const * np.tile(charges,(3,1,1)).transpose() * diffmat / np.tile(distmat,(3,1,1)).transpose() ** 3 , 1)#force on acting on each monpol charge
        self.torque = np.cross(self.ch_pos, self.force)#torque acting on each monopol charge in respect to point
        return self.energy


    @property
    def calc_energy(self):
        self.update_lists                                       #writes all pos and magnitudes from all charges in all components in one big list
        len_own = len(self.ch_pos)
        diffmat = np.tile(self.ch_pos, (len_own, 1, 1)) - np.transpose(np.tile(self.ch_pos, (len_own, 1, 1)), (1, 0, 2))
        distmat = np.sqrt(np.nansum(diffmat ** 2, 2))  # calculates the distance from every point to every other point
        np.fill_diagonal(distmat, 1)  # fills diagonal with ones to prevent zero divide for same charge distance
        removediag = np.ones((len_own, len_own)) - np.identity(len_own)  # matrix filled with ones and zeros along diagonal to remove same charge multiplication
        charges = np.tile(self.ch_mag, (len_own, 1)) * (np.ones((len_own, len_own)) *removediag)
        charges = charges * charges.transpose()
        self.energy = np.nansum(np.triu(self.const * charges / distmat, 1))  # potential energy due to magnetic interaction of point charges
        return self.energy

    @property
    def calc_energy_safe(self):
        # simple ineficent algo to calculate coulomb energy to compare validity of other algo
        len_own = len(self.ch_pos)
        self.energy = 0
        for l in range(len_own):
            for k in range(len_own):
                if l < k:
                    self.energy += self.const * self.ch_mag[l] * self.ch_mag[k] / np.linalg.norm(self.ch_pos[k] - self.ch_pos[l])
        return self.energy

    def calc_force_safe(self):
        half=int(len(self.ch_mag)/2)
        #force=0
        force=np.array((0,0,0))
        workload=len(self.ch_mag)
        i=0
        #pbar=Bar('Charging', max=len(self.ch_mag))
        for qi,xi in zip(self.ch_mag[:half],self.ch_pos[:half]):
            for qj, xj in zip(self.ch_mag[half:],self.ch_pos[half:]):
                #print(self.const*qi*qj*(xi-xj)/np.linalg.norm(xi-xj)**3)
                force= force+self.const*qi*qj*(xi-xj)/np.linalg.norm(xi-xj)**3
                #force= force+self.const*qi*qj/np.linalg.norm(xi-xj)**2
            #i+=1
            #print(100*i/workload)
        #self.force=force
        return force


    def calc_force_x(self,target):
        #calculates forces and torque acting on a specific component
        force=[]
        torque_m=[]
        torque_p=[]

        for i in range(len(self.comp)):
            if i!=target:
                self.comp[target].magnet.calc_kin_f(self.comp[i].get_charge_pos_abs,
                                                    self.comp[i].get_charge_mag,
                                                    self.comp[i].get_prev_hinge,)
            force.append(np.nansum(self.comp[target].magnet.force,0))
            torque_m.append(np.nansum(self.comp[target].magnet.torque_m,0))
            torque_p.append(np.nansum(self.comp[target].magnet.torque_p,0))
        return [np.nansum(force,0),
                np.nansum(torque_m,0),
                np.nansum(torque_p,0)]


    def calc_torque_x(self,
                      target,
                      axis=None):
        #calculates the general torque of components in regard to a certain point and axis
        point=self.comp[target].get_hinge_p
        comp_axis=self.comp[target].get_hinge_p_axis
        if type(axis)!=type(comp_axis):
            axis=comp_axis
        axis=axis/np.linalg.norm(axis)#noormalises axis
        ch_pos=np.array(self.ch_pos)
        ch_connect=np.array(self.ch_connect)
        print(target)
        torque_p = np.cross(np.array(ch_pos[ch_connect>=target])-point, self.force[ch_connect>=target])
        return np.inner(np.nansum(torque_p,0),axis)

    @property
    def movement_factor(self):
        #calculates two weighting factors: the change in angle from elongated position and the distance the centers traveled
        current_angles=self.get_angles
        current_centers=self.get_centers
        self.set(0)
        zero_centers=self.get_centers
        #print(current_centers)
        distance_factor=np.nansum(np.sqrt(np.nansum(np.square(current_centers-zero_centers),1)))
        self.set(current_angles)
        return [np.nansum(current_angles),distance_factor]

    @property
    def is_colliding(self):
        angles=self.get_angles
        if np.any(angles>self.upper_boundary) or np.any(angles<self.lower_boundary): #checks is any of the angles are out of bound
            return True
        else:
            centers=self.get_centers
            dist=fl.dist(centers,centers)+self.comp[0]dimension*np.identity(len(centers))
            if np.any(dist < 0.9*self.comp[0].dimension):
                return True

    @property
    def get_centers(self):
        centers=[]
        for comp in self.comp:
            centers.append(comp.get_center)
        return np.array(centers)

    @property
    def get_angles(self):
        angles=[]
        for comp in self.comp:
            angles.append(comp.get_angle)
        return np.array(angles)

    @property
    def calc_energy_safety(self):
        if self.is_colliding:
            return np.inf
        else:
            return self.calc_energy

    def cost_fun(self,x):
        self.set_angles(x)
        return self.calc_energy_safety

    @property
    def check_overlap(self):
        #todo
        return False

    def magnetise(self,
                  mag_vec=np.array((1,1,1)),
                  Brmax=np.NAN,
                  resolution=0):
        #remagnetises components
        if resolution==0:
            resolution=self.resolution
        if np.isnan(Brmax):
            Brmax=np.linalg.norm(mag_vec)
        if len(mag_vec) == 3:
            for comp in self.comp:
                comp.magnetise(charge_vec=mag_vec,
                               Brmax=Brmax,
                               resolution=resolution)
        elif len(mag_vec) == len(self.comp):
            for (vec,comp) in zip(mag_vec,self.comp):
                comp.magnetize(charge_vec=vec,
                               Brmax=Brmax,
                               resolution=resolution)
        else:
            print('error: input vector for magnetisation is neither 3D vector nor list of vectors of chain length')

    def magnetise_x(self,
                    index,
                    mag_vec=np.array((1, 1, 1)),
                    Brmax=np.NAN,
                    resolution=0):
        #magnetises specific component with given vector and br
        if resolution==0:
            resolution=self.resolution
        if np.isnan(Brmax):
            Brmax=np.linalg.norm(mag_vec)
        self.comp[index].magnetise(charge_vec=mag_vec,
                       Brmax=Brmax,
                       resolution=resolution)

    def set_angles(self,angles):
        #sets angles of all components
        for (phi,comp) in zip(angles,self.comp):
            comp.set_angle(phi)

    def set(self,param):
        if type(param) in (list, tuple, np.array, np.ndarray):
            self.set_angles(list(param))
        elif type(param) in (float,int):
            self.set_angles(param*np.ones(len(self.comp)))
        else:
            print('unknown data type to set angles')

    def set_anglex(self,angle,index):
        #sets angle of specific component
        self.comp[index].set_angle(angle)

    def plot(self,
             plt_hinges=True,             #if true displayes hinges
             plt_mag=False,                     #if true displayes magnetisation direction
             plt_force=False,                   #if true displayes net force acting on component
             plt_charges=True,                 #if true displayes charges
             plt_auto_zoom=True,
             edit_mode=False,
             color_mode='rb',
             alpha=0.5,                         #transparency of surfices
             axis=[[0,1],[-0.5,0.5],[-0.5,0.5]]): #limit for x,y and z axis
        #plots chain in figure or in new window
        if self.fig==0:#in case there is no figure window a new one is created
            self.fig= plt.figure().gca(projection='3d')
            dim=np.linalg.norm(self.comp[0].hinge_n.trans-self.comp[0].hinge_p.trans)
            length=len(self.comp)
            self.fig.set_xlim(fl.get_fix_mins_maxs(dim*length*axis[0][0], dim*length*axis[0][1]))
            self.fig.set_xlim(fl.get_fix_mins_maxs(dim*length*axis[0][0], dim*length*axis[0][1]))
            self.fig.set_ylim(fl.get_fix_mins_maxs(dim*length*axis[1][0], dim*length*axis[1][1]))
            self.fig.set_zlim(fl.get_fix_mins_maxs(dim*length*axis[2][0], dim*length*axis[2][1]))
            # Equally stretch all axes

        self.fig.cla()#clears figure
        for comp in self.comp[:-1]:#plot last component
            comp.plot(ax=self.fig,
                      plt_hinges=plt_hinges,
                      plt_mag=plt_mag,
                      plt_force=plt_force,
                      plt_charges=False,#plt_charges,
                      plt_color=color_mode,
                      alpha=alpha)
        #if plt_charges:
        if edit_mode:
            self.comp[-1].plot(ax=self.fig,
                          plt_hinges=plt_hinges,
                          plt_mag=plt_mag,
                          plt_force=plt_force,
                          plt_charges=False,#plt_charges,
                          plt_color='grey',
                          alpha=alpha/2)
        else:
            self.comp[-1].plot(ax=self.fig,
                      plt_hinges=plt_hinges,
                      plt_mag=plt_mag,
                      plt_force=plt_force,
                      plt_charges=False,#plt_charges,
                      plt_color=color_mode,
                      alpha=alpha)
        if plt_charges:
            self.update_lists
            ch_pos=np.array(self.ch_pos)
            self.fig.scatter(ch_pos[:,0],ch_pos[:,1],ch_pos[:,2])
        if plt_auto_zoom:
            dim=(self.comp[0].shape.dimension+self.find_max_dist)*np.array(axis)
        else:
            dim=self.comp[0].shape.dimension*len(self.comp)
        self.fig.set_xlim(fl.get_fix_mins_maxs(dim[0][0], dim[0][1]))
        self.fig.set_ylim(fl.get_fix_mins_maxs(dim[1][0], dim[1][1]))
        self.fig.set_zlim(fl.get_fix_mins_maxs(dim[2][0], dim[2][1]))
        # Equally stretch all axes
        plt.show(block=False)
        plt.draw()
        return self.fig

    @property
    def find_max_dist(self):
        return np.max(self.get_centers)

    def append(self,
               comp_type='A',
               phi_0=0,
               resolution=3,
               dtype=np.float32):
        self.comp.append(self.comp[-1].make_next(comp_type=comp_type,
                                             phi_0=phi_0,
                                             magnet=resolution,
                                             dtype=dtype))

    def remove(self,index=-1):
        del self.comp[index]

    def plot_to(self,
                fig=0,
                plt_hinges=False,
                plt_mag=False,
                plt_force=False,
                plt_charges=False,
                alpha=0.5,
                axis=[[0,1],[-0.5,0.5],[-0.5,0.5]]):
        self.fig=fig
        self.plot(plt_hinges=plt_hinges,
                  plt_mag=plt_mag,
                  plt_force=plt_force,
                  plt_charges=plt_charges,
                  alpha=alpha,
                  axis=axis)

    @staticmethod
    def from_types(sequence,
                   dimension=1,
                   inital_angles=0,
                   resolution=3,
                   charge_vec=np.array((1,1,1)), #vetor of magnetisation. Norm=magnitude
                   fig=0,
                   dtype=np.float32):
        #creates a kinetic chain from a list of component types and initial angles
        if type(inital_angles) in (float, int):
            inital_angles = inital_angles*np.ones(len(sequence))
        elif type(inital_angles) in (list,sequence) and len(sequence)==len(inital_angles):
            inital_angles = n
            p.array(inital_angles)
        elif len(sequence)!=len(inital_angles):
            print('length of initial angles must be one or equal to length of sequence')
            return -1
        for comp in sequence:
            if type(comp)==str or type(comp)==int:
                if comp in ('D','d',4,'4'):
                    comp=cds.CubeD
                elif comp in ('C','c',3,'3'):
                    comp=cds.CubeC
                elif comp in ('B','b',2,'2'):
                    comp=cds.CubeB
                else:
                    comp=cds.CubeA
        comps=[sequence[0](phi_0=inital_angles[0],
                           dimension=dimension,
                           magnet=resolution,
                           charge_vec=charge_vec,
                           dtype=dtype)]
        for (el_type,phi_0) in zip(sequence[1:],inital_angles[1:]):
            comps.append(comps[-1].make_next(comp_type=el_type,
                                             phi_0=phi_0,
                                             magnet=resolution,
                                             dtype=dtype))
        chain= KineticChain(sequence=comps,
                            resolution=resolution,
                            dimension=dimension,
                            upper_boundary=2*np.pi,
                            lower_boundary=0,
                            fig=fig,
                            dtype=dtype)
        chain.set(inital_angles)
        chain.magnetise(charge_vec)
        return chain

    @property
    def length(self):
        return len(self.comp)

    @property
    def destroy(self):
        plt.close('all')

    def get_phidot(self,
                   dof):
        #compute inertia
        I=self.get_I(dof)
        #I=2*M*r*r/5+M*r*r
        #solve newton
        t=self.calc_torque_x(dof)
        return t/I

    def get_I(self,dof):
        M=0.001
        r=0.25*0.5*inch
        return (2*M*r*r/5+M*r*r)*len(self.comp)/2

    def simulate(self,
                 frik=0,
                 dt=0.001,
                 time=None,
                 plot=False):
        time=time or 1
        n=len(self.comp)
        mass=self.comp[0].shape.get_m
        inertia=self.comp[0].shape.get_I
        Q=self.get_Q
        rao,rbo=self.get_r
        self.minangle=0
        self.maxangle=np.pi
        if plot:
            fig=plt.figure()
            self.ax=[fig,fig.add_subplot()]
        else:
            self.ax=False
        sim=cdl.KinChain(n=n,
                         mass=mass,
                         inertia=inertia,
                         Q=Q,
                         g=-np.array((0, 0, 0)),
                         fts=self.get_fts,
                         rao=rao,
                         rbo=rbo,
                         frik=frik,
                         dt=dt,
                         dtype=np.float32)
        return sim, sim.solver(dt=dt,
                               t_end=time)

    @property
    def get_Q(self):
        out=[]
        for comp in self.comp:
            out.append(comp.get_Q)
        return out

    @property
    def get_r(self):
        rao=[]
        rbo=[]
        for comp in self.comp:
            rao.append(comp.hinge_p.trans)
            rbo.append(comp.hinge_n.trans)
        return np.block(rao),np.block(rbo)

    def get_fts(self,
                state):
        state=self.set_state(state)
        #print(np.shape(state))
        self.calc_kin_f
        n=len(self.comp)
        #print(np.min(self.force))
        fs=np.nansum(np.split(self.force,n),1)
        fs[:,2]=np.zeros_like(fs[:,2])
        ts=np.nansum(np.split(self.torque,n),1)
        ts[:,0:1]=np.zeros_like(ts[:,0:1])
        #print(fs[0])
        self.ftsres.append([fs,ts])
        return np.block(list(fs)), np.block(list(ts)),state

    def set_state(self,
                  state):
        rotmat=fnl.undiag_mat(state[0])
        accel=fnl.undiag_mat(state[1])
        results=[]
        #print(np.max(np.abs(state[0])))
        #print(np.shape(rotmat[0]))
        #print(np.max(np.abs(state[1])))
        #print(np.shape(accel))
        #print(rotmat)
        Q=[]
        Qdt=[]
        angle_abs=0
        for comp,rot,acc in zip(self.comp,rotmat,accel):
            trans=rt.from_matrix(rot).as_rotvec()
            axis=comp.hinge_p.get_abs_axis
            #print(trans)
            #angle+=np.inner(trans,axis)
            angle_rel=trans[2]-angle_abs
            if np.isfinite(angle_rel):
                if angle_rel<self.minangle:
                    angle_rel=self.minangle
                    acc=np.zeros((3,3))
                if angle_rel>self.maxangle:
                    angle_rel=self.maxangle
                    acc=np.zeros((3,3))
            else:
                angle_rel=comp.get_angle
            comp.set_angle(angle_rel)
            results.append(angle_rel)
            angle_abs=angle_abs+angle_rel
            #print(rt.from_rotvec(angle*axis).as_matrix)
            Q.append(rt.from_rotvec(angle_abs*axis).as_matrix())
            Qdt.append(acc)
        self.results.append(results)
        self.update_lists
        return np.array((fnl.matlist2diag(Q),fnl.matlist2diag(Qdt)))

    def solve(self,
              dof,
              dt=0.01,
              steps=100,
              max=np.pi,
              min=0):
        out=[]
        #for
        steps=np.arange(0,steps*dt,dt)
        phi=0
        phidt=0
        for st in steps:
            phiddt=self.get_phidot(dof)
            phidt=phidt+phiddt/dt
            phi=phi+phidt
            if phi>max:
                phi=max
            if phi<min:
                phi=min
            print(phi)
            #get_phidot
            #update chain
            self.set_anglex(phi,dof)
            #save results
            out.append(phi)
        return out

""" def kin_simulation(self,
                       dt,
                       state):
        #define simulation
        sim=cdl.KinChain(self,dt,state)
        #set initial state
        sim.set_state(state)
        #get initial state
        self.set_state(state)
        steps=np.arrange(0,100*dt)
            #for
        for st in steps
            state=self.get_state
            #calc dp
            state=sim.func(state)

            #update chain
            self.update_state(state)"""


class Mag_Pull(KineticChain):
    def __init__(self,
                 diam=1.58*mm,
                 height=1.58*mm,
                 dist=np.linspace(0.3*mm,20*mm,30),
                 res=15,
                 Br_max=1.3):
        self.diam=diam
        self.height=height
        self.dist=dist
        self.distnow=dist[0]
        self.Br_max=Br_max/myz
        self.res=res
        self.ch_pos, self.ch_mag=self.make_points(dist[0])
        self.force=[]
        self.energy=[]
        self.torque=[]
        self.const=myz/(4*np.pi)#8.9875517923e9#

    def make_points(self,
                    dist):
        dist=dist+self.height
        points=[]
        charges=[]
        for x in np.linspace(-self.diam/2,self.diam/2,num=self.res):
            for y in np.linspace(-self.diam/2,self.diam/2,num=self.res):
                if np.sqrt(x**2+y**2)<=self.diam/2:
                    points.append((x,y,self.height/2))
                    points.append((x,y,-self.height/2))
                    charges.append(1)
                    charges.append(-1)
        for x in np.linspace(-self.diam/2,self.diam/2,num=self.res):
            for y in np.linspace(-self.diam/2,self.diam/2,num=self.res):
                if np.sqrt(x**2+y**2)<=self.diam/2:
                    points.append((x,y,dist+self.height/2))
                    points.append((x,y,dist-self.height/2))
                    charges.append(1)
                    charges.append(-1)
        area=np.pi*(self.diam**2)/4
        print(area)
        factor=(self.Br_max*area)/(len(charges)/4)
        print(factor)
        return np.array(points), factor*np.array(charges)

    def reposition(self,
                   dist):
        #dist=dist+self.height
        for p in self.ch_pos[int(len(self.ch_pos)/2):]:
            p[2]+=-self.distnow + dist
        self.distnow=dist

    def run(self):
        results=[]
        workload=len(self.dist)
        i=0
        for d in self.dist:
            self.reposition(d)
            results.append(np.linalg.norm(self.calc_force_safe()))
            #results.append(np.linalg.norm(np.nansum(self.force[:int(len(self.force)/2)],0)))
            i+=1
            print(str(100*i/workload)+'%')
        plt.plot(self.dist, results)
        return results, self.dist

    @property
    def plot(self):
        self.fig = plt.figure().gca(projection='3d')
        self.fig.scatter(self.ch_pos[:,0],self.ch_pos[:,1],self.ch_pos[:,2],c=self.ch_mag,cmap='RdBu')




class Mag_Ring(Mag_Pull):
    def __init__(self,
                 n=8,
                 diam=1.58*mm,
                 height=1.58*mm,
                 space=0.1*mm,
                 dist=np.linspace(0,30*mm,30),
                 res=10,
                 Br_max=1.3):
        self.n=n
        self.space=space
        r=(diam+space)/(2*np.sin(np.pi/n))
        self.angles=np.linspace(0,2*np.pi,num=n,endpoint=False)
        super().__init__(diam=diam,
                       height=height,
                       dist=dist,
                       res=res,
                       Br_max=Br_max)
        ch_pos=[]
        ch_mag=[]
        for phi in self.angles:
            pos, mag=self.make_points_single((np.sin(phi)*r,np.cos(phi)*r,0))
            ch_pos.append(pos)
            ch_mag.append(mag)
        for phi in self.angles:
            pos, mag=self.make_points_single((np.sin(phi)*r,np.cos(phi)*r,dist[0]+height))
            ch_pos.append(pos)
            ch_mag.append(mag)
        self.ch_pos=np.concatenate(ch_pos,0)
        self.ch_mag=np.concatenate(ch_mag,0)
        area=n*np.pi*self.diam**2/4
        factor=(self.Br_max*area)/(len(self.ch_mag)/4)
        self.ch_mag=self.ch_mag*factor


    def make_points_single(self,
                           center=np.array((0,0,0))):
        points=[]
        charges=[]
        for x in np.linspace(-self.diam/2,self.diam/2,num=self.res):
            for y in np.linspace(-self.diam/2,self.diam/2,num=self.res):
                if np.sqrt(x**2+y**2)<=self.diam/2:
                    points.append((center[0]+x,center[1]+y,self.height/2+center[2]))
                    points.append((center[0]+x,center[1]+y,-self.height/2+center[2]))
                    charges.append(1)
                    charges.append(-1)
        return np.array(points), np.array(charges)

class MSphere_testico(KineticChain):
    def __init__(self,
                 diam=1.58*mm,
                 dist=np.array((0,0,2.5*1.58*mm)),
                 angle=np.linspace(0,np.pi,50),
                 res=3,
                 Br_max=1.3):
        self.diam=diam
        self.angle=angle
        self.dist=dist
        self.Br_max=Br_max/myz
        self.res=res
        self.ch_pos, self.ch_mag=self.generate_point_charges(charge_vec=dist)
        self.force=[]
        self.energy=[]
        self.torque=[]
        self.const=myz/(4*np.pi)#8.9875517923e9#

    def generate_point_charges(self,
                 charge_vec=np.array((1,0,0))):
        mesh = tr.creation.icosphere(subdivisions=self.res, radius=1.0, color=None)
        normals = mesh.vertices
        ch_v=charge_vec/np.linalg.norm(charge_vec)
        points = []
        l=len(normals)
        area=np.array(mesh.area*(self.diam**2)/4)
        out=[]
        for nrm in normals:
            points.append(self.diam*nrm)
            out.append(np.dot(ch_v,nrm) * self.Br_max * area / (l/4))
        for nrm in normals:
            points.append(charge_vec+self.diam*nrm)
            out.append(np.dot(ch_v,nrm) * self.Br_max * area / (l/4))
        return np.array(points), np.array(out)

    def reposition(self,
                   angle):
        #dist=dist+self.height
        R=np.array(((np.cos(angle),0,np.sin(angle)),(0,1,0),(-np.sin(angle),0,np.cos(angle))))
        self.ch_pos, self.ch_mag=self.generate_point_charges(charge_vec=R@self.dist)

    def run(self):
        results=[]
        workload=len(self.angle)
        i=0
        for phi in self.angle:
            self.reposition(phi)
            results.append(np.linalg.norm(self.calc_force_safe()))
            #results.append(np.linalg.norm(np.nansum(self.force[:int(len(self.force)/2)],0)))
            i+=1
            print(str(100*i/workload)+'%')
        plt.plot(self.angle, results/results[0]*100-100)
        plt.xlabel("angle of mesh in rad")
        plt.ylabel("error in %")
        #plt.ylim(())
        return results, self.angle

    @property
    def plot(self):
        self.fig = plt.figure().gca(projection='3d')
        ch_pos=self.ch_pos
        self.fig.scatter(ch_pos[:,0],ch_pos[:,1],ch_pos[:,2],c=self.ch_mag,cmap='RdBu')

class MSphere_testspherical(MSphere_testico):
    def generate_point_charges(self,
                 charge_vec=np.array((1,0,0))):
        normals = self.spherical(res=self.res)
        ch_v=charge_vec/np.linalg.norm(charge_vec)
        points = []
        l=len(normals)
        area=np.pi*self.diam**2
        out=[]
        for nrm in normals:
            points.append(self.diam*nrm)
            out.append(np.dot(ch_v,nrm) * self.Br_max * area / (l/4))
        for nrm in normals:
            points.append(charge_vec+self.diam*nrm)
            out.append(np.dot(ch_v,nrm) * self.Br_max * area / (l/4))
        return np.array(points), np.array(out)

    def spherical(self,
                  res=10):
        out=[]
        for u in np.linspace(0,2*np.pi,num=res):
            for v in np.linspace(0.2, np.pi-0.2, num=res):
                    out.append((np.cos(u) * np.sin(v),np.sin(u) * np.sin(v),np.cos(v)))
        return np.array(out)

class Mag_selfRing(KineticChain):
        def __init__(self,
                     magnet_dim=6 * mm,
                     layers=3,
                     segments=8):
            self.layers = layers
            self.magnet_dim = magnet_dim
            self.segments = segments
            self.radius = magnet_dim * 1.2 * segments/(2*np.pi)
            sequence = [cds.MselfringFrag(2*np.pi*layers/segments,
                                          self.radius,
                                          layers=layers,
                                          dimension=magnet_dim,
                                          mode='up',
                                          charge_vec=np.array((0, 0, 1)))]
            for s in np.arange(1, segments):
                sequence.append(sequence[-1].make_next)
            super().__init__(sequence,  # list of components
                             resolution=2,  # resolution of point charge approxiamtion
                             dimension=self.magnet_dim,
                             upper_boundary=2 * np.pi,
                             lower_boundary=0,
                             fig=0,
                             dtype=np.float32)