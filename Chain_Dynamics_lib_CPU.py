import numpy as np
import matplotlib as plt
from scipy.spatial.transform.rotation import Rotation as rot
import Funklib_CPU as fnl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class KinChain:
    def __init__(self,
                 n=10,
                 mass=1,
                 inertia=(2/5)*np.eye(3),
                 Q=False,
                 w=np.array((0,0,0)),
                 fts=fnl.fts_none,
                 g=-np.array((0,0,9.8)),
                 rao=None,
                 rbo=None,
                 frik=0,
                 dt=0.001,
                 dtype=np.float32):
        self.n = n
        self.fts=fts
        if type(mass) in (int,float):
            mass = mass*np.eye(3*n)
        self.M=mass*np.eye(3*n)
        inertia = fnl.anyrep(inertia,n)
        self.inertia = np.array(fnl.matlist2diag(inertia))
        #calc component from length and angle
        if Q:
            self.Q=fnl.matlist2diag(Q)
        else:
            self.Q=np.eye(int(3*n))
        if type(rao)==type(None):
            self.rao=-np.array(fnl.blockrep(-np.array((0, 0, 1)) / 2, n, dtype=dtype))
            self.rbo=-rao
        else:
            self.rao=rao
            self.rbo=rbo
        self.ra=[]
        self.rb=[]

        self.z_0 = np.array([self.Q,np.zeros((3*n,3*n))])
        self.set_state(self.z_0)
        print(self.M)
        print(g)
        self.fg = np.array(self.M @ fnl.blockrep(g,n,dtype=dtype))
        self.dtype = np.float32
        self.dt = dt
        self.dt0 = dt
        #Clac E,B,C-
        self.B=np.array(np.diag(np.ones(3*(n-1)),k=-3))
        A=np.eye(3*n)-self.B
        self.Ci=np.linalg.inv(np.transpose(A))
        self.D=np.transpose(self.B)
        self.E=self.Ci @ np.array(self.M) @ np.linalg.inv(A)
        #initiate W
        self.frik=frik
        self.times=[]
        self.results=[]
        print('chain intiated')

    def solver(self,
               z_0=None,
               t_0=0,
               t_end=5,
               dt=None):
        if z_0 is None:
            z_0=self.z_0
        if dt is None:
            dt=self.dt
        t=np.arange(t_0,t_end,dt)
        self.results = fnl.rk4(self.DGLZ,z_0,t)
        self.times=t
        return self.results

    def DGLZ(self,z,t):
        dz=np.zeros_like(z)
        dz[0]=z[1] @ z[0]
        #dz[0]=z[1] @ z[0]/np.sqrt(np.max(np.abs(z[1])))
        #dz[0]=z[1]
        #z[1]=self.func(z[0],z[1])
        dz[1]=fnl.xdiag(self.func(z))
        return dz

    def func(self,state):
        fs,ts,state=self.fts(state)
        #print(state)
        self.set_state(state)
        RB,RD = self.getRBRD
        left = self.inertia - RD @ self.E @ RB
        right = RD @ (self.E @ (self.B @ self.Wx @ self.Wx @ self.rb - self.Wx @ self.Wx @ self.ra) - self.Ci @ fs) + ts
        #A=self.B @ self.Wx @ self.Wx @ self.rb - self.Wx @ self.Wx @ self.ra
        #right=RD @ (self.E @ A- self.Ci @ fs)+ ts
        return np.linalg.solve(left,right)

    @property
    def update_ft(self):
        return 'ft is constant'


    def set_state(self,state):
        self.trans=state[0]
        self.Wx=state[1]
        self.w=fnl.undiag(state[1],self.n)
        self.ra=self.trans@self.rao
        self.rb=self.trans@self.rbo

    @property
    def getRBRD(self):
        Ra=fnl.xdiag(self.ra,dtype=self.dtype)
        Rb=fnl.xdiag(self.rb,dtype=self.dtype)
        return [Ra -self.B @ Rb,
                Ra - Rb @ self.D]

    @property
    def calc_dt(self):
        self.dt = self.dt0/(np.linalg.norm(self.w) + 1)
        #print(self.dt)

    @property
    def get_coords(self):
        rs=fnl.split(-self.ra)
        out=[rs[0]]
        last=rs[0]
        for r in rs[1:]:
            out.append(out[-1]+last+r)
            last=r
        return np.reshape(out,(self.n*3))

    @property
    def get_vel(self):
        #calculates velocity of each component
        velrela = fnl.split(-self.Wx @ self.ra) #velocity of center of mass in respective frame
        velrelb = fnl.split(self.Wx @ self.rb)  # velocity of center of mass in respective frame
        velabs = [velrela[0]]
        for i in np.arange(1,len(velrela)):
            velabs.append(velrela[i]+velabs[-1]+velrelb[i-1])
        velabs=np.array(velabs)
        return np.reshape(velabs,(self.n*3))

    @property
    def get_kin(self):
        kinrot = np.sum(self.inertia@np.square(self.w)/2)
        mass = self.M[0,0]
        kinvel = np.sum(np.array(self.M) @ np.square(self.get_vel))/2
        return kinvel, kinrot

    @property
    def get_pot(self):
        x = self.get_coords
        p = np.dot(-self.fg,x)
        return p

    @property
    def get_state(self):
        return [self.trans,self.Wx]

    @property
    def get_en(self):
        kinv=0
        kinw=0
        pot=0


    def set(self,state):
        self.set_phi(state[0])
        self.w = state[1]

    @property
    def plot_en(self):
        #prepare plot
        fig = plt.figure()
        #get pot and kin
        pot = []
        kinv = []
        kinw = []
        vel = []
        for res in self.results:
            self.set_state(state=res)
            pot.append(self.get_pot)
            kv, kw = self.get_kin
            kinv.append(kv)
            kinw.append(kw)
            vel.append(np.sqrt(np.sum(np.square(self.get_vel))))
        pot=np.array(pot)
        pot=pot-pot[0]
        kinv=np.array(kinv)
        kinw=np.array(kinw)
        plt.plot(pot,color = 'blue')
        plt.plot(kinv+kinw, color = 'red')
        plt.plot(kinv,color='yellow')
        plt.plot(kinw,color='green')
        plt.plot(pot+kinv+kinw,color = 'black')

        return vel,kinv,kinw,pot

    @property
    def get_en(self):
        #prepare plot
        fig = plt.figure()
        #get pot and kin
        pot = []
        kinv = []
        kinw = []
        vel = []
        for res in self.results:
            self.set_state(state=res)
            pot.append(self.get_pot)
            kv, kw = self.get_kin
            kinv.append(kv)
            kinw.append(kw)
            vel.append(np.sqrt(np.sum(np.square(self.get_vel))))
        pot=np.array(pot)
        pot=pot-pot[0]
        kinv=np.array(kinv)
        kinw=np.array(kinw)
        return vel,kinv,kinw,pot

    @property
    def plot_en_II(self):
        fig = plt.figure()
        r=[]
        w=[]
        for res in self.results:
            self.set(state=res)
            r.append(self.ra)
            w.append(self.w)
        pot, kin = fnl.energy(rl=r,wl=w,m=self.M[0,0],I=self.inertia[0,0])
        plt.plot(pot,color = 'blue')
        plt.plot(kin, color = 'red')
        plt.plot(pot+kin,color = 'black')

    @property
    def fs(self):
        return self.fg

    @property
    def ts(self):
        return 0#-self.frik*self.w

    def plot(self,
             fig=None,
             z=None,
             col='r'):
        if fig == None:
            fig = fnl.Dfigure()
        if type(z) == type(None):
            z = self.get_state
        self.set_state(z)
        xyz=fnl.toxyz(self.get_coords)
        fig.scatter(xyz,col=col)
        return fig

    def plot_res(self,
                 dt=0):
        fig=fnl.Dfigure()
        fig.scatter(np.zeros(3),col='black')
        self.plot(fig=fig,z=self.results[0],col='black')
        for res in self.results[1:-1]:
            self.plot(fig=fig,z=res,col='r')
            if dt>0:
                fig.sleep(dt)
        self.plot(fig=fig,z=self.results[-1],col='blue')


    @property
    def get_lens(self):
        out=[]
        for i in range(self.n):
            out.append(np.linalg.norm(self.ra[3*i:3*i+3]))
        return out
"""
    def from_chain(self,
                   chain,
                   g=np.array((0,0,-9.8)),
                   fric=0,
                   mass=1,
                   inertia=(2/5)*np.eye(3),
                   minmax=(0,np.pi)):
        self.chain=chain
        self.minmax=minmax
        self.__init__(self,
                 n=len(chain.comp),
                 mass=mass,
                 inertia=(2/5)*np.eye(3),
                 trans=rot.from_rotvec((0,0,0)),
                 w=np.array((0,0,0)),
                 g=g,
                 r=np.array((0,0,1))/2,
                 frik=frik,
                 dt=0.001,
                 dtype=np.float32):"""

def fts_none(state):
    return 0,0