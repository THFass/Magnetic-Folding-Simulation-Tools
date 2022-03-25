"""----------------------------------------------------------------------------------------------------
Optimisation data structure
Offers classes and methods to optimise and handle kinematic chains
Version 0.0
    -empty
----------------------------------------------------------------------------------------------------"""
import numpy as np
import timeit as ti
import copy

class Optimization:
    def __init__(self,
                 obj,
                 cost_fun,
                 **kwargs):
        self.obj=obj
        self.cost_fun=cost_fun
        self.results=[]
        self.kwargs=kwargs


def Coulomb_energy(chain,**kwargs):
    return chain.calc_energy

def Coulomb_energy_distance_factor(chain,factor, initial_state,**kwargs):
    return chain.calc_energy+factor*(chain.distance-initial_state)

class Set_BruteForce(Optimization):
    def __init__(self,
                 obj,
                 cost_fun,
                 states=np.array((0,np.pi)),
                 nbest=10,
                 go=True,
                 ignore_first=True,
                 **kwargs):
        super().__init__(obj,cost_fun,**kwargs)

        self.nbest=nbest
        self.ignore_first=ignore_first
        length = self.obj.length
        if len(np.shape(states))==1:            #inpot is one dimensional and needs to be applied for all degrees of freedom
            states = np.reshape(np.tile(states,length),(length,len(states)))

        self.states=states
        if go:
            self.BruteForce

    def update_progress(self,progress):
        #todo
        #displays progress of simulation
        print(progress)
        return 0

    def iterate(self,state):
        #iterates state with all combination in states
        if self.ignore_first:
            for current,possible in enumerate(self.states[1:]):

                index=tuple(possible).index(state[current])
                if index+1<len(possible):       #can entry be raised by one index?
                    state[current+1]=possible[index+1]   #if yes, entry will be raised by one index
                    break                       #and exit loop
                else:
                    state[current+1]=possible[0]         #if not, entry is reset and next entry will be tried
            return state
        else:
            for current,possible in enumerate(self.states):

                index=tuple(possible).index(state[current])
                if index+1<len(possible):       #can entry be raised by one index?
                    state[current]=possible[index+1]   #if yes, entry will be raised by one index
                    break                       #and exit loop
                else:
                    state[current]=possible[0]         #if not, entry is reset and next entry will be tried
            return state

    @property
    def BruteForce(self):
        #optimises chain energy using bruteforce and trying all possible combinations given in states on all degres of freedom
        start_time=ti.timeit()
        combinations = 1
        if self.ignore_first:
            first=1
        else:
            first=0
        for st in self.states[first :]:
            combinations *= len(st) #calculates number of possible states

        value=[]
        for i in range(self.nbest):
            value.append((np.inf,[]))#np.NAN*np.ones(self.obj.length)))
        best=np.array(value,dtype=[('score',np.float),('state',np.ndarray)])
        print(type(best))
        state = copy.deepcopy(self.states[:,0])
        for index in range(combinations):
            self.obj.set(state)
            if self.obj.check_overlap:
                break
            score=self.cost_fun(self.obj,
                               **self.kwargs)
            #print(score)
            #self.update_progress(index/combinations)
            if score<best['score'][-1] and np.isfinite(score):
                print('hit!')
                best['score'][-1] = score
                best['state'][-1] = list(copy.deepcopy(state))
                print(best)
                best.sort(order='score')
            state = self.iterate(state)
        end_time=ti.timeit()
        print('total optimisation time:')
        print(end_time-start_time)
        self.results=best

    def plot(self,
                   index=0,
                   **args): #limit for x,y and z axis):
        #plots optmisation results
        self.obj.set(self.results['state'][index])
        self.obj.plot(args)

    def set_best(self,index=0):
        self.obj.set(self.results['state'][index])
        print(self.results['score'][index])


class QuasiNewton(Optimization):
    def __init__(self,
                 obj,
                 start,
                 step,
                 min,
                 max,
                 cost_fun,
                 go=True):
        self.start=start
        self.min=min
        self.max=max
        self.step=step
        super.__init__(obj,cost_fun)
        if go:
            self.optimise()
    #todo


class HingePerHinge(Optimization):
    def __init__(self,
                 obj,
                 cost_fun,
                 start=0,
                 step=0.01*np.pi,
                 min=0,
                 max=2*np.pi,
                 go=True):
        self.start=start
        self.min=min
        self.max=max
        self.step=step
        super.__init__(obj,cost_fun)
        if go:
            self.optimise()

class DynamicSym:
    def __init__(self,chain,integration):
        self.chain=chain
        self.integration=integration
        self.result=[]
    #todo
