"""----------------------------------------------------------------------------------------------------
GUI
Creates the GUI for handling, optimising, displaying chains
under construction V0.01
    -early Gui test with Tkinter
----------------------------------------------------------------------------------------------------"""
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog

import Chain_data_struct as chs
import Optimisation_data_structure as ops
import Component_data_struct as cps
import Function_lib as flp
import matplotlib.pyplot as plt

import numpy as np
import pickle
import time


class GUImain(tk.Frame):
    def __init__(self):
        self.root = tk.Tk()
        tk.Frame.__init__(self, self.root)
        self.master.title('magnet simulaton')

        # initialise key press events
        self.master.bind("<Key>", self.on_key_press)

        # initiate parameters
        input_scaling=0.001
        self.param = parameter()
        self.param.put('type', 'cube')
        self.param.put('dimension', input_scaling*9)
        self.param.put('mag_type','sphere')
        self.param.put('mag_dimension',input_scaling*7)
        self.param.put('length', 1)
        self.param.put('min_angle', 0)
        self.param.put('max_angle', np.pi)
        self.param.put('mag_vec', np.array([1, 1, 1]))
        self.param.put('mag_shape', 'cube')
        self.param.put('Br_max', 1)
        self.param.put('resolution', 5)
        self.param.put('Metod', 'Bruteforce')
        self.param.put('stepsize', 0)
        self.param.put('nbest', 10)
        self.param.put('starting_pos', 0)
        self.param.put('plt_hinges', True)
        self.param.put('plt_mag', False)  # if true displayes magnetisation direction
        self.param.put('plt_force', False)  # if true displayes net force acting on component
        self.param.put('plt_charges', True)  # if true displayes charges
        self.param.put('plt_auto_zoom', True)
        self.param.put('dsip_mode', 'dispaly')
        self.param.put('color_mode', 'rb')
        self.param.put('edit_mode', True)
        self.param.put('color', 'grey')
        self.param.put('alpha', 0.5)  # transparency of surfices
        self.param.put('axis', np.array([[0, 1], [-0.5, 0.5], [-0.5, 0.5]]))  # limit for x,y and z axis
        self.dtype = np.float32
        self.chain = self.new_chain('A')
        self.optim = ops.Optimization(self.chain, self.chain.cost_fun)
        self.fig = self.plot
        self.save_sequence = []
        self.animation = []

        # make menu bar
        self.menubar = tkinter.Menu(self.root)
        self.root.config(menu=self.menubar)
        # file submenu
        fileMenu = tk.Menu(self.menubar)
        fileMenu.add_command(label="New", command=self.new_chain)
        fileMenu.add_command(label="load", command=self.load_chain)
        fileMenu.add_command(label="save", command=self.save_chain)
        fileMenu.add_command(label="Export results", command=self.save_results)
        self.menubar.add_cascade(label="File", menu=fileMenu)
        # Edit submenu
        editMenu = tk.Menu(self.menubar)
        editMenu.add_command(label="add", command=self.add_multiple)
        set_angle_Menu = tk.Menu(self.menubar)
        set_angle_Menu.add_command(label="set all angles", command=self.set_angles)
        set_angle_Menu.add_command(label="set single angle", command=self.set_angle_x)
        editMenu.add_cascade(label="set angle", menu=set_angle_Menu)
        set_mag_Menu = tk.Menu(self.menubar)
        set_mag_Menu.add_command(label="set all mag", command=self.set_mag)
        set_mag_Menu.add_command(label="set single mag", command=self.set_mag_x)
        editMenu.add_cascade(label="set mag", menu=set_mag_Menu)
        editMenu.add_command(label="edit component", command=self.setup_component)
        editMenu.add_command(label="plot setup", command=self.setup_plot)
        self.menubar.add_cascade(label="Edit", menu=editMenu)
        # view submenu
        viewMenu = tk.Menu(self.menubar)
        viewMenu.add_command(label="display", command=self.display_mode)
        viewMenu.add_command(label="edit", command=self.edit_mode)
        viewMenu.add_command(label="expert edit", command=self.expert_edit_mode)
        viewMenu.add_command(label="Feed Folding", command=self.view_feed_fold)
        viewMenu.add_command(label="Free Folding", command=self.view_free_fold)
        viewMenu.add_command(label="info screen", command=self.info_screen)
        self.menubar.add_cascade(label="View", menu=viewMenu)
        # Simulate submenu
        simMenu = tk.Menu(self.menubar)
        simMenu.add_command(label="Newton Dynamics", command=self.newton_dynamic)
        simMenu.add_command(label="Lagrange Dynamics", command=self.lagrange_dynamic)
        simMenu.add_command(label="exaustive search", command=self.exaustive_search)
        simMenu.add_command(label="Optimisation", command=self.optimise)
        simMenu.add_command(label="Motion STudy", command=self.motion_study)
        self.menubar.add_cascade(label="Simulate", menu=simMenu)
        # help
        self.menubar.add_command(label='help', command=self.help)
        # test butten to be changed to test specific things
        self.menubar.add_command(label="test", command=self.test)
        # exit button
        self.menubar.add_command(label="Exit", command=self._quit)

        # user inteface
        self.mode = tk.StringVar(value='edit')
        display_button = tk.Radiobutton(self.root, text="Display", command=self.display_mode, variable=self.mode,
                                        indicatoron=False, value="display", width=8)
        display_button.grid(row=0, column=1)
        edit_button = tk.Radiobutton(self.root, command=self.edit_mode, text="Edit", variable=self.mode,
                                     indicatoron=False, value="edit", width=8)
        edit_button.grid(row=0, column=3)
        e_edit_button = tk.Radiobutton(self.root, command=self.edit_mode, text="Fast Edit", variable=self.mode,
                                       indicatoron=False, value="exedit", width=8)
        e_edit_button.grid(row=0, column=5)

        # buttons to add components
        # Creating a photoimage object to use image
        up_pic = tk.PhotoImage(file=r"grafic\up.png").subsample(2)
        self.button_up = tk.Button(master=self.root, command=self.add_a, image=up_pic)
        self.button_up.grid(row=1, column=3)
        enter_pic = tk.PhotoImage(file=r"grafic\enter.png").subsample(5)
        self.button_next = tk.Button(master=self.root, command=self.enter, image=enter_pic)
        self.button_next.grid(row=1, column=5)
        left_pic = tk.PhotoImage(file=r"grafic\left.png").subsample(2)
        self.button_up = tk.Button(master=self.root, command=self.add_b, image=left_pic)
        self.button_up.grid(row=3, column=1)
        center_pic = tk.PhotoImage(file=r"grafic\center.png").subsample(2)
        self.button_up = tk.Button(master=self.root, command=self.add_s, image=center_pic)
        self.button_up.grid(row=3, column=3)
        right_pic = tk.PhotoImage(file=r"grafic\right.png").subsample(2)
        self.button_up = tk.Button(master=self.root, command=self.add_c, image=right_pic)
        self.button_up.grid(row=3, column=5)
        down_pic = tk.PhotoImage(file=r"grafic\down.png").subsample(2)
        self.button_up = tk.Button(master=self.root, command=self.add_d, image=down_pic)
        self.button_up.grid(row=5, column=3)
        remove_pic = tk.PhotoImage(file=r"grafic\undo2.png").subsample(5)
        self.button_up = tk.Button(master=self.root, command=self.remove_last, image=remove_pic)
        self.button_up.grid(row=5, column=5)

        self.switch_variable = tk.StringVar(value="fold")
        self.fold_button = tk.Radiobutton(self.root, command=self.fold_unfold, text="Fold",
                                          variable=self.switch_variable,
                                          indicatoron=False, value="fold", width=5)
        self.fold_button.grid(row=6, column=1)
        self.unfold_button = tk.Radiobutton(self.root, command=self.fold_unfold, text="Unfold",
                                            variable=self.switch_variable,
                                            indicatoron=False, value="unfold", width=5)
        self.unfold_button.grid(row=7, column=1)

        # info box
        self.textbox = tk.Text(self.root, height=30, width=50)
        yscroll = tk.Scrollbar(self.root)
        xscroll = tk.Scrollbar(self.root, orient='horizontal')
        self.textbox.grid(row=0, column=6, rowspan=10)
        yscroll.grid(row=0, column=7, rowspan=10)
        xscroll.grid(row=11, column=6)
        yscroll.config(command=self.textbox.yview)
        xscroll.config(command=self.textbox.xview)
        self.textbox.config(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.update_textbox()

        tk.mainloop()

    def on_key_press(self, event):
        # function triggers when a key is pressed
        if event.char in ['w', 'W', 8, '8']:
            self.add_a()
        elif event.char in ['a', 'A', 4, '4']:
            self.add_b()
        elif event.char in ['z', 'Z', 2, '2', 'x', 'X']:
            self.add_c()
        elif event.char in ['d', 'D', 6, '6']:
            self.add_d()
        elif event.char in ['s', 'S', 5, '5']:
            self.add_s()
        elif event.keycode in [13, 32]:
            self.enter()
        elif event.keycode == 83:
            self.save_chain()
        elif event.keycode == 76:
            self.load_chain()
        elif event.keycode == 27:
            self._quit()
        elif event.keycode == 8:
            self.remove_last()
        # print("you pressed {}".format(event.char))
        # print(event)

    def _quit(self):
        self.chain.destroy
        self.quit()  # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent

    def test(self):
        for name, item in zip(self.param.name, self.param.value):
            print(name)
            print(str(item))

    def save_chain(self):
        filename = tk.filedialog.asksaveasfilename(title="Select save file", defaultextension='.opt',
                                                   filetypes=(("optimization files", "*.opt"), ("all files", "*.*")))
        file = open(filename, 'wb')
        pickle.dump(self.optim, file)
        file.close()

    def load_chain(self):
        filename = tk.filedialog.askopenfilename(title="Select save file",
                                                 filetypes=(("optimization files", "*.opt"), ("all files", "*.*")))
        file = open(filename, 'rb')
        self.optim = pickle.load(file)
        self.chain = self.optim.obj
        self.plot
        file.close()

    def save_results(self):
        print('todo')

    def set_angles(self):
        #sets the angle of one specific component
        #input for component and angle
        vars=[]
        for i in range(len(self.chain.comp)):
            vars.append('component'+str(i))
            values.append(self.chain.comp[i].get_angle)
        inp = self.input_window(title='Chain Properties', vars=vars, states=values)
        self.chain.set_angles(np.array(inp))
        self.redo_chain
        self.update_textbox()


    def set_angle_x(self):
        #sets the angle of one specific component
        #input for component and angle
        vars=['component index','new angle']
        values = [1,0]
        inp = self.input_window(title='Chain Properties', vars=vars, states=values)
        self.chain.set_anglex(float(inp[1]),int(inp[0]))
        self.redo_chain
        self.update_textbox()

    def view_feed_fold(self):
        for comp in self.chain.comp:
            self.animate(dof=[comp],
                         steps=10,
                         frame_time=1)
        print('todo')

    def view_free_fold(self):
        self.display_mode()
        self.animate(steps=10, frame_time=1)

    def info_screen(self):
        self.update_textbox()

    def animate(self,
                dof='all',  # defines which angles of the chain get animated
                steps=10,  # defines the number of steps for that chain to be animated
                frame_time=0):  # time for each frame. if <=0 frame steps by button pressed
        # todo:implement saving animation and or pics
        if dof == 'all':
            dof = self.chain.comp
        angles = []
        for comp in dof:
            angles.append(comp.get_angle)
            comp.set_angle(0)
        for frame_angle in np.linspace(np.zeros_like(angles), angles, steps):
            self.plot_frame(frame_time=frame_time)
            for index in range(len(dof)):
                dof[index].set_angle(frame_angle[index])
        self.plot_frame(frame_time=frame_time)

    def plot_frame(self, frame_time=1, save_to_file=''):
        self.plot
        if save_to_file != '':
            self.chain.fig.savefig(save_to_file)
        plt.pause(frame_time)

    @property
    def redo_chain(self):
        # recreates chain of same length and same angles
        self.plot
        print('todo')

    def setup_component(self):
        vars = ['type', 'dimension', 'min_angle', 'max_angle']
        values = self.param.get_multi(vars)
        newval = self.input_window(title='Chain Properties', vars=vars, states=values)
        self.redo_chain
        self.update_textbox()

    def set_mag(self):
        vars = ['mag_vec', 'mag_shape','mag_dimension', 'Br_max', 'resolution']
        values = self.param.get_multi(vars)
        newval = self.input_window(title='Magnet Properties', vars=vars, states=values)
        self.param.set_multi(vars, newval)
        self.chain.magnetise(mag_vec=self.param.get('mag_vec'),
                             Brmax=self.param.get('Br_max'),
                             resolution=self.param.get('resolution'))
        self.update_textbox()
        self.plot


    def set_mag_x(self):
        vars = ['mag_vec', 'mag_shape', 'Br_max', 'resolution','index']
        values = self.param.get_multi(vars[:-1])
        values.append(1)
        newval = self.input_window(title='Magnet Properties', vars=vars, states=values)
        self.param.set_multi(vars[:-1], newval[:-1])
        print(newval)
        self.chain.magnetise_x(mag_vec=self.param.get('mag_vec'),
                             Brmax=self.param.get('Br_max'),
                             resolution=self.param.get('resolution'),
                             index=newval[-1])
        self.update_textbox()
        self.plot

    def setup_plot(self):
        vars = ['starting_pos', 'plt_hinges', 'plt_mag', 'plt_force', 'plt_charges', 'plt_auto_zoom', 'color', 'alpha',
                'axis']
        values = self.param.get_multi(vars)
        newval = self.input_window(title='Plot Setup Properties', vars=vars, states=values)
        self.param.set_multi(vars, newval)
        self.plot
        self.update_textbox()

    def newton_dynamic(self):
        print('todo')

    def lagrange_dynamic(self):
        print('todo')

    def exaustive_search(self):
        print('todo')

    def optimise(self):
        print('todo')

    def motion_study(self):
        # plots coulomb energy over the folding of a specific angle
        self.display_mode()
        var = ['component', 'start', 'stop', 'step']
        state = [2,self.param.get('min_angle'), self.param.get('max_angle'), 100]
        [index, start, stop, steps] = self.input_window('Motion Study', var, state)
        index = int(index)
        energy = []
        torque = []
        angles = np.linspace(float(start), float(stop), num=int(steps))
        for phi in angles:
            self.chain.set_anglex(phi, index)
            self.chain.update_lists
            energy.append(self.chain.calc_kin_f)
            torque.append(self.chain.calc_torque_x(index))
        print('motion study done')
        self.param.put('motion study', [np.array(energy), np.array(torque), np.array(angles)])

    def help(self):
        print('todo')

    def input_window(self, title, vars, states=0):
        # generates window where list of vars can be input and returns these upon clicking a button
        window = tk.Toplevel(self.root)
        window.wm_title(title)
        if states == 0:
            states = np.zeros(len(vars))
        outvars = []
        for it in range(len(vars)):
            l = tk.Label(window, text=vars[it])
            l.pack()
            if type(states[it]) in [np.ndarray, list, tuple]:
                outvars.append(flp.array_edit_widget(window, states[it]))
            else:
                outvars.append(tk.StringVar(window, value=str(states[it])))
                tk.Entry(window, textvariable=outvars[-1]).pack()
        ok = tk.Button(window, text='OK', command=window.destroy)
        ok.pack()
        self.root.wait_window(window)
        out = []
        for it in range(len(outvars)):
            out.append(outvars[it].get())
            out[-1] = flp.sanatise(out[-1])
        self.update_textbox()
        return out

    @property
    def plot(self):
        # plots chain with set options
        # todo: integrate plot options menu
        plt_hinges = self.param.get('plt_hinges')  # if true displayes hinges
        plt_mag = self.param.get('plt_mag')  # if true displayes magnetisation direction
        plt_force = self.param.get('plt_force')  # if true displayes net force acting on component
        plt_charges = self.param.get('plt_charges')  # if true displayes charges
        plt_auto_zoom = self.param.get('plt_auto_zoom')
        alpha = self.param.get('alpha')  # transparency of surfices
        color_mode = self.param.get('color_mode')
        edit_mode = self.param.get('edit_mode')
        print(edit_mode)
        print(color_mode)
        #axis = [[0, 1], [-0.5, 0.5], [-0.5, 0.5]]  # limit for x,y and z axis

        return self.chain.plot(plt_hinges=plt_hinges,
                               plt_mag=plt_mag,
                               plt_force=plt_force,
                               plt_auto_zoom=plt_auto_zoom,
                               plt_charges=plt_charges,
                               edit_mode=edit_mode,
                               color_mode=color_mode,
                               alpha=alpha)
        # self.chain.plot(plt_hinges=plt_hinges,             #if true displayes hinges
        # plt_mag=plt_mag,                     #if true displayes magnetisation direction
        # plt_force=plt_force,                   #if true displayes net force acting on component
        # plt_charges=plt_charges,                 #if true displayes charges
        # plt_auto_zoom=plt_auto_zoom,
        # edit_mode=edit_mode,
        # alpha=alpha,                         #transparency of surfices
        # axis=axis) #limit for x,y and z axis)

    def fold(self):
        self.switch_variable.set('fold')
        self.fold_unfold()

    def fold_unfold(self):
        if self.switch_variable.get() == 'unfold':
            self.save_sequence = self.chain.get_angles
            self.chain.set(0)
            self.plot
            # self.chain.plot(plt_hinges=True,
            #                edit_mode=True)
            self.mode.set('display')
        elif self.switch_variable.get() == 'fold':
            if len(self.save_sequence) == len(self.chain.comp):
                self.chain.set(self.save_sequence)
                self.plot
                self.mode.set('edit')
            else:
                print('error: chain length in storage unequal to present chain length')

    def modify_chain(self, comp_type, angle):
        if self.mode.get() == 'exedit':
            self.chain.append(comp_type=comp_type,
                              phi_0=angle)
        elif self.mode.get() == 'edit':
            self.add(comp_type=comp_type,
                     angle=angle)
        if self.mode.get() in ['exedit', 'edit']:
            self.chain.magnetise(mag_vec=self.param.get('mag_vec'),
                                 Brmax=self.param.get('Br_max'),
                                 resolution=self.param.get('resolution'))
            self.plot
            self.update_textbox()

    def add(self, comp_type='A', angle=np.pi):
        if len(self.chain.comp) <= 2:
            self.chain = self.new_chain(comp_type=comp_type,
                                        angle=angle)
            self.chain.fig = self.fig
        else:
            self.chain.remove()
            oldangle = self.chain.comp[-1].get_angle
            self.chain.remove()
            self.chain.append(
                comp_type=comp_type,
                phi_0=oldangle,
                resolution=self.param.get('resolution'),
                dtype=self.dtype)
            self.chain.append(
                comp_type=cps.CubeA,
                phi_0=angle,
                resolution=self.param.get('resolution'),
                dtype=self.dtype)

    def new_chain(self, comp_type='A', angle=0):
        if type(comp_type) == str or type(comp_type) == int:
            if comp_type in ('D', 'd', 4, '4'):
                comp_type = cps.CubeD
            elif comp_type in ('C', 'c', 3, '3'):
                comp_type = cps.CubeC
            elif comp_type in ('B', 'b', 2, '2'):
                comp_type = cps.CubeB
            else:
                comp_type = cps.CubeA

        mag_vec = self.param.get('mag_vec')
        mag_vec = self.param.get('Br_max') * mag_vec / np.linalg.norm(mag_vec)
        mag_type = self.param.get('mag_type')
        mag_dim = self.param.get('mag_dimension')
        chain = chs.KineticChain.from_types([comp_type],
                                            dimension=self.param.get('dimension'),
                                            inital_angles=0,
                                            resolution=self.param.get('resolution'),
                                            #mag_type=mag_type,
                                            #mag_dim=mag_dim,
                                            charge_vec=mag_vec,
                                            dtype=self.dtype)
        chain.append(
            comp_type=cps.CubeA,
            phi_0=angle,
            resolution=self.param.get('resolution'),
            dtype=self.dtype)
        chain.magnetise(mag_vec=mag_vec,
                        resolution=self.param.get('resolution'))
        return chain

    def add_a(self):
        # print(self.param.get('max_angle'))
        self.modify_chain(comp_type='A',
                          angle=self.param.get('max_angle'))

    def add_b(self):
        self.modify_chain(comp_type='B',
                          angle=self.param.get('max_angle'))

    def add_c(self):
        self.modify_chain(comp_type='C',
                          angle=self.param.get('max_angle'))

    def add_d(self):
        self.modify_chain(comp_type='D',
                          angle=self.param.get('max_angle'))

    def add_s(self):
        self.modify_chain(comp_type='A',
                          angle=self.param.get('min_angle'))

    def enter(self):
        self.chain.append(comp_type='a', phi_0=0)
        mag_vec = self.param.get('mag_vec')
        mag_vec = self.param.get('Br_max') * mag_vec / np.linalg.norm(mag_vec)
        self.chain.magnetise(mag_vec=mag_vec)
        self.plot
        self.update_textbox()

    def remove_last(self, n=1):
        self.chain.remove()
        self.plot
        self.update_textbox()

    def add_multiple(self):
        n = self.g_inp_window('Add', ['how many to add?'])
        varnames = []
        states = []
        for it in range(n):
            varnames.append('Component ' + str(it) + ' type')
            varnames.append('Component ' + str(it) + ' angle')
            states.append('A')
            states.append('0')
        varvals = self.g_inp_window('Add Conponents', varnames, states=states)
        for it in range(int(len(varvals) / 2)):
            self.cain.append(type=varvals[2 * it], phi_0=varvals[2 * it + 1])

    def display_mode(self):
        self.param.set('edit_mode', False)
        self.remove_last()
        self.plot

    def edit_mode(self):
        if self.param.get('edit_mode') == False:
            self.add_s()
        self.param.set('edit_mode', True)
        self.plot

    def expert_edit_mode(self):
        if self.param.get('edit_mode') == False:
            self.add_s()
        self.param.set('edit_mode', True)
        self.plot

    def update_textbox(self):
        self.textbox.config(state=tk.NORMAL)
        self.textbox.delete(1.0, tk.END)
        self.textbox.insert(tk.END, self.generate_info_text)

    @property
    def generate_info_text(self):
        out_text = 'Number of components: ' + str(len(self.chain.comp)) + '\n'
        energy = 'Coulomb Energy: ' + str(self.chain.calc_energy) + '\n'
        length = 'Chain length: ' + str(len(self.chain.comp)) + '\n'
        [angle_factor, dist_factor] = self.chain.movement_factor
        factor = 'Angular Factor: ' + str(angle_factor) + ' |Distance Factor: ' + str(dist_factor) + '\n'
        sequence = 'Cain sequence: '
        for comp in self.chain.comp:
            sequence += comp.type + '; '
        sequence += '\n'
        return out_text + \
               energy + \
               length + \
               factor + \
               sequence + \
               str(self.param)


class parameter():
    def __init__(self, name=[], value=[]):
        if len(name) != len(value):
            print('inputs need same length')
        self.name = []
        self.value = []
        for (na, val) in zip(name, value):
            self.put(na, val)

    @property
    def iter(self):
        return self.value

    def put(self, name, value):
        self.name.append(name)
        self.value.append(value)

    def get_multi(self, names):
        if type(names) in [list, tuple, np.ndarray]:
            out = []
            for name in names:
                out.append(self.get(name))
        else:
            out = self.get(names)
        return out

    def get(self, name):
        try:
            value = self.value[self.name.index(name)]
            return value
        except:
            print('error: value ' + name + ' not in list')
            return np.NAN

    def set_multi(self, names, values):
        if len(names) != len(values):
            print('error: inputs must be the same length')
        else:
            for name, value in zip(names, values):
                self.set(name, value)

    def set(self, name, value):
        try:
            self.value[self.name.index(name)] = value
        except:
            print('error: value ' + name + ' not in list')

    def __str__(self):
        outstr = 'parameter:\n'
        for (name, value) in zip(self.name, self.value):
            outstr += name + ': ' + str(value) + '\n'
        return outstr

    def __repr__(self):
        return 'parameters'
