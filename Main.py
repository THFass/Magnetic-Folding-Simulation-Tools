print('test')
import GUI
import numpy as np
import matplotlib.pyplot as plt
import Function_lib as ful




gui=GUI.GUImain()
[energy, torque, angles]=gui.param.get('motion study')
ful.plot_and_save_motion_study(energy,torque,angles,'8Cubes_fold2')

