import sys
import os
sys.path.insert(0, './..')

import numpy as np
import tensorflow as tf
import logging
from scipy import linalg

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

import dev_ori_sel_RF
from dev_ori_sel_RF import integrator_tf,dynamics,network,run_onelayer
from dev_ori_sel_RF import data_dir
from dev_ori_sel_RF.tools import misc,update_params_dict

config_dict = misc.load_external_params("params_test")

N4 = config_dict["N4"]
Nlgn = config_dict["Nlgn"]
Nret = config_dict["Nret"]
Nlgnpop = 2
N4pop = config_dict["num_lgn_paths"] // Nlgnpop

for i in range(20):
    Version = misc.get_version(data_dir + "layer4/",version=None,readonly=False)
    print("Running with version =",Version)

    config_dict["Wlgn_to4_params"].update({
        "W_mode": "load_from_external",
        "load_from_prev_run" : Version-1})

    run_onelayer.parameter_sweep_layer4(Version,config_dict,not_saving_temp=False)
