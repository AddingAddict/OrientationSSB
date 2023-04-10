import sys
import os
sys.path.insert(0, './..')

import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('--initver', '-v', help='version',type=int, default=-1)
parser.add_argument('--nrep', '-n', help='version',type=int, default=20)
args = vars(parser.parse_args())
Version = int(args['initver'])
nrep = int(args['nrep'])

for i in range(nrep):
    config_dict = misc.load_external_params("params_test")

    if Version < 0:
        try:
            Version = misc.get_version(data_dir + "layer4/",version=None,readonly=False)
        except:
            Version = 0
    print("Running with version =",Version)

    config_dict["Wlgn_to4_params"].update({
        "W_mode": "load_from_external",
        "load_from_prev_run" : Version-1})

    run_onelayer.parameter_sweep_layer4(Version,config_dict,not_saving_temp=False)

    Version += 1
