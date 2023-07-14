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
from dev_ori_sel_RF import integrator_tf,dynamics,network_full_plastic,run_full_plastic
from dev_ori_sel_RF import data_dir
from dev_ori_sel_RF.tools import misc,update_params_dict

parser = argparse.ArgumentParser()
parser.add_argument('--initver', '-v', help='version',type=int, default=-1)
parser.add_argument('--nrep', '-n', help='version',type=int, default=20)
parser.add_argument('--config', '-c', help='version',type=str, default="test")
args = vars(parser.parse_args())
Version = int(args['initver'])
nrep = int(args['nrep'])
config_name = str(args['config'])

for i in range(nrep):
    config_dict = misc.load_external_params("params_"+config_name)
    config_dict.update({"config_name" : config_name})

    if Version < 0:
        try:
            Version = misc.get_version(data_dir + "ffrec/",version=None,readonly=False)
        except:
            Version = 0
    print("Running with version =",Version)

    if Version > 0:
        config_dict["Wlgn_to4_params"].update({
            "W_mode": "load_from_external",
            "load_from_prev_run" : Version-1})
        config_dict["W4to4_params"].update({
            "Wrec_mode": "load_from_external"})

    run_full_plastic.parameter_sweep_ffrec(Version,config_dict,not_saving_temp=True)

    Version += 1
