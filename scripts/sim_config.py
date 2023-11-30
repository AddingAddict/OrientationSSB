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
parser.add_argument('--maxver', '-m', help='version',type=int, default=10000)
parser.add_argument('--config', '-c', help='version',type=str, default="test")
parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=6)
args = vars(parser.parse_args())
Version = int(args['initver'])
nrep = int(args['nrep'])
maxver = int(args['maxver'])
config_name = str(args['config'])
gb = int(args['gb'])

for i in range(nrep):
    config_dict = misc.load_external_params("params_"+config_name)
    config_dict.update({"config_name" : config_name})

    if Version < 0:
        try:
            Version = misc.get_version(data_dir + "layer4/",version=None,readonly=False)
        except:
            Version = 0
    print("Running with version =",Version)

    if Version > 0:
        config_dict["Wlgn_to4_params"].update({
            "W_mode": "load_from_external",
            "load_from_prev_run" : Version-1})

    run_onelayer.parameter_sweep_layer4(Version,config_dict,not_saving_temp=True)

    Version += 1
    if Version >= maxver:
        break

if Version < maxver:
    os.system("python runjob_sim_config_ffrec.py -c {:s} -g {:d} -v {:d} -n {:d} -m {:d}".format(config_name,gb,Version,nrep,maxver));
