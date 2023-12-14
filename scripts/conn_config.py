import sys
import os
sys.path.insert(0, './..')

import argparse

import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt

import plot_func as pf
import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF import probe_RFs,network
from dev_ori_sel_RF.tools import analysis_tools

parser = argparse.ArgumentParser()
parser.add_argument('--maxver', '-v', help='version',type=int, default=8)
parser.add_argument('--nload', '-n', help='version',type=int, default=8)
parser.add_argument('--skip', '-s', help='how many RFs to skip for each plotted',type=int, default=0)
parser.add_argument('--config', '-c', help='version',type=str, default="test")
args = vars(parser.parse_args())
maxver = int(args['maxver'])
nload = int(args['nload'])
skip = int(args["skip"])
config_name = str(args['config'])

config_dict,N4pop,Nlgnpop,Nret,Nlgn,N4,rA = uf.get_network_size(config_name)

Vers = np.round(np.linspace(0,maxver,nload+1)-1).astype(int)[1:]

evals = np.zeros_like(Vers)

for i,Version in enumerate(Vers):
    config_dict.update({
        "config_name" : config_name,
        "system" : "one_layer"
    })
    config_dict["Wlgn_to4_params"].update({
        "W_mode": "load_from_external",
        "load_from_prev_run" : Version})
    n = network.Network(Version,config_dict)
    _,_,_,_,_,_,W4to4 = n.system
    
    evals[i],_ = sparse.linalg.eigs(W4to4,1,which='LR')
    print(i,evals[i])
    
np.savetxt('./../results/conn_evals_{:s}.txt'.format(config_name))