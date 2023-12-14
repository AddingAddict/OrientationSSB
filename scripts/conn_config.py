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

evals = np.zeros(len(Vers))

for i,Version in enumerate(Vers):
    with np.load('./../dev_ori_sel_RF/data/layer4/'+config_name+'/v'+str(Version)+'/y_v'+str(Version)+'.npz') as data:
        W4to4 = data['Wrec']
    
    evals[i] = np.real(sparse.linalg.eigs(W4to4,1,which='LR')[0][0])
    print(i,evals[i])
    
np.savetxt('./../results/conn_evals_{:s}.txt'.format(config_name),evals)