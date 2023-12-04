import sys
import os
sys.path.insert(0, './..')

import argparse

import numpy as np

import matplotlib.pyplot as plt

import plot_func as pf
import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF import probe_RFs,network_ffrec
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

Vers = np.round(np.linspace(0,maxver,nload+1)-1).astype(int)

config_dict["Wlgn_to4_params"].update({
    "W_mode": "load_from_external",
    "load_from_prev_run" : maxver})
if "2pop" in config_dict["W4to4_params"]["Wrec_mode"]:
    config_dict["W4to4_params"].update({
        "Wrec_mode": "load_from_external2pop"})
else:
    config_dict["W4to4_params"].update({
        "Wrec_mode": "load_from_external"})
n = network_ffrec.Network(maxver,config_dict)
_,Wlgn_to_4,_,_,_,_,_,_,_ = n.system

sd = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
sd = sd.reshape((N4,N4,Nlgn,Nlgn))

xs,ys = np.meshgrid(np.arange(Nlgn),np.arange(Nlgn))
xs[xs > Nlgn//2] -= Nlgn
ys[ys > Nlgn//2] -= Nlgn
ls = np.sqrt(xs**2+ys**2)
ls_bins = np.digitize(ls,np.arange(0,Nlgn//2)+0.5)

sd_fft = np.fft.fft2(sd,axes=(2,3))
sd_fft_mag = np.zeros((N4,N4,Nlgn//2))
for i in range(N4):
    for j in range(N4):
        for k in range(Nlgn//2):
            sd_fft_mag[i,j,k] = np.max(np.abs(sd_fft[i,j,ls_bins == k]))
sd_fft_mag = np.mean(sd_fft_mag,axis=(0,1))
pref_lam = np.argmax(sd_fft_mag)

for i,Version in enumerate(Vers):
    probe_RFs.probe_RFs_ffrec(Version,config_name,freqs=np.array([Nlgn*pref_lam,]))