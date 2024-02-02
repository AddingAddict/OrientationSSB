import sys
import os
sys.path.insert(0, './..')

import argparse

import numpy as np

import matplotlib.pyplot as plt

import plot_func as pf
import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF import probe_RFs,network
from dev_ori_sel_RF.tools import analysis_tools,misc

parser = argparse.ArgumentParser()
parser.add_argument('--maxver', '-v', help='version',type=int, default=8)
parser.add_argument('--nload', '-n', help='version',type=int, default=8)
parser.add_argument('--freq', '-f', help='spatial frequency',type=int, default=-1)
parser.add_argument('--nori', '-o', help='how many orientations to probe',type=int, default=4)
parser.add_argument('--nphs', '-p', help='how many phases to probe',type=int, default=8)
parser.add_argument('--grating', '-gr', help='whether the stimulus should be a full field grating',type=int, default=1)
parser.add_argument('--config', '-c', help='version',type=str, default="test")
args = vars(parser.parse_args())
maxver = int(args['maxver'])
nload = int(args['nload'])
freq = int(args["freq"])
nori = int(args["nori"])
nphs = int(args["nphs"])
grating = int(args["grating"])
config_name = str(args['config'])

config_dict,N4pop,Nlgnpop,Nret,Nlgn,N4,rA = uf.get_network_size(config_name)

Vers = np.round(np.linspace(0,maxver,nload+1)-1).astype(int)[1:]

if freq <= 1:
    config_dict.update({
        "config_name" : config_name,
        "system" : "one_layer"
    })
    config_dict["Wlgn_to4_params"].update({
        "W_mode": "load_from_external",
        "load_from_prev_run" : Vers[-1]})
    n = network.Network(Vers[-1],config_dict)
    _,Wlgnto4,_,_,_,_,_ = n.system

    sd = Wlgnto4[0,...] - Wlgnto4[1,...]
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
    
    freq = pref_lam

for i,Version in enumerate(Vers):
    act,inp,_,_ = probe_RFs.probe_RFs_one_layer(Version,config_name,freqs=np.array([Nlgn*freq,]),
                                                oris=np.linspace(0,np.pi,nori,endpoint=False),
                                                Nsur=nphs,outdir='./../plots/',grating=bool(grating))
    if grating:
        misc.ensure_path('./../results/grating_responses/{:s}/v{:d}_local/'.format(config_name,Version))
        np.save('./../results/grating_responses/{:s}/v{:d}_local/rates_f={:d}'.format(config_name,Version,freq),
                act.flatten())
        np.save('./../results/grating_responses/{:s}/v{:d}_local/inputs_f={:d}'.format(config_name,Version,freq),
                inp.flatten())
    else:
        misc.ensure_path('./../results/inputs/{:s}/v{:d}_local/'.format(config_name,Version))
        np.save('./../results/inputs/{:s}/v{:d}_local/rates_f={:d}'.format(config_name,Version,freq),
                act.flatten())
        np.save('./../results/inputs/{:s}/v{:d}_local/inputs_f={:d}'.format(config_name,Version,freq),
                inp.flatten())