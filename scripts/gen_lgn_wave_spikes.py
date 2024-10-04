import sys
import os
try:
    import pickle5 as pickle
except:
    import pickle
sys.path.insert(0, './..')

import argparse
import time

import numpy as np
from scipy.stats import poisson,zscore

import burst_func as bf

parser = argparse.ArgumentParser()
parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=20)
parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
args = vars(parser.parse_args())
n_wave = int(args['n_wave'])
n_grid = int(args['n_grid'])
seed = int(args['seed'])

n_bar = 2*n_grid
bar_len = 0.99/np.sqrt(2)
res = 1.001*bar_len/n_bar/np.sqrt(2)

rm = 10 # Hz
rb = 1 # Hz
dt = 0.25 # s
ibi = 26 # s
dur = 15 # s

rs = 0.35
ro = 0.05

rng = np.random.default_rng(seed)

start = time.process_time()

ss,ts = np.meshgrid(np.linspace(0.5/n_bar,1-0.5/n_bar,n_bar),np.linspace(0.5/n_bar,1-0.5/n_bar,n_bar))
ss,ts = ss.flatten(),ts.flatten()
xs,ys = np.meshgrid(np.linspace(0.5/n_grid,1-0.5/n_grid,n_grid),np.linspace(0.5/n_grid,1-0.5/n_grid,n_grid))
xs,ys = xs.flatten(),ys.flatten()

in_rf = np.sqrt((xs-0.5)**2 + (ys-0.5)**2) < 0.5/np.sqrt(2)
xs,ys = xs[in_rf],ys[in_rf]
n_in_rf = len(xs)

oris = rng.uniform(0,2*np.pi,n_wave)

pass_times = np.zeros((n_wave,n_in_rf))

for idx,ori in enumerate(oris):
    bar_to_box = bf.gen_mov_bar(0.5*(np.ones(2) - bar_len*np.array([np.cos(ori),np.sin(ori)])),
                                ori,bar_len,bar_len)
    bar_pos = bar_to_box(ss,ts)
    pass_times[idx] = bf.bar_pass_time(bar_pos,np.array([xs,ys]).T,ts,res)

burst_times = ibi*(0.25+0.5*pass_times + np.arange(n_wave)[:,None])
    
print('Generating burst times took',time.process_time() - start,'s\n')

start = time.process_time()

ts = np.linspace(0,ibi*n_wave,int(np.round(ibi*n_wave/dt))+1)

spike_ls = np.zeros((len(ts),n_in_rf))
spike_ls += rb

for widx in range(n_wave):
    for cidx in range(n_in_rf):
        spike_ls[:,cidx] += (rm-rb)*np.exp(-(np.abs(ts-burst_times[widx,cidx])/(0.5*dur))**2.4) * dt
spike_ls = np.concatenate([spike_ls,spike_ls],-1)
spike_ls = np.fmax(1e-5,spike_ls)

spike_rs = np.block([[rs*np.ones((n_in_rf,n_in_rf)),ro*np.ones((n_in_rf,n_in_rf))],
                     [ro*np.ones((n_in_rf,n_in_rf)),rs*np.ones((n_in_rf,n_in_rf))]])
np.fill_diagonal(spike_rs,1)
spike_rs = spike_rs[None,:,:] * np.ones((len(ts),1,1))

us = np.linspace(0,1,501)[1:-1]
ns = np.zeros((len(us),2*n_in_rf))

for idx,l in enumerate(spike_ls):
    ns[:] = poisson.ppf(us[:,None],l[None,:])
    ns[:] = zscore(ns,axis=0)

    # lo_bnd = np.einsum('ijk,ijl->jkl',ns,ns[::-1,:,:]) / len(us)
    # up_bnd = np.einsum('ijk,ijl->jkl',ns,ns) / len(us)
    lo_bnd = ns.T@ns[::-1,:] / len(us)
    up_bnd = ns.T@ns / len(us)

    spike_rs[idx] = np.fmax(np.fmin(spike_rs[idx],up_bnd),lo_bnd)

spike_rs = np.fmax(np.fmin(spike_rs,up_bnd),lo_bnd)
    
print('Generating spike statistics took',time.process_time() - start,'s\n')

start = time.process_time()

spikes = np.zeros((len(ts),2*n_in_rf),np.ushort)

for idx,l,r in zip(range(len(ts)),spike_ls,spike_rs):
    spikes[idx] = bf.gen_corr_pois_vars(l,r,rng)[:,0]
    
print('Generating spike counts took',time.process_time() - start,'s\n')

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'lgn_wave_spikes_nw={:d}_ng={:d}/'.format(n_wave,n_grid)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

res_dict['burst_times'] = burst_times
# res_dict['spike_ls'] = spike_ls
# res_dict['spike_rs'] = spike_rs
res_dict['spikes'] = spikes

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
