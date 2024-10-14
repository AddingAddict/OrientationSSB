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
parser.add_argument('--n_vis', '-nw', help='number of geniculate viss',type=int, default=60)
parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
args = vars(parser.parse_args())
n_vis = int(args['n_vis'])
n_grid = int(args['n_grid'])
seed = int(args['seed'])

n_bar = 2*n_grid
n_stim = 2
bar_len = 0.99/np.sqrt(2)
res = 1.001*bar_len/n_bar/np.sqrt(2)

rm = 10 # Hz
rb = 0.5 # Hz
dt = 0.1 # s
ibi = 3 # s
dur = 1/(2*n_stim) # s

rs = 0
ro = -1

corr_len = 0.05

rng = np.random.default_rng(seed)

start = time.process_time()

ss,ts = np.meshgrid(np.linspace(0.5/n_bar,1-0.5/n_bar,n_bar),np.linspace(0.5/n_bar,1-0.5/n_bar,n_bar))
ss,ts = ss.flatten(),ts.flatten()
xs,ys = np.meshgrid(np.linspace(0.5/n_grid,1-0.5/n_grid,n_grid),np.linspace(0.5/n_grid,1-0.5/n_grid,n_grid))
xs,ys = xs.flatten(),ys.flatten()

dxs = np.abs(xs[:,None]-xs[None,:])
dxs = np.fmin(dxs,1-dxs)
dys = np.abs(ys[:,None]-ys[None,:])
dys = np.fmin(dys,1-dys)
dists = np.sqrt(dxs**2 + dys**2)
dist_corrs = np.exp(-0.5*(dists/corr_len)**2)

oris = rng.uniform(0,2*np.pi,n_vis)
start_poss = rng.uniform(0,1,(n_vis,n_stim,2))
leading_on = rng.choice([1,-1],(n_vis,n_stim))

pass_times = np.zeros((n_vis,n_stim,n_grid**2))

for idx,ori,pos in zip(range(n_vis),oris,start_poss):
    for sidx in range(n_stim):
        bar_to_box = bf.gen_mov_bar(pos[sidx],ori,bar_len,bar_len)
        bar_pos = bar_to_box(ss,ts)
        pass_times[idx,sidx] = bf.bar_pass_time(bar_pos,np.array([xs,ys]).T,ts,res)

burst_times = np.zeros((n_vis,n_stim,2*n_grid**2))

for sidx in range(n_stim):
    burst_times[:,sidx,:n_grid**2] = ibi*(1/3 - 1/(12*n_stim) + sidx/(3*n_stim) +\
        1/(6*n_stim)*pass_times[:,sidx,:] - \
        1/(12*n_stim)*leading_on[:,sidx,None] + np.arange(n_vis)[:,None])
    burst_times[:,sidx,n_grid**2:] = ibi*(1/3 - 1/(12*n_stim) + sidx/(3*n_stim) +\
        1/(6*n_stim)*pass_times[:,sidx,:] + \
        1/(12*n_stim)*leading_on[:,sidx,None] + np.arange(n_vis)[:,None])
    
print('Generating burst times took',time.process_time() - start,'s\n')

start = time.process_time()

ts = np.linspace(0,ibi*n_vis,int(np.round(ibi*n_vis/dt))+1)

spike_ls = np.zeros((len(ts),2*n_grid**2))
spike_ls += rb * dt

for widx in range(n_vis):
    for cidx in range(2*n_grid**2):
        for sidx in range(n_stim):
            if np.isnan(burst_times[widx,sidx,cidx]):
                continue
            spike_ls[:,cidx] += (rm-rb)*np.exp(-(np.abs(ts-burst_times[widx,sidx,cidx])/(0.5*dur))**2) * dt
spike_ls = np.fmax(1e-5,spike_ls)

spike_rs = np.block([[rs*dist_corrs,ro*dist_corrs],
                     [ro*dist_corrs,rs*dist_corrs]])
np.fill_diagonal(spike_rs,1)
spike_rs = spike_rs[None,:,:] * np.ones((len(ts),1,1))

us = np.linspace(0,1,501)[1:-1]
ns = np.zeros((len(us),2*n_grid**2))

for idx,l in enumerate(spike_ls):
    ns[:] = poisson.ppf(us[:,None],l[None,:])
    ns[:] = zscore(ns,axis=0)

    # lo_bnd = np.einsum('ijk,ijl->jkl',ns,ns[::-1,:,:]) / len(us)
    # up_bnd = np.einsum('ijk,ijl->jkl',ns,ns) / len(us)
    lo_bnd = ns.T@ns[::-1,:] / len(us)
    up_bnd = ns.T@ns / len(us)

    spike_rs[idx] = np.fmax(np.fmin(spike_rs[idx],up_bnd),lo_bnd)

print('Generating spike statistics took',time.process_time() - start,'s\n')

start = time.process_time()

spikes = np.zeros((len(ts),2*n_grid**2),np.ushort)

for idx,l,r in zip(range(len(ts)),spike_ls,spike_rs):
    spikes[idx] = bf.gen_corr_pois_vars(l,r,rng)[:,0]
    
print('Generating spike counts took',time.process_time() - start,'s\n')

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + '2d_lgn_vis_spikes_nw={:d}_ng={:d}/'.format(n_vis,n_grid)
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
