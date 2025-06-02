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

n_bar = int(np.round(2*n_grid))
bar_len = 0.99/np.sqrt(2)
res = 1.001*bar_len/n_bar/np.sqrt(2)

rm = 16 # Hz
rb = 0 # Hz
dt = 0.1 # s
ibi = 14.4 # s
dur = 14.4 # s

rs = 0.00
ro = -0.1

corr_len = 0.1

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

oris = rng.uniform(0,2*np.pi,n_wave)
start_poss = rng.uniform(0,1,(n_wave,2))

spike_ls = np.zeros((int(np.round(ibi*n_wave/dt)),2*n_grid**2))
spike_ls += rb * dt

width = 0.25

for widx in range(n_wave):
    for tidx in range(int(np.round(dur/dt))):
        time_idx = int(np.round(ibi/dt))*widx + tidx
        edge_fact = 0.5 if tidx==0 or tidx==int(np.round(dur/dt))-1 else 1
        vel = bar_len * np.array([np.cos(oris[widx]),np.sin(oris[widx])])
        bar_to_box = bf.gen_mov_bar(start_poss[widx]+(tidx/int(dur/dt))*vel,
            oris[widx],bar_len,bar_len)
        bar_pos = bar_to_box(ss,ts)
        pass_times = np.abs(bf.bar_pass_time(bar_pos,np.array([xs,ys]).T,ts,res) - 0.5)
        for cidx in range(n_grid**2):
            if np.isnan(pass_times[cidx]):
                continue
            spike_ls[time_idx,cidx] += edge_fact*(rm-rb)*\
                np.exp(-(np.abs(pass_times[cidx])/(width))**2) * dt
            spike_ls[time_idx,n_grid**2+cidx] += edge_fact*(rm-rb)*\
                np.exp(-(np.abs(pass_times[cidx])/(width))**2) * dt
spike_ls = np.fmax(1*dt,spike_ls)

print('Generating spike statistics took',time.process_time() - start,'s\n')

start = time.process_time()

spikes = np.zeros((int(np.round(ibi*n_wave/dt)),2*n_grid**2),np.ushort)

for idx,l in enumerate(spike_ls):
    r = np.block([[rs*dist_corrs,ro*dist_corrs],
                  [ro*dist_corrs,rs*dist_corrs]])
    spikes[idx] = bf.gen_corr_pois_vars(l,r,rng)[:,0]
    
print('Generating spike counts took',time.process_time() - start,'s\n')

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + '2d_lgn_spont_spikes_nw={:d}_ng={:d}/'.format(n_wave,n_grid)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# res_dict['spike_ls'] = spike_ls
# res_dict['spike_rs'] = spike_rs
res_dict['spikes'] = spikes

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
