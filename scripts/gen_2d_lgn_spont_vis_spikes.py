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
parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=15)
parser.add_argument('--n_stim', '-ns', help='number of light/dark sweeping bars',type=int, default=2)
parser.add_argument('--n_shrink', '-nh', help='factor by which to shrink stimuli',type=float, default=1.0)
parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
args = vars(parser.parse_args())
n_wave = int(args['n_wave'])
n_vis = 4*n_wave
n_stim = int(args['n_stim'])
n_shrink = args['n_shrink']
n_grid = int(args['n_grid'])
seed = int(args['seed'])

n_bar = int(np.round(2*n_grid))
bar_len = 0.99/np.sqrt(2)
res = 1.001*bar_len/n_bar/np.sqrt(2)

dt = 0.1 # s

spnt_rb = 0 # Hz
spnt_rm = 16 # Hz
spnt_ibi = 14.4 # s
spnt_dur = 14.4 # s

vis_rb = 12 # Hz
vis_rm = 40 # Hz
vis_ibi = 3.6 # s
vis_dur = 1.2 # s
vis_stim_dur = vis_dur/n_stim # s

spnt_rs = 0.00
spnt_ro = -0.10
vis_rs = -0.20
vis_ro = -1.00

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

spike_ls = np.zeros((int(np.round(spnt_ibi*n_wave/dt)),2*n_grid**2))
times = np.arange(len(spike_ls)) * dt
sweeping = (np.mod(np.round(times/dt),np.round(vis_ibi/dt)) >= np.round(vis_dur/dt)) \
    & (np.mod(np.round(times/dt),np.round(vis_ibi/dt)) <= 2*np.round(vis_dur/dt))

spike_ls += spnt_rb * dt

spnt_oris = rng.uniform(0,2*np.pi,n_wave)
spnt_start_poss = rng.uniform(0,1,(n_wave,2))

spnt_width = 0.25

for widx in range(n_wave):
    for tidx in range(int(np.round(spnt_dur/dt))):
        time_idx = int(np.round(spnt_ibi/dt))*widx + tidx
        if sweeping[time_idx]:
            continue
        edge_fact = 1
        vel = bar_len * np.array([np.cos(spnt_oris[widx]),np.sin(spnt_oris[widx])])
        bar_to_box = bf.gen_mov_bar(spnt_start_poss[widx]+(tidx/int(spnt_dur/dt))*vel,
            spnt_oris[widx],bar_len,bar_len)
        bar_pos = bar_to_box(ss,ts)
        pass_times = np.abs(bf.bar_pass_time(bar_pos,np.array([xs,ys]).T,ts,res) - 0.5)
        for cidx in range(n_grid**2):
            if np.isnan(pass_times[cidx]):
                continue
            spike_ls[time_idx,cidx] += edge_fact*spnt_rm*\
                np.exp(-(np.abs(pass_times[cidx])/(spnt_width))**2) * dt
            spike_ls[time_idx,n_grid**2+cidx] += edge_fact*spnt_rm*\
                np.exp(-(np.abs(pass_times[cidx])/(spnt_width))**2) * dt

vis_oris = rng.uniform(0,2*np.pi,n_vis)
vis_start_poss = rng.uniform(0,1,(n_vis,n_stim,2))
vis_leading_on = rng.choice([1,-1],(n_vis,n_stim))

vis_offset = 0.125/n_shrink
vis_width = 0.3/n_shrink

for widx in range(n_vis):
    for sidx in range(n_stim):
        for tidx in range(int(np.round(vis_stim_dur/dt))+1):
            time_idx = int(np.round(vis_ibi/dt))*widx + int(np.round((vis_ibi-1)/2/dt)) \
                + int(np.round(vis_stim_dur/dt))*sidx + tidx - 1
            edge_fact = 0.5 if tidx==0 or tidx==int(np.round(vis_stim_dur/dt)) else 1
            vel = bar_len * np.array([np.cos(vis_oris[widx]),np.sin(vis_oris[widx])])
            bar_to_box = bf.gen_mov_bar(vis_start_poss[widx,sidx]+(tidx/int(vis_stim_dur/dt)+vis_offset*vis_leading_on[widx,sidx])*vel,
                vis_oris[widx],bar_len,bar_len)
            bar_pos = bar_to_box(ss,ts)
            n_pass_times = np.abs(bf.bar_pass_time(bar_pos,np.array([xs,ys]).T,ts,res) - 0.5)
            bar_to_box = bf.gen_mov_bar(vis_start_poss[widx,sidx]+(tidx/int(vis_stim_dur/dt)-vis_offset*vis_leading_on[widx,sidx])*vel,
                vis_oris[widx],bar_len,bar_len)
            bar_pos = bar_to_box(ss,ts)
            f_pass_times = np.abs(bf.bar_pass_time(bar_pos,np.array([xs,ys]).T,ts,res) - 0.5)
            for cidx in range(n_grid**2):
                spike_ls[time_idx,cidx] += edge_fact*vis_rb*dt
                if np.isnan(n_pass_times[cidx]):
                    continue
                spike_ls[time_idx,cidx] += edge_fact*vis_rm*\
                    np.exp(-(np.abs(n_pass_times[cidx])/(vis_width))**3) * dt
                spike_ls[time_idx,n_grid**2+cidx] -= 0.8*edge_fact*vis_rm*\
                    np.exp(-(np.abs(n_pass_times[cidx])/(vis_width))**3) * dt
            for cidx in range(n_grid**2):
                spike_ls[time_idx,n_grid**2+cidx] += edge_fact*vis_rb*dt
                if np.isnan(f_pass_times[cidx]):
                    continue
                spike_ls[time_idx,n_grid**2+cidx] += edge_fact*vis_rm*\
                    np.exp(-(np.abs(f_pass_times[cidx])/(vis_width))**3) * dt
                spike_ls[time_idx,cidx] -= 0.8*edge_fact*vis_rm*\
                    np.exp(-(np.abs(f_pass_times[cidx])/(vis_width))**3) * dt
spike_ls = np.fmax(1*dt,spike_ls)

print('Generating spike statistics took',time.process_time() - start,'s\n')

start = time.process_time()

spikes = np.zeros((int(np.round(spnt_ibi*n_wave/dt)),2*n_grid**2),np.ushort)

for idx,l in enumerate(spike_ls):
    if sweeping[idx]:
        r = np.block([[vis_rs*dist_corrs,vis_ro*dist_corrs],
                    [vis_ro*dist_corrs,vis_rs*dist_corrs]])
    else:
        r = np.block([[spnt_rs*dist_corrs,spnt_ro*dist_corrs],
                    [spnt_ro*dist_corrs,spnt_rs*dist_corrs]])
    spikes[idx] = bf.gen_corr_pois_vars(l,r,rng)[:,0]
    
print('Generating spike counts took',time.process_time() - start,'s\n')

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + '2d_lgn_spont_vis_spikes_nw={:d}_ns={:d}_nh={:.2f}_ng={:d}/'.format(n_wave,n_stim,n_shrink,n_grid)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# res_dict['spike_ls'] = spike_ls
# res_dict['spike_rs'] = spike_rs
res_dict['spikes'] = spikes

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
