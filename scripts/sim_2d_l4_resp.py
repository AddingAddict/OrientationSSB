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

import analyze_func as af
from kayser_model_2d import Model

parser = argparse.ArgumentParser()
parser.add_argument('--n_e', '-ne', help='number of excitatory cells',type=int, default=1)
parser.add_argument('--n_i', '-ni', help='number of inhibitory cells',type=int, default=1)
parser.add_argument('--load_iter', '-lit', help='2d L4 kayser iteration number to load',type=int, default=50)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--s_x', '-sx', help='feedforward arbor decay length',type=float, default=0.08)
parser.add_argument('--s_e', '-se', help='excitatory recurrent arbor decay length',type=float, default=0.08)
parser.add_argument('--s_i', '-si', help='inhibitory recurrent arbor decay length',type=float, default=0.08)
parser.add_argument('--gain_i', '-gi', help='gain of inhibitory cells',type=float, default=2.0)
parser.add_argument('--hebb_wei', '-hei', help='whether wei has Hebbian learning rule',type=int, default=0)
parser.add_argument('--hebb_wii', '-hii', help='whether wii has Hebbian learning rule',type=int, default=0)
parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=60)#20)
parser.add_argument('--n_stim', '-ns', help='number of light/dark sweeping bars',type=int, default=2)
parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
parser.add_argument('--f_vis', '-fv', help='spatial frequency of grating',type=float, default=5.0)
parser.add_argument('--n_ori', '-no', help='number of grating orientations',type=int, default=4)
parser.add_argument('--n_phs', '-np', help='number of grating phases',type=int, default=8)
parser.add_argument('--test', '-t', help='test?',type=int, default=0)
args = vars(parser.parse_args())
print(args)
n_e = int(args['n_e'])
n_i = int(args['n_i'])
load_iter = int(args['load_iter'])
seed = int(args['seed'])
s_x = args['s_x']
s_e = args['s_e']
s_i = args['s_i']
gain_i = args['gain_i']
hebb_wei = int(args['hebb_wei']) > 0
hebb_wii = int(args['hebb_wii']) > 0
n_wave = int(args['n_wave'])
n_stim = int(args['n_stim'])
n_grid = int(args['n_grid'])
f_vis = args['f_vis']
n_ori = int(args['n_ori'])
n_phs = int(args['n_phs'])
test = int(args['test']) > 0

max_spike_file = 50 # total number of lgn spike count files
dt_stim = 0.5#0.25 # simulation time between stimuli

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if test:
    res_dir = res_dir + 'sim_2d_l4_resp_ne={:d}_ni={:d}/'.format(n_e,n_i)
else:
    res_dir = res_dir + 'sim_2d_l4_resp_ng={:d}_ne={:d}_ni={:d}_sx={:.2f}_se={:.2f}_si={:.2f}_gi={:.1f}/'.format(
        n_grid,n_e,n_i,s_x,s_e,s_i,gain_i)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Define where L4 weights are saved
l4_dir = './../results/' + 'sim_2d_lgn_wave_rfs_ng={:d}_ne={:d}_ni={:d}_sx={:.2f}_se={:.2f}_si={:.2f}_gi={:.1f}/'.format(
        n_grid,n_e,n_i,s_x,s_e,s_i,gain_i)
if not os.path.exists(l4_dir):
    os.makedirs(l4_dir)

# Define where geniculate wave spikes are saved
lgn_dir = './../results/' + '2d_lgn_vis_spikes_nw={:d}_ns={:d}_ng={:d}/'.format(n_wave,n_stim,n_grid)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    
def init_net(
    n_iter: int,
    ):    
    # load weights, inputs, rates, averages, and learning rates from previous iteration
    with open(l4_dir + 'seed={:d}_iter={:d}.pkl'.format(seed,n_iter-1), 'rb') as handle:
        res_dict = pickle.load(handle)
    n_x = 1
        
    net = Model(n_grid=n_grid,n_e=n_e,n_i=n_i,n_x=n_x,
                s_x=s_x,s_e=s_e,s_i=s_i,gain_i=gain_i,hebb_wei=hebb_wei,hebb_wii=hebb_wii,
                init_dict=res_dict)
    
    return net

net = init_net(load_iter)

res_dict = {}

# Compute simulated feedforward responses to grating stimuli
start = time.process_time()

xs,ys = np.meshgrid(np.linspace(0.5/n_grid,1-0.5/n_grid,n_grid),np.linspace(0.5/n_grid,1-0.5/n_grid,n_grid))

grates = np.zeros((n_ori,n_phs,n_grid,n_grid))

for ori_idx in range(n_ori):
    ori = ori_idx/n_ori*np.pi
    kx,ky = np.round(f_vis*np.cos(ori)),np.round(f_vis*np.sin(ori))
    print(kx,ky,np.sqrt(kx**2+ky**2))
    for phs_idx in range(n_phs):
        phs = phs_idx/n_phs*2*np.pi+1e-12
        # grates[ori_idx,phs_idx] = 2*(np.heaviside(np.cos(phs+2*np.pi*(kx*xs+ky*ys)),0.5) - 0.5)
        grates[ori_idx,phs_idx] = 0.5*(1+np.cos(phs+2*np.pi*(kx*xs+ky*ys)))
        
grate_rx = np.concatenate((grates.reshape((n_ori,n_phs,n_grid**2)),1-grates.reshape((n_ori,n_phs,n_grid**2))),axis=-1)

resp_e = np.einsum('ij,klj->kli',net.wex,grate_rx).reshape((n_ori,n_phs,n_grid,n_grid))
resp_i = np.einsum('ij,klj->kli',net.wix,grate_rx).reshape((n_ori,n_phs,n_grid,n_grid))

resp_e -= np.mean(resp_e,axis=(0,1),keepdims=True)
resp_i -= np.mean(resp_i,axis=(0,1),keepdims=True)

resp_e = np.fmax(resp_e,0)
resp_i = np.fmax(resp_i,0)

opm_e,mr_e = af.calc_OPM_MR(resp_e.transpose(2,3,0,1))
opm_i,mr_i = af.calc_OPM_MR(resp_i.transpose(2,3,0,1))

res_dict['ff_resp_e'] = resp_e
res_dict['ff_opm_e'] = opm_e
res_dict['ff_mr_e'] = mr_e
res_dict['ff_resp_i'] = resp_i
res_dict['ff_opm_i'] = opm_i
res_dict['ff_mr_i'] = mr_i
            
print('Calculating feedforward grating evoked responses took',time.process_time() - start,'s')

# Compute firing rates to grating stimuli
start = time.process_time()

resp_e = np.zeros((n_ori,n_phs,n_grid,n_grid))
resp_i = np.zeros((n_ori,n_phs,n_grid,n_grid))

for ori_idx in range(n_ori):
    for phs_idx in range(n_phs):
        # update inputs and rates
        net.update_inps(grate_rx[ori_idx,phs_idx],dt_stim)
        
        resp_e[ori_idx,phs_idx] = np.fmax(net.uee - net.uei,0).reshape((n_grid,n_grid))
        resp_i[ori_idx,phs_idx] = np.fmax(net.uie - net.uii,0).reshape((n_grid,n_grid))

opm_e,mr_e = af.calc_OPM_MR(resp_e.transpose(2,3,0,1))
opm_i,mr_i = af.calc_OPM_MR(resp_i.transpose(2,3,0,1))

res_dict['resp_e'] = resp_e
res_dict['opm_e'] = opm_e
res_dict['mr_e'] = mr_e
res_dict['resp_i'] = resp_i
res_dict['opm_i'] = opm_i
res_dict['mr_i'] = mr_i
            
print('Calculating grating evoked responses took',time.process_time() - start,'s')

# Save grating evoked response data

res_file = res_dir + 'grate_evoked_resp.pkl'

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)

# Compute firing rates to geniculate wave stimuli

for lgn_idx in range(max_spike_file):
    start = time.process_time()
    res_dict = {}
    
    # compute number of LGN cells
    lgn_file = lgn_dir + 'seed={:d}.pkl'.format(lgn_idx)
    print('Opening spike counts from',lgn_file)
    with open(lgn_file, 'rb') as handle:
        lgn_dict = pickle.load(handle)
    lgn_spikes = lgn_dict['spikes']
    n_stim = lgn_spikes.shape[0]
    
    resp_e = np.zeros((n_stim,n_grid,n_grid))
    resp_i = np.zeros((n_stim,n_grid,n_grid))

    start = time.process_time()
    for idx in range(n_stim):
        rx = lgn_spikes[idx]
        
        # update inputs and rates
        net.update_inps(rx,dt_stim)
        
        resp_e[idx] = np.fmax(net.uee - net.uei,0).reshape((n_grid,n_grid))
        resp_i[idx] = np.fmax(net.uie - net.uii,0).reshape((n_grid,n_grid))
        
    res_dict['resp_e'] = resp_e
    res_dict['resp_i'] = resp_i
        
    # Save geinculate wave evoked response data
    res_file = res_dir + 'lgn_wave_seed={:d}.pkl'.format(lgn_idx)

    with open(res_file, 'wb') as handle:
        pickle.dump(res_dict,handle)
            
    print('Calculating geniculate wave evoked responses for seed {:d} took'.format(lgn_idx),time.process_time() - start,'s')