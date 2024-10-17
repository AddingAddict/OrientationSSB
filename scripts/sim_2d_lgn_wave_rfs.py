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
from scipy.integrate import solve_ivp

from kayser_model_2d import Model

parser = argparse.ArgumentParser()
parser.add_argument('--n_e', '-ne', help='number of excitatory cells',type=int, default=1)
parser.add_argument('--n_i', '-ni', help='number of inhibitory cells',type=int, default=1)
parser.add_argument('--init_iter', '-iit', help='initial iteration number',type=int, default=0)
parser.add_argument('--batch_iter', '-bit', help='number of iterations to run per batch',type=int, default=100)
parser.add_argument('--max_iter', '-mit', help='max iteration number',type=int, default=100)
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
parser.add_argument('--test', '-t', help='test?',type=int, default=0)
args = vars(parser.parse_args())
print(args)
n_e = int(args['n_e'])
n_i = int(args['n_i'])
init_iter = int(args['init_iter'])
batch_iter = int(args['batch_iter'])
max_iter = int(args['max_iter'])
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
test = int(args['test']) > 0

n_batch = 30#26 # number of batches to collect weight changes before adjusting weights
dt_stim = 0.1#0.25 # simulation time between stimuli

max_spike_file = 50 # total number of lgn spike count files

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if test:
    res_dir = res_dir + 'sim_2d_lgn_wave_rfs_ne={:d}_ni={:d}/'.format(n_e,n_i)
else:
    res_dir = res_dir + 'sim_2d_lgn_wave_rfs_ng={:d}_ne={:d}_ni={:d}_sx={:.2f}_se={:.2f}_si={:.2f}_gi={:.1f}/'.format(
        n_grid,n_e,n_i,s_x,s_e,s_i,gain_i)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Define where geniculate wave spikes are saved
lgn_dir = './../results/' + '2d_lgn_vis_spikes_nw={:d}_ns={:d}_ng={:d}/'.format(n_wave,n_stim,n_grid)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    
def init_net(
    n_iter: int,
    ):
    # compute number of LGN cells
    lgn_file = lgn_dir + 'seed={:d}.pkl'.format((n_iter + seed) % max_spike_file)
    print('Opening spike counts from',lgn_file)
    with open(lgn_file, 'rb') as handle:
        lgn_dict = pickle.load(handle)
    lgn_spikes = lgn_dict['spikes']
    n_lgn = lgn_spikes.shape[-1]
    n_x = n_lgn // n_grid**2 // 2
    n_stim = lgn_spikes.shape[0]
    print(n_lgn,'LGN cells')

    if n_iter==0: # starting a new simulation, must initialize the system
        net = Model(n_grid=n_grid,n_e=n_e,n_i=n_i,n_x=n_x,seed=seed,
                    s_x=s_x,s_e=s_e,s_i=s_i,gain_i=gain_i,hebb_wei=hebb_wei,hebb_wii=hebb_wii,
                    rx_wave_start=lgn_spikes[15])#lgn_spikes[26])
    else:
        # load weights, inputs, rates, averages, and learning rates from previous iteration
        with open(res_dir + 'seed={:d}_iter={:d}.pkl'.format(seed,n_iter-1), 'rb') as handle:
            res_dict = pickle.load(handle)
            
        net = Model(n_grid=n_grid,n_e=n_e,n_i=n_i,n_x=n_x,
                    s_x=s_x,s_e=s_e,s_i=s_i,gain_i=gain_i,hebb_wei=hebb_wei,hebb_wii=hebb_wii,
                    init_dict=res_dict)
        
    return net
    
def run_iter(
    n_iter: int,
    net,
    save: bool=False
    ):
    print('\nRunning iteration',n_iter,'\n')
    
    # compute number of LGN cells
    lgn_file = lgn_dir + 'seed={:d}.pkl'.format((n_iter + seed) % max_spike_file)
    print('Opening spike counts from',lgn_file)
    with open(lgn_file, 'rb') as handle:
        lgn_dict = pickle.load(handle)
    lgn_spikes = lgn_dict['spikes']
    n_lgn = lgn_spikes.shape[-1]
    n_stim = lgn_spikes.shape[0]
    print(n_lgn,'LGN cells')

    start = time.process_time()
    for idx in range(n_stim-1):
        rx = lgn_spikes[idx]
        if n_iter<10:
            inh_mult = 0.1 + 0.09*n_iter + 0.09*idx/(n_stim-1)
        else:
            inh_mult = 1.0
        
        # update inputs and rates
        net.update_inps(rx,dt_stim,inh_mult)
        
        # update averages
        net.update_avgs(rx)
        
        # collect weight changes
        net.collect_dw(rx)
        
        if (idx+1)%n_batch==0:
            # update learning rates if during first two iterations
            if n_iter<2:
                net.update_learn_rates()
                # print(net.wex_rate,net.wee_rate,net.wei_rate)
                # print(net.wix_rate,net.wie_rate,net.wii_rate)
            
            if ((idx+1)//n_batch - 1)%5==0:
                print('dwex rms = {:.2e}, dwee rms = {:.2e}, dwei rms = {:.2e}'.format(
                    np.sqrt(np.mean(net.dwex**2)),np.sqrt(np.mean(net.dwee**2)),np.sqrt(np.mean(net.dwei**2))))
                print('dwix rms = {:.2e}, dwie rms = {:.2e}, dwii rms = {:.2e}'.format(
                    np.sqrt(np.mean(net.dwix**2)),np.sqrt(np.mean(net.dwie**2)),np.sqrt(np.mean(net.dwii**2))))
            
            # update weights
            net.sum_norm_dw()
            net.update_weights()
            
            # initialize weight change accumulators
            net.reset_dw()
            
            print('batch {:d} took'.format((idx+1)//n_batch),time.process_time() - start,'s')
            
            start = time.process_time()

    if save:
        res_file = res_dir + 'seed={:d}_iter={:d}.pkl'.format(seed,n_iter)
    
        res_dict = {}

        res_dict['wex'] = net.wex
        res_dict['wix'] = net.wix
        res_dict['wee'] = net.wee
        res_dict['wei'] = net.wei
        res_dict['wie'] = net.wie
        res_dict['wii'] = net.wii
        res_dict['wex_rate'] = net.wex_rate
        res_dict['wix_rate'] = net.wix_rate
        res_dict['wee_rate'] = net.wee_rate
        res_dict['wei_rate'] = net.wei_rate
        res_dict['wie_rate'] = net.wie_rate
        res_dict['wii_rate'] = net.wii_rate
        res_dict['uee'] = net.uee
        res_dict['uei'] = net.uei
        res_dict['uie'] = net.uie
        res_dict['uii'] = net.uii
        res_dict['uee_avg'] = net.uee_avg
        res_dict['uei_avg'] = net.uei_avg
        res_dict['uie_avg'] = net.uie_avg
        res_dict['uii_avg'] = net.uii_avg
        res_dict['rx_avg'] = net.rx_avg

        with open(res_file, 'wb') as handle:
            pickle.dump(res_dict,handle)
            
net = init_net(init_iter)

for n_iter in range(init_iter,init_iter+batch_iter):
    run_iter(n_iter,net,save=(n_iter+1)%10==0)

if init_iter+batch_iter < max_iter:
    os.system("python runjob_sim_2d_lgn_wave_rfs.py " + \
            "-ne {:d} -ni {:d} -iit {:d} -bit {:d} -mit {:d} -s {:d} -nw {:d} -ns {:d} -ng {:d} -sx {:.2f} -se {:.2f} -si {:.2f} -gi {:.1f} -hei {:d} - hii {:d}".format(
            n_e,n_i,init_iter+batch_iter,batch_iter,max_iter,
            seed,n_wave,n_stim,n_grid,
            s_x,s_e,s_i,gain_i,hebb_wei,hebb_wii))
