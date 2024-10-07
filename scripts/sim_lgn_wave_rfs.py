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

parser = argparse.ArgumentParser()
parser.add_argument('--n_e', '-ne', help='number of excitatory cells',type=int, default=16)
parser.add_argument('--n_i', '-ni', help='number of inhibitory cells',type=int, default=4)
parser.add_argument('--n_iter', '-niter', help='current iteration number',type=int, default=0)
parser.add_argument('--max_iter', '-miter', help='max iteration number',type=int, default=100)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=20)
parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
args = vars(parser.parse_args())
n_e = int(args['n_e'])
n_i = int(args['n_i'])
n_iter = int(args['n_iter'])
max_iter = int(args['max_iter'])
seed = int(args['seed'])
n_wave = int(args['n_wave'])
n_grid = int(args['n_grid'])

# Define constants

# postsynaptic weight normalization
wff_sum = 1.0
wee_sum = 0.125
wie_sum = 0.5
wei_sum = 2.25
wii_sum = 0.25

# presynaptic weight normalization
w4e_sum = 0.25
w4i_sum = 9.25

# maximum allowed weights
max_wff = 0.018
max_wee = 0.5*0.125
max_wie = 0.5*0.5
max_wei = 0.5*2.25
max_wii = 0.5*0.25

# RELU gains
gain_e = 1
gain_i = 2
gain_mat = np.diag(np.concatenate((gain_e*np.ones(n_e),gain_i*np.ones(n_i))))

n_batch = 26 # number of batches to collect weight changes before adjusting weights
dt_stim = 0.25 # simulation time between stimuli
dt_dyn = 0.01 # timescale for voltage dynamics
a_avg = 1/30 # smoothing factor for average inputs
targ_dw_rms = 0.0001 # target root mean square weight change

max_spike_file = 50 # total number of lgn spike count files

# Define functions

# voltage dynamics function
def ode_func(
    t: float,
    u: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    ):
    r = np.fmax(u,0)
    np.matmul(gain_mat,r,out=r)
    np.matmul(w,r,out=r)
    r[:] += h - u
    return r / dt_dyn
    
# integrate rate dynamics and update inputs
def update_inps(
    wex: np.ndarray,
    wix: np.ndarray,
    wee: np.ndarray,
    wei: np.ndarray,
    wie: np.ndarray,
    wii: np.ndarray,
    uee: np.ndarray,
    uei: np.ndarray,
    uie: np.ndarray,
    uii: np.ndarray,
    rx: np.ndarray,
    int_time: float=dt_stim,
    ):
    # calculate feedforward inputs
    he,hi = wex@rx,wix@rx
    h = np.concatenate((he,hi))
    
    # create full recurrent weight matrix
    w = np.block([[wee,-wei],[wie,-wii]])
    
    # integrate dynamics
    sol = solve_ivp(ode_func,[0,int_time],np.concatenate((uee-uei,uie-uii)),args=(w,h),t_eval=[int_time],method='RK23')
    
    # compute rates and update inputs
    r = np.fmax(sol.y[:,-1],0)
    np.matmul(gain_mat,r,out=r)
    re,ri = r[:n_e],r[n_e:]
    uee[:] = wee@re + he
    uie[:] = wie@re + hi
    uei[:] = wei@ri
    uii[:] = wii@ri
    
# update input and rate averages
def update_avgs(
    uee: np.ndarray,
    uei: np.ndarray,
    uie: np.ndarray,
    uii: np.ndarray,
    rx: np.ndarray,
    uee_avg: np.ndarray,
    uei_avg: np.ndarray,
    uie_avg: np.ndarray,
    uii_avg: np.ndarray,
    rx_avg: np.ndarray,
    ):
    uee_avg[:] += a_avg * (uee - uee_avg)
    uei_avg[:] += a_avg * (uei - uei_avg)
    uie_avg[:] += a_avg * (uie - uie_avg)
    uii_avg[:] += a_avg * (uii - uii_avg)
    rx_avg[:] += a_avg * (rx - rx_avg)
    
# collect weight changes in a batch
def collect_dw(
    dwex: np.ndarray,
    dwix: np.ndarray,
    dwee: np.ndarray,
    dwei: np.ndarray,
    dwie: np.ndarray,
    dwii: np.ndarray,
    wex_rate: np.ndarray,
    wix_rate: np.ndarray,
    wee_rate: np.ndarray,
    wei_rate: np.ndarray,
    wie_rate: np.ndarray,
    wii_rate: np.ndarray,
    uee: np.ndarray,
    uei: np.ndarray,
    uie: np.ndarray,
    uii: np.ndarray,
    rx: np.ndarray,
    uee_avg: np.ndarray,
    uei_avg: np.ndarray,
    uie_avg: np.ndarray,
    uii_avg: np.ndarray,
    rx_avg: np.ndarray,
    ):
    ue = uee - uei
    ui = uie - uii
    ue_avg = uee_avg - uei_avg
    ui_avg = uie_avg - uii_avg
    
    dwex += wex_rate * np.outer(ue - ue_avg,rx - rx_avg)
    dwix += wix_rate * np.outer(ui - ui_avg,rx - rx_avg)
    dwee += wee_rate * np.outer(ue - ue_avg,ue - ue_avg)
    dwie += wie_rate * np.outer(ui - ui_avg,ue - ue_avg)
    dwei += wei_rate * (np.outer(np.fmax(uei - uei_avg,0),np.fmax(ui - ui_avg,0)) -\
                        np.outer(np.fmax(ue - ue_avg,0),np.fmax(ui - ui_avg,0)))
    dwii += wii_rate * (np.outer(np.fmax(uii - uii_avg,0),np.fmax(ui - ui_avg,0)) -\
                        np.outer(np.fmax(ui - ui_avg,0),np.fmax(ui - ui_avg,0)))
    
# update learning rates
def update_learn_rates(
    dwex: np.ndarray,
    dwix: np.ndarray,
    dwee: np.ndarray,
    dwei: np.ndarray,
    dwie: np.ndarray,
    dwii: np.ndarray,
    wex_rate: np.ndarray,
    wix_rate: np.ndarray,
    wee_rate: np.ndarray,
    wei_rate: np.ndarray,
    wie_rate: np.ndarray,
    wii_rate: np.ndarray,
    ):
    wex_rate *= targ_dw_rms / np.sqrt(np.mean(dwex**2))
    wix_rate *= targ_dw_rms / np.sqrt(np.mean(dwix**2))
    wee_rate *= targ_dw_rms / np.sqrt(np.mean(dwee**2))
    wei_rate *= targ_dw_rms / np.sqrt(np.mean(dwei**2))
    wie_rate *= targ_dw_rms / np.sqrt(np.mean(dwie**2))
    wii_rate *= targ_dw_rms / np.sqrt(np.mean(dwii**2))
    
# update weights with collected changes, then clip and normalize weights
def update_weights(
    wex: np.ndarray,
    wix: np.ndarray,
    wee: np.ndarray,
    wei: np.ndarray,
    wie: np.ndarray,
    wii: np.ndarray,
    dwex: np.ndarray,
    dwix: np.ndarray,
    dwee: np.ndarray,
    dwei: np.ndarray,
    dwie: np.ndarray,
    dwii: np.ndarray,
    ):
    wex += dwex
    wix += dwix
    wee += dwee
    wei += dwei
    wie += dwie
    wii += dwii
    
    # alternate clipping, presynaptic normalization, and postsynaptic normalization
    for _ in range(4):
        # clip weights
        np.clip(wex,1e-8,max_wff,out=wex)
        np.clip(wix,1e-8,max_wff,out=wix)
        np.clip(wee,1e-8,max_wee,out=wee)
        np.clip(wei,1e-8,max_wei,out=wei)
        np.clip(wie,1e-8,max_wie,out=wie)
        np.clip(wii,1e-8,max_wii,out=wii)
        
        # presynaptic normalization
        e_sum = np.sum(wee,axis=0,keepdims=True) + np.sum(wie,axis=0,keepdims=True)
        wee *= w4e_sum / e_sum
        wie *= w4e_sum / e_sum
        i_sum = np.sum(wei,axis=0,keepdims=True) + np.sum(wii,axis=0,keepdims=True)
        wei *= w4i_sum / i_sum
        wii *= w4i_sum / i_sum
        
        # clip weights
        np.clip(wex,0,max_wff,out=wex)
        np.clip(wix,0,max_wff,out=wix)
        np.clip(wee,0,max_wee,out=wee)
        np.clip(wei,0,max_wei,out=wei)
        np.clip(wie,0,max_wie,out=wie)
        np.clip(wii,0,max_wii,out=wii)
        
        # postsynaptic normalization
        wex *= wff_sum / np.sum(wex,axis=1,keepdims=True)
        wix *= wff_sum / np.sum(wix,axis=1,keepdims=True)
        wee *= wee_sum / np.sum(wee,axis=1,keepdims=True)
        wei *= wei_sum / np.sum(wei,axis=1,keepdims=True)
        wie *= wie_sum / np.sum(wie,axis=1,keepdims=True)
        wii *= wii_sum / np.sum(wii,axis=1,keepdims=True)

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'sim_lgn_wave_rfs_ne={:d}_ni={:d}/'.format(n_e,n_i)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}_iter={:d}.pkl'.format(seed,n_iter)

# Define where geniculate wave spikes are saved
lgn_dir = './../results/' + 'lgn_wave_spikes_nw={:d}_ng={:d}/'.format(n_wave,n_grid)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# compute number of LGN cells
lgn_file = lgn_dir + 'seed={:d}.pkl'.format((n_iter + seed) % max_spike_file)
print('Opening spike counts from',lgn_file)
with open(lgn_file, 'rb') as handle:
    lgn_dict = pickle.load(handle)
lgn_spikes = lgn_dict['spikes']
n_lgn = lgn_spikes.shape[-1]
n_stim = lgn_spikes.shape[0]

if n_iter==0: # starting a new simulation, must initialize the system
    rng = np.random.default_rng(seed)
    
    # initialize weights
    wex = rng.uniform(0.4,0.6,size=(n_e,n_lgn))
    wix = rng.uniform(0.4,0.6,size=(n_i,n_lgn))
    wee = rng.uniform(0.4,0.6,size=(n_e,n_e))
    wei = rng.uniform(0.4,0.6,size=(n_e,n_i))
    wie = rng.uniform(0.4,0.6,size=(n_i,n_e))
    wii = rng.uniform(0.4,0.6,size=(n_i,n_i))
    
    wex *= wff_sum / np.sum(wex,axis=1,keepdims=True)
    wix *= wff_sum / np.sum(wix,axis=1,keepdims=True)
    wee *= wee_sum / np.sum(wee,axis=1,keepdims=True)
    wei *= wei_sum / np.sum(wei,axis=1,keepdims=True)
    wie *= wie_sum / np.sum(wie,axis=1,keepdims=True)
    wii *= wii_sum / np.sum(wii,axis=1,keepdims=True)
    
    # initialize learning rates
    wex_rate = 1
    wix_rate = 1
    wee_rate = 1
    wei_rate = 1
    wie_rate = 1
    wii_rate = 1
    
    # initialize average inputs and rates
    uee = np.zeros(n_e)
    uei = np.zeros(n_e)
    uie = np.zeros(n_i)
    uii = np.zeros(n_i)
    
    # calculate average inputs and rates at the start of a geniculate wave
    rx_wave_start = lgn_spikes[26]
    update_inps(wex,wix,wee,wei,wie,wii,uee,uei,uie,uii,rx_wave_start,10*dt_stim)
    
    uee_avg = np.ones(n_e)*np.mean(uee)
    uei_avg = np.ones(n_e)*np.mean(uei)
    uie_avg = np.ones(n_i)*np.mean(uie)
    uii_avg = np.ones(n_i)*np.mean(uii)
    rx_avg = np.ones(n_lgn)*np.mean(rx_wave_start)
else:
    # load weights, inputs, rates, averages, and learning rates from previous iteration
    with open(res_dir + 'seed={:d}_iter={:d}.pkl'.format(seed,n_iter-1), 'rb') as handle:
        res_dict = pickle.load(handle)
    
    rng = res_dict['rng']
    wex = res_dict['wex']
    wix = res_dict['wix']
    wee = res_dict['wee']
    wei = res_dict['wei']
    wie = res_dict['wie']
    wii = res_dict['wii']
    wex_rate = res_dict['wex_rate']
    wix_rate = res_dict['wix_rate']
    wee_rate = res_dict['wee_rate']
    wei_rate = res_dict['wei_rate']
    wie_rate = res_dict['wie_rate']
    wii_rate = res_dict['wii_rate']
    uee = res_dict['uee']
    uei = res_dict['uei']
    uie = res_dict['uie']
    uii = res_dict['uii']
    uee_avg = res_dict['uee_avg']
    uei_avg = res_dict['uei_avg']
    uie_avg = res_dict['uie_avg']
    uii_avg = res_dict['uii_avg']
    rx_avg = res_dict['rx_avg']
    
dwex = np.zeros_like(wex)
dwix = np.zeros_like(wix)
dwee = np.zeros_like(wee)
dwei = np.zeros_like(wei)
dwie = np.zeros_like(wie)
dwii = np.zeros_like(wii)

start = time.process_time()
for idx in range(n_stim-1):
    
    rx = lgn_spikes[idx]
        
    # update inputs and rates
    update_inps(wex,wix,wee,wei,wie,wii,uee,uei,uie,uii,rx)
    
    # update averages
    update_avgs(uee,uei,uie,uii,rx,uee_avg,uei_avg,uie_avg,uii_avg,rx_avg)
    
    # collect weight changes
    collect_dw(dwex,dwix,dwee,dwei,dwie,dwii,wex_rate,wix_rate,wee_rate,wei_rate,wie_rate,wii_rate,
               uee,uei,uie,uii,rx,uee_avg,uei_avg,uie_avg,uii_avg,rx_avg)
    
    if (idx+1)%n_batch==0:
        # update learning rates if during first two iterations
        if n_iter<2:
            update_learn_rates(dwex,dwix,dwee,dwei,dwie,dwii,wex_rate,wix_rate,wee_rate,wei_rate,wie_rate,wii_rate)
        
        # update weights
        update_weights(wex,wix,wee,wei,wie,wii,dwex,dwix,dwee,dwei,dwie,dwii)
        
        # initialize weight change accumulators
        dwex[:] = 0
        dwix[:] = 0
        dwee[:] = 0
        dwei[:] = 0
        dwie[:] = 0
        dwii[:] = 0
        
        print('batch {:d} took'.format((idx+1)//n_batch),time.process_time() - start,'s')
        
        start = time.process_time()

res_dict = {}

res_dict['rng'] = rng
res_dict['wex'] = wex
res_dict['wix'] = wix
res_dict['wee'] = wee
res_dict['wei'] = wei
res_dict['wie'] = wie
res_dict['wii'] = wii
res_dict['wex_rate'] = wex_rate
res_dict['wix_rate'] = wix_rate
res_dict['wee_rate'] = wee_rate
res_dict['wei_rate'] = wei_rate
res_dict['wie_rate'] = wie_rate
res_dict['wii_rate'] = wii_rate
res_dict['uee'] = uee
res_dict['uei'] = uei
res_dict['uie'] = uie
res_dict['uii'] = uii
res_dict['uee_avg'] = uee_avg
res_dict['uei_avg'] = uei_avg
res_dict['uie_avg'] = uie_avg
res_dict['uii_avg'] = uii_avg
res_dict['rx_avg'] = rx_avg

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)

if n_iter < max_iter:
    os.system("python runjob_sim_lgn_wave_rfs.py -ne {:d} -ni {:d} -niter {:d} -miter {:d} -s {:d} -nw {:d} -ng {:d}".format(n_e,n_i,n_iter+1,max_iter,seed,n_wave,n_grid));
