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
from scipy.interpolate import interp1d
from scipy.stats import qmc
from scipy.special import ive
import matplotlib.pyplot as plt

import util_func as uf
import analyze_func as af

import dev_ori_sel_RF
from dev_ori_sel_RF import connectivity

parser = argparse.ArgumentParser()
parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=4)
parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=5)
parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--dens', '-d', help='selective cluster density for L4 map',type=float, default=0.002)
parser.add_argument('--grec', '-g', help='L2/3 recurrent weight strength',type=float, default=1.02)
parser.add_argument('--htro', '-ht', help='L2/3 recurrent weight heterogeneity',type=float, default=0.7)
parser.add_argument('--map', '-m', help='L4 orientation map type',type=str, default='act')
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=True)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
seed = int(args['seed'])
dens = args['dens']
grec = args['grec']
htro = args['htro']
map_type = args['map']
saverates = args['saverates']

n_inp = n_ori * n_rpt

res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Create heterogeneous recurrent connectivity
config_name = "big_hetero"
Version = -1
config_dict,_,_,_,_,N,_ = uf.get_network_size(config_name)
config_dict["W4to4_params"]["mean_eccentricity"] *= htro / 0.7
config_dict["W4to4_params"]["SD_eccentricity"] *= htro / 0.7
config_dict["W4to4_params"]["SD_size"] *= htro / 0.7

conn = connectivity.Connectivity_2pop((N,N),(N,N),\
                                    (N,N), (N,N),\
                                    random_seed=seed,\
                                    Nvert=1, verbose=True)

start = time.process_time()

H = htro
try:
    WL23 = np.load('./../notebooks/hetero_W_N={:d}_H={:.2f}_seed={:d}.npy'.format(N,H,seed))
except:
    WL23,_ = conn.create_matrix_2pop(config_dict["W4to4_params"],config_dict["W4to4_params"]["Wrec_mode"])
    np.save('./../notebooks/hetero_W_N={:d}_H={:.2f}_seed={:d}'.format(N,H,seed),WL23)

print('Creating heterogeneous recurrent connectivity took',time.process_time() - start,'s\n')

# Create L4 orientation map
with open(res_dir + 'L4_act_L23_sel_mod_dens={:.4f}_grec={:.3f}/seed={:d}.pkl'.format(
        dens,1.05,seed), 'rb') as handle:
    res_dict = pickle.load(handle)
rm = np.mean(res_dict['L4_rate_rm'])
meanCV = np.mean(np.sqrt(res_dict['L4_rate_rV'])/res_dict['L4_rate_rm'])
if map_type == 'act':
    # Get L4 orientation map from previous simulation
    L4_rate_opm = res_dict['L4_rate_opm']
    res_dir = res_dir + 'L23_sel_dens={:.4f}_grec={:.3f}_htro={:.2f}/'.format(
        dens,grec,htro)
else:
    rng = np.random.default_rng(seed+1234)
    opm_fft = rng.normal(size=(N,N)) + 1j * rng.normal(size=(N,N))
    opm_fft[0,0] = 0 # remove DC component
    freqs = np.fft.fftfreq(N,1/N)
    freqs = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)
    if 'band' in map_type:
        # assume map_type == 'per_{freq}_{width}'
        _,freq,width = map_type.split('_')
        freq = float(freq)
        width = float(width)
        opm_fft *= np.heaviside(0.5*width - np.abs(freqs-freq),0.5)
    elif 'low' in map_type:
        # assume map_type == 'low_{decay}'
        _,decay = map_type.split('_')
        decay = float(decay)
        opm_fft *= np.exp(-freqs/decay)
    else:
        raise ValueError('Unknown map type: {}'.format(map_type))
    L4_rate_opm = np.fft.ifft2(opm_fft)
    L4_rate_opm *= 0.5 / np.max(np.abs(L4_rate_opm)) # normalize to 0.5
    res_dir = res_dir + 'L23_sel_map={:s}_grec={:.3f}_htro={:.2f}/'.format(
        map_type,grec,htro)

# Define where to save results
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Define parameters for L2/3 input from databgnd_min = 0.05
bgnd_min = 0.00
bgnd_max = 0.3
meanOS = 0.18
maxOS = 0.6

# Define functions to model feedforward inputs with given orientation selectivity
def elong_inp(ksig,ori):
    return np.exp(-(ksig*np.sin(ori))**2/2) / ive(0,ksig**2/4)

def elong_os(ksig):
    return ive(1,ksig**2/4) / ive(0,ksig**2/4)

ksigs = np.linspace(0,2.5,251)
oss = elong_os(ksigs)

ksig_os_itp = interp1d(oss,ksigs,fill_value='extrapolate')

del ksigs,oss

ksmap = ksig_os_itp(np.abs(L4_rate_opm))

start = time.process_time()

# Create inputs
start = time.process_time()

pos = np.angle(L4_rate_opm)/2
inp_oris = np.arange(n_ori)/n_ori * np.pi
inps = np.zeros((n_ori,n_rpt,N,N))

rng = np.random.default_rng(seed)
for ori_idx,ori in enumerate(inp_oris):
    mean_inp = rm * elong_inp(ksmap,ori-pos)
    shape = 1/meanCV**2
    scale = mean_inp/shape
    for rpt_idx in range(n_rpt):
        inps[ori_idx,rpt_idx,:,:] = rng.gamma(shape=shape,scale=scale)
        
n_spnt = 100
spnt_inps = np.zeros((n_spnt,N,N))
for spnt_idx in range(n_spnt):
    mean_inp = rm*np.ones((N,N))
    shape = 1/meanCV**2
    scale = mean_inp/shape
    spnt_inps[spnt_idx,:,:] = rng.gamma(shape=shape,scale=scale)
    
print('Creating input patterns took',time.process_time() - start,'s\n')

# Define integration functions
def fio_rect(x):
    return np.fmax(x,0)

def dynamics_system(y,inp_ff,Wrec,gamma_rec,gamma_ff,tau):
    arg = gamma_rec * np.dot(Wrec,y) + gamma_ff * inp_ff.flatten()
    return 1./tau*( -y + fio_rect(arg))

def integrate(y0,inp,dt,Nt,Wrec,gamma_rec=1.02):
    y = y0
    for t_idx in range(Nt):
        out = dynamics_system(y,inp,Wrec,gamma_rec,1.0,1.0)
        dy = out
        y = y + dt*dy
    if len(y)==N**2:
        return y.reshape((N,N))
    else:
        return np.array([y[:N**2].reshape((N,N)),y[N**2:].reshape((N,N))])

start = time.process_time()

# Integrate to get firing rates
L23_rates = np.zeros((n_ori,n_rpt,2,N,N,n_int//100))

for ori_idx,ori in enumerate(inp_oris):
    for rpt_idx in range(n_rpt):
        L23_inp = np.concatenate((inps[ori_idx,rpt_idx].flatten(),
                                    inps[ori_idx,rpt_idx].flatten()))
        L23_rates[ori_idx,rpt_idx,:,:,:,0] = integrate(np.ones(2*N**2),
            L23_inp,0.25,100,WL23,grec)
        for i in range(1,n_int//100):
            L23_rates[ori_idx,rpt_idx,:,:,:,i] = integrate(L23_rates[ori_idx,rpt_idx,:,:,:,i-1].flatten(),
                L23_inp,0.25,100,WL23,grec)
            
L23_spnt_rates = np.zeros((n_spnt,2,N,N,n_int//100))
for spnt_idx in range(n_rpt):
    L23_inp = np.concatenate((spnt_inps[spnt_idx].flatten(),
                                spnt_inps[spnt_idx].flatten()))
    L23_spnt_rates[spnt_idx,:,:,:,0] = integrate(np.ones(2*N**2),
        L23_inp,0.25,100,WL23,grec)
    for i in range(1,n_int//100):
        L23_spnt_rates[spnt_idx,:,:,:,i] = integrate(L23_spnt_rates[spnt_idx,:,:,:,i-1].flatten(),
            L23_inp,0.25,100,WL23,grec)
    
print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['inputs'] = inps
    res_dict['rates'] = L23_rates
    
# calculate spontaneous dimensionality
L23_spnt_dim = np.zeros(n_int//100)
for i in range(n_int//100):
    mean_sub_rate = L23_spnt_rates[:,0,:,:,i].reshape((n_spnt,-1))
    mean_sub_rate -= np.mean(mean_sub_rate,0)
    rates_cov_mat = np.einsum('ij,ik->jk',mean_sub_rate,mean_sub_rate) / n_spnt
    cov_evals = np.linalg.eigvalsh(rates_cov_mat)
    L23_spnt_dim[i] = np.sum(cov_evals)**2/np.sum(cov_evals**2)

# Calculate z_fields from inputs and rates
inp_binned = inps.mean(1)
L23_rate_binned = L23_rates.mean(1)

inp_r0 = np.mean(inp_binned,0)
inp_opm = af.calc_OPM(inp_binned.transpose(1,2,0))
inp_os = np.abs(inp_opm)
inp_po = np.angle(inp_opm)*180/(2*np.pi)
inp_r1 = inp_os*inp_r0
inp_rm = np.mean(np.mean(inps,1),0)
inp_rV = np.mean(np.var(inps,1),0)

L23_rate_r0 = np.mean(L23_rate_binned,0)
L23_rate_opm = af.calc_OPM(L23_rate_binned.transpose(1,2,3,4,0))
L23_rate_os = np.abs(L23_rate_opm)
L23_rate_po = np.angle(L23_rate_opm)*180/(2*np.pi)
L23_rate_r1 = np.abs(L23_rate_opm)*L23_rate_r0
L23_rate_rm = np.mean(np.mean(L23_rates,1),0)
L23_rate_rV = np.mean(np.var(L23_rates,1),0)

# Calculate hypercolumn size and number of pinwheels
# _,z_fps = af.get_fps(L23_rate_opm[0])
# z_hc,_ = af.calc_hypercol_size(z_fps,N)
# z_pwcnt,z_pwpts = af.calc_pinwheels(af.bandpass_filter(L23_rate_opm[0],0.5*z_hc,1.5*z_hc))
# z_pwd = z_pwcnt/(N/z_hc)**2

# Lam = z_hc
# npws,pwpts = z_pwcnt,z_pwpts

res_dict['inp_r0'] = inp_r0
res_dict['inp_r1'] = inp_r1
res_dict['inp_rm'] = inp_rm
res_dict['inp_rV'] = inp_rV
res_dict['inp_opm'] = inp_opm

res_dict['L23_spnt_dim'] = L23_spnt_dim

res_dict['L23_rate_r0'] = L23_rate_r0
res_dict['L23_rate_r1'] = L23_rate_r1
res_dict['L23_rate_rm'] = L23_rate_rm
res_dict['L23_rate_rV'] = L23_rate_rV
res_dict['L23_rate_opm'] = L23_rate_opm

# res_dict['z_fps'] = z_fps
# res_dict['Lam'] = Lam
# res_dict['npws'] = npws
# res_dict['pwpts'] = pwpts

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
