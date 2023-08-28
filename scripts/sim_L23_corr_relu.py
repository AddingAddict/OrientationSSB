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

import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF import connectivity

parser = argparse.ArgumentParser()
parser.add_argument('--n_inp', '-ni', help='number of inputs',type=int, default=200)
parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
args = vars(parser.parse_args())
n_inp = int(args['n_inp'])
n_int= int(args['n_int'])
seed = int(args['seed'])
grec = 1.02

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L23_corr_relu/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Create heterogeneous recurrent connectivity
config_name = "big_hetero"
Version = -1
config_dict,N4pop,Nlgnpop,Nret,Nlgn,N4,rA = uf.get_network_size(config_name)

W4 = connectivity.Connectivity_2pop((N4,N4),(N4,N4),\
                                    (N4,N4), (N4,N4),\
                                    random_seed=seed,\
                                    Nvert=1, verbose=True)

start = time.process_time()

try:
    W4to4 = np.load('./../notebooks/hetero_W4to4_N4={:d}_seed={:d}.npy'.format(N4,seed))
except:
    W4to4,_ = W4.create_matrix_2pop(config_dict["W4to4_params"],config_dict["W4to4_params"]["Wrec_mode"])
    np.save('./../notebooks/hetero_W4to4_N4={:d}_seed={:d}'.format(N4,seed),W4to4)

print('Creating heterogeneous recurrent connectivity took',time.process_time() - start,'s\n')

# Create difference of Gaussians filter
x,y = np.meshgrid(np.linspace(-N4//2,N4//2-1,N4),np.linspace(-N4//2,N4//2-1,N4))
sig1 = 1.8
sig2 = 3.6
kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
diff_gauss = kern1-kern2

eta = 4e-3
inps = np.zeros((n_inp,2,N4,N4))

ring_image = np.fft.fft2(np.fft.fftshift(diff_gauss))

# Create inputs
rng = np.random.default_rng(seed)
for inp_idx in range(n_inp):
    for pop_idx in range(2):
        random_matrix = np.fft.fft2(rng.normal(size=(N4,N4)))
        ring_ifft = np.real(np.fft.ifft2(ring_image*random_matrix))
        ring_ifft = ring_ifft - np.mean(ring_ifft)
        ring_ifft = ring_ifft/np.std(ring_ifft)
        inps[inp_idx,pop_idx,:,:] = 1 + eta*ring_ifft

print('Creating input patterns took',time.process_time() - start,'s\n')

# Define integration functions
def fio_rect(x):
    return np.fmax(x,0)

def dynamics_system(y,inp_ff,Wrec,gamma_rec,gamma_ff,tau):
    arg = gamma_rec * np.dot(Wrec,y) + gamma_ff * inp_ff.flatten()
    return 1./tau*( -y + fio_rect(arg))

def integrate(y0,inp,Wrec,dt,Nt,gamma_rec=1.02):
    y = y0
    for t_idx in range(Nt):
        out = dynamics_system(y,inp,Wrec,gamma_rec,1.0,1.0)
        dy = out
        y = y + dt*dy
    return np.array([y[:N4**2].reshape((N4,N4)),y[N4**2:].reshape((N4,N4))])

# Integrate to get firing rates
rates = np.zeros_like(inps)

start = time.process_time()

for inp_idx in range(n_inp):
    rates[inp_idx] = integrate(np.ones(2*N4**2),inps[inp_idx].reshape((2,-1)),W4to4,0.25,n_int,grec)

res_dict['rates'] = rates

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)