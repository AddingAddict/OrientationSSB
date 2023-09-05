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
parser.add_argument('--H', '-H', help='connectivity heterogeneity level',type=float, default=0.7)
parser.add_argument('--eta', '-e', help='input modulation level',type=float, default=1e-2)
parser.add_argument('--grec', '-g', help='L2/3 recurrent weight strength',type=float, default=1.02)
parser.add_argument('--betx', '-b', help='Ratio of external input to inhibition vs excitation',type=float, default=1.)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
args = vars(parser.parse_args())
n_inp = int(args['n_inp'])
n_int= int(args['n_int'])
H = args['H']
eta = args['eta']
grec = args['grec']
betx = args['betx']
seed = int(args['seed'])

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L23_corr_ssn_H={:.1f}_eta={:.0e}_grec={:.3f}_betx={:.2f}/'.format(H,eta,grec,betx)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Create heterogeneous recurrent connectivity
config_name = "big_hetero"
Version = -1
config_dict,_,_,_,_,N,_ = uf.get_network_size(config_name)
config_dict["W4to4_params"]["mean_eccentricity"] = H
config_dict["W4to4_params"]["SD_eccentricity"] = 0.025*H
config_dict["W4to4_params"]["SD_size"] = 0.003*H

conn = connectivity.Connectivity_2pop((N,N),(N,N),\
                                    (N,N), (N,N),\
                                    random_seed=seed,\
                                    Nvert=1, verbose=True)

start = time.process_time()

try:
    Wrec = np.load('./../notebooks/hetero_W_N={:d}_H={:.1f}_seed={:d}.npy'.format(N,H,seed))
except:
    Wrec,_ = conn.create_matrix_2pop(config_dict["W4to4_params"],config_dict["W4to4_params"]["Wrec_mode"])
    np.save('./../notebooks/hetero_W_N={:d}_H={:.1f}_seed={:d}'.format(N,H,seed),Wrec)

Wrec = Wrec.reshape((2,N**2,2,N**2)).transpose((0,2,1,3))

print('Creating heterogeneous recurrent connectivity took',time.process_time() - start,'s\n')

# Figure out gammas for SSN to match RELU results
a = Wrec.sum(-1).mean(-1)
n = np.array([2,3])

I = np.array([1,betx])
R = np.array([2,10])

V = np.linalg.inv(np.eye(2)-grec * a/n[None,:])@I
k = R/(V**n)

print('k =',k)
print('n =',n)
print('I =',I)
print('V =',V)
print('R =',R)

gam = 1/(k*n*V**(n-1))

# Create difference of Gaussians filter
x,y = np.meshgrid(np.linspace(-N//2,N//2-1,N),np.linspace(-N//2,N//2-1,N))
sig1 = 1.8
sig2 = 3.6
kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
diff_gauss = kern1-kern2

inps = np.zeros((n_inp,2,N,N))

ring_image = np.fft.fft2(np.fft.fftshift(diff_gauss))

# Create inputs
rng = np.random.default_rng(seed)
for inp_idx in range(n_inp):
    for pop_idx in range(2):
        random_matrix = np.fft.fft2(rng.normal(size=(N,N)))
        ring_ifft = np.real(np.fft.ifft2(ring_image*random_matrix))
        inps[inp_idx,pop_idx,:,:] = I[pop_idx]*(1 + eta*ring_ifft)

print('Creating input patterns took',time.process_time() - start,'s\n')

# Define integration functions
def fio_rect(x):
    return np.fmax(x,0)

def dynamics_system(y,inp_ff,Wrec,gamma_rec,gamma_ff,k,n,tau):
    argE = gamma_rec[0] * np.dot(Wrec[0,0],y[0]) + gamma_rec[1] * np.dot(Wrec[0,1],y[1]) + gamma_ff * inp_ff[0]
    argI = gamma_rec[0] * np.dot(Wrec[1,0],y[0]) + gamma_rec[1] * np.dot(Wrec[1,1],y[1]) + gamma_ff * inp_ff[1]
    return 1./tau*( -y + np.stack([k[0]*fio_rect(argE)**n[0],k[1]*fio_rect(argI)**n[1]]))

def integrate(y0,inp,Wrec,gamma_rec,k,n,dt,Nt):
    y = y0
    for t_idx in range(Nt):
        out = dynamics_system(y,inp,Wrec,gamma_rec,1.0,k,n,1.0)
        dy = out
        y = y + dt*dy
    return np.array([y[0].reshape((N,N)),y[1].reshape((N,N))])

# Integrate to get firing rates
rates = np.zeros_like(inps)

start = time.process_time()

for inp_idx in range(n_inp):
    rates[inp_idx] = integrate(np.ones((2,N**2)),inps[inp_idx].reshape((2,-1)),Wrec,grec*gam,k,n,0.25,n_int)

res_dict['rates'] = rates

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)