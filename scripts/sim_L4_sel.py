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
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF import connectivity

parser = argparse.ArgumentParser()
parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=4)
parser.add_argument('--n_phs', '-np', help='number of orientations',type=int, default=15)
parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=10)
parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--ksel', '-k', help='selectivity shape',type=float, default=0.1)
parser.add_argument('--lker', '-l', help='arbor length from L4 to L2/3',type=float, default=0.01)
parser.add_argument('--grec', '-g', help='L4 recurrent weight strength',type=float, default=0.9)
parser.add_argument('--eta', '-e', help='input noise level',type=float, default=0.9)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
seed = int(args['seed'])
ksel = args['ksel']
lker = args['lker']
grec = args['grec']
maxos = args['maxos']
saverates = args['saverates']

freq = 4

lker2 = lker**2

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L23_sel_ksel={:.3f}_lker={:.3f}_grec={:.3f}_maxos={:.1f}/'.format(ksel,lker,grec,maxos)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Define parameters for L2/3 input from data
avg_OS = 0.15
avg_CV = 0.4

# Create heterogeneous recurrent connectivity
config_name = "lempel_big_L4_model"
Version = -1
config_dict,_,_,_,_,N,_ = uf.get_network_size(config_name)
config_dict['W4to4_params']['max_ew'] = grec
config_dict['W4to4_params']['max_ew'] = grec

conn = connectivity.Connectivity_2pop((N,N),(N,N),\
                                    (N,N), (N,N),\
                                    random_seed=seed,\
                                    Nvert=1, verbose=True)

start = time.process_time()

H = 0.7
try:
    Wrec = np.load('./../notebooks/hetero_W_N={:d}_H={:.1f}_seed={:d}.npy'.format(N,H,seed))
except:
    Wrec,_ = conn.create_matrix_2pop(config_dict["W4to4_params"],config_dict["W4to4_params"]["Wrec_mode"])
    np.save('./../notebooks/hetero_W_N={:d}_H={:.1f}_seed={:d}'.format(N,H,seed),Wrec)

print('Creating heterogeneous recurrent connectivity took',time.process_time() - start,'s\n')

# Define functions to calculate effect of clipping on orientation selectivity
def clip_r0(OSs):
    out = np.ones_like(OSs)
    out[OSs > 0.5] = (OSs[OSs > 0.5]*np.sqrt(4-1/OSs[OSs > 0.5]**2)+np.arccos(-0.5/OSs[OSs > 0.5]))/np.pi
    return out
    
def clip_r1(OSs):
    out = OSs.copy()
    out[OSs > 0.5] = (0.25*np.sqrt(4-1/OSs[OSs > 0.5]**2)+OSs[OSs > 0.5]*np.arccos(-0.5/OSs[OSs > 0.5]))/np.pi
    return out

def clip_OS(OSs):
    out = OSs.copy()
    out[OSs > 0.5] = (0.25*np.sqrt(4-1/OSs[OSs > 0.5]**2)+OSs[OSs > 0.5]*np.arccos(-0.5/OSs[OSs > 0.5]))/\
        (OSs[OSs > 0.5]*np.sqrt(4-1/OSs[OSs > 0.5]**2)+np.arccos(-0.5/OSs[OSs > 0.5]))
    return out

# Precalculate and interpolate clipping effect
OSs = np.linspace(0.0,10000.0,20001)

thy_clip_OSs = clip_OS(OSs)
thy_clip_r0s = clip_r0(OSs)
thy_clip_r1s = clip_r1(OSs)

clip_r0_itp = interp1d(OSs,thy_clip_r0s,fill_value='extrapolate')
inv_r1_itp = interp1d(thy_clip_r1s,OSs,fill_value='extrapolate')
inv_OS_itp = interp1d(thy_clip_OSs,OSs,fill_value='extrapolate')

# Define function to create activity with desired orientation selectivity
def gen_clip_act(ori,z):
    etas = inv_OS_itp(np.abs(z))
    return np.fmax(0,1 + 2*etas*np.real(np.exp(-1j*ori*2*np.pi/180) * z/np.abs(z))) / clip_r0_itp(etas)

start = time.process_time()

# Define function to build S&P OPM
def gen_sp_opm(r1,shape,seed=0,tol=1e-2):
    rng = np.random.default_rng(seed)
    
    z = np.exp(1j*2*np.pi*rng.random(size=(N,N)))
    
    x = rng.gamma(shape=shape,scale=r1/shape,size=(N,N)) + 0.001*rng.random((N,N))
    x = clip_OS(x)
    x = np.fmax(1e-12,x)
    
    z *= x
    
    return z

snp_z = gen_sp_opm(avg_OS,ksel,seed)

res_dict['snp_z'] = snp_z

# Calculate distance from all pairs of grid points
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
ds2 = dxs**2 + dys**2
ds = np.sqrt(ds2)

# Smooth S&P OPM with Gaussian kernel
if lker != 0.0:
    gauss = np.exp(-0.5*ds2/lker2)
else:
    gauss = (ds2 == 0.0).astype(float)
    
gauss /= np.sum(gauss,(-2,-1))[:,:,None,None]
    
z = np.einsum('ijkl,kl->ij',gauss,snp_z)

# scale magnitude of z field until its mean selectivity matches data
while np.abs(np.mean(np.fmin(maxos,clip_OS(np.abs(z)))) - avg_OS) > 1e-3:
    z *= 1 - (np.mean(np.fmin(maxos,clip_OS(np.abs(z)))) - avg_OS)

z *= np.fmin(maxos,clip_OS(np.abs(z))) / np.fmax(1e-12,np.abs(z))

res_dict['z'] = z

print('Creating input orientation map took',time.process_time() - start,'s\n')

# Interpolate relationship between squeezed gabor phase and OS
start = time.process_time()

def squeezed_gabor_r0(sigbar,phi,thresh):
    return quad(lambda θ: np.fmax(0,np.sqrt(np.sin(phi)**2+(sigbar*np.cos(θ))**2)/\
                                            np.sqrt(np.sin(phi)**2+sigbar**2)-thresh),0,np.pi)[0]

def squeezed_gabor_r1(sigbar,phi,thresh):
    return quad(lambda θ: np.cos(2*θ)*np.fmax(0,np.sqrt(np.sin(phi)**2+(sigbar*np.cos(θ))**2)/\
                                            np.sqrt(np.sin(phi)**2+sigbar**2)-thresh),0,np.pi)[0]

def squeezed_gabor_OS(sigbar,phi,thresh):
    r0 = squeezed_gabor_r0(sigbar,phi,thresh)
    r1 = squeezed_gabor_r1(sigbar,phi,thresh)
    return r1/r0

phis = np.linspace(0,np.pi/2,101)
RF_r0s = np.zeros_like(phis)
RF_r1s = np.zeros_like(phis)
RF_OSs = np.zeros_like(phis)

for phi_idx,phi in enumerate(phis):
    RF_r0s[phi_idx] = squeezed_gabor_r0(0.25,phi,0.72)
    RF_r1s[phi_idx] = squeezed_gabor_r1(0.25,phi,0.72)
    RF_OSs[phi_idx] = squeezed_gabor_OS(0.25,phi,0.72)
    
min_OS = np.min(RF_OSs)
max_OS = np.max(RF_OSs)
RF_r0_itp = interp1d(phis,RF_r0s,fill_value='extrapolate')
RF_r1_itp = interp1d(phis,RF_r1s,fill_value='extrapolate')
RF_OS_itp = interp1d(phis,RF_OSs,fill_value='extrapolate')
RF_OS_inv_itp = interp1d(RF_OSs,phis,fill_value='extrapolate')

L4_phis = RF_OS_inv_itp(np.fmax(min_OS,np.abs(z)))

# Create L4 RFs
σ = 0.04
k = 0.5/σ
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
xs[xs > 0.5] = xs[xs > 0.5] - 1
ys[ys > 0.5] = ys[ys > 0.5] - 1
ds = np.sqrt(xs**2 + ys**2)

def sq_gabor(ori,phi):
    return np.exp(-0.5*(xs**2+ys**2)/σ**2)*np.sin(k*(np.cos(ori*2*np.pi/180)*xs+np.sin(ori*2*np.pi/180)*ys)+phi)/np.sqrt(np.sin(phi)**2+(k*σ)**2)#np.sin(k*σ+phi)

RFs = np.zeros((N,N,N,N))
rng = np.random.default_rng(0)
for i in range(N):
    for j in range(N):
        this_rf = rng.choice([1,-1])*sq_gabor(np.angle(z[i,j])*180/(2*np.pi),rng.choice([1,-1])*L4_phis[i,j])
        this_rf[ds > 2.5*σ] = 0
        this_rf /= np.sum(np.abs(this_rf))
        RFs[i,j] = np.roll(this_rf,(i,j),(0,1))
        
on_conn = np.zeros_like(RFs)
of_conn = np.zeros_like(RFs)

on_conn[RFs > 0] = RFs[RFs > 0]
of_conn[RFs < 0] = -RFs[RFs < 0]

print('Creating L4 RFs took',time.process_time() - start,'s\n')