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
parser.add_argument('--n_inp', '-ni', help='number of inputs',type=int, default=500)
parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--ksel', '-k', help='selectivity shape',type=float, default=0.1)
parser.add_argument('--lker', '-l', help='arbor length from L4 to L2/3',type=float, default=0.01)
parser.add_argument('--grec', '-g', help='L2/3 recurrent weight strength',type=float, default=1.02)
parser.add_argument('--betx', '-b', help='Ratio of external input to inhibition vs excitation',type=float, default=1.)
parser.add_argument('--maxos', '-m', help='maximum input selectivity',type=float, default=1.0)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_inp = int(args['n_inp'])
n_int= int(args['n_int'])
seed = int(args['seed'])
ksel = args['ksel']
lker = args['lker']
grec = args['grec']
betx = args['betx']
maxos = args['maxos']
saverates = args['saverates']

lker2 = lker**2

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L23_sel_ssn_ksel={:.3f}_lker={:.3f}_grec={:.3f}_betx={:.2f}_maxos={:.1f}/'.format(
    ksel,lker,grec,betx,maxos)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Define parameters for L2/3 input from data
avg_OS = 0.15
avg_CV = 0.4

# Create heterogeneous recurrent connectivity
config_name = "big_hetero"
Version = -1
config_dict,_,_,_,_,N,_ = uf.get_network_size(config_name)

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

# Figure out gammas for SSN to match RELU results
a = Wrec.sum(-1).mean(-1)
n = np.array([2,2])

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
    z *= rng.gamma(shape=shape,scale=r1/shape,size=(N,N))
    
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
    
z = np.einsum('ijkl,kl->ij',gauss,snp_z)

# scale magnitude of z field until its mean selectivity matches data
while np.abs(np.mean(np.fmin(maxos,clip_OS(np.abs(z)))) - avg_OS) > 1e-3:
    z *= 1 - (np.mean(np.fmin(maxos,clip_OS(np.abs(z)))) - avg_OS)

z *= np.fmin(maxos,clip_OS(np.abs(z))) / np.fmax(1e-12,np.abs(z))

res_dict['z'] = z

print('Creating input orientation map took',time.process_time() - start,'s\n')

# Create inputs
start = time.process_time()

oris = np.arange(n_inp)/n_inp * 180
mean_inps = np.zeros((n_inp,N,N))
inps = np.zeros((n_inp,2,N,N))

rng = np.random.default_rng(seed)
for inp_idx in range(n_inp):
    ori = oris[inp_idx]
    mean_inps[inp_idx,:,:] = gen_clip_act(ori,z)
    shape = 1/avg_CV**2
    scale = mean_inps[inp_idx,:,:]/shape
    # scale = avg_FF
    # shape = mean_inps[inp_idx,:,:]/avg_FF
    for pop_idx in range(2):
        inps[inp_idx,pop_idx,:,:] = I[pop_idx]*rng.gamma(shape=shape,scale=scale)

# res_dict['mean_inps'] = mean_inps
# res_dict['inps'] = inps
    
print('Creating input patterns took',time.process_time() - start,'s\n')

# Define integration functions
def fio_rect(x):
    return np.fmax(x,0)

def dynamics_system(y,inp_ff,Wrec,gamma_rec,gamma_ff,k,n,tau):
    argE = gamma_rec[0] * np.dot(Wrec[0,0],y[0]) + gamma_rec[1] * np.dot(Wrec[0,1],y[1]) + gamma_ff * inp_ff[0]
    argI = gamma_rec[0] * np.dot(Wrec[1,0],y[0]) + gamma_rec[1] * np.dot(Wrec[1,1],y[1]) + gamma_ff * inp_ff[1]
    return np.stack([(-y[0] + k[0]*fio_rect(argE)**n[0])/tau[0],
                     (-y[1] + k[1]*fio_rect(argI)**n[1])/tau[1]])

def integrate(y0,inp,Wrec,gamma_rec,k,n,dt,Nt):
    y = y0
    for t_idx in range(Nt):
        out = dynamics_system(y,inp,Wrec,gamma_rec,1.0,k,n,np.array([1.0,2.0]))
        dy = out
        y = y + dt*dy
    return np.array([y[0].reshape((N,N)),y[1].reshape((N,N))])

# Integrate to get firing rates
rates = np.zeros_like(inps)

start = time.process_time()

for inp_idx in range(n_inp):
    rates[inp_idx] = integrate(np.ones((2,N**2)),inps[inp_idx].reshape((2,-1)),Wrec,grec*gam,k,n,0.25,n_int)
    
print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['rates'] = rates

# Calculate z_fields from inputs and rates
n_bins = 1
ori_binned = np.nanmean(np.ma.masked_invalid(oris.reshape(-1,n_bins)),1)
inp_binned = np.nanmean(np.ma.masked_invalid(inps.reshape(-1,n_bins,2,N,N)),(1,2))
rate_binned = np.nanmean(np.ma.masked_invalid(rates.reshape(-1,n_bins,2,N,N)),1)

rate_r0 = np.nanmean(np.ma.masked_invalid(rate_binned),0)
rate_rV = np.nanvar(np.ma.masked_invalid(rate_binned),0)
rate_rs = np.nanmean(np.ma.masked_invalid(np.sin(ori_binned*2*np.pi/180)[:,None,None,None]*rate_binned),0)
rate_rc = np.nanmean(np.ma.masked_invalid(np.cos(ori_binned*2*np.pi/180)[:,None,None,None]*rate_binned),0)
rate_r1 = np.sqrt(rate_rs**2 + rate_rc**2)

inp_r0 = np.nanmean(np.ma.masked_invalid(inp_binned),0)
inp_rV = np.nanvar(np.ma.masked_invalid(inp_binned),0)
inp_rs = np.nanmean(np.ma.masked_invalid(np.sin(ori_binned*2*np.pi/180)[:,None,None]*inp_binned),0)
inp_rc = np.nanmean(np.ma.masked_invalid(np.cos(ori_binned*2*np.pi/180)[:,None,None]*inp_binned),0)
inp_r1 = np.sqrt(inp_rs**2 + inp_rc**2)

rate_pref_ori = np.arctan2(rate_rs,rate_rc)*180/(2*np.pi)
rate_pref_ori[rate_pref_ori > 90] -= 180
rate_alt_pref_ori = ori_binned[rate_binned.argmax(0)]
rate_alt_pref_ori[rate_alt_pref_ori > 90] -= 180
rate_ori_sel = rate_r1/rate_r0

inp_pref_ori = np.arctan2(inp_rs,inp_rc)*180/(2*np.pi)
inp_pref_ori[inp_pref_ori > 90] -= 180
inp_alt_pref_ori = ori_binned[inp_binned.argmax(0)]
inp_alt_pref_ori[inp_alt_pref_ori > 90] -= 180
inp_ori_sel = inp_r1/inp_r0

rate_z = rate_ori_sel * np.exp(1j*rate_pref_ori*2*np.pi/180)
inp_z = inp_ori_sel * np.exp(1j*inp_pref_ori*2*np.pi/180)

res_dict['rate_r0'] = rate_r0
res_dict['rate_r1'] = rate_r1
res_dict['rate_rV'] = rate_rV
res_dict['inp_r0'] = inp_r0
res_dict['inp_r1'] = inp_r1
res_dict['inp_rV'] = inp_rV

res_dict['rate_z'] = rate_z
res_dict['inp_z'] = inp_z

res_dict['inp_OS'] = np.nanmean(np.ma.masked_invalid(inp_ori_sel))
res_dict['E_rate_OS'] = np.nanmean(np.ma.masked_invalid(rate_ori_sel[0]))
res_dict['I_rate_OS'] = np.nanmean(np.ma.masked_invalid(rate_ori_sel[1]))

opm_mismatch = np.abs(inp_pref_ori - rate_pref_ori)
opm_mismatch[opm_mismatch > 90] = 180 - opm_mismatch[opm_mismatch > 90]

res_dict['opm_mismatch'] = opm_mismatch
res_dict['E_mismatch'] = np.nanmean(np.ma.masked_invalid(opm_mismatch[0]))
res_dict['I_mismatch'] = np.nanmean(np.ma.masked_invalid(opm_mismatch[1]))

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)