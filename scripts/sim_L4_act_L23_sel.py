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
import matplotlib.pyplot as plt

import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF import connectivity

parser = argparse.ArgumentParser()
parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=60)
parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=10)
parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--bgnd', '-b', help='amplitude of background selectivity',type=float, default=0.001)
parser.add_argument('--ksel', '-k', help='selectivity shape',type=float, default=0.1)
parser.add_argument('--lker', '-l', help='smoothing kernel length scale for L4 map',type=float, default=0.01)
parser.add_argument('--Wlker_fact', '-w', help='ratio of arbor length from L4 to L2/3 vs correlation length scale of L4 map',type=float, default=1.0)
parser.add_argument('--grec', '-g', help='L2/3 recurrent weight strength',type=float, default=1.02)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
seed = int(args['seed'])
bgnd = args['bgnd']
ksel = args['ksel']
lker = args['lker']
Wlker_fact = args['Wlker_fact']
grec = args['grec']
saverates = args['saverates']

n_inp = n_ori * n_rpt

lker2 = lker**2

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L4_act_L23_sel_bgnd_{:.4}_ksel={:.4f}_lker={:.3f}_Wlker_fact={:.1f}_grec={:.3f}/'.format(
    bgnd,ksel,lker,Wlker_fact,grec)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Define parameters for L2/3 input from data
avg_OS = 0.17
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
    
    x = rng.gamma(shape=shape,scale=r1/shape,size=(N,N)) + bgnd*rng.random((N,N))
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

L4_z = np.einsum('ijkl,kl->ij',gauss,snp_z * inv_OS_itp(np.abs(snp_z)) / np.fmax(1e-12,np.abs(snp_z)))

# scale magnitude of z field until its mean selectivity matches data
while np.abs(np.mean(clip_OS(np.abs(L4_z))) - avg_OS) > 1e-3:
    L4_z *= 1 - (np.mean(clip_OS(np.abs(L4_z))) - avg_OS)

L4_z *= clip_OS(np.abs(L4_z)) / np.fmax(1e-12,np.abs(L4_z))

res_dict['L4_z'] = L4_z

Wlker = Wlker_fact*lker
Wlker2 = Wlker**2

if Wlker != 0.0:
    W = np.exp(-0.5*ds2/Wlker2)
else:
    W = (ds2 == 0.0).astype(float)
    
W *= np.random.default_rng(seed).random((N,N))
    
W /= np.sum(W,(-2,-1))[:,:,None,None]

z = np.einsum('ijkl,kl->ij',W,L4_z)

res_dict['z'] = z

print('Creating input orientation map took',time.process_time() - start,'s\n')

# Create inputs
start = time.process_time()

oris = np.repeat(np.arange(n_ori)/n_ori * 180,n_rpt)
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
        inps[inp_idx,pop_idx,:,:] = rng.gamma(shape=shape,scale=scale)

# res_dict['mean_inps'] = mean_inps
# res_dict['inps'] = inps
    
print('Creating input patterns took',time.process_time() - start,'s\n')

# Define integration functions
def fio_rect(x):
    return np.fmax(x,0)

def dynamics_system(y,inp_ff,Wrec,gamma_rec,gamma_ff,tau):
    arg = gamma_rec * np.dot(Wrec,y) + gamma_ff * inp_ff.flatten()
    return 1./tau*( -y + fio_rect(arg))

def integrate(y0,inp,dt,Nt,gamma_rec=1.02):
    y = y0
    for t_idx in range(Nt):
        out = dynamics_system(y,inp,Wrec,gamma_rec,1.0,1.0)
        dy = out
        y = y + dt*dy
    return np.array([y[:N**2].reshape((N,N)),y[N**2:].reshape((N,N))])

# Integrate to get firing rates
rates = np.zeros_like(inps)

start = time.process_time()

for inp_idx in range(n_inp):
    rates[inp_idx] = integrate(np.ones(2*N**2),inps[inp_idx].reshape((2,-1)),0.25,n_int,grec)
    
print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['rates'] = rates

# Calculate z_fields from inputs and rates
ori_binned = oris.reshape(-1,n_rpt).mean(1)
inp_binned = inps.reshape(-1,n_rpt,2,N,N).mean((1,2))
rate_binned = rates.reshape(-1,n_rpt,2,N,N).mean(1)

rate_r0 = np.mean(rate_binned,0)
rate_rs = np.mean(np.sin(ori_binned*2*np.pi/180)[:,None,None,None]*rate_binned,0)
rate_rc = np.mean(np.cos(ori_binned*2*np.pi/180)[:,None,None,None]*rate_binned,0)
rate_r1 = np.sqrt(rate_rs**2 + rate_rc**2)
rate_rm = np.mean(np.mean(rates.reshape(-1,n_rpt,2,N,N),1),0)
rate_rV = np.mean(np.var(rates.reshape(-1,n_rpt,2,N,N),1),0)

inp_r0 = np.mean(inp_binned,0)
inp_rs = np.mean(np.sin(ori_binned*2*np.pi/180)[:,None,None]*inp_binned,0)
inp_rc = np.mean(np.cos(ori_binned*2*np.pi/180)[:,None,None]*inp_binned,0)
inp_r1 = np.sqrt(inp_rs**2 + inp_rc**2)
inp_rm = np.mean(np.mean(inps.reshape(-1,n_rpt,2,N,N),1),0)
inp_rV = np.mean(np.var(inps.reshape(-1,n_rpt,2,N,N),1),0)

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

# Calculate hypercolumn size
z_unit = rate_z[0] / rate_ori_sel[0]
z_fft = np.abs(np.fft.fftshift(np.fft.fft2(rate_z[0] - np.nanmean(rate_z[0]))))
z_unit_fft = np.abs(np.fft.fftshift(np.fft.fft2(z_unit - np.nanmean(z_unit))))

z_fps = np.zeros(N//2)
z_unit_fps = np.zeros(N//2)

grid = np.arange(-N//2,N//2)
x,y = np.meshgrid(grid,grid)
bin_idxs = np.digitize(np.sqrt(x**2+y**2),np.arange(0,np.ceil(N//2*np.sqrt(2)))+0.5)
for idx in range(N//2):
    z_fps[idx] = np.mean(z_fft[bin_idxs == idx])
    z_unit_fps[idx] = np.mean(z_unit_fft[bin_idxs == idx])
    
freqs = np.arange(N//2)/N
Lam = 1/freqs[np.argmax(z_fps)]

# Calculate number of pinwheels
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def cross(A,B):
    return A[0]*B[1] - A[1]*B[0]

def intersectpt(A,B,C,D):
    qmp = [C[0]-A[0],C[1]-A[1]]
    r = [B[0]-A[0],B[1]-A[1]]
    s = [D[0]-C[0],D[1]-C[1]]
    rxs = cross(r,s)
    
    t = cross(qmp,s)/rxs
#     u = cross(qmp,r)/rxs
    
    return [A[0]+t*r[0],A[1]+t*r[1]]

def calc_pinwheels(A):
    rcont = plt.contour(np.real(A),levels=[0],colors="C0")
    icont = plt.contour(np.imag(A),levels=[0],colors="C1")

    rsegpts = []
    for pts in rcont.allsegs[0]:
        for i in range(len(pts)-1):
            rsegpts.append([pts[i],pts[i+1]])
    rsegpts = np.array(rsegpts)

    isegpts = []
    for pts in icont.allsegs[0]:
        for i in range(len(pts)-1):
            isegpts.append([pts[i],pts[i+1]])
    isegpts = np.array(isegpts)
    
    pwcnt = 0
    pwpts = []

    for rsegpt in rsegpts:
        for isegpt in isegpts:
            if intersect(rsegpt[0],rsegpt[1],isegpt[0],isegpt[1]):
                pwcnt += 1
                pwpts.append(intersectpt(rsegpt[0],rsegpt[1],isegpt[0],isegpt[1]))
    pwpts = np.array(pwpts)
    
    return pwcnt,pwpts

npws,pwpts = calc_pinwheels(rate_z[0])

res_dict['rate_r0'] = rate_r0
res_dict['rate_r1'] = rate_r1
res_dict['rate_rs'] = rate_rs
res_dict['rate_rc'] = rate_rc
res_dict['rate_rm'] = rate_rm
res_dict['rate_rV'] = rate_rV
res_dict['inp_r0'] = inp_r0
res_dict['inp_r1'] = inp_r1
res_dict['inp_rs'] = inp_rs
res_dict['inp_rc'] = inp_rc
res_dict['inp_rm'] = inp_rm
res_dict['inp_rV'] = inp_rV

res_dict['rate_z'] = rate_z
res_dict['inp_z'] = inp_z

res_dict['inp_OS'] = np.mean(inp_ori_sel)
res_dict['E_rate_OS'] = np.mean(rate_ori_sel[0])
res_dict['I_rate_OS'] = np.mean(rate_ori_sel[1])

opm_mismatch = np.abs(inp_pref_ori - rate_pref_ori)
opm_mismatch[opm_mismatch > 90] = 180 - opm_mismatch[opm_mismatch > 90]

res_dict['opm_mismatch'] = opm_mismatch
res_dict['E_mismatch'] = np.mean(opm_mismatch[0])
res_dict['I_mismatch'] = np.mean(opm_mismatch[1])

res_dict['z_fps'] = z_fps
res_dict['z_unit_fps'] = z_unit_fps
res_dict['Lam'] = Lam
res_dict['npws'] = npws
res_dict['pwpts'] = pwpts

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
