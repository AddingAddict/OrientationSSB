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
import matplotlib.pyplot as plt

import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF import connectivity

parser = argparse.ArgumentParser()
parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=60)
parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=10)
parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--dens', '-d', help='selective cluster density for L4 map',type=float, default=0.002)
parser.add_argument('--areaCV', '-a', help='CV of cluster area',type=float, default=0.0)
parser.add_argument('--grec', '-g', help='L2/3 recurrent weight strength',type=float, default=1.02)
parser.add_argument('--thresh', '-th', help='L2/3 activation threshold',type=float, default=0.0)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
seed = int(args['seed'])
dens = args['dens']
areaCV = args['areaCV']
grec = args['grec']
thresh = args['thresh']
saverates = args['saverates']

n_inp = n_ori * n_rpt

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L4_act_L23_sel_dens={:.4f}_areaCV={:.2f}_grec={:.3f}_thresh={:.2f}/'.format(
    dens,areaCV,grec,thresh)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Define parameters for L2/3 input from databgnd_min = 0.05
bgnd_min = 0.05
bgnd_max = 0.25
meanOS = 0.17
maxOS = 0.5
meanCV = 0.4

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

# Calculate distance from all pairs of grid points
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
ds2 = dxs**2 + dys**2
ds = np.sqrt(ds2)

# Define function to build L4 OPM
def gen_maps(N,dens,bgnd_min,bgnd_max,maxOS,meanOS,seed,areaCV=0):
    rng = np.random.default_rng(seed)
    
    bgndOS = 0.5*(bgnd_min+bgnd_max)

    nclstr = np.round(N**2*dens).astype(int)
    sig2 = (meanOS - bgndOS)/((maxOS - bgndOS)*dens*np.pi) / N**2

    rng = np.random.default_rng(seed)

    clstr_pts = qmc.Halton(d=2,scramble=False,seed=seed).random(nclstr)
    
    if np.isclose(areaCV,0):
        sig2s = sig2*np.ones(nclstr)
        rng.gamma(shape=1,scale=1,size=nclstr)
    else:
        shape = 1/areaCV**2
        scale = sig2/shape
        sig2s = rng.gamma(shape=shape,scale=scale,size=nclstr)
    
    xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
    dxs = np.abs(xs[None,:,:] - clstr_pts[:,0,None,None])
    dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
    dys = np.abs(ys[None,:,:] - clstr_pts[:,1,None,None])
    dys[dys > 0.5] = 1 - dys[dys > 0.5]
    ds2s = dxs**2 + dys**2

    omap = np.zeros((N,N),dtype='complex64')
    holes = np.zeros((N,N),dtype='float64')
    
    for i in range(nclstr):
        ori = 1j*2*np.pi*rng.random()
        omap += np.heaviside(1.01*sig2s[i]-ds2s[i],1)*np.exp(ori)
        holes += np.heaviside(1.01*sig2s[i]-ds2s[i],1)
            
    true_clstr_size = np.sum(np.abs(omap))
    omap *= maxOS*nclstr*np.pi*sig2*N**2/true_clstr_size

    ks = np.arange(N)/N
    ks[ks > 0.5] = ks[ks > 0.5] - 1
    kxs,kys = np.meshgrid(ks*N,ks*N)

    bgnd_ofield = np.fft.ifft2(np.exp(-0.25*(kxs**2+kys**2)*sig2*16)*np.fft.fft2(np.exp(1j*2*np.pi*rng.random((N,N)))))
    bgnd_ofield /= np.abs(bgnd_ofield)
    bgnd_sfield = np.real(np.fft.ifft2(np.exp(-0.25*(kxs**2+kys**2)*sig2*16)*np.fft.fft2(rng.random((N,N)))))
    bgnd_sfield -= np.min(bgnd_sfield)
    bgnd_sfield *= (bgnd_max-bgnd_min) / (np.max(bgnd_sfield) - np.min(bgnd_sfield))
    bgnd_sfield += bgnd_min
    bgnd_sfield /= nclstr*np.pi*sig2*N**2/true_clstr_size
    bgnd_sfield = bgnd_min+(bgnd_max-bgnd_min)*rng.random((N,N))
    omap += bgnd_sfield*bgnd_ofield*(1-holes)

    gauss = np.exp(-0.5*ds2/(sig2/4))
    gauss *= np.abs(omap[None,None,:,:])**1.3
    gauss /= np.sum(gauss,(-2,-1))[:,:,None,None]

    imap = np.einsum('ijkl,kl->ij',gauss,omap)
    
    return omap,imap,gauss

L4_rate_z,L23_inp_z,W4to23 = gen_maps(N,dens,bgnd_min,bgnd_max,maxOS,meanOS,seed,areaCV=areaCV)

res_dict['L4_z'] = L4_rate_z
res_dict['z'] = L23_inp_z

print('Creating input orientation map took',time.process_time() - start,'s\n')

# Create inputs
start = time.process_time()

oris = np.repeat(np.arange(n_ori)/n_ori * 180,n_rpt)
mean_inps = np.zeros((n_inp,N,N))
inps = np.zeros((n_inp,2,N,N))

rng = np.random.default_rng(seed)
for inp_idx in range(n_inp):
    ori = oris[inp_idx]
    mean_inps[inp_idx,:,:] = np.einsum('ijkl,kl->ij',W4to23,gen_clip_act(ori,L4_rate_z))
    shape = 1/meanCV**2
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
    # rates[inp_idx] = integrate(np.ones(2*N**2),inps[inp_idx].reshape((2,-1))-\
    #     thresh*np.concatenate((np.ones((1,N**2)),np.zeros((1,N**2))),axis=0),0.25,n_int,grec)
    rates[inp_idx] = integrate(np.ones(2*N**2),inps[inp_idx].reshape((2,-1))-\
        thresh,0.25,n_int,grec)
    
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

# Calculate hypercolumn size and number of pinwheels
z_unit = rate_z[0] / rate_ori_sel[0]

_,z_fps = uf.get_fps(rate_z[0])
z_hc,_ = uf.calc_hypercol_size(z_fps,N)
z_pwcnt,z_pwpts = uf.calc_pinwheels(uf.bandpass_filter(rate_z[0],0.5*z_hc,1.5*z_hc))
z_pwd = z_pwcnt/(N/z_hc)**2

_,z_unit_fps = uf.get_fps(z_unit)
# z_unit_hc,_ = uf.calc_hypercol_size(z_unit_fps,N)
# z_unit_pwcnt,z_unit_pwpts = uf.calc_pinwheels(uf.bandpass_filter(z_unit,0.5*z_unit_hc,1.5*z_unit_hc))
# z_unit_pwd = z_unit_pwcnt/(N/z_unit_hc)**2
    
Lam = z_hc
npws,pwpts = z_pwcnt,z_pwpts

res_dict['rate_r0'] = rate_r0
res_dict['rate_r1'] = rate_r1
res_dict['rate_rm'] = rate_rm
res_dict['rate_rV'] = rate_rV
res_dict['inp_r0'] = inp_r0
res_dict['inp_r1'] = inp_r1
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
