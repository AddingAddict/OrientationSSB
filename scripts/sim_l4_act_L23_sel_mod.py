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
parser.add_argument('--n_phs', '-np', help='number of phases',type=int, default=12)
parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=10)
parser.add_argument('--n_int', '-nt', help='number of integration steps',type=int, default=300)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--dens', '-d', help='selective cluster density for L4 map',type=float, default=0.002)
parser.add_argument('--grec', '-g', help='L2/3 recurrent weight strength',type=float, default=1.02)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_phs = int(args['n_phs'])
n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
seed = int(args['seed'])
dens = args['dens']
grec = args['grec']
saverates = args['saverates']

n_inp = n_ori * n_phs * n_rpt

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L4_act_L23_sel_mod_dens={:.4f}_grec={:.3f}/'.format(
    dens,grec)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Define parameters for L2/3 input from databgnd_min = 0.05
bgnd_min = 0.00
bgnd_max = 0.3
meanOS = 0.18
maxOS = 0.6
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
    WL23 = np.load('./../notebooks/hetero_W_N={:d}_H={:.1f}_seed={:d}.npy'.format(N,H,seed))
except:
    WL23,_ = conn.create_matrix_2pop(config_dict["W4to4_params"],config_dict["W4to4_params"]["Wrec_mode"])
    np.save('./../notebooks/hetero_W_N={:d}_H={:.1f}_seed={:d}'.format(N,H,seed),WL23)

print('Creating heterogeneous recurrent connectivity took',time.process_time() - start,'s\n')

# Define functions to model feedforward inputs with given orientation selectivity
def elong_inp(ksig,ori,phs,thr=0):
    return np.fmax(0,np.cos(phs)*np.exp(-(ksig*np.sin(ori))**2/2) - thr) / (ive(0,ksig**2/4)) * np.pi

def elong_os(ksig):
    return ive(1,ksig**2/4) / ive(0,ksig**2/4)

ksigs = np.linspace(0,2.5,251)
oss = elong_os(ksigs)

ksig_os_itp = interp1d(oss,ksigs,fill_value='extrapolate')

del ksigs,oss

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
def gen_maps(N,dens,bgnd_min,bgnd_max,maxOS,meanOS,seed=0,bgnd_scale=4,areaCV=0,bgnd_pow=1):
    rng = np.random.default_rng(seed)
    
    bgndOS = (bgnd_min+bgnd_pow*bgnd_max)/(bgnd_pow+1)

    nclstr = np.round(N**2*dens).astype(int)
    sig2 = (meanOS - bgndOS)/((maxOS - bgndOS)*dens*np.pi) / N**2

    rng = np.random.default_rng(seed)

    clstr_pts = qmc.Halton(d=2,scramble=False,seed=seed).random(nclstr)
    
    oris = 2*np.pi*rng.random(nclstr)
    
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
        omap += np.heaviside(1.01*sig2s[i]-ds2s[i],1)*np.exp(1j*oris[i])
        holes += np.heaviside(1.01*sig2s[i]-ds2s[i],1)
            
    true_clstr_size = np.sum(np.abs(omap))
    omap *= maxOS*nclstr*np.pi*sig2*N**2/true_clstr_size

    ks = np.arange(N)/N
    ks[ks > 0.5] = ks[ks > 0.5] - 1
    kxs,kys = np.meshgrid(ks*N,ks*N)

    bgnd_ofield = np.fft.ifft2(np.exp(-0.5*(kxs**2+kys**2)*sig2*bgnd_scale**2)*\
        np.fft.fft2(np.exp(1j*2*np.pi*rng.random((N,N)))))
    bgnd_ofield /= np.abs(bgnd_ofield)
    bgnd_sfield = bgnd_min+(bgnd_max-bgnd_min)*rng.random((N,N))**bgnd_pow
    
    # make orientation selectivities smooth
    min_clstr_dists = np.min(ds2s,0)
    min_dist = np.min(min_clstr_dists[holes==0])
    max_dist = np.max(min_clstr_dists[holes==0])
    min_clstr_dists[holes==1] = min_dist + (max_dist-min_dist)*rng.random(np.count_nonzero(holes))
    bgnd_sfield = bgnd_sfield.flatten()
    bgnd_sfield[np.argsort(min_clstr_dists.flatten())[::-1]] = np.sort(bgnd_sfield).flatten()
    bgnd_sfield = bgnd_sfield.reshape(N,N)
    
    # make orientation preferences smooth
    omap = (bgnd_sfield*(1-holes)+maxOS*nclstr*np.pi*sig2*N**2/true_clstr_size*holes)*bgnd_ofield
    
    return omap

L4_inp_z = gen_maps(N,dens,bgnd_min,bgnd_max,maxOS,meanOS,seed,bgnd_scale=6)
sig2 = (meanOS - 0.5*(bgnd_min+bgnd_max))/((maxOS - 0.5*(bgnd_min+bgnd_max))*dens*np.pi) / N**2

rec_scale = 0.3

WL4 = np.exp(-0.5*ds2/(sig2*rec_scale**2))
WL4 -= np.eye(N**2).reshape((N,N,N,N))
WL4 /= np.sum(WL4,(-2,-1),keepdims=True)
WL4 = WL4.reshape((N**2,N**2))

ksmap = ksig_os_itp(np.abs(L4_inp_z))
pos = np.angle(L4_inp_z)/2

res_dict['L4_inp_z'] = L4_inp_z
# res_dict['z'] = L23_inp_z

print('Creating L4 input orientation map took',time.process_time() - start,'s\n')

rf_sct_scale = 3
L_mm = N/11
mag_fact = 0.02
L_deg = L_mm / np.sqrt(mag_fact)
grate_freq = 0.06

print(np.sqrt(sig2)*rf_sct_scale*L_deg)

def gen_rf_sct_map(sig2,sct_scale,seed=0):
    rng = np.random.default_rng(seed)
    
    sctmap = rng.normal(loc=0,scale=np.sqrt(sig2)*sct_scale,size=(N,N,2))
    polmap = rng.binomial(n=1,p=0.5,size=(N,N))
    
    return sctmap,polmap

def gen_abs_phs_map(rf_sct_map,pol_map,ori,freq,Lgrid):
    xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
    abs_rf_centx = rf_sct_map[:,:,0] + xs
    abs_rf_centy = rf_sct_map[:,:,1] + ys
    
    abs_phs = 2*np.pi*np.mod(freq*Lgrid*(np.cos(ori)*abs_rf_centx + np.sin(ori)*abs_rf_centy) + 0.5*pol_map,1)
    return abs_phs

sctmap,polmap = gen_rf_sct_map(sig2,rf_sct_scale)

# Create inputs
start = time.process_time()

inp_oris = np.arange(n_ori)/n_ori * np.pi
inp_phss = np.arange(n_phs)/n_phs * 2*np.pi
inps = np.zeros((n_ori,n_phs,n_rpt,N,N))

rng = np.random.default_rng(seed)
for ori_idx,ori in enumerate(inp_oris):
    abs_phs = gen_abs_phs_map(sctmap,polmap,ori,grate_freq,L_deg)
    for phs_idx,phs in enumerate(inp_phss):
        mean_inp = elong_inp(ksmap,ori-pos,phs-abs_phs)
        shape = 1/meanCV**2
        scale = mean_inp/shape
        for rpt_idx in range(n_rpt):
            inps[ori_idx,phs_idx,rpt_idx,:,:] = rng.gamma(shape=shape,scale=scale)
    
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

# Integrate to get firing rates
L4_rates = np.zeros((n_ori,n_phs,n_rpt,N,N))
L23_rates = np.zeros((n_ori,n_phs,n_rpt,2,N,N))

start = time.process_time()

for ori_idx,ori in enumerate(inp_oris):
    for phs_idx,phs in enumerate(inp_phss):
        for rpt_idx in range(n_rpt):
            L4_rates[ori_idx,phs_idx,rpt_idx] = integrate(np.ones(N**2),
                inps[ori_idx,phs_idx,rpt_idx].flatten(),0.25,n_int,WL4,0.5)
            L23_inp = np.concatenate((L4_rates[ori_idx,phs_idx,rpt_idx].flatten(),
                                      L4_rates[ori_idx,phs_idx,rpt_idx].flatten()))
            L23_rates[ori_idx,phs_idx,rpt_idx] = integrate(np.ones(2*N**2),
                L23_inp,0.25,n_int,WL23,grec)
    
print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['L4_rates'] = L4_rates
    res_dict['L23_rates'] = L23_rates

# Calculate z_fields from inputs and rates
inp_binned = inps.mean(2)
L4_rate_binned = L4_rates.mean(2)
L23_rate_binned = L23_rates.mean(2)

inp_r0 = np.mean(inp_binned,(0,1))
inp_opm,inp_mr = af.calc_OPM_MR(inp_binned.transpose(2,3,0,1))
inp_os = np.abs(inp_opm)
inp_po = np.angle(inp_opm)*180/(2*np.pi)
inp_r1 = inp_os*inp_r0
inp_rm = np.mean(np.mean(inps,2),(0,1))
inp_rV = np.mean(np.var(inps,2),(0,1))

L4_rate_r0 = np.mean(L4_rate_binned,(0,1))
L4_rate_opm,L4_rate_mr = af.calc_OPM_MR(L4_rate_binned.transpose(2,3,0,1))
L4_rate_os = np.abs(L4_rate_opm)
L4_rate_po = np.angle(L4_rate_opm)*180/(2*np.pi)
L4_rate_r1 = np.abs(L4_rate_opm)*L4_rate_r0
L4_rate_rm = np.mean(np.mean(L4_rates,2),(0,1))
L4_rate_rV = np.mean(np.var(L4_rates,2),(0,1))

L23_rate_r0 = np.mean(L23_rate_binned,(0,1))
L23_rate_opm,L23_rate_mr = af.calc_OPM_MR(L23_rate_binned.transpose(2,3,0,1))
L23_rate_os = np.abs(L23_rate_opm)
L23_rate_po = np.angle(L23_rate_opm)*180/(2*np.pi)
L23_rate_r1 = np.abs(L23_rate_opm)*L23_rate_r0
L23_rate_rm = np.mean(np.mean(L23_rates,2),(0,1))
L23_rate_rV = np.mean(np.var(L23_rates,2),(0,1))

# Calculate hypercolumn size and number of pinwheels
_,z_fps = af.get_fps(L23_rate_opm[0])
z_hc,_ = af.calc_hypercol_size(z_fps,N)
z_pwcnt,z_pwpts = af.calc_pinwheels(af.bandpass_filter(L23_rate_opm[0],0.5*z_hc,1.5*z_hc))
z_pwd = z_pwcnt/(N/z_hc)**2
    
Lam = z_hc
npws,pwpts = z_pwcnt,z_pwpts

res_dict['inp_r0'] = inp_r0
res_dict['inp_r1'] = inp_r1
res_dict['inp_rm'] = inp_rm
res_dict['inp_rV'] = inp_rV
res_dict['inp_opm'] = inp_opm
res_dict['inp_mr'] = inp_mr

res_dict['L4_rate_r0'] = L4_rate_r0
res_dict['L4_rate_r1'] = L4_rate_r1
res_dict['L4_rate_rm'] = L4_rate_rm
res_dict['L4_rate_rV'] = L4_rate_rV
res_dict['L4_rate_opm'] = L4_rate_opm
res_dict['L4_rate_mr'] = L4_rate_mr

res_dict['L23_rate_r0'] = L23_rate_r0
res_dict['L23_rate_r1'] = L23_rate_r1
res_dict['L23_rate_rm'] = L23_rate_rm
res_dict['L23_rate_rV'] = L23_rate_rV
res_dict['L23_rate_opm'] = L23_rate_opm
res_dict['L23_rate_mr'] = L23_rate_mr

res_dict['L4_rate_OS'] = np.mean(L4_rate_os)
res_dict['L23E_rate_OS'] = np.mean(L23_rate_os[0])
res_dict['L23I_rate_OS'] = np.mean(L23_rate_os[1])

opm_mismatch = np.abs(L4_rate_po[None,:,:] - L23_rate_po)
opm_mismatch[opm_mismatch > 90] = 180 - opm_mismatch[opm_mismatch > 90]

res_dict['opm_mismatch'] = opm_mismatch
res_dict['E_mismatch'] = np.mean(opm_mismatch[0])
res_dict['I_mismatch'] = np.mean(opm_mismatch[1])

res_dict['z_fps'] = z_fps
res_dict['Lam'] = Lam
res_dict['npws'] = npws
res_dict['pwpts'] = pwpts

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
