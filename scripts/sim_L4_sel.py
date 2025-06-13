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

import analyze_func as af

parser = argparse.ArgumentParser()
parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=16)
parser.add_argument('--n_phs', '-np', help='number of orientations',type=int, default=16)
# parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=5)
parser.add_argument('--n_int', '-nt', help='number of integration steps between phases',type=int, default=5)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--dens', '-d', help='selective cluster density for L4 map',type=float, default=0.002)
parser.add_argument('--grec', '-g', help='L4 recurrent weight strength',type=float, default=0.95)
parser.add_argument('--lrec', '-l', help='L4 recurrent weight length',type=float, default=0.8)
parser.add_argument('--thresh', '-th', help='L4 activation threshold',type=float, default=1.0)
parser.add_argument('--actpow', '-p', help='L4 activation power',type=float, default=1.0)
parser.add_argument('--map', '-m', help='L4 orientation map type',type=str, default='act')
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_phs = int(args['n_phs'])
# n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
seed = int(args['seed'])
dens = args['dens']
grec = args['grec']
lrec = args['lrec']
thresh = args['thresh']
actpow = args['actpow']
map_type = args['map']
saverates = args['saverates']

N = 60

bgnd_min = 0.02
bgnd_max = 0.3
clst_min = 0.3
clst_max = 0.6
meanOS = 0.20

# Define function to generate clustered L4 input orientation map
def gen_maps(N,dens,bgnd_min,bgnd_max,clst_min,clst_max,meanOS,seed=0,bgnd_scale=4,areaCV=0,bgnd_pow=1,
             cont_oris=False,cont_sels=False):
    rng = np.random.default_rng(seed)
    
    bgndOS = (bgnd_min+bgnd_pow*bgnd_max)/(bgnd_pow+1)
    clstOS = (clst_min+clst_max)/2

    nclstr = np.round(N**2*dens).astype(int)
    sig2 = (meanOS - bgndOS)/(((clstOS) - bgndOS)*dens*np.pi) / N**2

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
    holes = np.clip(holes,0,1)
            
    true_clstr_size = np.sum(np.abs(omap))
    omap *= clstOS*nclstr*np.pi*sig2*N**2/true_clstr_size

    ks = np.arange(N)/N
    ks[ks > 0.5] = ks[ks > 0.5] - 1
    kxs,kys = np.meshgrid(ks*N,ks*N)

    bgnd_ofield = np.fft.ifft2(np.exp(-0.5*(kxs**2+kys**2)*sig2*bgnd_scale**2)*\
        np.fft.fft2(np.exp(1j*2*np.pi*rng.random((N,N)))))
    bgnd_ofield /= np.abs(bgnd_ofield)
    bgnd_sfield = bgnd_min+(bgnd_max-bgnd_min)*rng.random((N,N))**bgnd_pow
    clst_sfield = clst_min+(clst_max-clst_min)*rng.random((N,N))
    clst_sfield *= nclstr*np.pi*sig2*N**2/true_clstr_size
    if cont_sels:
        min_clstr_dists = np.min(ds2s,0)
        min_dist = np.min(min_clstr_dists[holes==0])
        max_dist = np.max(min_clstr_dists[holes==0])
        min_clstr_dists[holes==1] = min_dist + (max_dist-min_dist)*rng.random(np.count_nonzero(holes))
        bgnd_sfield = bgnd_sfield.flatten()
        bgnd_sfield[np.argsort(min_clstr_dists.flatten())[::-1]] = np.sort(bgnd_sfield).flatten()
        bgnd_sfield = bgnd_sfield.reshape(N,N)
        
        min_clstr_dists = np.min(ds2s,0)
        min_dist = np.min(min_clstr_dists[holes==1])
        max_dist = np.max(min_clstr_dists[holes==1])
        clst_sfield = clst_sfield.flatten()
        min_clstr_dists[holes==0] = min_dist + (max_dist-min_dist)*rng.random(np.count_nonzero(1-holes))
        clst_sfield[np.argsort(min_clstr_dists.flatten())[::-1]] = np.sort(clst_sfield).flatten()
        clst_sfield = clst_sfield.reshape(N,N)
    if cont_oris:
        omap = (bgnd_sfield*(1-holes)+clst_sfield*holes)*bgnd_ofield
    else:
        omap += bgnd_sfield*bgnd_ofield*(1-holes)
    
    return omap

# compute characteristic L4 length scale, derived from cluster density
sig2 = (meanOS - 0.5*(bgnd_min+bgnd_max))/((0.5*(clst_min+clst_max) - 0.5*(bgnd_min+bgnd_max))*dens*np.pi) / N**2
sig = np.sqrt(sig2)

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if map_type == 'act':
    res_dir = res_dir + 'L4_sel_dens={:.4f}_grec={:.2f}_lrec={:.1f}_thresh={:.3f}_actpow={:.2f}/'.format(
        dens,grec,lrec,thresh,actpow)
    L4_inp_opm = gen_maps(N,dens,bgnd_min,bgnd_max,clst_min,clst_max,meanOS,
                          seed,bgnd_scale=6,cont_oris=True,cont_sels=True)
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
    L4_inp_opm = np.fft.ifft2(opm_fft)
    L4_inp_opm *= meanOS / np.mean(np.abs(L4_inp_opm)) # normalize mean to meanOS
    res_dir = res_dir + 'L4_sel_map={:s}_grec={:.2f}_lrec={:.1f}_thresh={:.3f}_actpow={:.2f}/'.format(
            map_type,grec,lrec,thresh,actpow)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# Define thalamic input functions
kl2 = 2

def elong_inp(gam,ori,phs):
    return 1 + np.cos(phs)*np.exp(-kl2*(1+(1-gam**2)/gam**2*np.sin(ori)**2)/2)

# Precalculate and interpolate effect of activation function on orientation selectivity
gams = np.linspace(0.4,1,301)
oris = np.linspace(0,np.pi,120,endpoint=False)
phss = np.linspace(0,2*np.pi,120,endpoint=False)
resps = np.fmax(0,elong_inp(gams[:,None,None],oris[None,:,None],phss[None,None,:])-thresh)**actpow
oss,_ = af.calc_OS_MR(resps)

gam_os_itp = interp1d(oss,gams,fill_value='extrapolate')
gam_map = gam_os_itp(np.abs(L4_inp_opm))

# Generate rf scatter and ON/OFF bias maps
rf_sct_scale = 1.5
pol_scale = 0.7
L_mm = N/11
mag_fact = 0.02
L_deg = L_mm / np.sqrt(mag_fact)
grate_freq = 0.06

def gen_rf_sct_map(N,sig2,sct_scale,pol_scale,seed=0):
    rng = np.random.default_rng(seed)

    ks = np.arange(N)/N
    ks[ks > 0.5] = ks[ks > 0.5] - 1
    kxs,kys = np.meshgrid(ks*N,ks*N)
    ks = np.sqrt(kxs**2 + kys**2)
    kpol = 1/(np.sqrt(sig2)*pol_scale)

    polmap = (np.fft.ifft2(ks*np.exp(0.125-2*(ks - 0.75*kpol)**2/kpol**2)*\
        np.fft.fft2(rng.binomial(n=1,p=0.5,size=(N,N))-0.5)) > 0).astype(int)
    
    sctmap = rng.normal(loc=0,scale=np.sqrt(sig2)*sct_scale,size=(N,N,2))
    
    return sctmap,polmap

def gen_abs_phs_map(N,rf_sct_map,pol_map,ori,freq,Lgrid):
    xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
    abs_rf_centx = rf_sct_map[:,:,0] + xs
    abs_rf_centy = rf_sct_map[:,:,1] + ys
    
    abs_phs = 2*np.pi*np.mod(freq*Lgrid*(np.cos(ori)*abs_rf_centx + np.sin(ori)*abs_rf_centy) + 0.5*pol_map,1)
    return abs_phs

sctmap,polmap = gen_rf_sct_map(N,sig2,rf_sct_scale,pol_scale)
abs_phs = gen_abs_phs_map(N,sctmap,polmap,0,grate_freq,L_deg)

# Create L4 recurrent weights
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
ds2s = dxs**2 + dys**2

w = np.exp(-0.5*ds2s**2/(sig2*lrec**2)**2)
w -= 0.5*np.eye(N**2).reshape((N,N,N,N))
w /= np.mean(np.sum(w,(-2,-1)))

# Define integration functions
def fio_rect(x):
    return np.fmax(x,0)

def integrate_ampa_nmda(x0,inp,dt,Nt,Wrec,gamma_rec=1.02,ta=0.01,tn=0.325,frac_n=0.5):
    if np.isscalar(x0):
        xa = (1-frac_n)*x0
        xn = frac_n*x0
    else:
        xa = (1-frac_n)*x0[0]
        xn = frac_n*x0[1]
    
    if isinstance(inp, np.ndarray):
        inp_len = len(inp)
        for t_idx in range(Nt):
            net_inp = gamma_rec * np.dot(Wrec,fio_rect(xa+xn-thresh)**actpow) + inp[t_idx%inp_len]
            xa += ((1-frac_n)*net_inp - xa)*dt/ta
            xn += (frac_n*net_inp - xn)*dt/tn
    else:
        for t_idx in range(Nt):
            net_inp = gamma_rec * np.dot(Wrec,fio_rect(xa+xn-thresh)**actpow) + inp(t_idx*dt)
            xa += ((1-frac_n)*net_inp - xa)*dt/ta
            xn += (frac_n*net_inp - xn)*dt/tn
    return xa,xn,fio_rect(xa+xn-thresh)**actpow

def integrate_nmda(x0,inp,dt,Nt,Wrec,gamma_rec=1.02,tn=0.325):
    xn = x0
    
    if isinstance(inp, np.ndarray):
        inp_len = len(inp)
        for t_idx in range(Nt):
            net_inp = gamma_rec * np.dot(Wrec,fio_rect(xn-thresh)**actpow) + inp[t_idx%inp_len]
            xn += (net_inp - xn)*dt/tn
    else:
        for t_idx in range(Nt):
            net_inp = gamma_rec * np.dot(Wrec,fio_rect(xn-thresh)**actpow) + inp(t_idx*dt)
            xn += (net_inp - xn)*dt/tn
    return xn,fio_rect(xn-thresh)**actpow

def integrate_ampa_nmda_no_rec(x0,inp,dt,Nt,ta=0.01,tn=0.325,frac_n=0.5):
    if np.isscalar(x0):
        xa = (1-frac_n)*x0
        xn = frac_n*x0
    else:
        xa = (1-frac_n)*x0[0]
        xn = frac_n*x0[1]
        
    if isinstance(inp, np.ndarray):
        inp_len = len(inp)
        for t_idx in range(Nt):
            net_inp = inp[t_idx%inp_len]
            xa += ((1-frac_n)*net_inp - xa)*dt/ta
            xn += (frac_n*net_inp - xn)*dt/tn
    else:
        for t_idx in range(Nt):
            net_inp = inp(t_idx*dt)
            xa += ((1-frac_n)*net_inp - xa)*dt/ta
            xn += (frac_n*net_inp - xn)*dt/tn
    return xa,xn,fio_rect(xa+xn-thresh)**actpow

def integrate_nmda_no_rec(x0,inp,dt,Nt,tn=0.325):
    xn = x0
        
    if isinstance(inp, np.ndarray):
        inp_len = len(inp)
        for t_idx in range(Nt):
            net_inp = inp[t_idx%inp_len]
            xn += (net_inp - xn)*dt/tn
    else:
        for t_idx in range(Nt):
            net_inp = inp(t_idx*dt)
            xn += (net_inp - xn)*dt/tn
    return xn,fio_rect(xn-thresh)**actpow

# Integrate to get firing rates
L4_rf_rates = np.zeros((n_ori,n_phs,N,N))
L4_rates = np.zeros((n_ori,n_phs,N,N))

inp_oris = np.linspace(0,np.pi,n_ori,endpoint=False)

pref_oris = 0.5*np.angle(L4_inp_opm)
pref_oris[pref_oris < 0] += np.pi
gam_map_flat = gam_map.flatten()
pref_oris_flat = pref_oris.flatten()

start = time.process_time()

for ori_idx in range(n_ori):
    abs_phs = gen_abs_phs_map(N,sctmap,polmap,inp_oris[ori_idx],grate_freq,L_deg).flatten()
    def ff_inp(t):
        return elong_inp(gam_map_flat,np.abs(inp_oris[ori_idx]-pref_oris_flat),abs_phs+2*np.pi*3*t)
    dt = 1/(3*n_phs*n_int)
    ff_inp_array = ff_inp(np.arange(n_phs*n_int)[:,None]*dt)
    
    inp_xn,y = integrate_nmda_no_rec(0.001*np.ones(N**2),ff_inp_array,dt,6*3*n_phs*n_int)
    L4_rf_rates[ori_idx,0] = y.reshape((N,N))
    rec_xn,y = integrate_nmda(0.001*np.ones(N**2),ff_inp_array,dt,
                              6*3*n_phs*n_int,np.array(w.reshape((N**2,N**2))),grec)
    L4_rates[ori_idx,0] = y.reshape((N,N))
    for phs_idx in range(n_phs-1):
        inp_xn,y = integrate_nmda_no_rec(inp_xn,ff_inp_array[n_int*phs_idx:n_int*(phs_idx+1)], dt,n_int)
        L4_rf_rates[ori_idx,phs_idx+1] = y.reshape((N,N))
        rec_xn,y = integrate_nmda(rec_xn,ff_inp_array[n_int*phs_idx:n_int*(phs_idx+1)],
                                  dt,n_int,np.array(w.reshape((N**2,N**2))),grec)
        L4_rates[ori_idx,phs_idx+1] = y.reshape((N,N))
    
print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['L4_rf_rates'] = L4_rf_rates
    res_dict['L4_rates'] = L4_rates

# Calculate CV of inputs and responses
inp_r0 = np.mean(L4_rf_rates,(0,1))
inp_opm,inp_mr = af.calc_OPM_MR(L4_rf_rates.transpose(2,3,0,1))
# inp_os = np.abs(inp_opm)
# inp_po = np.angle(inp_opm)*180/(2*np.pi)
inp_r1 = np.abs(inp_opm)*inp_r0

L4_rate_r0 = np.mean(L4_rates,(0,1))
L4_rate_opm,L4_rate_mr = af.calc_OPM_MR(L4_rates.transpose(2,3,0,1))
# L4_rate_os = np.abs(L4_rate_opm)
# L4_rate_po = np.angle(L4_rate_opm)*180/(2*np.pi)
L4_rate_r1 = np.abs(L4_rate_opm)*L4_rate_r0

res_dict['inp_r0'] = inp_r0
res_dict['inp_r1'] = inp_r1
res_dict['inp_opm'] = inp_opm
res_dict['inp_mr'] = inp_mr

res_dict['L4_rate_r0'] = L4_rate_r0
res_dict['L4_rate_r1'] = L4_rate_r1
res_dict['L4_rate_opm'] = L4_rate_opm
res_dict['L4_rate_mr'] = L4_rate_mr

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
