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
from scipy.interpolate import CubicSpline
from scipy import integrate

import analyze_func as af

parser = argparse.ArgumentParser()
parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=16)
parser.add_argument('--n_phs', '-np', help='number of orientations',type=int, default=16)
parser.add_argument('--n_int', '-nt', help='number of integration steps between phases',type=int, default=4)
parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
parser.add_argument('--add_phase', '-ap', help='add phase selectivity to L4 inputs or not',type=bool, default=False)
parser.add_argument('--add_orisel', '-aos', help='add orientation selectivity to L4 inputs or not',type=bool, default=False)
parser.add_argument('--add_sandp', '-asp', help='make L4 inputs salt and pepper or not',type=bool, default=False)
parser.add_argument('--map', '-m', help='whether to switch to a different L4 map',type=str, default=None)
parser.add_argument('--static', '-st', help='static or dynamic input',type=bool, default=False)
parser.add_argument('--saverates', '-r', help='save rates or not',type=bool, default=False)
parser.add_argument('--saveweights', '-w', help='save weights or not',type=bool, default=False)
args = vars(parser.parse_args())
n_ori = int(args['n_ori'])
n_phs = int(args['n_phs'])
# n_rpt = int(args['n_rpt'])
n_int= int(args['n_int'])
seed = int(args['seed'])
add_phase = args['add_phase']
add_orisel = args['add_orisel']
add_sandp = args['add_sandp']
static = args['static']
saverates = args['saverates']
saveweights = args['saveweights']

N = 60

# Define parameters for connectivity
params = np.load("./../notebooks/l23_params.npy")

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'L23_sel/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if static:
    res_dir = res_dir + 'static_'
if args['map'] is not None:
    res_dir = res_dir + args['map'] + '_'
if add_phase:
    res_dir = res_dir + 'phase_'
if add_orisel:
    res_dir = res_dir + 'orisel_'
if add_sandp:
    res_dir = res_dir + 'sandp_'
res_file = res_dir + 'seed={:d}.pkl'.format(seed)

res_dict = {}

# load L4 responses
if args['map'] is None:
    with open('./../results/L4_sel/seed={:d}.pkl'.format(seed), 'rb') as handle:
        L4_res_dict = pickle.load(handle)
else:
    with open('./../results/L4_sel/{:s}_seed={:d}.pkl'.format(args['map'],seed), 'rb') as handle:
        L4_res_dict = pickle.load(handle)
    
L4_rates = L4_res_dict['L4_rates'][0]
L4_rate_opm = L4_res_dict['L4_rate_opm'][0]

L4_rates /= np.nanmean(L4_rates,axis=(-2,-1),keepdims=True)
if add_phase:
    _,_,phs = af.calc_dc_ac_comp(L4_rates)
    L4_phase_rates = np.fmax(0,np.cos(np.linspace(0,2*np.pi,n_phs,endpoint=False)[None,None,:]-phs[:,:,None]))
    L4_phase_rates *= np.nanmean(L4_rates,axis=(-1),keepdims=True) / np.nanmean(L4_phase_rates,axis=(-1),keepdims=True)
    L4_rates = L4_phase_rates
if add_orisel:
    _,_,doub_po = af.calc_dc_ac_comp(L4_rates.mean(-1))
    L4_orisel_rates = np.fmax(0,np.cos(np.linspace(0,2*np.pi,n_ori,endpoint=False)[None,:]-doub_po[:,None]))
    L4_orisel_rates *= np.nanmean(L4_rates.mean(-1),axis=(-1),keepdims=True) / np.nanmean(L4_orisel_rates,axis=(-1),keepdims=True)
    L4_norm_phase_tuning = np.fmax(1e-12,L4_rates / np.nanmean(L4_rates,axis=(-1),keepdims=True))
    L4_rates = L4_norm_phase_tuning * L4_orisel_rates[:,:,None]
if add_sandp:
    rng = np.random.default_rng(seed)
    L4_rates = rng.permutation(L4_rates)

# Compute distance matrix for connectivity kernel
xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)
    
# define L4 rate interpolation function after L4 to L2/3 scattering
L4_rates_itp = CubicSpline(np.arange(0,n_phs+1) * 1/(3*n_phs),
                           np.concatenate((L4_rates,L4_rates[:,:,0:1]),axis=-1),
                           axis=-1,bc_type='periodic')

# define simulation functions
def integrate_sheet(xea0,xen0,xeg0,xia0,xin0,xig0,inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,
                    het_lev,N,ne,ni,threshe,threshi,
                    t0,dt,Nt,tsamp=None,ta=0.01,tn=0.300,tg=0.01,frac_n=0.7):
    '''
    Integrate 2D sheet with AMPA, NMDA, and GABA receptor dynamics.
    xe0, xi0: initial excitatory and inhibitory activity
    inp: input function, takes time t and returns input at that time
    Jee, Jei, Jie, Jii: connectivity strengths per connection type
    kern: connectivity kernel for the sheet
    ne, ni: rate activation exponents for excitatory and inhibitory neurons
    threshe, threshi: activation thresholds for excitatory and inhibitory neurons
    t0: initial time
    dt: time step for integration
    Nt: number of time steps to integrate
    ta, tn, tg: time constants for AMPA, NMDA, and GABA receptor dynamics
    frac_n: fraction of NMDA vs NMDA+AMPA receptors in the excitatory population
    '''
    
    if tsamp is None:
        tsamp = [Nt-1]
    samp_idx = 0
    
    xea = xea0.copy()
    xen = xen0.copy()
    xeg = xeg0.copy()
    xia = xia0.copy()
    xin = xin0.copy()
    xig = xig0.copy()
    
    rng = np.random.default_rng(seed)
    
    noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
    Wee = Jee*kerne.reshape(N**2,N**2)*noise
    noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
    Wei = Jei*kernei.reshape(N**2,N**2)*noise
    noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
    Wie = Jie*kerne.reshape(N**2,N**2)*noise
    noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
    Wii = Jii*kernii.reshape(N**2,N**2)*noise
    
    if len(xea.shape) == 1:
        xea = xea
        xen = xen
        xeg = xeg
        xia = xia
        xin = xin
        xig = xig
    
    resps = np.zeros((2,N**2,len(tsamp)))
    
    # for t_idx in range(Nt):
    #     ff_inp = inp(t0+t_idx*dt)
    #     ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
    #     yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
    #     if t_idx in tsamp:
    #         resps[0,:,samp_idx] = ye
    #         resps[1,:,samp_idx] = yi
    #         samp_idx += 1
    #     net_ee = Wee@ye + ff_inp
    #     net_ei = Wei@yi
    #     net_ie = Wie@ye + ff_inp
    #     net_ii = Wii@yi
    #     xea += ((1-frac_n)*net_ee - xea)*dt/ta
    #     xen += (frac_n*net_ee - xen)*dt/tn
    #     xeg += (net_ei - xeg)*dt/tg
    #     xia += ((1-frac_n)*net_ie - xia)*dt/ta
    #     xin += (frac_n*net_ie - xin)*dt/tn
    #     xig += (net_ii - xig)*dt/tg
        
    def dyn_func(t,x,ncell):
        xea = x[0*ncell:1*ncell]
        xen = x[1*ncell:2*ncell]
        xeg = x[2*ncell:3*ncell]
        xia = x[3*ncell:4*ncell]
        xin = x[4*ncell:5*ncell]
        xig = x[5*ncell:6*ncell]
        
        ff_inp = inp(t)

        ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
        yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
        
        net_ee = Wee@ye + ff_inp
        net_ei = Wei@yi
        net_ie = Wie@ye + ff_inp
        net_ii = Wii@yi
        
        dx = np.zeros_like(x)
        dx[0*ncell:1*ncell] = ((1-frac_n)*net_ee - xea)/ta
        dx[1*ncell:2*ncell] = (frac_n*net_ee - xen)/tn
        dx[2*ncell:3*ncell] = (net_ei - xeg)/tg
        dx[3*ncell:4*ncell] = ((1-frac_n)*net_ie - xia)/ta
        dx[4*ncell:5*ncell] = (frac_n*net_ie - xin)/tn
        dx[5*ncell:6*ncell] = (net_ii - xig)/tg
        
        return dx.flatten()
    
    x0 = np.concatenate((xea,xen,xeg,xia,xin,xig),axis=0).flatten()
    
    start_time = time.process_time()
    max_time = 60
    def time_event(t,x,ncell):
        int_time = (start_time + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True
    
    sol = integrate.solve_ivp(dyn_func,(0,dt*Nt),y0=x0,t_eval=tsamp*dt,args=(N**2,),method='RK23')#,events=time_event)
    if sol.status != 0:
        x = np.nan*np.ones((6*N**2,len(tsamp)))
    else:
        x = sol.y
    x = x.reshape((-1,len(tsamp)))
    
    xea = x[0*N**2:1*N**2,:]
    xen = x[1*N**2:2*N**2,:]
    xeg = x[2*N**2:3*N**2,:]
    xia = x[3*N**2:4*N**2,:]
    xin = x[4*N**2:5*N**2,:]
    xig = x[5*N**2:6*N**2,:]
        
    ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
    yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
    
    resps[0] = ye
    resps[1] = yi
    # return xea,xen,xeg,xia,xin,xig,np.concatenate((ye,yi))
    return resps

def get_sheet_resps(params,N):
    '''
    params[0] = log10[|Jee]]
    params[1] = log10[|Jei]]
    params[2] = log10[|Jie]]
    params[3] = log10[|Jii]]
    params[4] = s_e
    params[5] = s_ei / s_e
    params[6] = s_ii / s_ei
    params[7] = p_ker
    params[8] = het_level
    params[9] = inp_mult
    
    returns: [q1_os,q2_os,q3_os,mu_os,sig_os,q1_mr,q2_mr,q3_mr,mu_mr,sig_mr]
    os = excitatory orientation selectivity
    mr = excitatory modulation ratio
    '''
    Jee,Jei,Jie,Jii = 10**params[:4]
    Jei *= -1
    Jii *= -1
    
    thresh = 0
    nori = n_ori
    nphs = n_phs
    nint = 5
    nwrm = 4 * nint * nphs
    dt = 1 / (nint * nphs * 3)
    
    s_e = params[4]
    s_ei = s_e * params[5]
    s_ii = s_ei * params[6]
    kerne = np.exp(-(dss/s_e)**params[None,None,7])
    norm = kerne.sum(axis=1).mean(axis=0)
    kerne /= norm
    kernei = np.exp(-(dss/s_ei)**params[None,None,7])
    norm = kernei.sum(axis=1).mean(axis=0)
    kernei /= norm
    kernii = np.exp(-(dss/s_ii)**params[None,None,7])
    norm = kernii.sum(axis=1).mean(axis=0)
    kernii /= norm
    
    inp_mult = params[9]
    
    tsamp = nwrm-1 + np.arange(0,nphs) * nint
    resps = np.zeros((2,N**2,nori,nphs))
    for ori_idx in range(nori):
        if static:
            for phs_idx,phs in enumerate(np.linspace(0,2*np.pi,n_phs,endpoint=False)):
                print(ori_idx,phs_idx)
                def ff_inp(t):
                    return inp_mult * L4_rates_itp(phs/(2*np.pi*3))[:,ori_idx]
                resps[:,:,ori_idx,phs_idx] = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                        np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                        ff_inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,params[8],N,2,2,
                                        thresh,thresh,0,dt,nwrm/2,tsamp[0:1]/2)[:,:,-1]
        else:
            def ff_inp(t):
                return inp_mult * L4_rates_itp(t)[:,ori_idx]
            resps[:,:,ori_idx,:] = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                    ff_inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,params[8],N,2,2,
                                    thresh,thresh,0,dt,nwrm+nint*nphs,tsamp)
        
    return resps

# Integrate to get firing rates
start = time.process_time()

L23_rates = get_sheet_resps(params,N)
    
print('Simulating rate dynamics took',time.process_time() - start,'s\n')

if saverates:
    res_dict['L23_rates'] = L23_rates
if saveweights:
    Jee,Jei,Jie,Jii = 10**params[:4]
    
    s_e = params[4]
    s_ei = s_e * params[5]
    s_ii = s_ei * params[6]
    kerne = np.exp(-(dss/s_e)**params[None,None,7])
    norm = kerne.sum(axis=1).mean(axis=0)
    kerne /= norm
    kernei = np.exp(-(dss/s_ei)**params[None,None,7])
    norm = kernei.sum(axis=1).mean(axis=0)
    kernei /= norm
    kernii = np.exp(-(dss/s_ii)**params[None,None,7])
    norm = kernii.sum(axis=1).mean(axis=0)
    kernii /= norm
    
    het_lev = params[8]
    
    rng = np.random.default_rng(seed)
    
    noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
    Wee = Jee*kerne.reshape(N**2,N**2)*noise
    noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
    Wei = Jei*kernei.reshape(N**2,N**2)*noise
    noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
    Wie = Jie*kerne.reshape(N**2,N**2)*noise
    noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
    Wii = Jii*kernii.reshape(N**2,N**2)*noise
    
    res_dict['weights'] = (Wee,Wei,Wie,Wii)

# Calculate CV of inputs and responses
L23_rate_r0 = np.mean(L23_rates,(-2,-1))
L23_rate_opm,L23_rate_mr = af.calc_OPM_MR(L23_rates)
L23_rate_r1 = np.abs(L23_rate_opm)*L23_rate_r0
L23_inp_opm,L23_inp_mr = af.calc_OPM_MR(L4_rates**2)

res_dict['L23_rate_r0'] = L23_rate_r0
res_dict['L23_rate_r1'] = L23_rate_r1
res_dict['L23_rate_opm'] = L23_rate_opm
res_dict['L23_rate_mr'] = L23_rate_mr
res_dict['L23_inp_opm'] = L23_inp_opm
res_dict['L23_inp_mr'] = L23_inp_mr

# Calculate hypercolumn size and number of pinwheels
_,L23_rate_raps = af.get_fps(L23_rate_opm[0].reshape(N,N))
L23_rate_hc,_ = af.calc_hypercol_size(L23_rate_raps,N)
freqs = np.arange(len(L23_rate_raps))/60
pwd,popt = af.calc_pinwheel_density_from_raps(freqs,L23_rate_raps,return_fit=True)

Lam = L23_rate_hc

res_dict['L23_rate_raps'] = L23_rate_raps
res_dict['L23_rate_hc'] = L23_rate_hc
res_dict['L23_rate_pwd'] = pwd

# Calculate orientation mismatch
L23_rate_pref_ori = np.angle(L23_rate_opm)*180/(2*np.pi)
L23_rate_pref_ori[L23_rate_pref_ori > 90] -= 180
L4_rate_pref_ori = np.angle(L4_rate_opm)*180/(2*np.pi)
L4_rate_pref_ori[L4_rate_pref_ori > 90] -= 180
opm_mismatch = np.abs(L4_rate_pref_ori - L23_rate_pref_ori)
opm_mismatch[opm_mismatch > 90] = 180 - opm_mismatch[opm_mismatch > 90]

res_dict['opm_mismatch'] = opm_mismatch
res_dict['E_mismatch'] = np.mean(opm_mismatch[0])
res_dict['I_mismatch'] = np.mean(opm_mismatch[1])

with open(res_file, 'wb') as handle:
    pickle.dump(res_dict,handle)
