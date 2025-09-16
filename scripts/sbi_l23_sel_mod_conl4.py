import os
import pickle
import time
import argparse

import numpy as np
import torch
from scipy import interpolate
from scipy import integrate
from scipy.sparse.linalg import eigs

from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior

import analyze_func as af
import map_func as mf
from sbi_func import PostTimesBoxUniform

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', '-i', help='completely arbitrary job id label',type=int, default=0)
parser.add_argument('--num_samp', '-ns', help='number of samples',type=int, default=200)
parser.add_argument('--bayes_iter', '-bi', help='bayessian inference interation (0 = use prior, 1 = use first posterior)',type=int, default=0)
args = vars(parser.parse_args())
job_id = int(args['job_id'])
num_samp = int(args['num_samp'])
bayes_iter = int(args['bayes_iter'])

print("Bayesian iteration:", bayes_iter)
print("Job ID:", job_id)

device = torch.device("cpu")

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'sbi_l23_sel_mod_conl4/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

# set prior and grid size for model
N = 60

se0 = 0.02375
sei0 = 0.0175
sii0 = 0.0075
Jee0 = 22.2
Jie0 = 21.6
Jei0 = 21.6
Jii0 = 20.8

# create L4 orientation map
rng = np.random.default_rng(0)
opm_fft = rng.normal(size=(N,N)) + 1j * rng.normal(size=(N,N))
opm_fft[0,0] = 0 # remove DC component
freqs = np.fft.fftfreq(N,1/N)
freqs = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)

decay = 5
opm_fft *= np.exp(-freqs/decay)

omap = np.fft.ifft2(opm_fft)
omap *= np.abs(omap)**1.6/np.abs(omap)
omap *= 0.16 / np.median(np.abs(omap)) # normalize median to data
omap *= np.clip(np.abs(omap),0,0.8) / np.abs(omap) # clip max os to 0.8

# assign modulation ratios
mrmap = rng.chisquare(4,size=(N,N))
mrmap *= 0.52/np.median(mrmap)
mrmap = np.clip(mrmap,0.02,1.57)

# compute elongation of each rf and dc/ac ratio
oris = np.linspace(0,np.pi,100,endpoint=False) - np.pi/2
phss = np.linspace(0,2*np.pi,100,endpoint=False) - np.pi

kl2 = 2

def ori_fact(gam,ori):
    return np.exp(-kl2*(1+(1-gam**2)/gam**2*np.sin(ori)**2)/2)

def phs_fact(dc,phs):
    return np.fmax(0,dc + np.cos(phs))

gams = np.linspace(0.4,1,301)
resps = ori_fact(gams[:,None],np.linspace(0,np.pi,36,endpoint=False)[None,:])
r0,r1,_ = af.calc_dc_ac_comp(resps)
oss = np.where(r0 > 0, r1/r0, 0)

gam_os_itp = interpolate.interp1d(oss,gams,fill_value='extrapolate')
r0_os_itp = interpolate.interp1d(oss,r0,fill_value='extrapolate')

gam_map = gam_os_itp(np.abs(omap))

dcs = np.linspace(0,51,511)
resps = phs_fact(dcs[:,None],np.linspace(0,2*np.pi,36,endpoint=False)[None,:])
r0,r1,_ = af.calc_dc_ac_comp(resps)
mrs = np.where(r0 > 0, 2*r1/r0, 0)

dc_mr_itp = interpolate.interp1d(mrs,dcs,fill_value='extrapolate')
r0_mr_itp = interpolate.interp1d(mrs,r0,fill_value='extrapolate')

dc_map = dc_mr_itp(mrmap)

r0_map = r0_mr_itp(mrmap) * r0_os_itp(np.abs(omap))

# create L4 phase map
sig2 = 0.00095

rf_sct_scale = 1.5
pol_scale = 0.7
L_mm = N/11
mag_fact = 0.02
L_deg = L_mm / np.sqrt(mag_fact)
grate_freq = 0.06

rf_sct_map,pol_map = mf.gen_rf_sct_map(N,sig2,rf_sct_scale,pol_scale)
abs_phs = mf.gen_abs_phs_map(N,rf_sct_map,pol_map,0,grate_freq,L_deg)

# create prior distribution
if bayes_iter == 0:
    # params:
    # 0: Jee mult
    # 1: Jei mult
    # 2: Jie mult
    # 3: Jii mult
    # 4: s mult
    # 5: sei mult
    # 6: sii mult
    # 7: noise level
    # 8: grec
    full_prior = BoxUniform(low =torch.tensor([0.9,0.9,0.9,0.9,0.5,0.9,0.9,0.01,0.9],device=device),
                            high=torch.tensor([1.1,1.1,1.1,1.1,2.0,1.1,1.1,0.50,1.1],device=device),)

    full_prior,_,_ = process_prior(full_prior)
else:
    with open(f'./../notebooks/l23_sel_mod_conl4_posterior_{bayes_iter:d}.pkl','rb') as handle:
        full_prior = pickle.load(handle)

# create distances between grid points
N = 60

xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

# define simulation functions
def integrate_sheet(xe0,xi0,inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,
                    het_lev,N,
                    t0,dt,Nt,tsamp=None,te=0.02,ti=0.01):
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
    
    xe = xe0.copy()
    xi = xi0.copy()
    
    rng = np.random.default_rng(0)
    
    if np.isscalar(Jee):
        noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
        Wee = Jee*kerne.reshape(N**2,N**2)*noise[:,:]
        noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
        Wei = Jei*kernei.reshape(N**2,N**2)*noise[:,:]
        noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
        Wie = Jie*kerne.reshape(N**2,N**2)*noise[:,:]
        noise = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
        Wii = Jii*kernii.reshape(N**2,N**2)*noise[:,:]
        
        Wee = Wee[:,:,None]
        Wei = Wei[:,:,None]
        Wie = Wie[:,:,None]
        Wii = Wii[:,:,None]
        
        if len(xe.shape) == 1:
            xe = xe[:,None]
            xi = xi[:,None]
        
        nprm = 1
        resps = np.zeros((2,N**2,1,len(tsamp)))
    else:
        noise = rng.gamma(shape=1/het_lev[None,None,:]**2,scale=het_lev[None,None,:]**2,
                          size=(N**2,N**2,len(Jee)))
        Wee = Jee[None,None,:]*kerne.reshape(N**2,N**2,-1)*noise
        noise = rng.gamma(shape=1/het_lev[None,None,:]**2,scale=het_lev[None,None,:]**2,
                          size=(N**2,N**2,len(Jee)))
        Wei = Jei[None,None,:]*kernei.reshape(N**2,N**2,-1)*noise
        noise = rng.gamma(shape=1/het_lev[None,None,:]**2,scale=het_lev[None,None,:]**2,
                          size=(N**2,N**2,len(Jee)))
        Wie = Jie[None,None,:]*kerne.reshape(N**2,N**2,-1)*noise
        noise = rng.gamma(shape=1/het_lev[None,None,:]**2,scale=het_lev[None,None,:]**2,
                          size=(N**2,N**2,len(Jee)))
        Wii = Jii[None,None,:]*kernii.reshape(N**2,N**2,-1)*noise
        
        if len(xe.shape) == 1:
            xe = xe[:,None] * np.ones(len(Jee))[None,:]
            xi = xi[:,None] * np.ones(len(Jee))[None,:]
            
        nprm = len(Jee)
        resps = np.zeros((2,N**2,len(Jee),len(tsamp)))
    
    # for t_idx in range(Nt):
    #     ff_inp = inp(t0+t_idx*dt)
    #     ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg))
    #     yi = np.fmin(1e5,np.fmax(0,xia+xin+xig))
    #     if t_idx in tsamp:
    #         resps[0,:,:,samp_idx] = ye
    #         resps[1,:,:,samp_idx] = yi
    #         samp_idx += 1
    #     net_ee = np.einsum('ijk,jk->ik',Wee,ye) + ff_inp
    #     net_ei = np.einsum('ijk,jk->ik',Wei,yi)
    #     net_ie = np.einsum('ijk,jk->ik',Wie,ye) + ff_inp
    #     net_ii = np.einsum('ijk,jk->ik',Wii,yi)
    #     xea += ((1-frac_n)*net_ee - xea)*dt/ta
    #     xen += (frac_n*net_ee - xen)*dt/tn
    #     xeg += (net_ei - xeg)*dt/tg
    #     xia += ((1-frac_n)*net_ie - xia)*dt/ta
    #     xin += (frac_n*net_ie - xin)*dt/tn
    #     xig += (net_ii - xig)*dt/tg
        
    # ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
    # yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
    # return xea,xen,xeg,xia,xin,xig,np.concatenate((ye,yi))
    
    def dyn_func(t,x,ncell,nprm=1):
        x = x.reshape((-1,nprm))
        xe = x[0*ncell:1*ncell,:]
        xi = x[1*ncell:2*ncell,:]
        
        ff_inp = inp(t)

        ye = np.fmin(1e5,np.fmax(0,xe))
        yi = np.fmin(1e5,np.fmax(0,xi))
        
        net_ee = np.einsum('ijk,jk->ik',Wee,ye) + ff_inp[:,None]
        net_ei = np.einsum('ijk,jk->ik',Wei,yi)
        net_ie = np.einsum('ijk,jk->ik',Wie,ye) + ff_inp[:,None]
        net_ii = np.einsum('ijk,jk->ik',Wii,yi)
        
        dx = np.zeros_like(x)
        dx[0*ncell:1*ncell,:] = (net_ee + net_ei - xe)/te
        dx[1*ncell:2*ncell,:] = (net_ie + net_ii - xi)/ti
        
        return dx.flatten()
    
    x0 = np.concatenate((xe,xi),axis=0).flatten()
    
    start_time = time.process_time()
    max_time = 60
    def time_event(t,x,ncell,nprm):
        int_time = (start_time + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True
    
    sol = integrate.solve_ivp(dyn_func,(0,dt*Nt),y0=x0,t_eval=tsamp*dt,args=(N**2,nprm),method='RK23')#,events=time_event)
    if sol.status != 0:
        x = np.nan*np.ones((2*N**2*nprm,len(tsamp)))
    else:
        x = sol.y
    x = x.reshape((-1,nprm,len(tsamp)))
    
    xe = x[0*N**2:1*N**2,:]
    xi = x[1*N**2:2*N**2,:]
    
    ye = np.fmin(1e5,np.fmax(0,xe))
    yi = np.fmin(1e5,np.fmax(0,xi))
    
    resps[0] = ye
    resps[1] = yi
        
    return resps

def get_J(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (|Jee|-|Jii|)/(|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    
    returns: [Jee,Jei,Jie,Jii]
    '''
    Jei = -10**(theta[:,2] + theta[:,3])
    Jie =  10**(theta[:,2] - theta[:,3])
    Jee_m_Jii = (-Jei + Jie) * theta[:,1]
    Jee_p_Jii_2 = 4*((theta[:,0] - 1) * Jei*Jie) + Jee_m_Jii**2
    Jee = 0.5*(Jee_m_Jii + torch.sqrt(Jee_p_Jii_2))
    Jii = -(Jee - Jee_m_Jii)
    
    return Jee,Jei,Jie,Jii

def get_sheet_resps(theta,N):
    nori = 8
    nphs = 8
    nint = 5
    nwrm = 4 * nint * nphs
    dt = 1 / (nint * nphs * 3)
    
    oris = np.linspace(0,np.pi,nori,endpoint=False)
    
    tsamp = nwrm-1 + np.arange(0,nphs) * nint
    resps = np.zeros((theta.shape[0],2,N**2,nori,nphs))
    for prm_idx in range(theta.shape[0]):
        Jee = theta[prm_idx,0].item() * Jee0
        Jei = -theta[prm_idx,1].item() * Jei0
        Jie = theta[prm_idx,2].item() * Jie0
        Jii = -theta[prm_idx,3].item() * Jii0
        
        s_e = se0 * theta[prm_idx,4].item()
        s_ei = sei0 * theta[prm_idx,4].item() * theta[prm_idx,5].item()
        s_ii = sii0 * theta[prm_idx,4].item() * theta[prm_idx,6].item()
        
        kerne = np.exp(-(dss/(s_e))**2)
        norm = kerne.sum(axis=1).mean(axis=0)
        kerne /= norm
        kernei = np.exp(-(dss/(s_ei))**2)
        norm = kernei.sum(axis=1).mean(axis=0)
        kernei /= norm
        kernii = np.exp(-(dss/(s_ii))**2)
        norm = kernii.sum(axis=1).mean(axis=0)
        kernii /= norm
        
        het_lev = theta[prm_idx,7].item()
        
        max_eig = eigs(np.block([
            [Jee*kerne.reshape(N**2,N**2), Jei*kernei.reshape(N**2,N**2)],
            [Jie*kerne.reshape(N**2,N**2), Jii*kernii.reshape(N**2,N**2)]
        ]),k=1,which='LR',return_eigenvectors=False)
        
        Jee *= theta[prm_idx,8].item()/np.real(max_eig[0])
        Jei *= theta[prm_idx,8].item()/np.real(max_eig[0])
        Jie *= theta[prm_idx,8].item()/np.real(max_eig[0])
        Jii *= theta[prm_idx,8].item()/np.real(max_eig[0])
        
        for ori_idx,ori in enumerate(oris):
            phs_map_flat = mf.gen_abs_phs_map(N,rf_sct_map,pol_map,ori,grate_freq,L_deg).flatten()
            gam_map_flat = gam_map.flatten()
            dc_map_flat = dc_map.flatten()
            ori_map_flat = np.angle(omap).flatten() / 2
            r0_map_flat = r0_map.flatten()
            def ff_inp(t):
                return ori_fact(gam_map_flat,ori_map_flat-ori) * \
                       phs_fact(dc_map_flat,phs_map_flat + 2*np.pi*3*t) / r0_map_flat
            resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),
                                    ff_inp,Jee,Jei,Jie,Jii,
                                    kerne,kernei,kernii,het_lev,N,0,dt,nwrm+nint*nphs,tsamp)
            resps[prm_idx,:,:,ori_idx,:] = resp.transpose((2,0,1,3))
        
    return resps

def sheet_simulator(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (|Jee|-|Jii|)/(|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = s_e
    theta[:,5] = s_ei / s_e
    theta[:,6] = s_ii / s_ei
    theta[:,7] = p_ker
    theta[:,8] = het_level
    theta[:,9] = inp_mult
    theta[:,10] = s_mult
    theta[:,11] = het_mult
    theta[:,12] = J_mult
    
    returns: [q1_os,q2_os,q3_os,mu_os,sig_os,q1_mr,q2_mr,q3_mr,mu_mr,sig_mr,mu_mm]
    os = excitatory orientation selectivity
    mr = excitatory modulation ratio
    mm = input-output mismatch
    '''
    
    resps = get_sheet_resps(theta,N)
    
    opm,mr = af.calc_OPM_MR(resps[:,0,:,:,:])
    os = np.abs(opm)
    
    inp_po = np.angle(omap.flatten())*180/(2*np.pi)
    inp_po[inp_po > 90] -= 180
    out_po = np.angle(opm)*180/(2*np.pi)
    out_po[out_po > 90] -= 180

    mm = np.abs(inp_po - out_po)
    mm[mm > 90] = 180 - mm[mm > 90]
    
    _,raps = af.get_fps(opm.reshape(-1,N,N))
    pwd = af.calc_pinwheel_density_from_raps(np.arange(raps.shape[-1])[None,:]/N,
                                             raps,continuous=True)
    
    out = torch.zeros((theta.shape[0],12),dtype=theta.dtype).to(theta.device)
    out[:,0:3] = torch.tensor(np.quantile(os,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out[:,3] = torch.tensor(np.mean(os,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,4] = torch.tensor(np.std(os,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,5:8] = torch.tensor(np.quantile(mr,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out[:,8] = torch.tensor(np.mean(mr,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,9] = torch.tensor(np.std(mr,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,10] = torch.tensor(np.mean(mm,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,11] = torch.tensor(pwd,dtype=theta.dtype).to(theta.device)
    
    valid_idx = torch.all(torch.tensor(resps) < 5e4,axis=(1,2,3,4))
    
    return torch.where(valid_idx[:,None],out,torch.tensor([torch.nan])[:,None])

start = time.process_time()

# sample from prior
theta = full_prior.sample((num_samp,))

# simulate sheet
x = sheet_simulator(theta)

print(f'Simulating samples took',time.process_time() - start,'s\n')

# save results
with open(res_file, 'wb') as handle:
    pickle.dump({
        'theta': theta,
        'x': x,
    }, handle)
