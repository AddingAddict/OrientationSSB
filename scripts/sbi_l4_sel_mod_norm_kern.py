import os
import pickle
import time
import argparse

import numpy as np
import torch
from scipy import interpolate

from sbi.utils.user_input_checks import process_prior

import analyze_func as af
import map_func as mf
from sbi_func import PostTimesBoxUniform

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', '-i', help='completely arbitrary job id label',type=int, default=0)
parser.add_argument('--num_inner', '-ni', help='number of samples per outer loop',type=int, default=50)
parser.add_argument('--num_outer', '-no', help='number of outer loops',type=int, default=10)
parser.add_argument('--bayes_iter', '-bi', help='bayessian inference interation (0 = use prior, 1 = use first posterior)',type=int, default=0)
args = vars(parser.parse_args())
job_id = int(args['job_id'])
num_inner = int(args['num_inner'])
num_outer = int(args['num_outer'])
bayes_iter = int(args['bayes_iter'])

print("Bayesian iteration:", bayes_iter)
print("Job ID:", job_id)

device = torch.device("cpu")

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'sbi_l4_sel_mod_norm_kern/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

# create prior distribution
if bayes_iter == 0:
    # load posterior of phase ring connectivity parameters
    with open('./../notebooks/phase_ring_posterior.pkl','rb') as handle:
        posterior = pickle.load(handle)
        
    full_prior = PostTimesBoxUniform(posterior,
                                     post_low =torch.tensor([ 0.0, 0.0,-3.0,-2.0, 0.2],device=device),
                                     post_high=torch.tensor([ 1.0, 1.0,-0.0, 1.0, 2.0],device=device),
                                     low =torch.tensor([0.1, 0.3, 2.0],device=device),
                                     high=torch.tensor([1.0, 1.0, 4.0],device=device),)

    full_prior,_,_ = process_prior(full_prior)
else:
    with open(f'./../notebooks/l4_norm_kern_posterior_{bayes_iter:d}.pkl','rb') as handle:
        full_prior = pickle.load(handle)

# create L4 orientation map
N = 60

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

# compute elongation of each rf
oris = np.linspace(0,np.pi,100,endpoint=False) - np.pi/2
phss = np.linspace(0,2*np.pi,100,endpoint=False) - np.pi

kl2 = 2

def elong_inp(gam,ori,phs):
    return 1 + np.cos(phs)*np.exp(-kl2*(1+(1-gam**2)/gam**2*np.sin(ori)**2)/2)

gams = np.linspace(0.4,1,301)
resps = np.fmax(0,elong_inp(gams[:,None,None],np.linspace(0,np.pi,36,endpoint=False)[None,:,None],np.linspace(0,2*np.pi,36,endpoint=False)[None,None,:])-1)**2
oss,_ = af.calc_OS_MR(resps)

gam_os_itp = interpolate.interp1d(oss,gams,fill_value='extrapolate')

gam_map = gam_os_itp(np.abs(omap))

# compute rf scatter and ON/OFF bias maps
sig2 = 0.00095

rf_sct_scale = 1.5
pol_scale = 0.7
L_mm = N/11
mag_fact = 0.02
L_deg = L_mm / np.sqrt(mag_fact)
grate_freq = 0.06

sctmap,polmap = mf.gen_rf_sct_map(N,sig2,rf_sct_scale,pol_scale)
abs_phs = mf.gen_abs_phs_map(N,sctmap,polmap,0,grate_freq,L_deg)

xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

# define simulation functions
def integrate_sheet(xea0,xen0,xeg0,xia0,xin0,xig0,inp,Jee,Jei,Jie,Jii,kern,N,ne,ni,threshe,threshi,
                    t0,dt,Nt,ta=0.01,tn=0.300,tg=0.01,frac_n=0.7,lat_frac=1.0):
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
    
    xea = xea0.copy()
    xen = xen0.copy()
    xeg = xeg0.copy()
    xia = xia0.copy()
    xin = xin0.copy()
    xig = xig0.copy()
    
    if np.isscalar(Jee):
        Wee = Jee*np.eye(N**2)
        Wei = Jei*np.eye(N**2)
        Wie = Jie*np.eye(N**2)
        Wii = Jii*np.eye(N**2)
        
        Wee += lat_frac*Jee*kern.reshape(N**2,N**2)
        Wie += lat_frac*Jie*kern.reshape(N**2,N**2)
        
        Wee = Wee[:,:,None]
        Wei = Wei[:,:,None]
        Wie = Wie[:,:,None]
        Wii = Wii[:,:,None]
        
        if len(xea.shape) == 1:
            xea = xea[:,None]
            xen = xen[:,None]
            xeg = xeg[:,None]
            xia = xia[:,None]
            xin = xin[:,None]
            xig = xig[:,None]
    else:
        Wee = Jee[None,None,:]*np.eye(N**2)[:,:,None]
        Wei = Jei[None,None,:]*np.eye(N**2)[:,:,None]
        Wie = Jie[None,None,:]*np.eye(N**2)[:,:,None]
        Wii = Jii[None,None,:]*np.eye(N**2)[:,:,None]
        
        Wee += lat_frac[None,None,:]*Jee[None,None,:]*kern.reshape(N**2,N**2,-1)
        Wie += lat_frac[None,None,:]*Jie[None,None,:]*kern.reshape(N**2,N**2,-1)
        
        if len(xea.shape) == 1:
            xea = xea[:,None] * np.ones(len(Jee))[None,:]
            xen = xen[:,None] * np.ones(len(Jee))[None,:]
            xeg = xeg[:,None] * np.ones(len(Jee))[None,:]
            xia = xia[:,None] * np.ones(len(Jee))[None,:]
            xin = xin[:,None] * np.ones(len(Jee))[None,:]
            xig = xig[:,None] * np.ones(len(Jee))[None,:]
    
    for t_idx in range(Nt):
        ff_inp = inp(t0+t_idx*dt)
        ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
        yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
        net_ee = np.einsum('ijk,jk->ik',Wee,ye) + ff_inp[:,None]
        net_ei = np.einsum('ijk,jk->ik',Wei,yi)
        net_ie = np.einsum('ijk,jk->ik',Wie,ye) + ff_inp[:,None]
        net_ii = np.einsum('ijk,jk->ik',Wii,yi)
        xea += ((1-frac_n)*net_ee - xea)*dt/ta
        xen += (frac_n*net_ee - xen)*dt/tn
        xeg += (net_ei - xeg)*dt/tg
        xia += ((1-frac_n)*net_ie - xia)*dt/ta
        xin += (frac_n*net_ie - xin)*dt/tn
        xig += (net_ii - xig)*dt/tg
        
    ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
    yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
    return xea,xen,xeg,xia,xin,xig,np.concatenate((ye,yi))

def get_J(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (Ω_I - Ω_E)/(|Jei| + |Jie|) = 1 - (|Jee| + |Jii|) / (|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    
    returns: [Jee,Jei,Jie,Jii]
    '''
    Jei = -10**(theta[:,2] + theta[:,3])
    Jie =  10**(theta[:,2] - theta[:,3])
    Jee_p_Jii = (-Jei + Jie) * (1 - theta[:,1])
    Jee_m_Jii_2 = 4*(theta[:,0] * Jei*Jie) + Jee_p_Jii**2
    Jee = 0.5*(Jee_p_Jii + torch.sqrt(Jee_m_Jii_2))
    Jii = -(Jee_p_Jii - Jee)
    
    return Jee,Jei,Jie,Jii

def get_sheet_resps(theta,N,gam_map,ori_map,rf_sct_map,pol_map):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (Ω_I - Ω_E)/(|Jei| + |Jie|) = 1 - (|Jee| + |Jii|) / (|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = J_lat / J_pair
    theta[:,5] = J_fact
    theta[:,6] = l_ker
    theta[:,7] = p_ker
    
    returns: resps, array of shape (theta.shape[0],2,N**2,nori=8,nphs=8)
    '''
    Jee,Jei,Jie,Jii = get_J(theta)
    Jee *= theta[:,5]
    Jei *= theta[:,5]
    Jie *= theta[:,5]
    Jii *= theta[:,5]
    
    c = 100
    thresh = c
    nori = 8
    nphs = 8
    nint = 5
    nwrm = 12 * nint * nphs
    dt = 1 / (nint * nphs * 3)
    oris = np.linspace(0,np.pi,nori,endpoint=False)
    
    kern = np.exp(-(dss[:,:,None]/(np.sqrt(sig2)*theta[None,None,:,6]))**theta[None,None,:,7])
    norm = kern.sum(axis=1).mean(axis=0)
    kern /= norm[None,None,:]
    
    resps = np.zeros((theta.shape[0],2,N**2,nori,nphs))
    for ori_idx,ori in enumerate(oris):
        phs_map_flat = mf.gen_abs_phs_map(N,rf_sct_map,pol_map,ori,grate_freq,L_deg).flatten()
        gam_map_flat = gam_map.flatten()
        ori_map_flat = ori_map.flatten()
        def ff_inp(t):
            return c*elong_inp(gam_map_flat,ori-ori_map_flat,phs_map_flat+2*np.pi*3*t)
        xea,xen,xeg,xia,xin,xig,resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                 np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                 ff_inp,Jee,Jei,Jie,Jii,kern,N,2,2,
                                 thresh,thresh,0,dt,nwrm,lat_frac=theta[:,4])
        resps[:,:,:,ori_idx,0] = resp.T.reshape(theta.shape[0],2,N**2)
        for phs_idx in range(nphs-1):
            xea,xen,xeg,xia,xin,xig,resp = integrate_sheet(xea,xen,xeg,xia,xin,xig,
                                 ff_inp,Jee,Jei,Jie,Jii,kern,N,2,2,
                                 thresh,thresh,phs_idx*nint*dt,dt,nint,lat_frac=theta[:,4])
            resps[:,:,:,ori_idx,phs_idx+1] = resp.T.reshape(theta.shape[0],2,N**2)
        
    return resps

def sheet_simulator(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (Ω_I - Ω_E)/(|Jei| + |Jie|) = 1 - (|Jee| + |Jii|) / (|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = J_lat / J_pair
    theta[:,5] = J_fact
    theta[:,6] = l_ker
    theta[:,7] = p_ker
    
    returns: [q1_os,q2_os,q3_os,mu_os,sig_os,q1_mr,q2_mr,q3_mr,mu_mr,sig_mr]
    os = excitatory orientation selectivity
    mr = excitatory modulation ratio
    '''
    
    _,_,_,Jii = get_J(theta)
    
    resps = get_sheet_resps(theta,N,gam_map,np.angle(omap),sctmap,polmap)
    
    resp_opm,mr = af.calc_OPM_MR(resps[:,0,:,:,:])
    os = np.abs(resp_opm)
    
    out = torch.zeros((theta.shape[0],10),dtype=theta.dtype).to(theta.device)
    out[:,0:3] = torch.tensor(np.quantile(os,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out[:,3] = torch.tensor(np.mean(os,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,4] = torch.tensor(np.std(os,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,5:8] = torch.tensor(np.quantile(mr,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out[:,8] = torch.tensor(np.mean(mr,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,9] = torch.tensor(np.std(mr,axis=1),dtype=theta.dtype).to(theta.device)
    
    inp_PO = np.angle(omap.flatten())*180/(2*np.pi)
    inp_PO[inp_PO > 90] -= 180
    rate_PO = np.angle(resp_opm)*180/(2*np.pi)
    rate_PO[rate_PO > 90] -= 180

    mm = np.abs(inp_PO - rate_PO)
    mm[mm > 90] = 180 - mm[mm > 90]
    
    out_mm = torch.zeros((theta.shape[0],5),dtype=theta.dtype).to(theta.device)
    out_mm[:,0:3] = torch.tensor(np.quantile(mm,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out_mm[:,3] = torch.tensor(np.mean(mm,axis=1),dtype=theta.dtype).to(theta.device)
    out_mm[:,4] = torch.tensor(np.std(mm,axis=1),dtype=theta.dtype).to(theta.device)
    
    valid_idx = torch.all(torch.tensor(resps) < 5e4,axis=(1,2,3,4)) & (Jii < 0)
    
    return torch.where(valid_idx[:,None],out,torch.tensor([torch.nan])[:,None]), \
        torch.where(valid_idx[:,None],out_mm,torch.tensor([torch.nan])[:,None])

start = time.process_time()

theta = torch.zeros((0,8),dtype=torch.float32,device=device)
x = torch.zeros((0,10),dtype=torch.float32,device=device)
x_mm = torch.zeros((0,5),dtype=torch.float32,device=device)
for outer_idx in range(num_outer):
    print(f'Outer loop {outer_idx+1}/{num_outer}')
    start_outer = time.process_time()

    # sample from prior
    theta_samp = full_prior.sample((num_inner,))

    # simulate sheet
    x_samp,mm_samp = sheet_simulator(theta_samp)

    # append results
    theta = torch.cat((theta,theta_samp),dim=0)
    x = torch.cat((x,x_samp),dim=0)
    x_mm = torch.cat((x_mm,mm_samp),dim=0)

    print(f'  Outer loop took',time.process_time() - start_outer,'s\n')

print(f'Simulating samples took',time.process_time() - start,'s\n')

# save results
with open(res_file, 'wb') as handle:
    pickle.dump({
        'theta': theta,
        'x': x,
        'x_mm': x_mm,
    }, handle)
