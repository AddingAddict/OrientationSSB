import os
import pickle
import time
import argparse

import numpy as np
import torch
from scipy.interpolate import CubicSpline

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

res_dir = res_dir + 'sbi_l23_sel_mod/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

# load L4 responses
with open('./../results/L4_sel/seed=0.pkl', 'rb') as handle:
    L4_rates = pickle.load(handle)['L4_rates'][0]
L4_rates /= np.nanmean(L4_rates,axis=(-2,-1),keepdims=True)

L4_rates_itp = CubicSpline(np.arange(0,8+1) * 1/(3*8),
                           np.concatenate((L4_rates,L4_rates[:,:,0:1]),axis=-1),
                           axis=-1,bc_type='periodic')

# create prior distribution
if bayes_iter == 0:
    # load posterior of phase ring connectivity parameters
    with open(f'./../notebooks/l23_patt_posterior_5.pkl', 'rb') as handle:
        posterior = pickle.load(handle)
        
    full_prior = PostTimesBoxUniform(posterior,
        post_low =torch.tensor([ 0.0,-2.0,-4.0,-2.0, 0.01, 0.5, 0.3, 2.0, 0.01],device=device),
        post_high=torch.tensor([ 1.0, 2.0,-0.0, 2.0, 0.06, 0.9, 0.9, 4.0, 0.5],device=device),
        low =torch.tensor([0.5, 0.5, 0.5],device=device),
        high=torch.tensor([1.5, 1.5, 1.5],device=device),)

    full_prior,_,_ = process_prior(full_prior)
else:
    with open(f'./../notebooks/l23_sel_mod_{bayes_iter:d}.pkl','rb') as handle:
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
        
        if len(xea.shape) == 1:
            xea = xea[:,None]
            xen = xen[:,None]
            xeg = xeg[:,None]
            xia = xia[:,None]
            xin = xin[:,None]
            xig = xig[:,None]
        
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
        
        if len(xea.shape) == 1:
            xea = xea[:,None] * np.ones(len(Jee))[None,:]
            xen = xen[:,None] * np.ones(len(Jee))[None,:]
            xeg = xeg[:,None] * np.ones(len(Jee))[None,:]
            xia = xia[:,None] * np.ones(len(Jee))[None,:]
            xin = xin[:,None] * np.ones(len(Jee))[None,:]
            xig = xig[:,None] * np.ones(len(Jee))[None,:]
            
        resps = np.zeros((2,N**2,len(Jee),len(tsamp)))
    
    for t_idx in range(Nt):
        ff_inp = inp(t0+t_idx*dt)
        ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
        yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
        if t_idx in tsamp:
            resps[0,:,:,samp_idx] = ye
            resps[1,:,:,samp_idx] = yi
            samp_idx += 1
        net_ee = np.einsum('ijk,jk->ik',Wee,ye) + ff_inp
        net_ei = np.einsum('ijk,jk->ik',Wei,yi)
        net_ie = np.einsum('ijk,jk->ik',Wie,ye) + ff_inp
        net_ii = np.einsum('ijk,jk->ik',Wii,yi)
        xea += ((1-frac_n)*net_ee - xea)*dt/ta
        xen += (frac_n*net_ee - xen)*dt/tn
        xeg += (net_ei - xeg)*dt/tg
        xia += ((1-frac_n)*net_ie - xia)*dt/ta
        xin += (frac_n*net_ie - xin)*dt/tn
        xig += (net_ii - xig)*dt/tg
        
    # ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg-threshe)**ne)
    # yi = np.fmin(1e5,np.fmax(0,xia+xin+xig-threshi)**ni)
    # return xea,xen,xeg,xia,xin,xig,np.concatenate((ye,yi))
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
    Jee,Jei,Jie,Jii = get_J(theta)
    Jee *= theta[:,5]
    Jei *= theta[:,5]
    Jie *= theta[:,5]
    Jii *= theta[:,5]
    
    thresh = 0
    nori = 8
    nphs = 8
    nint = 5
    nwrm = 12 * nint * nphs
    dt = 1 / (nint * nphs * 3)
    
    s_e = theta[:,10]*theta[:,4]
    s_ei = s_e * theta[:,5]
    s_ii = s_ei * theta[:,6]
    kerne = np.exp(-(dss[:,:,None]/(s_e[None,None,:]))**theta[None,None,:,7])
    norm = kerne.sum(axis=1).mean(axis=0)
    kerne /= norm[None,None,:]
    kernei = np.exp(-(dss[:,:,None]/(s_ei[None,None,:]))**theta[None,None,:,7])
    norm = kernei.sum(axis=1).mean(axis=0)
    kernei /= norm[None,None,:]
    kernii = np.exp(-(dss[:,:,None]/(s_ii[None,None,:]))**theta[None,None,:,7])
    norm = kernii.sum(axis=1).mean(axis=0)
    kernii /= norm[None,None,:]
    
    inp_mult = theta[:,9].numpy()
    
    tsamp = nwrm-1 + np.arange(0,nphs) * nint
    resps = np.zeros((theta.shape[0],2,N**2,nori,nphs))
    for ori_idx in range(nori):
        def ff_inp(t):
            return inp_mult[None,:] * L4_rates_itp(t)[:,ori_idx,None]
        # xea,xen,xeg,xia,xin,xig,resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
        #                          np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
        #                          ff_inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,theta[:,11]*theta[:,8],N,2,2,
        #                          thresh,thresh,0,dt,nwrm)
        resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                 np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                 ff_inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,theta[:,11]*theta[:,8],N,2,2,
                                 thresh,thresh,0,dt,nwrm+nint*nphs,tsamp)
        resps[:,:,:,ori_idx,:] = resp.transpose((2,0,1,3))
        # for phs_idx in range(nphs-1):
        #     xea,xen,xeg,xia,xin,xig,resp = integrate_sheet(xea,xen,xeg,xia,xin,xig,
        #                          ff_inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,theta[:,11]*theta[:,8],N,2,2,
        #                          thresh,thresh,0,dt,nwrm)
        #     resps[:,:,:,ori_idx,phs_idx+1] = resp.T.reshape(theta.shape[0],2,N**2)
        
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
    
    returns: [q1_os,q2_os,q3_os,mu_os,sig_os,q1_mr,q2_mr,q3_mr,mu_mr,sig_mr]
    os = excitatory orientation selectivity
    mr = excitatory modulation ratio
    '''
    
    resps = get_sheet_resps(theta,N)
    
    os,mr = af.calc_OS_MR(resps[:,0,:,:,:])
    
    out = torch.zeros((theta.shape[0],10),dtype=theta.dtype).to(theta.device)
    out[:,0:3] = torch.tensor(np.quantile(os,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out[:,3] = torch.tensor(np.mean(os,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,4] = torch.tensor(np.std(os,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,5:8] = torch.tensor(np.quantile(mr,[0.25,0.50,0.75],axis=1).T,dtype=theta.dtype).to(theta.device)
    out[:,8] = torch.tensor(np.mean(mr,axis=1),dtype=theta.dtype).to(theta.device)
    out[:,9] = torch.tensor(np.std(mr,axis=1),dtype=theta.dtype).to(theta.device)
    
    valid_idx = torch.all(torch.tensor(resps) < 5e4,axis=(1,2,3,4))
    
    return torch.where(valid_idx[:,None],out,torch.tensor([torch.nan])[:,None])

start = time.process_time()

theta = torch.zeros((0,12),dtype=torch.float32,device=device)
x = torch.zeros((0,10),dtype=torch.float32,device=device)
for outer_idx in range(num_outer):
    print(f'Outer loop {outer_idx+1}/{num_outer}')
    start_outer = time.process_time()

    # sample from prior
    theta_samp = full_prior.sample((num_inner,))

    # simulate sheet
    x_samp = sheet_simulator(theta_samp)

    # append results
    theta = torch.cat((theta,theta_samp),dim=0)
    x = torch.cat((x,x_samp),dim=0)

    print(f'  Outer loop took',time.process_time() - start_outer,'s\n')

print(f'Simulating samples took',time.process_time() - start,'s\n')

# save results
with open(res_file, 'wb') as handle:
    pickle.dump({
        'theta': theta,
        'x': x,
    }, handle)
