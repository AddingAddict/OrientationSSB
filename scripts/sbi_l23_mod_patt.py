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

res_dir = res_dir + 'sbi_l23_mod_patt/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

# create prior distribution
if bayes_iter == 0:
    prior = BoxUniform(low =torch.tensor([ 0.0,-2.0,-3.0,-2.0, 0.01, 0.5, 0.3, 2.0, 0.01],device=device),
                   high=torch.tensor([ 1.0, 2.0,-0.0, 1.0, 0.04, 0.9, 0.9, 4.0, 0.5],device=device),)

    prior,_,_ = process_prior(prior)
else:
    with open(f'./../notebooks/l23_patt_posterior_{bayes_iter:d}.pkl','rb') as handle:
        prior = pickle.load(handle)

# create distances between grid points
N = 60

xs,ys = np.meshgrid(np.arange(N)/N,np.arange(N)/N)
dxs = np.abs(xs[:,:,None,None] - xs[None,None,:,:])
dxs[dxs > 0.5] = 1 - dxs[dxs > 0.5]
dys = np.abs(ys[:,:,None,None] - ys[None,None,:,:])
dys[dys > 0.5] = 1 - dys[dys > 0.5]
dss = np.sqrt(dxs**2 + dys**2).reshape(N**2,N**2)

idxs = np.digitize(dss,np.linspace(0,np.max(dss),nbins+1))

# define simulation functions
def integrate_sheet_no_nmda(xea0,xeg0,xia0,xig0,inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,
                    het_lev,N,ne,ni,threshe,threshi,
                    t0,dt,Nt,ta=0.02,tg=0.01):
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
    xeg = xeg0.copy()
    xia = xia0.copy()
    xig = xig0.copy()
    
    rng = np.random.default_rng(0)
    
    if np.isscalar(Jee):
        if het_lev > 0:
            noise_ee = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
            noise_ei = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
            noise_ie = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
            noise_ii = rng.gamma(shape=1/het_lev**2,scale=het_lev**2,size=(N**2,N**2))
        else:
            noise_ee = np.ones((N**2,N**2))
            noise_ei = np.ones((N**2,N**2))
            noise_ie = np.ones((N**2,N**2))
            noise_ii = np.ones((N**2,N**2))
        
        Wee = Jee*kerne.reshape(N**2,N**2)*noise_ee[:,:]
        Wei = Jei*kernei.reshape(N**2,N**2)*noise_ei[:,:]
        Wie = Jie*kerne.reshape(N**2,N**2)*noise_ie[:,:]
        Wii = Jii*kernii.reshape(N**2,N**2)*noise_ii[:,:]
        
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
        if het_lev > 0:
            noise_ee = rng.gamma(shape=1/het_lev[None,None,:]**2,scale=het_lev[None,None,:]**2,
                                 size=(N**2,N**2,len(Jee)))
            noise_ei = rng.gamma(shape=1/het_lev[None,None,:]**2,scale=het_lev[None,None,:]**2,
                                 size=(N**2,N**2,len(Jee)))
            noise_ie = rng.gamma(shape=1/het_lev[None,None,:]**2,scale=het_lev[None,None,:]**2,
                                 size=(N**2,N**2,len(Jee)))
            noise_ii = rng.gamma(shape=1/het_lev[None,None,:]**2,scale=het_lev[None,None,:]**2,
                                 size=(N**2,N**2,len(Jee)))
        else:
            noise_ee = np.ones((N**2,N**2,1))
            noise_ei = np.ones((N**2,N**2,1))
            noise_ie = np.ones((N**2,N**2,1))
            noise_ii = np.ones((N**2,N**2,1))
        
        Wee = Jee[None,None,:]*kerne.reshape(N**2,N**2,-1)*noise_ee
        Wei = Jei[None,None,:]*kernei.reshape(N**2,N**2,-1)*noise_ei
        Wie = Jie[None,None,:]*kerne.reshape(N**2,N**2,-1)*noise_ie
        Wii = Jii[None,None,:]*kernii.reshape(N**2,N**2,-1)*noise_ii
        
        if len(xea.shape) == 1:
            xea = xea[:,None] * np.ones(len(Jee))[None,:]
            xeg = xeg[:,None] * np.ones(len(Jee))[None,:]
            xia = xia[:,None] * np.ones(len(Jee))[None,:]
            xig = xig[:,None] * np.ones(len(Jee))[None,:]
    
    for t_idx in range(Nt):
        ff_inp = inp
        ye = np.fmin(1e5,np.fmax(0,xea+xeg-threshe)**ne)
        yi = np.fmin(1e5,np.fmax(0,xia+xig-threshi)**ni)
        net_ee = np.einsum('ijk,jk->ik',Wee,ye) + ff_inp[:,None]
        net_ei = np.einsum('ijk,jk->ik',Wei,yi)
        net_ie = np.einsum('ijk,jk->ik',Wie,ye) + ff_inp[:,None]
        net_ii = np.einsum('ijk,jk->ik',Wii,yi)
        xea += (net_ee - xea)*dt/ta
        xeg += (net_ei - xeg)*dt/tg
        xia += (net_ie - xia)*dt/ta
        xig += (net_ii - xig)*dt/tg
        
    ye = np.fmin(1e5,np.fmax(0,xea+xeg-threshe)**ne)
    yi = np.fmin(1e5,np.fmax(0,xia+xig-threshi)**ni)
    return xea,xeg,xia,xig,np.concatenate((ye,yi))

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

ng = np.random.default_rng(0)

npatt = 200
patts = 1 + 0.01*rng.normal(size=(npatt,N**2))

def get_sheet_resps(theta,N):
    Jee,Jei,Jie,Jii = get_J(theta)
    Jee *= theta[:,5]
    Jei *= theta[:,5]
    Jie *= theta[:,5]
    Jii *= theta[:,5]
    
    c = 100
    thresh = c
    nint = 3
    nwrm = 100 * nint
    dt = 0.01 / nint
    
    s_e = theta[:,4]
    s_ei = s_e * theta[:,5]
    s_ii = s_ei * theta[:,6]
    kerne = np.exp(-(dss[:,:,None]/(s_e[None,None,:]))**theta[None,None,:,7])
    kernei = np.exp(-(dss[:,:,None]/(s_ei[None,None,:]))**theta[None,None,:,7])
    kernii = np.exp(-(dss[:,:,None]/(s_ii[None,None,:]))**theta[None,None,:,7])
    
    resps = np.zeros((theta.shape[0],2,N**2,npatt))
    for patt_idx,patt in enumerate(patts):
        ff_inp = c * patt
        _,_,_,_,resp = integrate_sheet_no_nmda(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                 ff_inp,Jee,Jei,Jie,Jii,kerne,kernei,kernii,theta[:,8],N,2,2,
                                 thresh,thresh,0,dt,nwrm)
        resps[:,:,:,patt_idx] = resp.T.reshape(theta.shape[0],2,N**2)
        
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
    
    returns: [mod,min_r]
    mod = excitatory response modularity
    min_r = average minimum excitatory response relative to the maximum
    '''
    
    resps = get_sheet_resps(theta,N)
    
    resp_z = resps[:,0,:,:]
    resp_z = resp_z - np.mean(resp_z,axis=-1,keepdims=True)
    resp_z = resp_z / np.std(resp_z,axis=-1,keepdims=True)
    corr = np.mean(resp_z[:,None,:,:] * resp_z[:,:,None,:],-1)
    
    corr_curve = np.zeros((theta.shape[0],nbins))
    for i in range(nbins):
        corr_curve[:,i] = np.mean(corr[:,idxs == i+1],axis=(1,2))
    arg_mins = np.argmin(corr_curve,axis=1)
    corr_mins = np.array([corr_curve[i,arg_mins[i]] for i in range(theta.shape[0])])
    corr_maxs = np.array([np.max(corr_curve[i,arg_mins[i]:]) for i in range(theta.shape[0])])
    mod = corr_maxs - corr_mins
    
    min_r = np.mean(np.min(resps[:,0,:,:],axis=-2) / np.max(resps[:,0,:,:],axis=-2),axis=-1)
    
    out = torch.zeros((theta.shape[0],2),dtype=theta.dtype).to(theta.device)
    out[:,0] = torch.tensor(mod,dtype=theta.dtype).to(theta.device)
    out[:,1] = torch.tensor(min_r,dtype=theta.dtype).to(theta.device)
    
    valid_idx = torch.all(torch.tensor(resps) < 5e4,axis=(1,2,3))
    
    return torch.where(valid_idx[:,None],out,torch.tensor([torch.nan])[:,None])

start = time.process_time()

theta = torch.zeros((0,9),dtype=torch.float32,device=device)
x = torch.zeros((0,2),dtype=torch.float32,device=device)
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
