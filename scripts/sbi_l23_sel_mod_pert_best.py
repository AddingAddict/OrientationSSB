import os
import pickle
import time
import argparse

import numpy as np
import torch
from scipy.interpolate import CubicSpline
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

res_dir = res_dir + 'sbi_l23_sel_mod_pert_best/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

# set prior and grid size for model
N = 60

# load L4 responses
with open('./../results/L4_sel/seed=0.pkl', 'rb') as handle:
    L4_res_dict = pickle.load(handle)
    L4_rates = L4_res_dict['L4_rates'][0]
    L4_rate_opm = L4_res_dict['L4_rate_opm'][0]
L4_rates /= np.nanmean(L4_rates,axis=(-2,-1),keepdims=True)

L4_rates_itp = CubicSpline(np.arange(0,8+1) * 1/(3*8),
                           np.concatenate((L4_rates,L4_rates[:,:,0:1]),axis=-1),
                           axis=-1,bc_type='periodic')

full_prior = BoxUniform(low =torch.tensor([0.7,0.7,0.7,0.7,0.5,0.7,0.7,0.7,0.7],device=device),
                        high=torch.tensor([1.3,1.3,1.3,1.3,3.0,1.3,1.3,1.3,1.3],device=device),)
# create prior distribution
if bayes_iter == 0:
    params = np.load("./../notebooks/l23_params_base.npy")
else:
    params = np.load(f"./../notebooks/l23_params_base_{bayes_iter}.npy")
Jee0,Jei0,Jie0,Jii0 = 10**params[:4]

se0 = params[4]
sei0 = se0 * params[5]
sii0 = sei0 * params[6]

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
                    het_lev,N,
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
        
        if len(xea.shape) == 1:
            xea = xea[:,None] * np.ones(len(Jee))[None,:]
            xen = xen[:,None] * np.ones(len(Jee))[None,:]
            xeg = xeg[:,None] * np.ones(len(Jee))[None,:]
            xia = xia[:,None] * np.ones(len(Jee))[None,:]
            xin = xin[:,None] * np.ones(len(Jee))[None,:]
            xig = xig[:,None] * np.ones(len(Jee))[None,:]
            
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
        xea = x[0*ncell:1*ncell,:]
        xen = x[1*ncell:2*ncell,:]
        xeg = x[2*ncell:3*ncell,:]
        xia = x[3*ncell:4*ncell,:]
        xin = x[4*ncell:5*ncell,:]
        xig = x[5*ncell:6*ncell,:]
        
        ff_inp = inp(t)

        ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg)**2)
        yi = np.fmin(1e5,np.fmax(0,xia+xin+xig)**2)
        
        net_ee = np.einsum('ijk,jk->ik',Wee,ye) + ff_inp[:,None]
        net_ei = np.einsum('ijk,jk->ik',Wei,yi)
        net_ie = np.einsum('ijk,jk->ik',Wie,ye) + ff_inp[:,None]
        net_ii = np.einsum('ijk,jk->ik',Wii,yi)
        
        dx = np.zeros_like(x)
        dx[0*ncell:1*ncell,:] = ((1-frac_n)*net_ee - xea)/ta
        dx[1*ncell:2*ncell,:] = (frac_n*net_ee - xen)/tn
        dx[2*ncell:3*ncell,:] = (net_ei - xeg)/tg
        dx[3*ncell:4*ncell,:] = ((1-frac_n)*net_ie - xia)/ta
        dx[4*ncell:5*ncell,:] = (frac_n*net_ie - xin)/tn
        dx[5*ncell:6*ncell,:] = (net_ii - xig)/tg
        
        return dx.flatten()
    
    x0 = np.concatenate((xea,xen,xeg,xia,xin,xig),axis=0).flatten()
    
    start_time = time.process_time()
    max_time = 60
    def time_event(t,x,ncell,nprm):
        int_time = (start_time + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True
    
    sol = integrate.solve_ivp(dyn_func,(0,dt*Nt),y0=x0,t_eval=tsamp*dt,args=(N**2,nprm),method='RK23')#,events=time_event)
    if sol.status != 0:
        x = np.nan*np.ones((6*N**2*nprm,len(tsamp)))
    else:
        x = sol.y
    x = x.reshape((-1,nprm,len(tsamp)))
    
    xea = x[0*N**2:1*N**2,:]
    xen = x[1*N**2:2*N**2,:]
    xeg = x[2*N**2:3*N**2,:]
    xia = x[3*N**2:4*N**2,:]
    xin = x[4*N**2:5*N**2,:]
    xig = x[5*N**2:6*N**2,:]
    
    ye = np.fmin(1e5,np.fmax(0,xea+xen+xeg)**2)
    yi = np.fmin(1e5,np.fmax(0,xia+xin+xig)**2)
    
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
    nwrm = 6 * nint * nphs
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
        
        kerne = np.exp(-(dss/(s_e))**params[7])
        norm = kerne.sum(axis=1).mean(axis=0)
        kerne /= norm
        kernei = np.exp(-(dss/(s_ei))**params[7])
        norm = kernei.sum(axis=1).mean(axis=0)
        kernei /= norm
        kernii = np.exp(-(dss/(s_ii))**params[7])
        norm = kernii.sum(axis=1).mean(axis=0)
        kernii /= norm
        
        het_lev = params[8]*theta[prm_idx,7].item()
        inp_mult = params[9]*theta[prm_idx,8].item()
        
        for ori_idx,ori in enumerate(oris):
            def ff_inp(t):
                return inp_mult * L4_rates_itp(t)[:,ori_idx]
            resp = integrate_sheet(np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
                                   np.zeros(N**2),np.zeros(N**2),np.zeros(N**2),
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
    
    inp_po = np.angle(L4_rate_opm.flatten())*180/(2*np.pi)
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
