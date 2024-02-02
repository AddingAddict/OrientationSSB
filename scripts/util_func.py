import sys
import os
sys.path.insert(0, './..')

import pickle
from math import floor, ceil
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import dev_ori_sel_RF
from dev_ori_sel_RF import data_dir,integrator_tf,dynamics,network, network_ffrec,run_onelayer,probe_RFs
from dev_ori_sel_RF.tools import misc,update_params_dict,analysis_tools

def get_network_size(config_name,verbose=True):
    config_dict = misc.load_external_params("params_"+config_name,verbose=verbose)
    config_dict.update({"config_name" : config_name})

    N4 = config_dict["N4"]
    Nlgn = config_dict["Nlgn"]
    Nret = config_dict["Nret"]
    Nlgnpop = 2
    N4pop = config_dict["num_lgn_paths"] // Nlgnpop
    rA = ceil(config_dict["Wlgn_to4_params"]["r_A_on"] * config_dict["Wlgn_to4_params"].get("r_lim",1.) * N4)
    
    return config_dict,N4pop,Nlgnpop,Nret,Nlgn,N4,rA

def get_network_system(Version,config_name):
    if Version == -1:
        config_dict = misc.load_external_params("params_"+config_name,False)
        config_dict.update({
                        "RF_mode" : "initialize",
                        "system" : "one_layer",
                        "Version" : Version,
                        })
        net = network.Network(Version,config_dict,verbose=False)
    else:
        load_location = 'local'
        load_path = data_dir + "layer4/{s}/v{v}/".format(s=config_name,v=Version)
        config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
        config_dict.update({"config_name" : config_name})
        config_dict["Wlgn_to4_params"].update({
            "W_mode": "load_from_external",
            "load_from_prev_run" : Version})
        net = network.Network(Version,config_dict,load_location=load_location,verbose=False)
    return net.system

def get_adaptive_system(Version,config_name):
    if Version == -1:
        config_dict = misc.load_external_params("params_"+config_name,False)
        N4 = config_dict["N4"]
        Nvert = config_dict["Nvert"]
        N4pop = config_dict["num_lgn_paths"] // 2
        l4_avg = config_dict["W4to4_params"].get(l4_avg,0) * np.ones(N4pop*N4**2*Nvert)
        theta_4 = config_dict["W4to4_params"].get(theta_4,0) * np.ones(N4pop*N4**2*Nvert)
    else:
        load_path = data_dir + "layer4/{s}/v{v}/".format(s=config_name,v=Version)
        data_dict = np.load(open(load_path + "y_v{v}.npz".format(v=Version),"rb"))
        l4_avg = data_dict["l4_avg"]
        theta_4 = data_dict["theta_4"]
    return l4_avg, theta_4

def get_network_system_ffrec(Version,config_name):
    if Version == -1:
        config_dict = misc.load_external_params("params_"+config_name,False)
        config_dict.update({
                        "RF_mode" : "initialize",
                        "system" : "one_layer",
                        "Version" : Version,
                        })
        net = network_ffrec.Network(Version,config_dict,verbose=False)
    else:
        load_location = 'local'
        load_path = data_dir + "ffrec/{s}/v{v}/".format(s=config_name,v=Version)
        config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
        config_dict.update({"config_name" : config_name})
        config_dict["Wlgn_to4_params"].update({
            "W_mode": "load_from_external",
            "load_from_prev_run" : Version})
        if "2pop" in config_dict["W4to4_params"]["Wrec_mode"]:
            config_dict["W4to4_params"].update({
                "Wrec_mode": "load_from_external2pop"})
        else:
            config_dict["W4to4_params"].update({
                "Wrec_mode": "load_from_external"})
        net = network_ffrec.Network(Version,config_dict,load_location=load_location,verbose=False)
    return net.system

def get_adaptive_system_ffrec(Version,config_name):
    if Version == -1:
        config_dict = misc.load_external_params("params_"+config_name,False)
        N4 = config_dict["N4"]
        Nvert = config_dict["Nvert"]
        N4pop = config_dict["num_lgn_paths"] // 2
        l4_avg = config_dict["W4to4_params"].get(l4_avg,0) * np.ones(N4pop*N4**2*Nvert)
        theta_4 = config_dict["W4to4_params"].get(theta_4,0) * np.ones(N4pop*N4**2*Nvert)
    else:
        load_path = data_dir + "ffrec/{s}/v{v}/".format(s=config_name,v=Version)
        data_dict = np.load(open(load_path + "y_v{v}.npz".format(v=Version),"rb"))
        l4_avg = data_dict["l4_avg"]
        theta_4 = data_dict["theta_4"]
    return l4_avg, theta_4

def get_network_weights(Version,config_name,N4pop,N4,Nlgn):
    if Version == -1:
        config_dict = misc.load_external_params("params_"+config_name,False)
        config_dict.update({
                        "RF_mode" : "initialize",
                        "system" : "one_layer",
                        "Version" : Version,
                        })
        net = network.Network(Version,config_dict,verbose=False)
        _,Wlgnto4,_,_,_,_,_ = net.system
    else:
        load_path = data_dir + "layer4/{s}/v{v}/".format(s=config_name,v=Version)
        Wlgnto4 = np.load(load_path+'y_v{v}.npz'.format(v=Version))['W']
        Wlgnto4 = Wlgnto4.reshape(2*N4pop,N4**2,Nlgn**2)
    return Wlgnto4

def get_network_weights_ffrec(Version,config_name,N4pop,N4,Nlgn):
    if Version == -1:
        config_dict = misc.load_external_params("params_"+config_name,False)
        config_dict.update({
                        "RF_mode" : "initialize",
                        "system" : "one_layer",
                        "Version" : Version,
                        })
        net = network_ffrec.Network(Version,config_dict,verbose=False)
        _,Wlgnto4,_,_,_,_,W4to4,_,_ = net.system
    else:
        load_path = data_dir + "ffrec/{s}/v{v}/".format(s=config_name,v=Version)
        Wlgnto4 = np.load(load_path+'y_v{v}.npz'.format(v=Version))['W']
        Wlgnto4 = Wlgnto4.reshape(2*N4pop,N4**2,Nlgn**2)
        W4to4 = np.load(load_path+'y_v{v}.npz'.format(v=Version))['Wrec']
        W4to4 = W4to4.reshape(N4pop*N4**2,N4pop*N4**2)
    return Wlgnto4,W4to4

def get_fps(A,axes=None,zero_mean=True):
    if axes is None:
        Nax = A.shape[-2]
    else:
        Nax = A.shape[axes[0]]
    if zero_mean:
        fft = np.abs(np.fft.fftshift(np.fft.fft2(A - np.nanmean(A))))
    else:
        fft = np.abs(np.fft.fftshift(np.fft.fft2(A)))
    fps = np.zeros(int(np.ceil(Nax//2*np.sqrt(2))))

    grid = np.arange(-Nax//2,Nax//2)
    x,y = np.meshgrid(grid,grid)
    bin_idxs = np.digitize(np.sqrt(x**2+y**2),np.arange(0,np.ceil(Nax//2*np.sqrt(2)))+0.5)
    for idx in range(0,int(np.ceil(Nax//2*np.sqrt(2)))):
        fps[idx] = np.mean(fft[bin_idxs == idx])
    
    return fft,fps

def get_ori_sel(opm,calc_fft=True):
    N4 = opm.shape[0]
    sel = np.abs(opm)
    ori = np.angle(opm)/2
    ori = ori - (np.sign(ori)-1)*0.5*np.pi
    ori *= 180/np.pi
    
    if calc_fft:
        opm_fft,opm_fps = get_fps(opm)
        return ori,sel,opm_fft,opm_fps
    else:
        return ori,sel
    
def calc_hypercol_size(fps,N):
    def fps_fn(k,a0,a1,a2,a3,a4,a5):
        return a0*np.exp(-0.5*(k-a1)**2/a2**2) + a3 + a4*k + a5*k**2

    freqs = np.arange(N//2)/N

    popt,pcov = curve_fit(fps_fn,freqs,fps[:N//2],
                          p0=(fps[1],freqs[np.argmax(fps[:N//2])],0.5*freqs[np.argmax(fps[:N//2])],0,0,0))
    
    hcsize = 1/popt[1]
    return hcsize,fps_fn(np.arange(len(fps))/N,*popt)

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

def bandpass_filter(A,ll,lu):
    Nax = A.shape[0]
    
    xs,ys = np.meshgrid(np.arange(Nax)/Nax,np.arange(Nax)/Nax)
    xs[xs > 0.5] = 1 - xs[xs > 0.5]
    ys[ys > 0.5] = 1 - ys[ys > 0.5]
    ks = 2*np.pi*np.sqrt(xs**2 + ys**2)

    def fermi_kernel(lamlp,beta=0.05):
        return 1 / (1 + np.exp(-(2*np.pi/lamlp - ks)/beta))
    
    A_fft = np.fft.fft2(A)
    
    this_kern = fermi_kernel(ll) * (1-fermi_kernel(lu))
    return np.fft.ifft2(A_fft*this_kern)
    
def calc_dc_ac_comp(A,axis=-1):
    A_xpsd = np.moveaxis(A,axis,-1)
    Nax = A.shape[axis]
    angs = np.arange(Nax) * 2*np.pi/Nax
    A0 = np.mean(A_xpsd,axis=-1)
    As = np.mean(A_xpsd*np.sin(angs),axis=-1)
    Ac = np.mean(A_xpsd*np.cos(angs),axis=-1)
    A1mod = np.sqrt(As**2+Ac**2)
    A1phs = np.arctan2(As,Ac)
    
    return A0,A1mod,A1phs
