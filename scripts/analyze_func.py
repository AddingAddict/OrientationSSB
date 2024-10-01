import sys
import os
sys.path.insert(0, './..')

import pickle
from math import floor, ceil
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
    peak_idx = np.argmax(np.concatenate(([0,0],fps[2:N//2])))

    popt,pcov = curve_fit(fps_fn,freqs,fps[:N//2],
                          p0=(fps[peak_idx],freqs[peak_idx],0.5*freqs[peak_idx],0,0,0))
    
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
    if A.ndim > 1:
        A_xpsd = np.moveaxis(A,axis,-1)
    else:
        A_xpsd = A.copy()
    Nax = A.shape[axis]
    angs = np.arange(Nax) * 2*np.pi/Nax
    A0 = np.mean(A_xpsd,axis=-1)
    As = np.mean(A_xpsd*np.sin(angs),axis=-1)
    Ac = np.mean(A_xpsd*np.cos(angs),axis=-1)
    A1mod = np.sqrt(As**2+Ac**2)
    A1phs = np.arctan2(As,Ac)
    
    return A0,A1mod,A1phs

def calc_OS_MR(A):
    noris = np.shape(A)[-2]
    
    # calculate phase DC and AC per orientation
    F0,F1,_ = calc_dc_ac_comp(A)
    
    # calculate orientation DC, AC, and center from phase-averaged response
    A0,A1,PO = calc_dc_ac_comp(F0)
    
    # calculate OS from DC and AC of phase-averaged response
    OS = np.where(A1==0,0,A1/A0)
    
    # infer index of preferred orientation
    PO += np.pi/noris
    PO = np.where(PO<0,2*np.pi + PO,PO)
    PO = np.array(PO / (2*np.pi) * noris).astype(int)
    
    # calculate MR at preferred orientation
    # pref_F0,pref_F1 = np.zeros(np.shape(A)[:-2]),np.zeros(np.shape(A)[:-2])
    F0,F1 = np.take_along_axis(F0,PO[...,None],-1)[...,0],np.take_along_axis(F1,PO[...,None],-1)[...,0]
    MR = np.where(F1==0,0,2*F1/F0)
    
    return OS,MR