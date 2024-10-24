import sys
import os
sys.path.insert(0, './..')

import argparse

from math import floor, ceil
import numpy as np

import matplotlib.pyplot as plt

import plot_func as pf
import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF import probe_RFs,network_ffrec
from dev_ori_sel_RF.tools import analysis_tools,misc

parser = argparse.ArgumentParser()
parser.add_argument('--maxver', '-v', help='version',type=int, default=8)
parser.add_argument('--nload', '-n', help='version',type=int, default=8)
parser.add_argument('--freq', '-f', help='spatial frequency',type=int, default=-1)
parser.add_argument('--nori', '-o', help='how many orientations to probe',type=int, default=4)
parser.add_argument('--nphs', '-p', help='how many phases to probe',type=int, default=8)
parser.add_argument('--nbin', '-b', help='how many bins for sorting recurrent weights',type=int, default=5)
parser.add_argument('--skip', '-s', help='how many cells to skip when plotting averaged RFs',type=int, default=0)
parser.add_argument('--config', '-c', help='version',type=str, default="test")
args = vars(parser.parse_args())
maxver = int(args['maxver'])
nload = int(args['nload'])
freq = int(args["freq"])
nori = int(args["nori"])
nphs = int(args["nphs"])
nbin = int(args["nbin"])
skip = int(args["skip"])
config_name = str(args['config'])

config_dict,N4pop,Nlgnpop,Nret,Nlgn,N4,rA = uf.get_network_size(config_name)
dA = 2*rA+1

Vers = np.round(np.linspace(0,maxver,nload+1)-1).astype(int)[1:]

if freq <= 1:
    config_dict.update({
        "config_name" : config_name,
        "system" : "one_layer"
    })
    config_dict["Wlgn_to4_params"].update({
        "W_mode": "load_from_external",
        "load_from_prev_run" : Vers[-1]})
    if "2pop" in config_dict["W4to4_params"]["Wrec_mode"]:
        config_dict["W4to4_params"].update({
            "Wrec_mode": "load_from_external2pop"})
    else:
        config_dict["W4to4_params"].update({
            "Wrec_mode": "load_from_external"})
    n = network_ffrec.Network(Vers[-1],config_dict)
    _,Wlgnto4,_,_,_,_,_,_,_ = n.system

    sd = Wlgnto4[0,...] - Wlgnto4[1,...]
    sd = sd.reshape((N4,N4,Nlgn,Nlgn))

    xs,ys = np.meshgrid(np.arange(Nlgn),np.arange(Nlgn))
    xs[xs > Nlgn//2] -= Nlgn
    ys[ys > Nlgn//2] -= Nlgn
    ls = np.sqrt(xs**2+ys**2)
    ls_bins = np.digitize(ls,np.arange(0,Nlgn//2)+0.5)

    sd_fft = np.fft.fft2(sd,axes=(2,3))
    sd_fft_mag = np.zeros((N4,N4,Nlgn//2))
    for i in range(N4):
        for j in range(N4):
            for k in range(Nlgn//2):
                sd_fft_mag[i,j,k] = np.max(np.abs(sd_fft[i,j,ls_bins == k]))
    sd_fft_mag = np.mean(sd_fft_mag,axis=(0,1))
    pref_lam = np.argmax(sd_fft_mag)
    
    freq = pref_lam

for i,Version in enumerate(Vers):
    # load connection weights
    config_dict.update({
        "config_name" : config_name,
        "system" : "one_layer"
    })
    config_dict["Wlgn_to4_params"].update({
        "W_mode": "load_from_external",
        "load_from_prev_run" : Version})
    if "2pop" in config_dict["W4to4_params"]["Wrec_mode"]:
        config_dict["W4to4_params"].update({
            "Wrec_mode": "load_from_external2pop"})
    else:
        config_dict["W4to4_params"].update({
            "Wrec_mode": "load_from_external"})
    n = network_ffrec.Network(Version,config_dict)
    _,Wlgnto4,_,_,_,_,W4to4,_,_ = n.system

    wff = np.zeros((2,N4**2,Nlgn**2))
    wff[0] = Wlgnto4[0,...]
    wff[1] = Wlgnto4[1,...]
    abs_wff = wff.reshape((2,N4,N4,Nlgn,Nlgn))
    rel_wff = abs_wff.copy()
    for i in range(N4):
        for j in range(N4):
            rel_wff[:,i,j] = np.roll(rel_wff[:,i,j],(rA-i,rA-j),axis=(-2,-1))
    rel_wff = rel_wff[:,:,:,:dA,:dA]
    
    # select reference cells, bin presynaptic cells by strength of recurrent connectivity
    ref_locs = np.arange(0,N4,1+skip)
    ei_weighted_rfs = np.zeros((len(ref_locs),len(ref_locs),2,3*dA,3*dA))
    ei_bin_avg_rfs = np.zeros((len(ref_locs),len(ref_locs),nbin,2,dA,dA))
    et_weighted_rfs = np.zeros((len(ref_locs),len(ref_locs),2,3*dA,3*dA))
    et_bin_avg_rfs = np.zeros((len(ref_locs),len(ref_locs),nbin,2,dA,dA))
    
    avg_resp = False
    if os.path.isfile('./../results/grating_responses/{:s}/v{:d}_local/rates_f={:d}.npy'.format(
            config_name,Version,freq)):
        avg_resp = True
        
        # Load FF input and response to full field grating data
        inputs = np.load('./../results/grating_responses/{:s}/v{:d}_local/inputs_f={:d}.npy'.format(
            config_name,Version,freq)).reshape((1,nori,nphs,2,N4,N4))
        rates = np.load('./../results/grating_responses/{:s}/v{:d}_local/rates_f={:d}.npy'.format(
            config_name,Version,freq)).reshape((1,nori,nphs,2,N4,N4))

        # Fix inverted orientations and phases
        inputs[:,1:] = inputs[:,-1:0:-1]
        inputs[:,0,1:] = inputs[:,0,-1:0:-1]
        rates[:,1:] = rates[:,-1:0:-1]
        rates[:,0,1:] = rates[:,0,-1:0:-1]
        
        # Calculate feature preferences and properties of inputs and responses
        inp_F0,inp_F1,inp_APP = uf.calc_dc_ac_comp(rates[0,:,:,0,:,:],1)
        inp_F1 *= 2
        inp_MR = inp_F1/inp_F0
        # inp_APP *= 360/(2*np.pi)
        inp_avg,inp_OS,inp_PO = uf.calc_dc_ac_comp(inp_F1,0)
        inp_OS = inp_OS / inp_avg
        # inp_PO *= 180/(2*np.pi)
        resp_F0,resp_F1,resp_APP = uf.calc_dc_ac_comp(rates[0,:,:,0,:,:],1)
        resp_F1 *= 2
        resp_MR = resp_F1/resp_F0
        # resp_APP *= 360/(2*np.pi)
        resp_avg,resp_OS,resp_PO = uf.calc_dc_ac_comp(resp_F0,0)
        resp_OS = resp_OS / resp_avg
        # resp_PO *= 180/(2*np.pi)
        # inp_phs_props = np.concatenate((inp_MR[:,:,:,None],
        #                                 inp_MR[:,:,:,None]*np.exp(1j*inp_APP[:,:,:,None])),axis=-1)
        # inp_ori_props = np.concatenate((inp_OS[:,:,None],
        #                                 inp_OS[:,:,None]*np.exp(1j*inp_PO[:,:,None])),axis=-1)
        # resp_phs_props = np.concatenate((resp_MR[:,:,:,None],
        #                                 resp_MR[:,:,:,None]*np.exp(1j*resp_APP[:,:,:,None])),axis=-1)
        # resp_ori_props = np.concatenate((resp_OS[:,:,None],
        #                                 resp_OS[:,:,None]*np.exp(1j*resp_PO[:,:,None])),axis=-1)
        inp_phs_props = inp_MR*np.exp(1j*inp_APP)
        inp_ori_props = inp_OS*np.exp(1j*inp_PO)
        resp_phs_props = resp_MR*np.exp(1j*resp_APP)
        resp_ori_props = resp_OS*np.exp(1j*resp_PO)
        ei_bin_avg_inp_phs_props = np.zeros((len(ref_locs),len(ref_locs),nbin,nori,2))
        ei_bin_avg_inp_ori_props = np.zeros((len(ref_locs),len(ref_locs),nbin,2))
        ei_bin_avg_resp_phs_props = np.zeros((len(ref_locs),len(ref_locs),nbin,nori,2))
        ei_bin_avg_resp_ori_props = np.zeros((len(ref_locs),len(ref_locs),nbin,2))
        et_bin_avg_inp_phs_props = np.zeros((len(ref_locs),len(ref_locs),nbin,nori,2))
        et_bin_avg_inp_ori_props = np.zeros((len(ref_locs),len(ref_locs),nbin,2))
        et_bin_avg_resp_phs_props = np.zeros((len(ref_locs),len(ref_locs),nbin,nori,2))
        et_bin_avg_resp_ori_props = np.zeros((len(ref_locs),len(ref_locs),nbin,2))
        
    print('Average input/response properties?',avg_resp)

    for i,ref_x in enumerate(ref_locs):
        for j,ref_y in enumerate(ref_locs):
            ee_weights = np.abs(W4to4[:N4**2,:N4**2]).reshape(N4,N4,N4,N4)[ref_x,ref_y,:,:]
            ei_weights = np.abs(W4to4[:N4**2,N4**2:]).reshape(N4,N4,N4,N4)[ref_x,ref_y,:,:]
            et_weights = ee_weights - ei_weights
            
            # Calculate weighted absolute RFs
            
            # Calculate weight bin averaged relative RFs and properties
            nz_ei_weights = np.sort(ei_weights[ei_weights > 0])
            nz_ei_weights = np.concatenate((nz_ei_weights,nz_ei_weights[-1:]+1))
            nz_et_weights = np.sort(et_weights[np.logical_or(ee_weights > 0,ei_weights > 0)])
            nz_et_weights = np.concatenate((nz_et_weights,nz_et_weights[-1:]+1))
            for k in range(nbin):
                bin_idxs = np.logical_and(ei_weights >= nz_ei_weights[int((len(nz_ei_weights)-1)*k/nbin)],
                                        ei_weights < nz_ei_weights[int((len(nz_ei_weights)-1)*(k+1)/nbin)])
                ei_bin_avg_rfs[i,j,k] = np.mean(wff[:,bin_idxs],axis=1)
                if avg_resp:
                    ei_bin_avg_inp_phs_props[i,j,k] = np.concatenate((
                                            np.abs(np.mean(inp_phs_props[:,bin_idxs],axis=1))[:,None],
                                            np.angle(np.mean(inp_phs_props[:,bin_idxs],axis=1))[:,None]*360/(2*np.pi)),
                                            axis=-1)
                    ei_bin_avg_inp_ori_props[i,j,k] = np.array([np.abs(np.mean(inp_ori_props[bin_idxs],axis=0)),
                                                np.angle(np.mean(inp_ori_props[bin_idxs],axis=0))*180/(2*np.pi)])
                    ei_bin_avg_resp_phs_props[i,j,k] = np.concatenate((
                                            np.abs(np.mean(resp_phs_props[:,bin_idxs],axis=1))[:,None],
                                            np.angle(np.mean(resp_phs_props[:,bin_idxs],axis=1))[:,None]*360/(2*np.pi)),
                                            axis=-1)
                    ei_bin_avg_resp_ori_props[i,j,k] = np.array([np.abs(np.mean(resp_ori_props[bin_idxs],axis=0)),
                                                np.angle(np.mean(resp_ori_props[bin_idxs],axis=0))*180/(2*np.pi)])
                    
                bin_idxs = np.logical_and(et_weights >= nz_et_weights[int((len(nz_et_weights)-1)*k/nbin)],
                                        et_weights < nz_et_weights[int((len(nz_et_weights)-1)*(k+1)/nbin)])
                bin_idxs = np.logical_and(bin_idxs,np.logical_or(ee_weights > 0,ei_weights > 0))
                et_bin_avg_rfs[i,j,k] = np.mean(wff[:,bin_idxs],axis=1)
                if avg_resp:
                    et_bin_avg_inp_phs_props[i,j,k] = np.concatenate((
                                            np.abs(np.mean(inp_phs_props[:,bin_idxs],axis=1))[:,None],
                                            np.angle(np.mean(inp_phs_props[:,bin_idxs],axis=1))[:,None]*360/(2*np.pi)),
                                            axis=-1)
                    et_bin_avg_inp_ori_props[i,j,k] = np.array([np.abs(np.mean(inp_ori_props[bin_idxs],axis=0)),
                                                np.angle(np.mean(inp_ori_props[bin_idxs],axis=0))*180/(2*np.pi)])
                    et_bin_avg_resp_phs_props[i,j,k] = np.concatenate((
                                            np.abs(np.mean(resp_phs_props[:,bin_idxs],axis=1))[:,None],
                                            np.angle(np.mean(resp_phs_props[:,bin_idxs],axis=1))[:,None]*360/(2*np.pi)),
                                            axis=-1)
                    et_bin_avg_resp_ori_props[i,j,k] = np.array([np.abs(np.mean(resp_ori_props[bin_idxs],axis=0)),
                                                np.angle(np.mean(resp_ori_props[bin_idxs],axis=0))*180/(2*np.pi)])
                    
    misc.ensure_path('./../results/grating_responses/{:s}/v{:d}_local/'.format(config_name,Version))
    np.save('./../results/grating_responses/{:s}/v{:d}_local/ei_avg_RFs'.format(config_name,Version),
            ei_bin_avg_rfs.flatten())
    np.save('./../results/grating_responses/{:s}/v{:d}_local/et_avg_RFs'.format(config_name,Version),
            et_bin_avg_rfs.flatten())
    if avg_resp:
        np.save('./../results/grating_responses/{:s}/v{:d}_local/ei_avg_inp_phs_props_f={:d}'.format(config_name,
                Version,freq),
                ei_bin_avg_inp_phs_props.flatten())
        np.save('./../results/grating_responses/{:s}/v{:d}_local/ei_avg_inp_ori_props_f={:d}'.format(config_name,
                Version,freq),
                ei_bin_avg_inp_ori_props.flatten())
        np.save('./../results/grating_responses/{:s}/v{:d}_local/ei_avg_resp_phs_props_f={:d}'.format(config_name,
                Version,freq),
                ei_bin_avg_resp_phs_props.flatten())
        np.save('./../results/grating_responses/{:s}/v{:d}_local/ei_avg_resp_ori_props_f={:d}'.format(config_name,
                Version,freq),
                ei_bin_avg_resp_ori_props.flatten())
        np.save('./../results/grating_responses/{:s}/v{:d}_local/et_avg_inp_phs_props_f={:d}'.format(config_name,
                Version,freq),
                et_bin_avg_inp_phs_props.flatten())
        np.save('./../results/grating_responses/{:s}/v{:d}_local/et_avg_inp_ori_props_f={:d}'.format(config_name,
                Version,freq),
                et_bin_avg_inp_ori_props.flatten())
        np.save('./../results/grating_responses/{:s}/v{:d}_local/et_avg_resp_phs_props_f={:d}'.format(config_name,
                Version,freq),
                et_bin_avg_resp_phs_props.flatten())
        np.save('./../results/grating_responses/{:s}/v{:d}_local/et_avg_resp_ori_props_f={:d}'.format(config_name,
                Version,freq),
                et_bin_avg_resp_ori_props.flatten())

    misc.ensure_path('./../plots/grating_responses/{:s}/v{:d}_local/'.format(config_name,Version))
    for k in range(nbin):
        ei_avg_RFs = np.zeros((2,len(ref_locs)*(dA+1)+1,len(ref_locs)*(dA+1)+1))
        et_avg_RFs = np.zeros((2,len(ref_locs)*(dA+1)+1,len(ref_locs)*(dA+1)+1))
        for i,ref_x in enumerate(ref_locs):
            for j,ref_y in enumerate(ref_locs):
                ei_avg_RFs[:,1+i*(dA+1):1+i*(dA+1)+dA,1+j*(dA+1):1+j*(dA+1)+dA] = ei_bin_avg_rfs[i,j,k]
                et_avg_RFs[:,1+i*(dA+1):1+i*(dA+1)+dA,1+j*(dA+1):1+j*(dA+1)+dA] = et_bin_avg_rfs[i,j,k]

        fig,axs = plt.subplots(1,2,figsize=(len(ref_locs),0.5*len(ref_locs)),dpi=300)
        pf.doubcontbar(fig,axs[0],ei_avg_RFs[0],-ei_avg_RFs[1],
                       cmap='RdBu',levels=np.linspace(-np.max(np.abs(ei_avg_RFs)),np.max(np.abs(ei_avg_RFs)),13),
                       linewidths=0.8,origin='lower')
        pf.doubimshbar(fig,axs[1],ei_avg_RFs[0],-ei_avg_RFs[1],cmap='RdBu',vmin=-np.max(np.abs(ei_avg_RFs)),
                       vmax=np.max(np.abs(ei_avg_RFs)),origin='lower')
        # pf.imshowbar(fig,axs,ei_avg_RFs,cmap='RdBu',
        #            vmin=-np.max(np.abs(ei_avg_RFs)),vmax=np.max(np.abs(ei_avg_RFs)),origin='lower')

        fig.tight_layout()
        plt.savefig("./../plots/grating_responses/{:s}/v{:d}_local/ei_bin{:d}_avg_RFs".format(config_name,Version,k)+\
            ".pdf")

        fig,axs = plt.subplots(1,2,figsize=(len(ref_locs),0.5*len(ref_locs)),dpi=300)
        pf.doubcontbar(fig,axs[0],et_avg_RFs[0],-et_avg_RFs[1],
                       cmap='RdBu',levels=np.linspace(-np.max(np.abs(et_avg_RFs)),np.max(np.abs(et_avg_RFs)),13),
                       linewidths=0.8,origin='lower')
        pf.doubimshbar(fig,axs[1],et_avg_RFs[0],-et_avg_RFs[1],cmap='RdBu',vmin=-np.max(np.abs(et_avg_RFs)),
                       vmax=np.max(np.abs(et_avg_RFs)),origin='lower')
        # pf.imshowbar(fig,axs,et_avg_RFs,cmap='RdBu',
        #            vmin=-np.max(np.abs(et_avg_RFs)),vmax=np.max(np.abs(et_avg_RFs)),origin='lower')

        fig.tight_layout()
        plt.savefig("./../plots/grating_responses/{:s}/v{:d}_local/et_bin{:d}_avg_RFs".format(config_name,Version,k)+\
            ".pdf")

rI = ceil(config_dict["W4to4_params"]["rA_EI"] * config_dict["W4to4_params"].get("r_lim",1.) * N4)
dI = 2*rI+1
Nshow = N4//(1+skip)
wei = np.zeros((Nshow*(dI+1)+1,Nshow*(dI+1)+1))
for i in range(Nshow):
    for j in range(Nshow):
        this_wei = W4to4[:N4**2,N4**2:].reshape(N4,N4,N4,N4)[i*(1+skip),j*(1+skip),:,:]
        wei[1+i*(dI+1):1+i*(dI+1)+dI,1+j*(dI+1):1+j*(dI+1)+dI] =\
               np.roll(this_wei,(rI-i*(1+skip),rI-j*(1+skip)),axis=(-2,-1))[:dI,:dI]

fig,axs = plt.subplots(1,1,figsize=(0.5*Nshow,0.5*Nshow),dpi=300)
fig.subplots_adjust(hspace=.1, wspace=.3)

pf.imshowbar(fig,axs,-wei,cmap='Blues',vmin=0,vmax=np.max(np.abs(wei)),origin='lower')

fig.tight_layout()
plt.savefig("./../plots/WEIs/WEIs_"+config_name+".pdf")

rT = np.fmax(ceil(config_dict["W4to4_params"]["rA_EI"] * config_dict["W4to4_params"].get("r_lim",1.) * N4),
             ceil(config_dict["W4to4_params"]["rA_EE"] * config_dict["W4to4_params"].get("r_lim",1.) * N4))
dT = 2*rT+1
Nshow = N4//(1+skip)
wet = np.zeros((Nshow*(dT+1)+1,Nshow*(dT+1)+1))
for i in range(Nshow):
    for j in range(Nshow):
        this_wet = (W4to4[:N4**2,:N4**2]+W4to4[:N4**2,N4**2:]).reshape(N4,N4,N4,N4)[i*(1+skip),j*(1+skip),:,:]
        wet[1+i*(dT+1):1+i*(dT+1)+dT,1+j*(dT+1):1+j*(dT+1)+dT] =\
               np.roll(this_wet,(rT-i*(1+skip),rT-j*(1+skip)),axis=(-2,-1))[:dT,:dT]

fig,axs = plt.subplots(1,1,figsize=(0.5*Nshow,0.5*Nshow),dpi=300)
fig.subplots_adjust(hspace=.1, wspace=.3)

pf.imshowbar(fig,axs,wet,cmap='RdBu_r',vmin=-np.max(np.abs(wet)),vmax=np.max(np.abs(wet)),origin='lower')

fig.tight_layout()
plt.savefig("./../plots/WEIs/WETs_"+config_name+".pdf")
