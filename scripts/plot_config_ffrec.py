import sys
import os
sys.path.insert(0, './..')

import argparse

import numpy as np

import matplotlib.pyplot as plt

import plot_func as pf
import util_func as uf

import dev_ori_sel_RF
from dev_ori_sel_RF.tools import analysis_tools

parser = argparse.ArgumentParser()
parser.add_argument('--maxver', '-v', help='version',type=int, default=8)
parser.add_argument('--nload', '-n', help='version',type=int, default=8)
parser.add_argument('--skip', '-s', help='how many RFs to skip for each plotted',type=int, default=0)
parser.add_argument('--flip', '-f', help='flip on vs off?',type=int, default=0)
parser.add_argument('--config', '-c', help='version',type=str, default="test")
args = vars(parser.parse_args())
maxver = int(args['maxver'])
nload = int(args['nload'])
skip = int(args["skip"])
flip = int(args["flip"])
config_name = str(args['config'])

config_dict,N4pop,Nlgnpop,Nret,Nlgn,N4,rA = uf.get_network_size(config_name)
DA = 2*rA + 5

Vers = np.round(np.linspace(0,maxver,nload+1)-1).astype(int)

oris_e = np.zeros((len(Vers),N4,N4))
oris_i = np.zeros((len(Vers),N4,N4))
sels_e = np.zeros((len(Vers),N4,N4))
sels_i = np.zeros((len(Vers),N4,N4))
ori_ffts_e = np.zeros((len(Vers),N4,N4))
ori_ffts_i = np.zeros((len(Vers),N4,N4))
opm_fpss_e = np.zeros((len(Vers),int(np.ceil(N4//2*np.sqrt(2)))))
opm_fpss_i = np.zeros((len(Vers),int(np.ceil(N4//2*np.sqrt(2)))))
seps_e = np.zeros((len(Vers),N4,N4))
seps_i = np.zeros((len(Vers),N4,N4))
bals_e = np.zeros((len(Vers),N4,N4))
bals_i = np.zeros((len(Vers),N4,N4))

for idx,Version in enumerate(Vers):
    Wlgnto4,W4to4 = uf.get_network_weights_ffrec(Version,config_name,N4pop,N4,Nlgn)

    ss_e = Wlgnto4[0,...] + Wlgnto4[1,...]
    sd_e = Wlgnto4[0,...] - Wlgnto4[1,...]
    ss_i = Wlgnto4[2,...] + Wlgnto4[3,...]
    sd_i = Wlgnto4[2,...] - Wlgnto4[3,...]
    ss_e = ss_e.reshape((N4,N4,Nlgn,Nlgn))
    ss_i = ss_i.reshape((N4,N4,Nlgn,Nlgn))
    sd_e = sd_e.reshape((N4,N4,Nlgn,Nlgn))
    sd_i = sd_i.reshape((N4,N4,Nlgn,Nlgn))
    opm_e,Rn_e = analysis_tools.get_response(sd_e,DA)
    opm_i,Rn_i = analysis_tools.get_response(sd_i,DA)
    
    oris_e[idx],sels_e[idx],ori_ffts_e[idx],opm_fpss_e[idx] = uf.get_ori_sel(opm_e)
    seps_e[idx] = np.abs(sd_e).sum((-2,-1))/ss_e.sum((-2,-1))
    bals_e[idx] = 1-np.abs(sd_e.sum((-2,-1)))/ss_e.sum((-2,-1))

    oris_i[idx],sels_i[idx],ori_ffts_i[idx],opm_fpss_i[idx] = uf.get_ori_sel(opm_i)
    seps_i[idx] = np.abs(sd_i).sum((-2,-1))/ss_i.sum((-2,-1))
    bals_i[idx] = 1-np.abs(sd_i.sum((-2,-1)))/ss_i.sum((-2,-1))

fig,axs = plt.subplots(2*9,len(Vers),figsize=(4*len(Vers),4*2*9),dpi=300,sharex='row',sharey='row')
for i,Version in enumerate(Vers):
    pf.imshowbar(fig,axs[0,i],oris_e[i],cmap=pf.hue_cmap,vmin=0,vmax=180,origin='lower')
    pf.imshowbar(fig,axs[1,i],oris_i[i],cmap=pf.hue_cmap,vmin=0,vmax=180,origin='lower')    
    pf.imshowbar(fig,axs[2,i],sels_e[i],cmap='binary',vmin=0,vmax=np.max(sels_e),origin='lower')
    pf.imshowbar(fig,axs[3,i],sels_i[i],cmap='binary',vmin=0,vmax=np.max(sels_i),origin='lower')
    pf.imshowbar(fig,axs[4,i],seps_e[i],cmap='binary',vmin=0,vmax=1,origin='lower')
    pf.imshowbar(fig,axs[5,i],seps_i[i],cmap='binary',vmin=0,vmax=1,origin='lower')
    pf.imshowbar(fig,axs[6,i],bals_e[i],cmap='binary',vmin=0,vmax=1,origin='lower')
    pf.imshowbar(fig,axs[7,i],bals_i[i],cmap='binary',vmin=0,vmax=1,origin='lower')
    nbin = 20
    def bin_edges(data,nbin):
        return np.linspace(np.min(data),np.max(data),nbin+1)
    axs[8,i].hist(oris_e[i].flatten(),bin_edges(oris_e,nbin))
    axs[8,i].axhline(N4*N4/nbin,color='k',ls='--')
    axs[9,i].hist(oris_i[i].flatten(),bin_edges(oris_i,nbin))
    axs[9,i].axhline(N4*N4/nbin,color='k',ls='--')

    axs[10,i].hist(sels_e[i].flatten(),bin_edges(sels_e,nbin))
    axs[10,i].axvline(np.mean(sels_e[i].flatten()),color='k',ls='--')
    axs[11,i].hist(sels_i[i].flatten(),bin_edges(sels_i,nbin))
    axs[11,i].axvline(np.mean(sels_i[i].flatten()),color='k',ls='--')
    axs[12,i].hist(seps_e[i].flatten(),bin_edges([0,1],nbin))
    axs[12,i].axvline(np.mean(seps_e[i].flatten()),color='k',ls='--')
    axs[13,i].hist(seps_i[i].flatten(),bin_edges([0,1],nbin))
    axs[13,i].axvline(np.mean(seps_i[i].flatten()),color='k',ls='--')
    axs[14,i].hist(bals_e[i].flatten(),bin_edges([0,1],nbin))
    axs[14,i].axvline(np.mean(bals_e[i].flatten()),color='k',ls='--')
    axs[15,i].hist(bals_i[i].flatten(),bin_edges([0,1],nbin))
    axs[15,i].axvline(np.mean(bals_i[i].flatten()),color='k',ls='--')
    
    pf.imshowbar(fig,axs[16,i],ori_ffts_e[i],cmap='binary',vmin=0,vmax=np.max(ori_ffts_e),origin='lower')
    pf.imshowbar(fig,axs[17,i],ori_ffts_i[i],cmap='binary',vmin=0,vmax=np.max(ori_ffts_i),origin='lower')
    axs[16,i].plot(np.arange(N4//2,N4),(N4//4)*opm_fpss_e[i][:N4//2]/np.nanmax(opm_fpss_e[i][:N4//2-1]))
    axs[17,i].plot(np.arange(N4//2,N4),(N4//4)*opm_fpss_i[i][:N4//2]/np.nanmax(opm_fpss_i[i][:N4//2-1]))

    axs[0,i].set_title('Simulation Step {:d}'.format(Version+1))
    
axs[0,0].set_ylabel('Preferred Orientation (E)')
axs[1,0].set_ylabel('Preferred Orientation (I)')
axs[2,0].set_ylabel('Orientation Selectivity (E)')
axs[3,0].set_ylabel('Orientation Selectivity (I)')
axs[4,0].set_ylabel('Subregion Separation Index (E)')
axs[5,0].set_ylabel('Subregion Separation Index (I)')
axs[6,0].set_ylabel('Subregion Balance Index (E)')
axs[7,0].set_ylabel('Subregion Balance Index (I)')
axs[8,0].set_ylabel('Preferred Orientation (Count) (E)')
axs[9,0].set_ylabel('Preferred Orientation (Count) (I)')
axs[10,0].set_ylabel('Orientation Selectivity (Count) (E)')
axs[11,0].set_ylabel('Orientation Selectivity (Count) (I)')
axs[12,0].set_ylabel('Subregion Separation Index (Count) (E)')
axs[13,0].set_ylabel('Subregion Separation Index (Count) (I)')
axs[14,0].set_ylabel('Subregion Balance Index (Count) (E)')
axs[15,0].set_ylabel('Subregion Balance Index (Count) (I)')
axs[16,0].set_ylabel('Preferred Orientation (DFT) (E)')
axs[17,0].set_ylabel('Preferred Orientation (DFT) (I)')

plt.savefig("./../plots/OS_Dev_"+config_name+".pdf")

dA = 2*rA+1
Nshow = N4//(1+skip)
wff = np.zeros((N4pop,Nlgnpop,Nshow*(dA+1)+1,Nshow*(dA+1)+1))
for i in range(Nshow):
    for j in range(Nshow):
        this_wff = Wlgnto4.reshape(N4pop,Nlgnpop,N4,N4,Nlgn,Nlgn)[:,:,i*(1+skip),j*(1+skip),:,:]
        wff[:,:,1+i*(dA+1):1+i*(dA+1)+dA,1+j*(dA+1):1+j*(dA+1)+dA] =\
               np.roll(this_wff,(rA-i*(1+skip),rA-j*(1+skip)),axis=(-2,-1))[:,:,:dA,:dA]
rf = wff[:,0]-wff[:,1]

fig,axs = plt.subplots(2,2,figsize=(Nshow,Nshow),dpi=300)
fig.subplots_adjust(hspace=.1, wspace=.3)

if flip:
    pf.doubcontbar(fig,axs[0,0],wff[0,1],-wff[0,0],
                cmap='RdBu',levels=np.linspace(-np.max(np.abs(wff[0])),np.max(np.abs(wff[0])),13),linewidths=0.8,
                origin='lower')
    pf.doubimshbar(fig,axs[0,1],wff[0,1],-wff[0,0],cmap='RdBu',vmin=-np.max(np.abs(wff[0])),vmax=np.max(np.abs(wff[0])),
                origin='lower')

    pf.doubcontbar(fig,axs[1,0],wff[1,1],-wff[1,0],
                cmap='RdBu',levels=np.linspace(-np.max(np.abs(wff[1])),np.max(np.abs(wff[1])),13),linewidths=0.8,
                origin='lower')
    pf.doubimshbar(fig,axs[1,1],wff[1,1],-wff[1,0],cmap='RdBu',vmin=-np.max(np.abs(wff[1])),vmax=np.max(np.abs(wff[1])),
                origin='lower')
else:
    pf.doubcontbar(fig,axs[0,0],wff[0,0],-wff[0,1],
                cmap='RdBu',levels=np.linspace(-np.max(np.abs(wff[0])),np.max(np.abs(wff[0])),13),linewidths=0.8,
                origin='lower')
    pf.doubimshbar(fig,axs[0,1],wff[0,0],-wff[0,1],cmap='RdBu',vmin=-np.max(np.abs(wff[0])),vmax=np.max(np.abs(wff[0])),
                origin='lower')

    pf.doubcontbar(fig,axs[1,0],wff[1,0],-wff[1,1],
                cmap='RdBu',levels=np.linspace(-np.max(np.abs(wff[1])),np.max(np.abs(wff[1])),13),linewidths=0.8,
                origin='lower')
    pf.doubimshbar(fig,axs[1,1],wff[1,0],-wff[1,1],cmap='RdBu',vmin=-np.max(np.abs(wff[1])),vmax=np.max(np.abs(wff[1])),
                origin='lower')

fig.tight_layout()
plt.savefig("./../plots/RFs/RFs_"+config_name+".pdf")
