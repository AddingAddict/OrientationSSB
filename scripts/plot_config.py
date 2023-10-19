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
parser.add_argument('--config', '-c', help='version',type=str, default="test")
args = vars(parser.parse_args())
maxver = int(args['maxver'])
nload = int(args['nload'])
skip = int(args["skip"])
config_name = str(args['config'])

config_dict,N4pop,Nlgnpop,Nret,Nlgn,N4,rA = uf.get_network_size(config_name)
DA = 2*rA + 5

Vers = np.round(np.linspace(0,maxver,nload+1)-1).astype(int)

oris = np.zeros((len(Vers),N4,N4))
sels = np.zeros((len(Vers),N4,N4))
opm_ffts = np.zeros((len(Vers),N4,N4))
opm_fpss = np.zeros((len(Vers),int(np.ceil(N4//2*np.sqrt(2)))))
seps = np.zeros((len(Vers),N4,N4))
bals = np.zeros((len(Vers),N4,N4))

for idx,Version in enumerate(Vers):
    Wlgnto4 = uf.get_network_weights(Version,config_name,N4pop,N4,Nlgn)

    ss = Wlgnto4[0,...] + Wlgnto4[1,...]
    sd = Wlgnto4[0,...] - Wlgnto4[1,...]
    ss = ss.reshape((N4,N4,Nlgn,Nlgn))
    sd = sd.reshape((N4,N4,Nlgn,Nlgn))
    opm,Rn = analysis_tools.get_response(sd,DA)
    
    oris[idx],sels[idx],opm_ffts[idx],opm_fpss[idx] = uf.get_ori_sel(opm)
    seps[idx] = np.abs(sd).sum((-2,-1))/ss.sum((-2,-1))
    bals[idx] = 1-np.abs(sd.sum((-2,-1)))/ss.sum((-2,-1))

fig,axs = plt.subplots(9,len(Vers),figsize=(4*len(Vers),4*9),dpi=300,sharex='row',sharey='row')
for i,Version in enumerate(Vers):
    pf.imshowbar(fig,axs[0,i],oris[i],cmap='twilight',vmin=0,vmax=180,origin='lower')
    pf.imshowbar(fig,axs[1,i],sels[i],cmap='binary',vmin=0,vmax=np.max(sels),origin='lower')
    pf.imshowbar(fig,axs[2,i],seps[i],cmap='binary',vmin=0,vmax=1,origin='lower')
    pf.imshowbar(fig,axs[3,i],bals[i],cmap='binary',vmin=0,vmax=1,origin='lower')
    nbin = 20
    def bin_edges(data,nbin):
        return np.linspace(np.min(data),np.max(data),nbin+1)
    axs[4,i].hist(oris[i].flatten(),bin_edges(oris,nbin))
    axs[4,i].axhline(N4*N4/nbin,color='k',ls='--')
    axs[5,i].hist(sels[i].flatten(),bin_edges(sels,nbin))
    axs[5,i].axvline(np.mean(sels[i].flatten()),color='k',ls='--')
    axs[6,i].hist(seps[i].flatten(),bin_edges([0,1],nbin))
    axs[6,i].axvline(np.mean(seps[i].flatten()),color='k',ls='--')
    axs[7,i].hist(bals[i].flatten(),bin_edges([0,1],nbin))
    axs[7,i].axvline(np.mean(bals[i].flatten()),color='k',ls='--')
    
    pf.imshowbar(fig,axs[8,i],opm_ffts[i],cmap='binary',vmin=0,vmax=np.max(opm_ffts),origin='lower')
    axs[8,i].plot(np.arange(N4//2,N4),(N4//4)*opm_fpss[i][:N4//2]/np.nanmax(opm_fpss[i][:N4//2-1]))
    
    axs[0,i].set_title('Simulation Step {:d}'.format(Version+1))
    
axs[0,0].set_ylabel('Preferred Orientation')
axs[1,0].set_ylabel('Orientation Selectivity')
axs[2,0].set_ylabel('Subregion Separation Index')
axs[3,0].set_ylabel('Subregion Balance Index')
axs[4,0].set_ylabel('Preferred Orientation (Count)')
axs[5,0].set_ylabel('Orientation Selectivity (Count)')
axs[6,0].set_ylabel('Subregion Separation Index (Count)')
axs[7,0].set_ylabel('Subregion Balance Index (Count)')
axs[8,0].set_ylabel('Preferred Orientation (DFT)')

plt.savefig("./../plots/Ori_Sel_Dev_FF_Plasticity_"+config_name+".pdf")

dA = 2*rA+1
Nshow = N4//(1+skip)
wff = np.zeros((N4pop,Nlgnpop,Nshow*(dA+1)+1,Nshow*(dA+1)+1))
for i in range(Nshow):
    for j in range(Nshow):
        this_wff = Wlgnto4.reshape(N4pop,Nlgnpop,N4,N4,Nlgn,Nlgn)[:,:,i*(1+skip),j*(1+skip),:,:]
        wff[:,:,1+i*(dA+1):1+i*(dA+1)+dA,1+j*(dA+1):1+j*(dA+1)+dA] =\
               np.roll(this_wff,(rA-i*(1+skip),rA-j*(1+skip)),axis=(-2,-1))[:,:,:dA,:dA]
rf = wff[:,0]-wff[:,1]

fig,axs = plt.subplots(1,2,figsize=(Nshow,0.5*Nshow),dpi=300)
fig.subplots_adjust(hspace=.1, wspace=.3)

pf.doubcontbar(fig,axs[0],wff[0,0],-wff[0,1],
               cmap='RdBu',levels=np.linspace(-np.max(np.abs(wff[0])),np.max(np.abs(wff[0])),13),linewidths=0.8,origin='lower')
pf.doubimshbar(fig,axs[1],wff[0,0],-wff[0,1],cmap='RdBu',vmin=-np.max(np.abs(wff[0])),vmax=np.max(np.abs(wff[0])),origin='lower')

fig.tight_layout()
plt.savefig("./../plots/RFs/RFs_"+config_name+".pdf")
