import sys
import os
sys.path.insert(0, './..')

import pickle
from math import floor, ceil
import numpy as np

import dev_ori_sel_RF
from dev_ori_sel_RF import data_dir,integrator_tf,dynamics,network,run_onelayer,probe_RFs
from dev_ori_sel_RF.tools import misc,update_params_dict,analysis_tools

def get_network_size(config_name):
    config_dict = misc.load_external_params("params_"+config_name)
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

def get_ori_sel(opm,calc_fft=True):
    N4 = opm.shape[0]
    sel = np.abs(opm)
    ori = np.angle(opm)/2
    ori = ori - (np.sign(ori)-1)*0.5*np.pi
    ori *= 180/np.pi
    
    if calc_fft:
        ori_fft = np.abs(np.fft.fftshift(np.fft.fft2(ori - np.nanmean(ori))))
        ori_fps = np.zeros(int(np.ceil(N4//2*np.sqrt(2))))

        grid = np.arange(-N4//2,N4//2)
        x,y = np.meshgrid(grid,grid)
        bin_idxs = np.digitize(np.sqrt(x**2+y**2),np.arange(0,np.ceil(N4//2*np.sqrt(2)))+0.1)
        for idx in range(0,int(np.ceil(N4//2*np.sqrt(2)))):
            ori_fps[idx] = np.mean(ori_fft[bin_idxs == idx])
        
        return ori,sel,ori_fft,ori_fps
    else:
        return ori,sel