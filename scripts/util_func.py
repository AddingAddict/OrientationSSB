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

def get_two_layer_network_size(config_name,verbose=True):
    config_dict = misc.load_external_params("params_"+config_name,verbose=verbose)
    config_dict.update({"config_name" : config_name})

    N4 = config_dict["N4"]
    N23 = config_dict["N23"]
    Nlgn = config_dict["Nlgn"]
    Nret = config_dict["Nret"]
    Nlgnpop = 2
    N4pop = config_dict["num_lgn_paths"] // Nlgnpop
    rA = ceil(config_dict["Wlgn_to4_params"]["r_A_on"] * config_dict["Wlgn_to4_params"].get("r_lim",1.) * N4)
    
    return config_dict,N4pop,Nlgnpop,Nret,Nlgn,N4,N23,rA

def get_network_system(Version,config_name):
    if Version == -1:
        try:
            load_location = 'local'
            load_path = data_dir + "layer4/{s}/v{v}/".format(s=config_name,v=Version)
            config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
        except:
            config_dict = misc.load_external_params("params_"+config_name,False)
            config_dict.update({
                            "RF_mode" : "initialize",
                            "system" : "one_layer",
                            "Version" : Version,
                            })
            net = network.Network(Version,config_dict,verbose=False)
        else:
            config_dict.update({
                            "config_name" : config_name,
                            "system" : "one_layer",
                            })
            config_dict["Wlgn_to4_params"].update({
                "W_mode": "load_from_external",
                "load_from_prev_run" : Version})
            net = network.Network(Version,config_dict,load_location=load_location,verbose=False)
    else:
        load_location = 'local'
        load_path = data_dir + "layer4/{s}/v{v}/".format(s=config_name,v=Version)
        config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
        config_dict.update({
                        "config_name" : config_name,
                        "system" : "one_layer",
                        })
        config_dict["Wlgn_to4_params"].update({
            "W_mode": "load_from_external",
            "load_from_prev_run" : Version})
        net = network.Network(Version,config_dict,load_location=load_location,verbose=False)
    return net.system

def get_two_layer_network_system(Version,config_name):
    if Version == -1:
        try:
            load_location = 'local'
            load_path = data_dir + "two_layer/{s}/v{v}/".format(s=config_name,v=Version)
            config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
        except:
            config_dict = misc.load_external_params("params_"+config_name,False)
            config_dict.update({
                            "RF_mode" : "initialize",
                            "system" : "two_layer",
                            "Version" : Version,
                            })
            net = network.Network(Version,config_dict,verbose=False)
        else:
            config_dict.update({
                            "config_name" : config_name,
                            "system" : "two_layer",
                            })
            config_dict["Wlgn_to4_params"].update({
                "W_mode": "load_from_external",
                "load_from_prev_run" : Version})
            net = network.Network(Version,config_dict,load_location=load_location,verbose=False)
    else:
        load_location = 'local'
        load_path = data_dir + "two_layer/{s}/v{v}/".format(s=config_name,v=Version)
        config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
        config_dict.update({
                        "config_name" : config_name,
                        "system" : "two_layer",
                        })
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