#!/usr/bin/python

"""
layer 4 receiving ff input from lgn receiving input from retina
"""

import os
import sys
import numpy as np
import tensorflow as tf
import logging
from scipy import linalg

from dev_ori_sel_RF import integrator_tf, connectivity,\
dynamics,data_dir,network
from dev_ori_sel_RF.tools import misc



if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

set_tf_loglevel(logging.FATAL)

def parameter_sweep_twolayer(Version,config_dict,**kwargs):
    ## Parameters
    Nvert = config_dict["Nvert"]
    N4 = config_dict["N4"]
    N23 = config_dict["N23"]
    Nlgn = config_dict["Nlgn"]
    Nret = config_dict["Nret"]
    dt = config_dict["dt"]
    t = np.arange(0,config_dict["Inp_params"]["pattern_duration"],1).astype("float64")

    last_timestep = config_dict["runtime"]/dt
    # number of input patterns needed
    T_pd = config_dict["Inp_params"]["pattern_duration"]
    T_exp = config_dict["Inp_params"]["expanse_time"]
    config_dict["Inp_params"].update({"Nsur" : int(np.ceil(1.*last_timestep/T_pd/(T_exp+1) ))})


    config_dict.update({
                    "last_timestep" : last_timestep,
                    "RF_mode" : "initialize",
                    "system" : "two_layer",
                    "Version" : Version,
                    })
    n = network.Network(Version,config_dict)
    Wret_to_lgn,Wlgn_to_4,arbor_on,arbor_off,arbor2,init_weights,W4to4,arbor4to4,\
                            W23to23,arbor23,W4to23,arbor4to23,init_weights_4to23,W23to4 = n.system
    Wrec_mode = config_dict["W4to4_params"]["Wrec_mode"]
    max_ew = config_dict["W4to4_params"]["max_ew"]
    arbor_hstack = np.hstack([arbor4to23,0*arbor4to23])
    arbor4to23_full = np.concatenate([arbor_hstack,arbor_hstack])

    c_orth,s_orth = misc.get_projection_operators(config_dict,config_dict["Wlgn_to4_params"],\
                                              arbor_on,arbor_off,\
                                              config_dict["Wlgn_to4_params"]["constraint_mode"],\
                                              "layer4")
    # Normalisation projector for connectivity from L4 to L23
    c_orth_4to23,s_orth_4to23 = misc.get_projection_operators(config_dict,\
                                                config_dict["W4to23_params"],\
                                                arbor4to23,0,\
                                                config_dict["W4to23_params"]["constraint_mode"],\
                                                "layer23")

    # normalise learning rate by number of projections per projection field
    proj_field = np.sum(Wlgn_to_4[0,:N4**2,0]>0)
    config_dict["Wlgn_to4_params"]["beta_P"] /= proj_field
    print("PLASTICITY RATE",config_dict["Wlgn_to4_params"]["beta_P"])

    if config_dict["W4to23_params"]["plasticity_rule"]!="None":
        proj_field = np.sum(W4to23[:N23**2,0]>0)
        config_dict["W4to23_params"]["beta_P"] /= proj_field

    if config_dict["W4to4_params"]["plasticity_rule"]!="None":
        proj_field = np.sum(np.abs(W4to4[:N4**2,N4**2*Nvert+1])>0)
        config_dict["W4to4_params"]["beta_P"] /= proj_field

    if config_dict["W23_params"]["plasticity_rule"]!="None":
        proj_field = np.sum(np.abs(W23to23[:N23**2,N23**2+1])>0)
        config_dict["W23_params"]["beta_P"] /= proj_field

    ##================================= initialization ====================================
    tf.random.set_seed(config_dict["random_seed"]*113)
    l40 = tf.random.uniform([N4*N4*2*Nvert], minval=0, maxval=1, dtype=tf.float32)*0.1
    l230 = tf.random.uniform([N23*N23*2], minval=0, maxval=1, dtype=tf.float32)*0.1

    if config_dict["tau"]!=1:
        tau = np.ones((N4**2*2*Nvert),dtype=float)
        tau[N4**2*Nvert:] *= config_dict["tau"]
    else:
        tau = 1.

    ## run network
    if arbor4to4 is not None:
        arbor4to4 = tf.convert_to_tensor(arbor4to4,dtype=tf.float32)
    if arbor23 is not None:
        arbor23 = tf.convert_to_tensor(arbor23,dtype=tf.float32)
    params_dict = {
                "Version" : tf.constant(Version, dtype=tf.int32),
                "Nlgn" : tf.constant(Nlgn, dtype=tf.int32),
                "N4" : tf.constant(N4, dtype=tf.int32),
                "N23" : tf.constant(N23, dtype=tf.int32),
                "Nret" : tf.constant(Nret, dtype=tf.int32),
                "Nvert" : tf.constant(Nvert, dtype=tf.int32),

                "Wret_to_lgn" : tf.convert_to_tensor(Wret_to_lgn,dtype=tf.float32),
                "init_weights" : tf.convert_to_tensor(init_weights,dtype=tf.float32),
                "W4to4" : tf.convert_to_tensor(W4to4, dtype=tf.float32),
                "W23to23" : tf.convert_to_tensor(W23to23, dtype=tf.float32),
                "W4to23" : tf.convert_to_tensor(W4to23, dtype=tf.float32),
                "W23to4" : tf.convert_to_tensor(W23to4, dtype=tf.float32),
                "init_weights_4to23" : tf.convert_to_tensor(init_weights_4to23,dtype=tf.float32),

                "arbor_on" : tf.convert_to_tensor(arbor_on,dtype=tf.float32),
                "arbor_off" : tf.convert_to_tensor(arbor_off,dtype=tf.float32),
                "arbor2" : tf.convert_to_tensor(arbor2,dtype=tf.float32),
                "arbor4to23" : tf.convert_to_tensor(arbor4to23,dtype=tf.float32),
                "arbor4to23_full" : tf.convert_to_tensor(arbor4to23_full,dtype=tf.float32),
                "arbor4to4_full" : arbor4to4,
                "arbor23to23_full" : arbor23,
                # "arbor4to4_full" : None,
                # "arbor23to23_full" : None,

                "c_orth" : tf.convert_to_tensor(c_orth,dtype=tf.float32),
                "s_orth" : tf.convert_to_tensor(s_orth,dtype=tf.float32),
                "c_orth_4to23" : tf.convert_to_tensor(c_orth_4to23,dtype=tf.float32),
                "s_orth_4to23" : tf.convert_to_tensor(s_orth_4to23,dtype=tf.float32),

                "config_dict" : config_dict,
                }

    s = N4*N4*Nlgn*Nlgn*Nvert
    print("Starting simulation. This might take a while...")
    print("...")
    sys.stdout.flush()
    if config_dict["W4to23_params"]["plasticity_rule"]!="None":
        y0 = tf.concat([Wlgn_to_4.flatten(), l40, l230, W4to23.flatten()], axis=0)
    else:
        y0 = tf.concat([Wlgn_to_4.flatten(), l40, l230], axis=0)
    yt,time_dep_dict = integrator_tf.odeint_new(dynamics.dynamics_twolayer,\
                                            y0, t, dt, params_dict, mode="dynamic")
    del params_dict["config_dict"]["W4to4_params"]['l4_avg']
    del params_dict["config_dict"]["W4to4_params"]['theta_4']

    l4t = np.array(time_dep_dict["l4t"])[:,:2*N4**2*Nvert]
    l23t = np.array(time_dep_dict["l4t"])[:,2*N4**2*Nvert:]
    y = yt[-1,:]
    l4 = l4t[-1,:]
    l23 = l23t[-1,:]

    #################################################################################
    ############################# SAVE PARAMS AND DATA ##############################
    if not (isinstance(config_dict["gamma_lgn"],int) or\
            isinstance(config_dict["gamma_lgn"],float)):
        config_dict["gamma_lgn"] = 4.
    if not os.path.exists(data_dir + "two_layer/v{v}".format(v=Version)):
        os.makedirs(data_dir + "two_layer/v{v}".format(v=Version))
    print("Version",Version,l4t.shape,yt.shape,yt[:,2*s+2*N4**2*Nvert:].shape)
    sys.stdout.flush()

    data_dict_time = {
                    "Wt"		:	yt[:,:config_dict["num_lgn_paths"]*s],\
                    "l4t"		:	l4t,\
                    "l23t"		:	l23t,
                    }
    print("KEYS",time_dep_dict.keys())
    for further_key in time_dep_dict.keys():
        if further_key!="l4t":
            data_dict_time.update({\
             further_key : np.array(time_dep_dict[further_key])})

    if config_dict["W4to4_params"]["plasticity_rule"]!="None":
        data_dict_time.update({"W4to4" : params_dict["W4to4"]})

    if config_dict["W23_params"]["plasticity_rule"]!="None":
        data_dict_time.update({"W23to23" : params_dict["W23to23"]})

    print("data_dict_time",data_dict_time.keys())
    filename = "two_layer/v{v}/yt_v{v}.npz".format(v=Version)
    misc.save_data(Version, filename, data_dict_time)
    ## save ff connections and activity of last timestep separately
    filename = "two_layer/v{v}/y_v{v}.npz".format(v=Version)
    data_dict = {"W" : y[:config_dict["num_lgn_paths"]*s], "l4" : l4, "l23" : l23}
    if config_dict["W4to23_params"]["plasticity_rule"]!="None":
        data_dict.update({"W4to23" : y[config_dict["num_lgn_paths"]*s+L4_size+L23_size:]})
    misc.save_data(Version, filename, data_dict)


    filename = "two_layer/v{v}/config_v{v}".format(v=Version)
    misc.save_params(Version,filename,config_dict)
    #################################################################################
    #################################################################################




if __name__=="__main__":
    # from bettina.modeling.ori_dev_model import config_dict
    from bettina.modeling.ori_dev_model.tools import parse_args,update_params_dict

    args_dict = vars(parse_args.args)

    if args_dict["load_params_file"] is not None:
        config_dict = misc.load_external_params(args_dict["load_params_file"])
    else:
        config_dict = misc.load_external_params("params_default")

    config_dict = update_params_dict.update_params_dict(config_dict,args_dict)

    if (config_dict["Wlgn_to4_params"]["het_ampl"] and\
     config_dict["Inp_params"]["simulate_activity"]=="antolik_etal"):
        rng = np.random.RandomState(19*90)
        Nlgn = config_dict["Nlgn"]
        noise_per_neuron = rng.uniform(-0.5,0.5,Nlgn**2)
        gamma_ON = config_dict["gamma_lgn"] * 0.9 - noise_per_neuron
        gamma_OFF = config_dict["gamma_lgn"] * 1.1 + noise_per_neuron
        config_dict["gamma_lgn"] = np.stack([gamma_ON,gamma_OFF,gamma_ON,gamma_OFF])
        print("gamma_lgn",config_dict["gamma_lgn"].shape)

    if args_dict["V"] is not None:
        Version = args_dict["V"]
    else:
        Version = misc.get_version(data_dir + "layer4/",version=None,readonly=False)

    print("args_dict",args_dict)
    print(" ")
    # print("config_dict, Wret_to_lgn_params",config_dict["Wret_to_lgn_params"])
    # print("config_dict, W4to4_params",config_dict["W4to4_params"])

    # print(" ")
    # print("config_dict, Wlgn_to4_params",config_dict["Wlgn_to4_params"])
    # print(" ")


    # config_dict["tau"] = 0.25

    parameter_sweep_twolayer(Version,config_dict,**args_dict)
    print("done")