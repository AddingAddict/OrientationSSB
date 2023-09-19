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

from dev_ori_sel_RF import integrator_tf,\
dynamics, data_dir, network_ffrec
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


# def parameter_sweep_layer4(Version,sigma_rec,sigma_cc,r_A):
def parameter_sweep_ffrec(Version,config_dict,**kwargs):
    ## Parameters
    Nvert = config_dict["Nvert"]
    N4 = config_dict["N4"]
    Nlgn = config_dict["Nlgn"]
    Nret = config_dict["Nret"]
    dt = config_dict["dt"]
    # t = np.arange(0,config_dict["runtime"]/dt,1).astype("float64")
    t = np.arange(0,config_dict["Inp_params"]["pattern_duration"],1).astype("float64")


    last_timestep = config_dict["runtime"]/dt
    ## number of input patterns needed
    T_pd = config_dict["Inp_params"]["pattern_duration"]
    T_exp = config_dict["Inp_params"]["expanse_time"]
    config_dict["Inp_params"].update({"Nsur" : int(np.ceil(1.*last_timestep/T_pd/(T_exp+1) ))})
    print("# of stimuli: {}".format(config_dict["Inp_params"]["Nsur"]));sys.stdout.flush()

    config_dict.update({
                    "last_timestep" : last_timestep,
                    "RF_mode" : "initialize",
                    "system" : "one_layer",
                    "Version" : Version,
                    })

    # =================== Network system ===============================================
    n = network_ffrec.Network(Version,config_dict)
    Wret_to_lgn,Wlgn_to_4,arbor_on,arbor_off,arbor2,init_weights,W4to4,arbor4to4,init_weights_4to4 = n.system
    arbor_e = arbor4to4[:N4*N4*Nvert,:N4*N4*Nvert]
    arbor_i = arbor4to4[:N4*N4*Nvert,N4*N4*Nvert:]
    Wrec_mode = config_dict["W4to4_params"]["Wrec_mode"]
    max_ew = config_dict["W4to4_params"]["max_ew"]

    adaptive = config_dict["Inp_params"]["simulate_activity"] in (
            "dynamics_adaptive","stevens_etal","antolik_etal")
    N4pop = config_dict["num_lgn_paths"] // 2
    if adaptive:
        if config_dict["Wlgn_to4_params"]["W_mode"]=="load_from_external":
            if config_dict.get("config_name",False):
                if not os.path.exists(data_dir + "ffrec/{s}".format(s=config_dict["config_name"])):
                    os.makedirs(data_dir + "ffrec/{s}".format(s=config_dict["config_name"]))
                if not os.path.exists(data_dir + "ffrec/{s}/v{v}".format(s=config_dict["config_name"],v=Version)):
                    os.makedirs(data_dir + "ffrec/{s}/v{v}".format(s=config_dict["config_name"],v=Version))
                filename = "ffrec/{s}/v{v}/yt_v{v}.npz".format(s=config_dict["config_name"],v=Version)
            else:
                if not os.path.exists(data_dir + "ffrec/v{v}".format(v=Version)):
                    os.makedirs(data_dir + "ffrec/v{v}".format(v=Version))
                filename = "ffrec/v{v}/yt_v{v}.npz".format(v=Version)
            data_dict = np.load(open(filename,"rb"))
            l4_avg = data_dict["l4_avg"]
            theta_4 = data_dict["theta_4"]
        else:
            l4_avg = config_dict["W4to4_params"].get("l4_avg",0) * np.ones(N4pop*N4**2*Nvert)
            theta_4 = config_dict["W4to4_params"].get("theta_4",0) * np.ones(N4pop*N4**2*Nvert)
    else:
        l4_avg = np.zeros(N4pop*N4**2*Nvert)
        theta_4 = np.zeros(N4pop*N4**2*Nvert)

    print("W4to4",W4to4.shape)

    # ================== Normalisation projector for ff connectivity =======================
    c_orth,s_orth = misc.get_projection_operators(config_dict,config_dict["Wlgn_to4_params"],\
                                                  arbor_on,arbor_off,\
                                                  config_dict["Wlgn_to4_params"]["constraint_mode"],\
                                                  "ffrec")
    c_orth_4to4_e,s_orth_4to4_e = misc.get_projection_operators(config_dict,config_dict["W4to4_params"],\
                                                  arbor_e,0,\
                                                  config_dict["W4to4_params"]["constraint_mode"],\
                                                  "rec4_E")
    c_orth_4to4_i,s_orth_4to4_i = misc.get_projection_operators(config_dict,config_dict["W4to4_params"],\
                                                  arbor_i,0,\
                                                  config_dict["W4to4_params"]["constraint_mode"],\
                                                  "rec4_I")

    if config_dict["Wlgn_to4_params"]["mult_norm"]=="xalpha":
        num_pops = Wlgn_to_4.shape[0]//2
        init_weights = 0
        ## do normalisation separately for E and I population
        for i in range(num_pops):
            Wpop = Wlgn_to_4[i*2:(i+1)*2,...]
            arbpop = arbor2[i*2:(i+1)*2,...]

            dot_product = np.dot(c_orth,Wpop[arbpop>0])
            if isinstance(init_weights,int):
                init_weights = 1.*dot_product
            else:
                init_weights = np.stack([init_weights,dot_product])

    if config_dict["W4to4_params"]["mult_norm"]=="postprex":
        raise Exception("TODO: Code init_weights for recurrent L4 xalpha normalization")

    ##================================= initialization ====================================
    tf.random.set_seed(config_dict["random_seed"]*113)
    if "2pop" in Wrec_mode:
        l40 = tf.random.uniform([N4*N4*2*Nvert], minval=0, maxval=1, dtype=tf.float32)
    else:
        l40 = tf.random.uniform([N4*N4*Nvert], minval=0, maxval=1, dtype=tf.float32)

    if config_dict["tau"]!=1:
        tau = np.ones((N4**2*2*Nvert),dtype=float)
        tau[N4**2*Nvert:] *= config_dict["tau"]
    else:
        tau = 1.
    ## run network
    params_dict = {
                    "Version" : tf.constant(Version, dtype=tf.int32),
                    "Nlgn" : tf.constant(Nlgn, dtype=tf.int32),
                    "N4" : tf.constant(N4, dtype=tf.int32),
                    "Nret" : tf.constant(Nret, dtype=tf.int32),
                    "Nvert" : tf.constant(Nvert, dtype=tf.int32),
                    "invert_rec" : True,

                    "init_weights" : [tf.convert_to_tensor(init_weights[0],dtype=tf.float32),
                            tf.convert_to_tensor(init_weights[1],dtype=tf.float32)]
                        if "ffrec_postpre_approx" in config_dict["Wlgn_to4_params"]["mult_norm"]
                        else tf.convert_to_tensor(init_weights,dtype=tf.float32),
                    "Wret_to_lgn" : tf.convert_to_tensor(Wret_to_lgn,dtype=tf.float32),
                    "W4to4" : tf.convert_to_tensor(W4to4, dtype=tf.float32),
                    "W23to23" : tf.convert_to_tensor(np.array([]), dtype=tf.float32),
                    "W4to23" : tf.convert_to_tensor(np.array([]), dtype=tf.float32),
                    "W23to4" : tf.convert_to_tensor(np.array([]), dtype=tf.float32),
                    "init_weights_4to23" : None,
                    "init_weights_4to4" : [tf.convert_to_tensor(init_weights_4to4[0],dtype=tf.float32),
                            tf.convert_to_tensor(init_weights_4to4[1],dtype=tf.float32)]
                        if "ffrec_postpre_approx" in config_dict["Wlgn_to4_params"]["mult_norm"]
                        else tf.convert_to_tensor(init_weights_4to4,dtype=tf.float32),

                    "l4_avg" : tf.convert_to_tensor(l4_avg,dtype=tf.float32),
                    "theta_4" : tf.convert_to_tensor(theta_4,dtype=tf.float32),

                    "arbor_on" : tf.convert_to_tensor(arbor_on,dtype=tf.float32),
                    "arbor_off" : tf.convert_to_tensor(arbor_off,dtype=tf.float32),
                    "arbor2" : tf.convert_to_tensor(arbor2,dtype=tf.float32),
                    "arbor4to23" : None,
                    "arbor4to23_full" : None,
                    "arbor4to4" : tf.convert_to_tensor(arbor4to4,dtype=tf.float32),
                    "arbor23to23" : None,
                    "arbor23to23_full" : None,
                    "arbor4to4_e" : tf.convert_to_tensor(arbor_e,dtype=tf.float32),
                    "arbor4to4_i" : tf.convert_to_tensor(arbor_i,dtype=tf.float32),
                    "arbor4to4_full" : tf.convert_to_tensor(arbor4to4,dtype=tf.float32),

                    "c_orth" : tf.convert_to_tensor(c_orth,dtype=tf.float32),
                    "s_orth" : tf.convert_to_tensor(s_orth,dtype=tf.float32),
                    "c_orth_4to4_e" : tf.convert_to_tensor(c_orth_4to4_e,dtype=tf.float32),
                    "s_orth_4to4_e" : tf.convert_to_tensor(s_orth_4to4_e,dtype=tf.float32),
                    "c_orth_4to4_i" : tf.convert_to_tensor(c_orth_4to4_i,dtype=tf.float32),
                    "s_orth_4to4_i" : tf.convert_to_tensor(s_orth_4to4_i,dtype=tf.float32),
                    "c_orth_4to4" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),
                    "s_orth_4to4" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),
                    "c_orth_4to23" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),
                    "s_orth_4to23" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),

                    "Corr" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),

                    "integrator" : config_dict["integrator"],
                    "config_dict" : config_dict,
                    }


    s = N4*N4*Nlgn*Nlgn*Nvert
    print("Starting simulation. This might take a while...")
    print("...")


    sys.stdout.flush()
    if config_dict["Inp_params"]["simulate_activity"]:
        if True:
            y0 = tf.concat([Wlgn_to_4.flatten(), l40], axis=0)
            if config_dict["test_lowDsubset"]:
                yt,time_dep_dict = integrator_tf.odeint_new(dynamics.lowD_GRF_l4,\
                                                y0, t, dt, params_dict, mode="dynamic")
            else:
                yt,time_dep_dict = integrator_tf.odeint_new(dynamics.dynamics_l4_new,\
                                                y0, t, dt, params_dict, mode="dynamic")
            # yt,l4t = integrator_tf.odeint(dynamics.dynamics_l4,\
            #                                y0, t, dt, params_dict, mode="dynamic")
            l4t = np.array(time_dep_dict["l4t"])[:,:2*N4**2*Nvert]

            print("CHECK SHAOE",yt.shape,l4t.shape)
            y = yt[-1,:]
            l4 = l4t[-1,:]
            W4to4t = time_dep_dict["W4to4t"]
            W4to4 = W4to4t[-1]
            if adaptive:
                l4_avgt = time_dep_dict["l4_avgt"]
                l4_avg = l4_avgt[-1]
                theta_4t = time_dep_dict["theta_4t"]
                theta_4 = theta_4t[-1]
            else:
                try:
                    del params_dict["config_dict"]["W4to4_params"]['l4_avg']
                    del params_dict["config_dict"]["W4to4_params"]['theta_4']
                except:
                    pass

        else:
            t = t[:-config_dict["Inp_params"]["pattern_duration"]]
            y0 = tf.concat([Wlgn_to_4.flatten(), l40], axis=0)
            y,_ = integrator_tf.odeint(dynamics.dynamics_l4_sgl, y0, t, dt, params_dict,\
                                        mode="single_stim_update")
            l4 = y[2*s:]

    else:
        y0 = tf.concat([Wlgn_to_4.flatten(), l40], axis=0)
        if config_dict["test_lowDsubset"]:
            yt,time_dep_dict = integrator_tf.odeint_new(dynamics.lowD_GRF_l4,\
                                            y0, t, dt, params_dict, mode="dynamic")
        else:
            yt,time_dep_dict = integrator_tf.odeint_new(dynamics.dynamics_l4_new,y0,t,dt,params_dict,\
                                                mode="static")
        l4t = np.array(time_dep_dict["l4t"])[:,:2*N4**2*Nvert]
        y = yt[-1,:]
        l4 = l4t[-1,:]
        W4to4 = time_dep_dict["W4to4t"][-1]
        if adaptive:
            l4_avg = time_dep_dict["l4_avgt"][-1]
            theta_4 = time_dep_dict["theta_4t"][-1]
    #################################################################################
    ############################# SAVE PARAMS AND DATA ##############################
    if config_dict.get("config_name",False):
        if not os.path.exists(data_dir + "ffrec/{s}".format(s=config_dict["config_name"])):
            os.makedirs(data_dir + "ffrec/{s}".format(s=config_dict["config_name"]))
        if not os.path.exists(data_dir + "ffrec/{s}/v{v}".format(s=config_dict["config_name"],v=Version)):
            os.makedirs(data_dir + "ffrec/{s}/v{v}".format(s=config_dict["config_name"],v=Version))
        filename = "ffrec/{s}/v{v}/yt_v{v}.npz".format(s=config_dict["config_name"],v=Version)
    else:
        if not os.path.exists(data_dir + "ffrec/v{v}".format(v=Version)):
            os.makedirs(data_dir + "ffrec/v{v}".format(v=Version))
        filename = "ffrec/v{v}/yt_v{v}.npz".format(v=Version)
    print("Version",Version,s);sys.stdout.flush()
    if config_dict["Inp_params"]["simulate_activity"]:
        if not kwargs["not_saving_temp"]:
            data_dict_time = {
                "Wt"        :   yt[:,:config_dict["num_lgn_paths"]*s],\
                "Wrect"     :   W4to4t,
                #optional:
                # "lgn_inp" :   lgn,\
                # "cct"     :   cct,\
                "l4t"       :   l4t
            }
            if adaptive:
                data_dict_time.update({
                    "l4_avgt"       :   l4_avgt,
                    "theta_4t"       :   theta_4t
                })
        data_dict = {"W" : y[:config_dict["num_lgn_paths"]*s], "Wrec" : W4to4, "l4" : l4}
        if adaptive:
            data_dict.update({"l4_avg" : l4_avg, "theta_4" : theta_4})
    else:
        data_dict_time = {
                "Wt"        :   yt[:,:config_dict["num_lgn_paths"]*s],\
                "Wrect"     :   W4to4t,
                "l4t"       :   l4t
        }
        data_dict = {"W" : y[:config_dict["num_lgn_paths"]*s], "Wrec" : W4to4, "l4" : l4}
        if adaptive:
            data_dict_time.update({
                "l4_avgt"       :   l4_avgt,
                "theta_4t"       :   theta_4t
            })
            data_dict.update({"l4_avg" : l4_avg, "theta_4" : theta_4})
    ## save time development of ff connections and activity
    if not kwargs["not_saving_temp"]:
        misc.save_data(Version, filename, data_dict_time)

    ## save ff connections and activity of last timestep separately
    if config_dict.get("config_name",False):
        filename = "ffrec/{s}/v{v}/y_v{v}.npz".format(s=config_dict["config_name"],v=Version)
    else:
        filename = "ffrec/v{v}/y_v{v}.npz".format(v=Version)
    misc.save_data(Version, filename, data_dict)

    ## save parameter settings
    if config_dict.get("config_name",False):
        filename = "ffrec/{s}/v{v}/config_v{v}".format(s=config_dict["config_name"],v=Version)
    else:
        filename = "ffrec/v{v}/config_v{v}".format(v=Version)
    config_dict.update({
                "maxew"     : np.array([max_ew])\
        })
    misc.save_params(Version,filename,config_dict)
    # #################################################################################
    # #################################################################################


    try:
        del yt
    except:
        pass




if __name__=="__main__":
    # import argparse
    # from bettina.modeling.ori_dev_model import config_dict
    from bettina.modeling.ori_dev_model.tools import parse_args,update_params_dict

    args_dict = vars(parse_args.args)
    print("args_dict",args_dict)

    if args_dict["load_params_file"] is not None:
        config_dict = misc.load_external_params(args_dict["load_params_file"])
    else:
        config_dict = misc.load_external_params("params_default")

    config_dict = update_params_dict.update_params_dict(config_dict,args_dict)

    if args_dict["V"] is not None:
        Version = args_dict["V"]
    else:
        Version = misc.get_version(data_dir + "ffrec/",version=None,readonly=False)



    print("Version",Version)
    print(" ")
    print("config_dict, Wret_to_lgn_params",config_dict["Wret_to_lgn_params"])

    print(" ")
    print("config_dict, W4to4_params",config_dict["W4to4_params"])
    print(" ")

    print(" ")
    print("config_dict, Wlgn_to4_params",config_dict["Wlgn_to4_params"])
    print(" ")

    print(" ")
    print("config_dict, Inp_params",config_dict["Inp_params"])
    print(" ")
    print('gamma_lgn',config_dict["gamma_lgn"])

    parameter_sweep_ffrec(Version,config_dict,**args_dict)
    print("done")







