#!/usr/bin/python

import os
import sys
import numpy as np
import tensorflow as tf
current_user = os.environ["USER"]
if current_user=="hein":
    import matplotlib
    matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

from dev_ori_sel_RF import integrator_tf, connectivity,\
dynamics, inputs
from dev_ori_sel_RF import misc
from dev_ori_sel_RF import Nlgn, Nret, N4, W4to4_params,\
Wret_to_lgn_params, Wlgn_to4_params, inp_params, image_dir, data_dir,\
config_dict


## layer 4 receiving ff input from lgn receiving input from retina

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


def parameter_sweep_layer4(sigma_rec,sigma_cc,r_A):
    ## Parameters
    dt = 0.05
    t = np.arange(0,config_dict["runtime"]/dt,1).astype("float32")
    T_pd = inp_params["pattern_duration"]
    T_exp = inp_params["expanse_time"]
    inp_params.update({"Nsur" : int(np.ceil(1.*t.size/T_pd/(T_exp+1) ))})
    print("# of stimuli: {}".format(inp_params["Nsur"]))

    print("sigma_cc={}, sigma_rec={}, r_A={}".format(sigma_cc,sigma_rec,r_A));sys.stdout.flush()

    Version = 0#misc.get_version(data_dir + "layer4/",version=None)
    # image_dir_param = image_dir +\
    #  "layer4/tmp/v{}_rA{:.1f}_srec{:.2f}_scc{:.2f}_{:s}_AVG/".format(\
    #   Version,\
    #   Wlgn_to4_params["r_A"],
    #   sigma_rec,sigma_cc,
    #   config_dict["normalisation_mode"])
    # print("image_dir_param",image_dir_param);sys.stdout.flush()
    # if not os.path.exists(image_dir_param):
    #   os.makedirs(image_dir_param)


    # inp_params["mean_width"] = sigma_cc
    Wret_to_lgn_params["sigma"] = sigma_cc
    W4to4_params["sigma"] = sigma_rec
    Wlgn_to4_params["r_A"] = r_A

    ## ====================== Retinal input ============================================
    Wret_to_lgn = connectivity.Connectivity((Nret,Nret),(Nlgn,Nlgn),\
        random_seed=config_dict["random_seed"]).create_matrix(\
        Wret_to_lgn_params,"Gaussian") * Wret_to_lgn_params["gamma_ret"]
    # lgn = inputs.Inputs_lgn((Nret,Nret),Version,\
    #   config_dict["random_seed"]).create_lgn_input(\
    #   inp_params, "white_noise", Wret_to_lgn)
    lgn = []
    for i in range(1,int((t[-1]+1)/inp_params["avg_no_inp"]/T_pd)):
        for it in range(inp_params["avg_no_inp"]):
            if (inp_params["simulate_activity"] and ((int(np.floor((i+1)/T_pd)) - it)%2)==0):
                continue
            rng_seed = config_dict["random_seed"]*1000 + i*inp_params["avg_no_inp"]-1 - it
            lgn.append( inputs.Inputs_lgn((Nret,Nret),Version,rng_seed).create_lgn_input(\
                inp_params, "white_noise_online", Wret_to_lgn) )
    lgn = np.swapaxes(np.swapaxes(np.array(lgn),0,1),1,2)
    print("lgn",lgn.shape,rng_seed);sys.stdout.flush()
    ## ====================== Retinal input end ========================================
    ## =================================================================================


    ## determine input properties
    if False:
        figl = plt.figure()
        ax = figl.add_subplot(221)
        ax.plot(np.nanmean(lgn[0,:,:],axis=0),'-')
        ax.plot(np.nanmean(lgn[1,:,:],axis=0),'-')
        ax = figl.add_subplot(222)
        # ax.plot(np.nanmean(lgn[0,:,:],axis=0),'-')
        # ax.plot(np.nanmean(lgn[1,:,:],axis=0),'-')
        ax.plot(lgn[0,Nlgn,:].T,"-")
        ax.plot(lgn[1,Nlgn,:].T,"-")
        ax = figl.add_subplot(223)
        im=ax.imshow(tf.reshape(lgn[0,:,0],[Nlgn,Nlgn]).numpy(),interpolation="nearest",\
            cmap="binary")
        plt.colorbar(im,ax=ax)
        ax = figl.add_subplot(224)
        im=ax.imshow(tf.reshape(lgn[1,:,0],[Nlgn,Nlgn]).numpy(),interpolation="nearest",\
            cmap="binary")
        plt.colorbar(im,ax=ax)
        # plt.savefig(image_dir_param + "lgn_input.pdf")
        # plt.close(figl)

        # for i in range(10):
        #   fig = plt.figure()
        #   ax = fig.add_subplot(111)
        #   im=ax.imshow(tf.reshape(lgn[0,:,i],[Nlgn,Nlgn]),interpolation="nearest",\
        #       cmap="binary")
        #   plt.colorbar(im,ax=ax)
        # plt.show()
        # sys.exit()
    if True:
    ## check lgn correlations
        N = Nlgn
        # lgn = lgn.reshape(2,N,N,-1)
        lgn = lgn.reshape(2,N*N,-1)
        lgnc = np.delete(lgn,np.arange(1,inp_params["Nsur"]+1)*(T_exp+1)-1,axis=2)
        print("lgnc",lgnc.shape,lgn.shape)
        lgn_norm = (lgnc - 0*np.nanmean(lgnc,axis=2)[:,:,None])/np.nanstd(lgnc,axis=2)[:,:,None]
        cc = np.nanmean(lgn_norm[:,None,:,None,:] * lgn_norm[None,:,None,:,:],axis=4)
        # cc = np.nanmean(lgn[:,None,:,None,:] * lgn[None,:,None,:,:],axis=4)
        cc = cc.reshape(2,2,N,N,N,N)
        fig = plt.figure()
        ax = fig.add_subplot(141)
        ax.set_title("onon")
        im=ax.imshow(cc[0,0,N//2,N//2].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
        plt.colorbar(im,ax=ax)
        ax = fig.add_subplot(142)
        ax.set_title("onoff")
        im=ax.imshow(cc[0,1,N//2,N//2].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
        plt.colorbar(im,ax=ax)
        ax = fig.add_subplot(143)
        ax.set_title("offoff")
        im=ax.imshow(cc[1,1,N//2,N//2].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
        plt.colorbar(im,ax=ax)
        ax = fig.add_subplot(144)
        ax.set_title("onon-onoff")
        diff = (0.5*cc[0,0,N//2,N//2]+0.5*cc[1,1,N//2,N//2]-cc[0,1,N//2,N//2]).reshape(N,N)
        im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
        plt.colorbar(im,ax=ax)
        print("Peak CC",cc[:,:,N//2,N//2,N//2,N//2])
        # plt.savefig(image_dir_param + "cc_lgn.pdf")

        ## average correlation pattern
        cc_avg = 0
        cc[np.logical_not(np.isfinite(cc))] = 0.
        for i in range(N):
            for j in range(N):
                cc_avg += np.roll(np.roll(cc[:,:,i,j,:,:],N//2-i,axis=2),N//2-j,axis=3)
        cc_avg /= N*N
        fig = plt.figure()
        plt.suptitle("Avg corr")
        ax = fig.add_subplot(141)
        ax.set_title("onon")
        im=ax.imshow(cc_avg[0,0].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
        plt.colorbar(im,ax=ax)
        ax = fig.add_subplot(142)
        ax.set_title("onoff")
        im=ax.imshow(cc_avg[0,1].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
        plt.colorbar(im,ax=ax)
        ax = fig.add_subplot(143)
        ax.set_title("offoff")
        im=ax.imshow(cc_avg[1,1].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
        plt.colorbar(im,ax=ax)
        ax = fig.add_subplot(144)
        ax.set_title("onon-onoff")
        diff = (0.5*cc_avg[0,0]+0.5*cc_avg[1,1]-cc_avg[0,1]).reshape(N,N)
        im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
        plt.colorbar(im,ax=ax)
        print("Avg Peak CC",cc_avg[:,:,N//2,N//2])
        # plt.savefig(image_dir_param + "avg_cc_lgn.pdf")
        # plt.close()

        x = np.linspace(-0.5,0.5,N,endpoint=False)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,diff[N//2,:],'-k')
        sigma = r_A * 0.2
        ax.plot(x,np.nanmax(diff[N//2,:])*np.exp(-x**2/2./sigma**2),'--r')

        plt.show()
        sys.exit()

    ## LGN to Layer 4
    ## TODO: add noise/heterogeneity in W
    Wlgn4 = connectivity.Connectivity((Nlgn,Nlgn), (N4,N4),\
     rng=np.random.RandomState(20200204))
    arbor_profile = "heaviside"
    arbor = Wlgn4.create_arbor(radius=Wlgn_to4_params["r_A"],profile=arbor_profile)
    Won_to_4 = Wlgn4.create_matrix(Wlgn_to4_params, "random_delta",\
        r_A=Wlgn_to4_params["r_A"],profile_A="heaviside")
    Wof_to_4 = Wlgn4.create_matrix(Wlgn_to4_params, "random_delta",\
        r_A=Wlgn_to4_params["r_A"],profile_A="heaviside")
    Wlgn_to_4 = np.stack([Won_to_4,Wof_to_4])

    ## init normalization
    ## syn norm over x
    init_weights = np.sum(Wlgn_to_4,axis=1)
    ## syn norm over alpha
    # init_weights = np.sum(Wlgn_to_4,axis=2)

    # normalisation projector for ff connectivity
    # P_orth = dynamics.generate_simIO_normalisation_coarse(Nlgn*Nlgn,N4*N4)
    # row_x,row_alpha = dynamics.generate_elements_simIO_normalisation(Nlgn*Nlgn,N4*N4,N4*N4)
    try:
        constraint_vec = np.load(data_dir + "layer4/P_orth/N4{}_Nlgn{}_rA{}_{}.npy".format(\
            N4,Nlgn,np.around(Wlgn_to4_params["r_A"],2),arbor_profile))
        c_orth = constraint_vec[:constraint_vec.shape[0]//2,:]
        s_orth = constraint_vec[constraint_vec.shape[0]//2:,:]
    except Exception as e:
        print(e)
        print("No file found for projection operator\n generating operator now...")
        c_orth,s_orth = dynamics.generate_simIO_normalisation(Nlgn,N4,arbor)
        np.save(data_dir + "layer4/P_orth/N4{}_Nlgn{}_rA{}_{}.npy".format(N4,\
            Nlgn,np.around(Wlgn_to4_params["r_A"],2),arbor_profile),\
            np.concatenate([c_orth,s_orth]))


    ## Layer 4
    W4 = connectivity.Connectivity((N4,N4), (N4,N4))
    x_I = np.clip(sigma_rec*2,0.1,N4/2.)
    W4to4 = W4.create_matrix(W4to4_params, "Gaussian")#, r_A=x_I)

    ## initialiyation
    tf.random.set_seed(20200128)
    l40 = tf.random.uniform([N4*N4], minval=0, maxval=1, dtype=tf.float32)

    ## run network
    params_dict = {
                    "Version" : tf.constant(Version, dtype=tf.int32),
                    "Nlgn" : tf.constant(Nlgn, dtype=tf.int32),
                    "N4" : tf.constant(N4, dtype=tf.int32),
                    "Nret" : tf.constant(Nret, dtype=tf.int32),
                    "init_weights" : tf.convert_to_tensor(init_weights,dtype=tf.float32),
                    "W4to4" : tf.convert_to_tensor(W4to4, name='W4to4', dtype=tf.float32),
                    "arbor" : tf.convert_to_tensor(arbor,name="arbor",dtype=tf.float32),
                    # "lgn" : tf.convert_to_tensor(lgn,name="lgn",dtype=tf.float32),
                    "Wret_to_lgn" : tf.convert_to_tensor(Wret_to_lgn,dtype=tf.float32),
                    "gamma_4" : tf.constant(0.99, dtype=tf.float32),
                    "gamma_lgn" : tf.constant(.9, dtype=tf.float32),
                    "beta_P" : tf.constant(0.1, dtype=tf.float32),
                    "beta_O" : tf.constant(0.01, dtype=tf.float32),
                    "tau" : tf.constant(1.0, dtype=tf.float32),
                    "Wlim" : tf.constant(12, dtype=tf.float32),
                    "pattern_duration" : tf.constant(T_pd,dtype=tf.float32),
                    "expanse_time" : tf.constant(T_exp,dtype=tf.float32),
                    "normalisation_mode" : tf.constant(config_dict["normalisation_mode"]),
                    # "P_orth" : tf.convert_to_tensor(P_orth,dtype=tf.float32)
                    "c_orth" : tf.convert_to_tensor(c_orth,dtype=tf.float32),
                    "s_orth" : tf.convert_to_tensor(s_orth,dtype=tf.float32),
                    "integrator" : "Euler"#"RK4"
                    }

    s = N4*N4*Nlgn*Nlgn
    y0 = tf.concat([Wlgn_to_4.reshape(2*s), l40], axis=0)
    yt = integrator_tf.odeint(dynamics.dynamics_l4, y0, t, dt, params_dict)
    timepoints = yt.shape[0]

    Wlgn_to_4 = yt[:,:2*s]
    Wlgn_to_4 = tf.reshape(Wlgn_to_4, [timepoints,2,N4*N4,Nlgn*Nlgn])
    after_weights = np.sum(Wlgn_to_4[-1,...],axis=1)
    wtmp = Wlgn_to_4[-1,0,:,:]
    print("weights before/after plasticity",init_weights[0,:5],after_weights[0,:5])
    print("weights outside arbor",np.sum(wtmp[np.logical_not(arbor)]),\
        np.sum(wtmp[np.logical_not(arbor)]))

    l4 = yt[:,2*s:].numpy()


    #################################################################################
    ############################# SAVE PARAMS AND DATA ##############################
    if not os.path.exists(data_dir + "layer4/v{v}".format(v=Version)):
        os.makedirs(data_dir + "layer4/v{v}".format(v=Version))
    print("Version",Version)
    filename = "layer4/v{v}/yt_v{v}.npy".format(v=Version)
    misc.save_data(Version,filename,yt)

    filename = "layer4/v{v}/config_v{v}".format(v=Version)
    config_dict.update({"Wlim" : params_dict["Wlim"].numpy(),\
        "integrator" : params_dict["integrator"],\
        "beta_P" : params_dict["beta_P"].numpy(),\
        "gamma_lgn" : params_dict["gamma_lgn"].numpy(),\
        "gamma_4" : params_dict["gamma_4"].numpy(),\
        "arbor_profile" : arbor_profile\
        })
    misc.save_params(Version,filename,config_dict)
    #################################################################################
    #################################################################################


    #################################################################################
    ############################# FIGURES ###########################################
    fig = plt.figure(figsize=(18,5))
    ax = fig.add_subplot(131)
    deltaW_on = Wlgn_to_4[1:,0,...] - Wlgn_to_4[:-1,0,...]
    deltaW_of = Wlgn_to_4[1:,1,...] - Wlgn_to_4[:-1,1,...]
    avg_deltaW_t = np.nanmean(deltaW_on * deltaW_of,axis=(0))
    avg_deltaW_xa = np.nanmean(deltaW_on * deltaW_of,axis=(1,2))
    DA = int(Wlgn_to4_params["r_A"]*Nlgn*2)
    sf = avg_deltaW_t.reshape(N4,N4,Nlgn,Nlgn)
    delta_RF = np.zeros((DA*N4,DA*N4))
    for i in range(N4):
        for j in range(N4):
            son_ij = np.roll(np.roll(sf[j,i,:,:],shift=Nlgn//2-j,axis=0),\
                shift=Nlgn//2-i,axis=1)
            delta_RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
             son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]

    im=ax.imshow(delta_RF,interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax.set_ylabel("avg cov between change in Won/off")
    ax = fig.add_subplot(132)
    ax.plot(avg_deltaW_xa,"-k")
    ax.set_ylabel("avg cov between change in Won/off")
    ax = fig.add_subplot(133)
    ax.plot(np.nanmean(np.abs(deltaW_on),axis=(1,2)),"-k")
    ax.plot(np.nanmean(np.abs(deltaW_of),axis=(1,2)),"-m")
    ax.set_ylabel("avg abs change in W_on/off")
    fig.savefig(image_dir_param + "change_Weights.pdf")
    plt.close(fig)


    ## check whether sum of RF and PF decay to zero
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(121)
    ax.plot(after_weights.flatten(),'-k')
    ax.set_title("# of RF whose sum is 0: {}".format(np.sum(after_weights==0)))
    ax = fig.add_subplot(122)
    after_weights_a = np.sum(Wlgn_to_4[-1,...],axis=2)
    ax.plot(after_weights_a.flatten(),'-k')
    ax.set_title("# of PF whose sum is 0: {}".format(np.sum(after_weights_a==0)))
    fig.savefig(image_dir_param + "sum_RF-PF.pdf")
    plt.close(fig)


    ## visualize ff W and activity
    fig = plt.figure()
    ax = fig.add_subplot(331)
    ax.set_title("input on(t=-11)")
    im=ax.imshow(tf.reshape(lgn[0,:,0],(Nlgn,Nlgn)),interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(332)
    ax.set_title("input off(t=-11)")
    im=ax.imshow(tf.reshape(lgn[1,:,0],(Nlgn,Nlgn)),interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(333)
    ax.set_title("FF conn (t=-1)")
    im=ax.imshow(tf.reshape(Wlgn_to_4[-1,1,:,Nlgn*Nlgn//2],(N4,N4)),\
        interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax.contour(tf.reshape(arbor[:,Nlgn*Nlgn//2],(N4,N4)),[1],colors="m")

    ax = fig.add_subplot(334)
    ax.set_title("Avg lgn input on")
    im=ax.imshow(tf.reshape(np.nanmean(lgn[0,:,:],axis=1),(Nlgn,Nlgn)),\
        interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(335)
    ax.set_title("Avg lgn input off")
    im=ax.imshow(tf.reshape(np.nanmean(lgn[1,:,:],axis=1),(Nlgn,Nlgn)),\
        interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)

    ax = fig.add_subplot(337)
    ax.set_title("L23 act (t=10)")
    im=ax.imshow(l4[1,:].reshape(N4,N4),interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(338)
    ax.set_title("L23 act (t=-1)")
    im=ax.imshow(l4[-1,:].reshape(N4,N4),interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(339)
    ax.set_title("Rec conn (t=-1)")
    try:
        im=ax.imshow(tf.reshape(W4to4[:,int((ctrs[0,1]*25+ctrs[0,0])*N4)],(N4,N4)),\
            interpolation="nearest",cmap="binary")
    except:
        im=ax.imshow(tf.reshape(W4to4[:,N4],(N4,N4)),\
            interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    fig.savefig(image_dir_param + "IO_test.pdf")
    plt.close(fig)


    ## temporal behaviour
    figt = plt.figure(figsize=(24,5))
    ax = figt.add_subplot(141)
    # ax.plot(l4[:,::100],"-")
    im=ax.imshow(l4,interpolation="nearest",cmap="binary",aspect=1.*l4.shape[1]/l4.shape[0])
    plt.colorbar(im,ax=ax)
    ax.set_ylabel("Timesteps")
    ax.set_xlabel("Act l23")
    ax = figt.add_subplot(142)
    # ax.plot(np.nanmean(l4,axis=1),np.nanmean(lgn[:,:,:l4.shape[0]],axis=(0,1)),'o')
    ax.plot(np.nanmean(l4,axis=1),'-k')
    ax.plot(l4[:,0],'-m')
    ax.plot(l4[:,N4//2*(N4+1)],'-c')
    # ax.plot(np.nanmean(lgn[:,:,:l4.shape[0]],axis=(0,1)),'-')
    # ax.plot(np.nanmean(lgn[0,:,:],axis=(0)),'-m')
    # ax.plot(np.nanmean(lgn[1,:,:],axis=(0)),'-g')
    # ax.set_ylabel("on/off")
    # ax.set_xlabel("l4")
    ax.legend(loc="best")
    ax = figt.add_subplot(143)
    try:
        ax.plot(Wlgn_to_4[:,0,int((ctrs[0,1]*25+ctrs[0,0])*N4),:],"-")
    except:
        ax.plot(Wlgn_to_4[:,0,N4//2+N4//2*N4,:],"-")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Won conn")
    ax = figt.add_subplot(144)
    try:
        ax.plot(Wlgn_to_4[:,1,int((ctrs[0,1]*25+ctrs[0,0])*N4),:],"-")
    except:
        ax.plot(Wlgn_to_4[:,1,N4//2+N4//2*N4,:],"-")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Woff conn")
    figt.savefig(image_dir_param + "dyn_test.pdf")
    plt.close(figt)


    ## correlation between cortical layer and lgn input
    lgn_norm = (lgn - np.nanmean(lgn,axis=1)[:,None,:])/np.nanstd(lgn,axis=1)[:,None,:]
    l4_norm = (l4 - np.nanmean(l4,axis=1)[:,None])/np.nanstd(l4,axis=1)[:,None]
    cc_rec = np.nanmean(l4_norm[:,None,:]*l4_norm[None,:,:],axis=2)
    cc_recff = []
    print("CHECK TIMe",l4_norm.shape,lgn_norm.shape)
    for it in range(min([l4.shape[0],lgn_norm.shape[2]])):
        cc_recff.append( np.nanmean(l4_norm[it,None,:] * lgn_norm[:,:,it],axis=1) )
    cc_recff = np.array(cc_recff)
    print("cc_recff",cc_recff.shape,np.sum(np.isfinite(cc_recff)))
    pp = PdfPages(image_dir_param + "corr_ff_rec.pdf")
    fig = plt.figure(figsize=(18,5))
    ax = fig.add_subplot(131)
    ax.plot(cc_recff[:,0],'-o')
    ax.plot(cc_recff[:,1],'-o')
    ax = fig.add_subplot(132)
    ax.imshow(cc_rec,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    ax = fig.add_subplot(133)
    im=ax.imshow(np.sum(l4,axis=0).reshape(N4,N4),interpolation="nearest",\
        cmap="binary")
    plt.colorbar(im,ax=ax)
    pp.savefig(fig,dpi=300,bbox_inches='tight')
    plt.close(fig)

    inp_on,inp_of = 0,0
    for it in range(min([l4.shape[0],lgn_norm.shape[2]])):
        inp_on += np.clip(np.dot(Wlgn_to_4[it,0,...],lgn[0,:,it]).reshape(N4,N4),0,1000)
        inp_of += np.clip(np.dot(Wlgn_to_4[it,1,...],lgn[1,:,it]).reshape(N4,N4),0,1000)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    im=ax.imshow(inp_on/lgn_norm.shape[2],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(122)
    im=ax.imshow(inp_of/lgn_norm.shape[2],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    pp.savefig(fig,dpi=300,bbox_inches='tight')
    plt.close(fig)

    # for it in range(min([l4.shape[0],lgn_norm.shape[2]])):
    # # for it in range(0,l4.shape[0],20):
    #   fig = plt.figure()
    #   ax = fig.add_subplot(131)
    #   im=ax.imshow(l4[it,:].reshape(N4,N4),interpolation="nearest",cmap="binary")
    #   plt.colorbar(im,ax=ax)
    #   try:
    #       ax = fig.add_subplot(132)
    #       inp_on = np.dot(Wlgn_to_4[it,0,...],lgn[0,:,it]).reshape(N4,N4)
    #       inp_of = np.dot(Wlgn_to_4[it,1,...],lgn[1,:,it]).reshape(N4,N4)
    #       im=ax.imshow(np.clip(inp_on+inp_of,0,1000),interpolation="nearest",cmap="binary")
    #       plt.colorbar(im,ax=ax)
    #       ax = fig.add_subplot(133)
    #       im=ax.imshow(inp_of,interpolation="nearest",cmap="binary")
    #       plt.colorbar(im,ax=ax)
    #   except:
    #       pass
    #   pp.savefig(fig,dpi=300,bbox_inches='tight')
    #   plt.close(fig)
    pp.close()




    ## visualization of SD = S_on - S_off
    #sd = (s1[:,:,0]-s1[:,:,1]).reshape(Nlgn,Nlgn,N4,x_crtx)
    DA = int(Wlgn_to4_params["r_A"]*Nlgn*2)
    sf = (Wlgn_to_4[-1,...]).numpy()
    sf = sf.reshape(2,N4,N4,Nlgn,Nlgn)
    RF = np.zeros((3,DA*N4,DA*N4))
    PF = np.zeros((3,DA*Nlgn,DA*Nlgn))
    for i in range(N4):
        for j in range(N4):
            son_ij = np.roll(np.roll(sf[0,j,i,:,:],shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
            sof_ij = np.roll(np.roll(sf[1,j,i,:,:],shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
            RF[0,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
             son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2] -\
             sof_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
            RF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
             son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
            RF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
             sof_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
            # print(i,j,np.sum(RF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA]),\
            #   np.sum(RF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA]),\
            #   np.sum(sf[0,j,i,:,:]))
            # print(son_ij)

    for i in range(Nlgn):
        for j in range(Nlgn):
            son_ij = np.roll(np.roll(sf[0,:,:,j,i],shift=N4//2-j,axis=0),shift=N4//2-i,axis=1)
            sof_ij = np.roll(np.roll(sf[1,:,:,j,i],shift=N4//2-j,axis=0),shift=N4//2-i,axis=1)
            PF[0,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
             son_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2] -\
             sof_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2]
            PF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
             son_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2]
            PF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
             sof_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4//2-DA//2:N4//2+DA//2+DA%2]
            # print(i,j,np.sum(PF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA]),np.sum(PF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA]))


    fig = plt.figure(figsize=(18,10))
    ## receptive field
    ax = fig.add_subplot(231)
    ax.set_title("S_D")
    im = ax.imshow(RF[0,:,:],interpolation='nearest',cmap='RdBu_r')
    plt.colorbar(im,ax=ax)

    ax = fig.add_subplot(232)
    ax.set_title("S_on")
    im = ax.imshow(RF[1,:,:],interpolation='nearest',cmap='binary')
    plt.colorbar(im,ax=ax)

    ax = fig.add_subplot(233)
    ax.set_title("S_of")
    im = ax.imshow(RF[2,:,:],interpolation='nearest',cmap='binary')
    plt.colorbar(im,ax=ax)

    ## projective field
    ax = fig.add_subplot(234)
    ax.set_title("S_D (PF)")
    im = ax.imshow(PF[0,:,:],interpolation='nearest',cmap='RdBu_r')
    plt.colorbar(im,ax=ax)

    ax = fig.add_subplot(235)
    ax.set_title("S_on (PF)")
    im = ax.imshow(PF[1,:,:],interpolation='nearest',cmap='binary')
    plt.colorbar(im,ax=ax)

    ax = fig.add_subplot(236)
    ax.set_title("S_of (PF)")
    im = ax.imshow(PF[2,:,:],interpolation='nearest',cmap='binary')
    plt.colorbar(im,ax=ax)
    plt.savefig(image_dir_param + "rec_field.pdf")
    plt.close(fig)

    ## OPM
    opm,rn = misc.get_response(sf[0,...]-sf[1,...])
    fig = plt.figure()
    ax = fig.add_subplot(121)
    im=ax.imshow(0.5*np.angle(opm),interpolation="nearest",cmap='hsv')
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(122)
    im=ax.imshow(np.abs(opm),interpolation="nearest",cmap='binary')
    plt.colorbar(im,ax=ax)
    plt.savefig(image_dir_param + "opm.pdf")
    plt.close(fig)

    # plt.show()

    try:
        del Wlgn_to_4
        del l4
        del yt
    except:
        pass


if __name__=="__main__":
    import argparse
    ## ==================Optional Input Arguments ===========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--rA", dest='rA', required=False, type=float)
    parser.add_argument("--sI", dest='sI', required=False, type=float)
    parser.add_argument("--rC", dest='rC', required=False, type=float)
    args = parser.parse_args()

    if args.rA is not None:
        r_A = args.rA
    else:
        r_A = Wlgn_to4_params["r_A"]

    if args.sI is not None:
        sigma_rec_list = np.array([args.sI,]) * r_A
    else:
        sigma_rec_list = np.array([2.5]) * r_A

    if args.rC is not None:
        sigma_cc_list = np.array([args.rC,]) * r_A
    else:
        sigma_cc_list = np.array([0.4,]) * r_A

    for sigma_rec in sigma_rec_list:
        for sigma_cc in sigma_cc_list:
            print(sigma_rec,sigma_cc)
            parameter_sweep_layer4(sigma_rec,sigma_cc,r_A)