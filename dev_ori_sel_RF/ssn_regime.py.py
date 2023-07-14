"""
analyse RF of specific form (either generated via simulation or theoretically assuming eg. 
gabor shape) to test phase sensitivity, orientation tuning
one or two layer system with moving grating input
"""

import numpy as np

def fio_lin(x):
    return x
def fio_rect(x):
    x[x<0] = 0
    return x
def fio_powerlaw(x):
    x[x<0] = 0
    return x**2


def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                    W23to23,W23to4,gamma_rec,gamma_ff,N4,N23):
    return dynamics_np.dynamics_twolayer_fullinput(y,inp_ff,Wff,W4to4,W4to23,W23to23,\
                                                      W23to4,gamma_rec,gamma_ff,N4,N23,\
                                                      fio=fio_powerlaw)


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from bettina.modeling.ori_dev_model import image_dir,data_dir,config_dict,dynamics_np,\
    system_generation,network
    from bettina.modeling.ori_dev_model.tools import analysis_tools,misc

    ## set up system
    constrasts = np.array([0.0001,0.001,0.01,0.05,0.075,0.1,0.2,0.5,1.0,])#,10.,20.])
    RF_mode = "initialize"#"load_from_external"#"gabor"#
    system_mode = "one_layer" #"two_layer" "one_layer"
    connectivity_type = "EI"
    N = 20
    Version = 608
    Nvert = 1

    dt = config_dict["dt"]
    T_pd = 50#config_dict["Inp_params"]["pattern_duration"]
    t = np.arange(0,T_pd/dt,1).astype(int)


    Nret,Nlgn,N4,N23 = N,N,N,N
    config_dict["Wlgn_to4_params"].update({"load_from_prev_run" : Version,\
                                            "connectivity_type" : connectivity_type,\
                                            # "r_A" : 0.1
                                            })
    config_dict["Inp_params"].update({"input_type" : "white_noise_online"})
    config_dict.update({"Nlgn" : Nlgn,
                        "Nvert" : Nvert,
                        "N23" : N23,
                        "N4" : N4,
                        "Nret" : Nret,
                        "tau" : 0.5})
    config_dict["W4to4_params"]["max_ew"] = "orig"#0.8##0.013#
    suffix = "_tau05_MH_ahmadianFig2A"#"_MH"

    last_timestep = t[-1]
    config_dict.update({
                        "last_timestep" : last_timestep,
                        "RF_mode" : RF_mode,
                        "system" : system_mode,
                        "Version" : Version
                        })
    n = network.Network(Version,config_dict)
    lgn = n.generate_inputs(full_lgn_output=True,last_timestep=last_timestep,\
                            same_EI_input=True)
    if config_dict["tau"]!=1:
        tau = np.ones((N4**2*2*Nvert),dtype=float)
        tau[N4**2*Nvert:] *= config_dict["tau"]
    else:
        tau = 1.
    # Wret_to_lgn,Wlgn_to_4,arbor,arbor2,init_weights,W4to4,W23to23,W4to23,W23to4 = n.system
    # sf = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
    # rA = int(config_dict["Wlgn_to4_params"]["r_A"] * N4)
    # DA = 2*rA + 5
    # RF,_,_,_ = analysis_tools.get_RF_form(sf,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,mode="o")
    # ext = np.nanmax(np.abs(RF))
    # print("RF",RF.shape)
    # fig = plt.figure(figsize=(6,5))
    # ax = fig.add_subplot(111)
    # im=ax.imshow(RF,interpolation="nearest",cmap="RdBu_r",vmin=-ext,vmax=ext)
    # plt.colorbar(im,ax=ax)
    # plt.show()


    if system_mode=="one_layer":
        Wret_to_lgn,Wlgn_to_4,arbor,arbor2,init_weights,W4to4 = n.system
        ################################# initialization ###############################
        np.random.seed(config_dict["random_seed"]*113)
        l40 = np.random.uniform(0,1,2*N4*N4*Nvert)
        y0 = l40
        def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                            W23to23,W23to4,gamma_rec,gamma_ff,N4,N23):
            return dynamics_np.dynamics_onelayer(y,inp_ff,Wff,W4to4,gamma_rec,gamma_ff,N4,\
                                                tau=tau,fio=fio_powerlaw)
        if connectivity_type=="EI":
            def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                                W23to23,W23to4,gamma_rec,gamma_ff,N4,N23):
                return dynamics_np.dynamics_onelayer_fullinput(y,inp_ff,Wff,W4to4,gamma_rec,\
                                                                gamma_ff,N4,tau=tau,\
                                                                fio=fio_powerlaw)
        W4to23 = 0
        W23to4 = 0
        W23to23= 0
    elif system_mode=="two_layer":
        Wret_to_lgn,Wlgn_to_4,arbor,arbor2,init_weights,W4to4,W23to23,W4to23,W23to4 = n.system
        ################################# initialization ###############################
        np.random.seed(config_dict["random_seed"]*113)
        l40 = np.random.uniform(0,1,2*N4*N4*Nvert)
        l230 = np.random.uniform(0,1,2*N23**2)
        y0 = np.concatenate([l40,l230])

        def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                            W23to23,W23to4,gamma_rec,gamma_ff,N4,N23):
            return dynamics_np.dynamics_twolayer(y,inp_ff,Wff,W4to4,W4to23,W23to23,W23to4,\
                                                 gamma_rec,gamma_ff,N4,N23,tau=tau,\
                                                 fio=fio_powerlaw)

        if connectivity_type=="EI":
            def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                                W23to23,W23to4,gamma_rec,gamma_ff,N4,N23):
                return dynamics_np.dynamics_twolayer_fullinput(y,inp_ff,Wff,W4to4,W4to23,\
                                                                W23to23,W23to4,gamma_rec,\
                                                                 gamma_ff,N4,N23,tau=tau,\
                                                                 fio=fio_powerlaw)

        # W23to4 *= 0
    
    ################################# DYNAMICS ###############################
    gamma_rec = config_dict["gamma_4"]
    if system_mode=="one_layer":
        if RF_mode=="initialize":
            filename = "regime{}.pdf".format(suffix)
        elif RF_mode=="load_from_external":
            filename = "regime_V{}{}.pdf".format(config_dict["Version"],suffix)
    elif system_mode=="two_layer":
        if RF_mode=="initialize":
            filename = "regime_2layer{}.pdf".format(suffix)
        elif RF_mode=="load_from_external":
            filename = "regime_V{}_2layer{}.pdf".format(config_dict["Version"],suffix)
    pp = PdfPages(image_dir + "contrast_response/" + filename)
    act_last_timestep = []
    timesteps = t.size
    for i,contrast in enumerate(constrasts):
        print("Contrast at {}".format(contrast))
        gamma_ff = contrast
        yt = []
        y = y0
        for kt in t:
            dy = dynamics_system(y,lgn[:,:,0],Wlgn_to_4,W4to4,W4to23,\
                                 W23to23,W23to4,gamma_rec,gamma_ff,N4*N4*Nvert,N23**2)
            y = y + dt*dy
            yt.append( y )
        yt = np.array(yt)

        last_response = yt[-1,...]
        act_last_timestep.append(last_response)
        labels = ["L4,E","L4,I","L23,E","L23,I"]
        L4_size = N4*N4*Nvert
        L23_size = N23**2
        if system_mode=="one_layer":
            yt_list = [yt[:,:L4_size].reshape(timesteps,N4,N4*Nvert),\
                        yt[:,L4_size:L4_size*2].reshape(timesteps,N4,N4*Nvert)]
            yfinal_list = [last_response[:L4_size].reshape(N4,N4*Nvert),\
                            last_response[L4_size:L4_size*2].reshape(N4,N4*Nvert)]
            ncols,nrows = 3,1
        elif system_mode=="two_layer":
            yt_list = [yt[:,:L4_size].reshape(timesteps,N4,N4*Nvert),\
                        yt[:,L4_size:L4_size*2].reshape(timesteps,N4,N4*Nvert),\
                        yt[:,L4_size*2:L4_size*2+L23_size].reshape(timesteps,N23,N23),\
                        yt[:,L4_size*2+L23_size:].reshape(timesteps,N23,N23)]
            yfinal_list = [last_response[:L4_size].reshape(N4,N4*Nvert),\
                            last_response[L4_size:L4_size*2].reshape(N4,N4*Nvert),\
                            last_response[L4_size*2:L4_size*2+L23_size].reshape(N23,N23),\
                            last_response[L4_size*2+L23_size:].reshape(N23,N23)]
            ncols,nrows = 3,2
        trace_list = []
        for yt_pop in yt_list:
            trace_list.append(np.nanmean(yt_pop,axis=(1,2)))


        figc = plt.figure(figsize=(6*ncols,5*nrows))
        figc.suptitle("Response L4, contrast={}".format(contrast))
        axes = []
        for k in range(nrows):
            for l in range(ncols):
                axes.append(figc.add_subplot(nrows,ncols,l+k*ncols+1))
        for k,(yt_pop,label) in enumerate(zip(yt_list,labels)):
            ## activity trace
            axes[0].plot(t,trace_list[k],"-",label=label)
            ## max/mean resposne over last moving grating
            im=axes[1+k].imshow(yfinal_list[k],interpolation="nearest",cmap="binary")
            plt.colorbar(im,ax=axes[1+k])
            axes[1+k].set_title(label)
        axes[0].set_ylabel("Activity")
        axes[0].set_xlabel("Timesteps")
        axes[0].legend(loc="best")
        axes[0].legend(loc="best")
        axes[1].set_title("Activity last timestep")
        pp.savefig(figc,dpi=300,bbox_inches="tight")
        plt.close(figc)

    ## input
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(121)
    ax.set_title("Input ON")
    im=ax.imshow(lgn[0,:,0].reshape(Nlgn,Nlgn),interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(122)
    ax.set_title("Input OFF")
    im=ax.imshow(lgn[1,:,0].reshape(Nlgn,Nlgn),interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    pp.savefig(fig,dpi=300,bbox_inches="tight")
    plt.close(fig)


    act_last_timestep = np.array(act_last_timestep)
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(121)
    # ax.plot(constrasts,act_last_timestep[:,N4*(N4+1)//2],"-o",label="Center E")
    # ax.plot(constrasts,act_last_timestep[:,N4*(N4+1)//2+N4**2],"-o",label="Center I")
    ax.plot(constrasts,np.nanmean(act_last_timestep[:,:N4**2],axis=1),"-o",label="Avg E")
    ax.axvline(0.05,ymin=0,ymax=1,ls="--",c="gray")
    ax.legend(loc="best")
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Activity, Center")
    ax = fig.add_subplot(122)
    # ax.plot(constrasts,np.nanmean(act_last_timestep[:,:N4**2],axis=1),"-o",label="Avg E")
    ax.plot(constrasts,np.nanmean(act_last_timestep[:,N4**2:],axis=1),"-o",label="Avg I")
    ax.axvline(0.05,ymin=0,ymax=1,ls="--",c="gray")
    ax.legend(loc="best")
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Avg activity")
    pp.savefig(fig,dpi=300,bbox_inches="tight")
    plt.close(fig)	
    pp.close()

print("Contrast analysis done")
    