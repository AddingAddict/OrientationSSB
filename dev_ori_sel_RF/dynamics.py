import tensorflow as tf
import numpy as np
from dev_ori_sel_RF import inputs, image_dir



def dynamics_twolayer(scan_func, y, t, dt, params_dict, **kwargs):
    """Layer 4 and Layer 23 dynamics with vertical units"""
    nl_4 = params_dict["nonlinearity_l4"]
    nl_23 = params_dict["nonlinearity_l23"]
    lgn = params_dict["lgn"]

    ##network params
    N23 = params_dict["N23"]
    N4 = params_dict["N4"]
    Nlgn = params_dict["Nlgn"]
    Nret = params_dict["Nret"]
    Nvert = params_dict["Nvert"]
    num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

    ## cortical layer parameters
    gamma_lgn = params_dict["config_dict"]["gamma_lgn"]
    W4to4 = params_dict["W4to4"]
    W23to4 = params_dict["W23to4"]
    W4to23 = params_dict["W4to23"]
    W23to23 = params_dict["W23to23"]
    tau = params_dict["config_dict"]["tau"]
    gamma_4 = params_dict["config_dict"]["gamma_4"]
    gamma_23 = params_dict["config_dict"]["gamma_23"]
    dt = params_dict["config_dict"]["dt"]
    nu_4 = params_dict["config_dict"]["W4to4_params"]["nu_4"]
    nu_23 = params_dict["config_dict"]["W23_params"]["nu_23"]
    theta_4 = params_dict["theta_4"]

    s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
    l4_size = N4**2*Nvert * 2
    l23_size = N23**2 * 2
    Wlgn_to_4 = y[:s]
    Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])

    l4 = y[s:s+l4_size]
    l23 = y[s+l4_size:s+l4_size+l23_size]
    I4 = kwargs["I_crt"]
    I23 = kwargs["I_crt23"]
    rhs_l4EI = kwargs["rhs_l4"]
    rhs_l23EI = kwargs["rhs_l23"]

    if params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="dynamics":
        l4_avg,theta_4 = 0,0
        for kt in t:
            l4 = scan_func(rhs_l4EI,l4,(kt,dt),N=N4**2*Nvert,inp_ff=lgn,inp_fb=l23,\
                            gamma_FF=gamma_lgn,gamma_rec=gamma_4, Wff_to_l=Wlgn_to_4,\
                            Wfb_to_l=W23to4,W_rec=W4to4, tau=tau, nl=nl_4)

            l23 = scan_func(rhs_l23EI,l23,(kt,dt),N=N23**2,inp_ff=l4,gamma_FF=1.,\
                          gamma_rec=1.,Wff_to_l=W4to23,W_rec=W23to23,tau=tau,nl=nl_23)

    elif params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="antolik_etal":
        lgn_inp = tf.linalg.matvec(Wlgn_to_4[0,:,:],lgn[0,:]*gamma_lgn[0,:]) +\
                    tf.linalg.matvec(Wlgn_to_4[1,:,:],lgn[1,:]*gamma_lgn[1,:])
        # print("lgn_inp",np.nanmax(lgn),np.nanmax(lgn_inp),np.nanmax(Wlgn_to_4))
        if num_lgn_paths==4:
            lgn_inp_I = tf.linalg.matvec(Wlgn_to_4[2,:,:],lgn[0,:]*gamma_lgn[2,:]) +\
                           tf.linalg.matvec(Wlgn_to_4[3,:,:],lgn[1,:]*gamma_lgn[3,:])
            lgn_inp = tf.concat([lgn_inp,lgn_inp_I],0)

        # tf.random.set_seed(2323)
        # epsilon_4 = tf.random.normal(l4.shape,mean=0.0,stddev=1.0,dtype=tf.dtypes.float32)

        lambd = 0.3
        l4_avg = params_dict["l4_avg"]
        for k in range(200):
            total_inp = tf.linalg.matvec(W4to4,l4) + lgn_inp + \
                        tf.linalg.matvec(W23to4,l23)
            epsilon = tf.random.normal(l4.shape,mean=0.0,stddev=1.0,dtype=tf.dtypes.float32)

            l4_new = l4 * (1-lambd) + lambd * nu_4 * nl_4(total_inp,theta_4)\
                     + 0.02 * epsilon
            l4_new = nl_4(l4_new,0)	#added to force L4 activity to be positive
            # l4_new = nl_4(epsilon_4) * 0.1
            l4_avg = l4_new*0.002 + l4_avg*0.998
            theta_4 = theta_4 + 0.02*(l4_avg-0.09)
            # print("l4",k,np.nanmin(l4_new),np.nanmax(l4_new),np.nanmin(l4_avg),np.nanmax(l4_avg),\
            # 	np.nanmin(theta_4))

            total_inp = tf.linalg.matvec(W4to23,l4) + tf.linalg.matvec(W23to23,l23)
            epsilon = tf.random.normal(l23.shape,mean=0.0,stddev=1.0,dtype=tf.dtypes.float32)
            l23_new = l23 * (1-lambd) + nu_23 * lambd*nl_23(total_inp) + 0.02 * epsilon
            l4,l23 = l4_new,l23_new

    else:
        l4_avg,theta_4 = 0,0
        inpE = tf.linalg.matvec(Wlgn_to_4[0,:,:],lgn[0,:]) +\
                                  tf.linalg.matvec(Wlgn_to_4[1,:,:],lgn[1,:])
        l4 = tf.linalg.matvec(I4[:,:N4**2*Nvert],inpE)

        if num_lgn_paths==4:
            inpI = tf.linalg.matvec(Wlgn_to_4[2,:,:],lgn[2,:]) +\
                                      tf.linalg.matvec(Wlgn_to_4[3,:,:],lgn[3,:])
            l4_toI = tf.linalg.matvec(I4[:,N4**2*Nvert:], inpI)
            l4 += l4_toI
        l4 *= gamma_lgn
        l4 = np.clip(l4,0,np.nanmax(l4))

        l23 = tf.linalg.matvec(I23[:,:N4**2*Nvert], inpE)

        if num_lgn_paths==4:
            l23_toI = tf.linalg.matvec(I23[:,N4**2*Nvert:],inpI)
            l23 += l23_toI
        l23 *= gamma_lgn
        l23 = np.clip(l23,0,np.nanmax(l23))

    out = tf.concat([y[:s], l4, l23, y[s+l4_size+l23_size:]], axis=0)

    # dW = tf.zeros_like(Wlgn_to_4)
    # dW = tf.reshape(dW, [s])
    # out = tf.concat([dW, dl4, dl23, 0*y[s+l4_size+l23_size:]], axis=0)
    return out,l4_avg,theta_4

def dynamics_l4_sgl(y, t, params_dict, yt):
    """ff plasticity update after each stimlus, with vertical cortical units"""
    N4 = params_dict["N4"]
    Nlgn = params_dict["Nlgn"]
    Nret = params_dict["Nret"]
    Nvert = params_dict["Nvert"]

    T_pd = params_dict["config_dict"]["Inp_params"]["pattern_duration"]
    T_exp = params_dict["config_dict"]["Inp_params"]["expanse_time"]
    avg_no_inp = params_dict["config_dict"]["Inp_params"]["avg_no_inp"]


    if params_dict["config_dict"]["nonlinearity_l4"]=="rectifier":
        def nl(x):
            out = tf.where(tf.greater(x, 0), x, tf.zeros(tf.shape(x),dtype=tf.float32) )
            # out = x
            return out
    elif params_dict["config_dict"]["nonlinearity_l4"]=="linear":
        def nl(x):
            out = x
            return out

    # lgn = params_dict["lgn"][...,int(tf.math.floor(t/T_pd))]
    if (int(tf.math.floor(t/T_pd))%2)==1:
        lgn = tf.zeros((2,Nlgn*Nlgn))
        # print("zero",t,int(tf.math.floor(t/T_pd)))
    else:
        # rng_seed = params_dict["Version"].numpy()*1000 + int(tf.math.floor(t/T_pd))
        rng_seed = params_dict["config_dict"]["random_seed"]*1000 + int(tf.math.floor(t/T_pd))
        lgn = inputs.Inputs_lgn((Nret,Nret),params_dict["Version"].numpy(),\
              rng_seed).create_lgn_input(params_dict["config_dict"]["Inp_params"],\
              "white_noise_online", params_dict["Wret_to_lgn"].numpy())
        lgn = tf.convert_to_tensor(lgn,dtype=tf.float32)

    arbor2 = params_dict["arbor2"]
    Wlim = params_dict["config_dict"]["Wlgn_to4_params"]["Wlim"]
    num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

    gamma_lgn = params_dict["config_dict"]["gamma_lgn"]
    W4to4 = params_dict["W4to4"]
    tau = params_dict["config_dict"]["tau"]
    beta_P = params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
    beta_O = params_dict["config_dict"]["Wlgn_to4_params"]["beta_O"]
    gamma_4 = params_dict["config_dict"]["gamma_4"]

    s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
    Wlgn_to_4 = y[:s]
    Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])

    l4 = y[s:]

    # dl4 = rhs_l4(t, l4, lgn, gamma_lgn, gamma_4, Wlgn_to_4, W4to4, tau, nl)
    dl4 = rhs_l4EI(t, l4, N4*N4*Nvert, lgn, gamma_lgn, gamma_4, Wlgn_to_4, W4to4, tau, nl)
    ## update ff weights after activity has converged for specific input
    if (t%T_pd)==(T_pd-1):
        notfrozen = tf.math.logical_and(Wlgn_to_4>0, Wlgn_to_4<Wlim)
        mask = tf.math.logical_and( notfrozen, arbor[tf.newaxis,:,:]>0 )
        mode = params_dict["config_dict"]["Wlgn_to4_params"]["normalisation_mode"]
        if mode=="xalpha":
            c_orth = params_dict["c_orth"]
            s_orth = params_dict["s_orth"]
        else:
            c_orth,s_orth = None,None
        dW = unconstrained_plasticity(t, l4[:N4**2*Nvert], lgn, Wlgn_to_4, beta_P, beta_O)
        mask_fl = tf.cast(mask, tf.float32)
        dW = constrain_plasticity_update(dW*arbor[tf.newaxis,:,:],mask_fl,arbor2,mode,\
                                         c_orth,s_orth)
    else:
        dW = tf.zeros_like(Wlgn_to_4)

    dW = tf.reshape(dW, [s])
    out = tf.concat([dW, dl4], axis=0)
    return out

def lowD_GRF_l4(scan_func,y,t,dt,params_dict,**kwargs):
    ndim = params_dict["lowD_subset"].shape[1]
    idx = tf.random.uniform([1],minval=0,maxval=ndim,dtype=tf.int32)
    l4 = params_dict["lowD_subset"][:,idx[0]]

    ##network params
    N4 = params_dict["N4"]
    Nlgn = params_dict["Nlgn"]
    Nret = params_dict["Nret"]
    Nvert = params_dict["Nvert"]
    num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]
    s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert

    out = tf.concat([y[:s], l4], axis=0)
    return out

def dynamics_l4_new(scan_func,y,t,dt,params_dict,**kwargs):
    """only Layer 4 dynamics after number of stimuli with vertical units"""
    nl = params_dict["nonlinearity_l4"]
    lgn = params_dict["lgn"]

    ##network params
    N4 = params_dict["N4"]
    Nlgn = params_dict["Nlgn"]
    Nret = params_dict["Nret"]
    Nvert = params_dict["Nvert"]
    num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

    ## cortical layer parameters
    gamma_lgn = params_dict["config_dict"]["gamma_lgn"]
    W4to4 = params_dict["W4to4"]
    tau = params_dict["config_dict"]["tau"]
    gamma_4 = params_dict["config_dict"]["gamma_4"]
    nu_4 = params_dict["config_dict"]["W4to4_params"]["nu_4"]
    l4_avg = params_dict["l4_avg"]
    theta_4 = params_dict["theta_4"]
    l4e_target = params_dict["config_dict"]["W4to4_params"]["l4e_target"]
    l4i_target = params_dict["config_dict"]["W4to4_params"]["l4i_target"]

    s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
    Wlgn_to_4 = y[:s]
    Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])
    l4 = y[s:]
    rhs_l4EI = kwargs["rhs_l4"]
    I_crt = kwargs["I_crt"]
    dt = params_dict["config_dict"]["dt"]

    if params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="dynamics":
        l4_avg,theta_4 = 0,0
        for kt in t:
            l4 = scan_func(rhs_l4EI, l4, (kt,dt), N=N4*N4*Nvert, inp=lgn, gamma_FF=gamma_lgn,\
                          gamma_rec=gamma_4, Wff_to_l=Wlgn_to_4, W_rec=W4to4, tau=tau, nl=nl)
            
    elif params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="dynamics_adaptive":
        beta = params_dict["config_dict"]["W4to4_params"]["l4_beta"]
        lam = params_dict["config_dict"]["W4to4_params"]["l4_lam"]
        for kt in t:
            l4 = scan_func(rhs_l4EI, l4, (kt,dt), N=N4*N4*Nvert, inp=lgn, gamma_FF=gamma_lgn,\
                          gamma_rec=gamma_4, Wff_to_l=Wlgn_to_4, W_rec=W4to4, tau=tau, nl=nl, theta=theta_4)

        l4_avg = l4*(1-beta) + l4_avg*beta
        theta_4e = theta_4[:N4*N4*Nvert] + lam*(l4_avg[:N4*N4*Nvert]-l4e_target)
        theta_4i = theta_4[N4*N4*Nvert:] + lam*(l4_avg[N4*N4*Nvert:]-l4i_target)
        theta_4 = tf.concat([theta_4e,theta_4i], axis=0)
            
    elif params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="stevens_etal":
        for kt in t:
            l4 = scan_func(rhs_l4EI, l4, (kt,dt), N=N4*N4*Nvert, inp=lgn, gamma_FF=gamma_lgn,\
                          gamma_rec=gamma_4, Wff_to_l=Wlgn_to_4, W_rec=W4to4, tau=tau, nl=nl, theta=theta_4)

        l4_avg = l4*0.009 + l4_avg*0.991
        theta_4e = theta_4[:N4*N4*Nvert] + 0.01*(l4_avg[:N4*N4*Nvert]-l4e_target)
        theta_4i = theta_4[N4*N4*Nvert:] + 0.01*(l4_avg[N4*N4*Nvert:]-l4i_target)
        theta_4 = tf.concat([theta_4e,theta_4i], axis=0)

    elif params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="antolik_etal":
        lgn_inp = tf.linalg.matvec(Wlgn_to_4[0,:,:],lgn[0,:]) +\
                    tf.linalg.matvec(Wlgn_to_4[1,:,:],lgn[1,:])
        if num_lgn_paths==4:
            lgn_inp = tf.concat([lgn_inp,lgn_inp],0)
        lambd = 0.3
        total_inp = tf.linalg.matvec(W4to4,l4)*gamma_4 + lgn_inp*gamma_lgn
        epsilon = tf.random.normal(l4.shape,mean=0.0,stddev=1.0,dtype=tf.dtypes.float32)

        l4_avg = total_inp*0.002 + params_dict["l4_avg"]*0.998
        theta_4 = theta_4 + 0.02*(l4_avg-0.003)

        l4 = l4 * (1-lambd) + lambd*nl(nu_4*total_inp,theta_4) + 0.02 * epsilon

    else:
        l4_avg,theta_4 = 0,0
        l4 = tf.linalg.matvec(I_crt[:,:N4**2*Nvert],tf.linalg.matvec(Wlgn_to_4[0,:,:],lgn[0,:]) +\
                                                       tf.linalg.matvec(Wlgn_to_4[1,:,:],lgn[1,:]))

        if num_lgn_paths==4:
            l4_toI = tf.linalg.matvec(I_crt[:,N4**2*Nvert:],tf.linalg.matvec(Wlgn_to_4[2,:,:],lgn[2,:]) +\
                                                                tf.linalg.matvec(Wlgn_to_4[3,:,:],lgn[3,:]))
            l4 += l4_toI
        l4 *= gamma_lgn
        l4 = np.clip(l4,0,np.nanmax(l4))

    out = tf.concat([y[:s], l4], axis=0)
    return out,l4_avg,theta_4

def dynamics_Wonly(y, t, params_dict, yt):
    """plasticity dynamics only, cortical activity assumed to be converged
    in linear approximation"""
    N4 = params_dict["N4"]
    Nlgn = params_dict["Nlgn"]
    Nret = params_dict["Nret"]
    Nvert = params_dict["Nvert"]

    arbor2 = params_dict["arbor2"]
    Wlim = params_dict["config_dict"]["Wlim"]
    avg_no_inp = params_dict["config_dict"]["Inp_params"]["avg_no_inp"]
    num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

    gamma_lgn = params_dict["config_dict"]["gamma_lgn"]
    beta_P = params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
    beta_O = params_dict["config_dict"]["Wlgn_to4_params"]["beta_O"]
    gamma_4 = params_dict["config_dict"]["gamma_4"]
    W4to4 = params_dict["W4to4"]

    s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
    Wlgn_to_4 = y
    Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])

    ## update ff weights every
    if (((t+1)%avg_no_inp)==0 and t>0):
        dW = 0
        notfrozen = arbor2>0#tf.math.logical_and(Wlgn_to_4>0, Wlgn_to_4<Wlim)
        mask = tf.math.logical_and( notfrozen, arbor2>0 )
        mode = params_dict["config_dict"]["Wlgn_to4_params"]["normalisation_mode"]
        if mode=="xalpha":
            c_orth = params_dict["c_orth"]
            s_orth = params_dict["s_orth"]
        else:
            c_orth,s_orth = None,None
            P_orth = None
        for it in range(avg_no_inp):
            rng_seed = params_dict["config_dict"]["random_seed"]*1000 + int(t)*2 - it*2
            lgn = inputs.Inputs_lgn((Nret,Nret),params_dict["Version"].numpy(),\
                rng_seed).create_lgn_input(params_dict["config_dict"]["Inp_params"],\
                "white_noise_online", params_dict["Wret_to_lgn"].numpy())
            lgn = tf.convert_to_tensor(lgn,dtype=tf.float32)

            ## assume here that W4to4 is Gaussian as in Francescos derivation
            ## ## take only activity from exc units for plasticity update, that means here
            ## W4to4 is E to E connectivity matrix
            l4 = tf.linalg.matvec(W4to4, tf.linalg.matvec(Wlgn_to_4[0,:,:],lgn[0,:])) +\
                    tf.linalg.matvec(W4to4, tf.linalg.matvec(Wlgn_to_4[1,:,:],lgn[1,:]))
            l4 = l4 * gamma_lgn

            # dW += constrained_plasticity(t, l4, lgn, Wlgn_to_4, beta_P, beta_O, mask,\
            # 	arbor, mode, c_orth, s_orth)
            dW += unconstrained_plasticity(t, l4, lgn, Wlgn_to_4, beta_P, beta_O)

        mask_fl = tf.cast(mask, tf.float32)
        dW = constrain_plasticity_update(dW*arbor2,mask_fl,arbor2,mode,\
                                        c_orth,s_orth)

    else:
        dW = tf.zeros_like(Wlgn_to_4)

    dW = tf.reshape(dW, [s])
    return dW

def rhs_l4(l_act,t,inp,gamma_FF,gamma_rec,Wff_to_l,W_rec,tau,nl):
    '''
    input:
    inp			:	input to cortical layer
    l_act		:	activity of purely E crt layer
    '''
    arg = gamma_FF * tf.linalg.matvec(Wff_to_l[0,:,:],inp[0,:]) +\
          gamma_FF * tf.linalg.matvec(Wff_to_l[1,:,:],inp[1,:]) +\
          gamma_rec * tf.linalg.matvec(W_rec, l_act)

    return 1./tau * (nl(arg) - l_act)

def arg_l4_1pop(l_act,inp_on,inp_off,gamma_FF,gamma_rec,Won_to_l,Woff_to_l,W_rec,tau,theta=0): #argument of non-linear function f
    arg_ff = gamma_FF * tf.linalg.matvec(Won_to_l,inp_on) +\
             gamma_FF * tf.linalg.matvec(Woff_to_l,inp_off)
    arg_rec = gamma_rec * tf.linalg.matvec(W_rec, l_act)
    return arg_ff + arg_rec - theta

def rhs_l4EI(l_act,t,**kwargs):
    '''
    input:
    inp			:	input to cortical layer
    l_act		:	activity of E/I crt layer
    '''
    N = kwargs["N"]
    inp = kwargs["inp"]
    gamma_FF = kwargs["gamma_FF"]
    gamma_rec = kwargs["gamma_rec"]
    Wff_to_l = kwargs["Wff_to_l"]
    W_rec = kwargs["W_rec"]
    tau = kwargs["tau"]
    nl = kwargs["nl"]
    theta_4 = kwargs.get("theta_4",None)

    if theta_4 is None:
        argE = arg_l4_1pop(l_act,inp[0,:],inp[1,:],gamma_FF,gamma_rec,Wff_to_l[0,:,:],\
                            Wff_to_l[1,:,:],W_rec[:N,:],tau)
        argI = gamma_rec * tf.linalg.matvec(W_rec[N:,:], l_act)
    else:
        argE = arg_l4_1pop(l_act,inp[0,:],inp[1,:],gamma_FF,gamma_rec,Wff_to_l[0,:,:],\
                            Wff_to_l[1,:,:],W_rec[:N,:],tau,theta=theta_4[:N])
        argI = gamma_rec * tf.linalg.matvec(W_rec[N:,:], l_act)
    # argE = gamma_FF * tf.linalg.matvec(Wff_to_l[0,:,:],inp[0,:]) +\
    # 	   gamma_FF * tf.linalg.matvec(Wff_to_l[1,:,:],inp[1,:]) +\
    # 	   gamma_rec * tf.linalg.matvec(W_rec[:N,:], l_act)
    # argI = gamma_rec * tf.linalg.matvec(W_rec[N:,:], l_act)
    return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - l_act)

def rhs_l4EI_full_LGN_input(l_act,t,**kwargs):
    '''
    input:
    inp			:	input to cortical layer
    l_act		:	activity of E/I crt layer
    '''
    N = kwargs["N"]
    inp = kwargs["inp"]
    gamma_FF = kwargs["gamma_FF"]
    gamma_rec = kwargs["gamma_rec"]
    Wff_to_l = kwargs["Wff_to_l"]
    W_rec = kwargs["W_rec"]
    tau = kwargs["tau"]
    nl = kwargs["nl"]
    theta_4 = kwargs.get("theta_4",None)

    if theta_4 is None:
        argE = arg_l4_1pop(l_act,inp[0,:],inp[1,:],gamma_FF,gamma_rec,Wff_to_l[0,:,:],\
                            Wff_to_l[1,:,:],W_rec[:N,:],tau)
        argI = arg_l4_1pop(l_act,inp[2,:],inp[3,:],gamma_FF,gamma_rec,Wff_to_l[2,:,:],\
                            Wff_to_l[3,:,:],W_rec[N:,:],tau)
    else:
        argE = arg_l4_1pop(l_act,inp[0,:],inp[1,:],gamma_FF,gamma_rec,Wff_to_l[0,:,:],\
                            Wff_to_l[1,:,:],W_rec[:N,:],tau,theta=theta_4[:N])
        argI = arg_l4_1pop(l_act,inp[2,:],inp[3,:],gamma_FF,gamma_rec,Wff_to_l[2,:,:],\
                            Wff_to_l[3,:,:],W_rec[N:,:],tau,theta=theta_4[N:])
    return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - l_act)

def rhs_twolayer_l4EI(act,t,**kwargs):
    N = kwargs["N"]
    inp_ff = kwargs["inp_ff"]
    inp_fb = kwargs["inp_fb"]
    gamma_FF = kwargs["gamma_FF"]
    gamma_rec = kwargs["gamma_rec"]
    Wff_to_l = kwargs["Wff_to_l"]
    Wfb_to_l = kwargs["Wfb_to_l"]
    W_rec = kwargs["W_rec"]
    tau = kwargs["tau"]
    nl = kwargs["nl"]

    argE = gamma_ff * tf.linalg.matvec(Wff_to_l[0,:,:],inp_ff[0,:]) +\
           gamma_ff * tf.linalg.matvec(Wff_to_l[1,:,:],inp_ff[1,:]) +\
           gamma_rec * tf.linalg.matvec(W_rec[:N,:], act) +\
                         tf.linalg.matvec(Wfb_to_l[:N,:], inp_fb)

    argI = gamma_rec * tf.linalg.matvec(W_rec[N:,:], act) +\
                          tf.linalg.matvec(Wfb_to_l[N:,:], inp_fb)
    return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - act)

def rhs_twolayer_l4EI_full_LGN_input(act,t,**kwargs):
    N = kwargs["N"]
    inp_ff = kwargs["inp_ff"]
    inp_fb = kwargs["inp_fb"]
    gamma_ff = kwargs["gamma_FF"]
    gamma_rec = kwargs["gamma_rec"]
    Wff_to_l = kwargs["Wff_to_l"]
    Wfb_to_l = kwargs["Wfb_to_l"]
    W_rec = kwargs["W_rec"]
    tau = kwargs["tau"]
    nl = kwargs["nl"]

    arg_ff = gamma_ff * tf.linalg.matvec(Wff_to_l[0,:,:],inp_ff[0,:]) +\
             gamma_ff * tf.linalg.matvec(Wff_to_l[1,:,:],inp_ff[1,:])
    argE =  arg_ff +\
            gamma_rec * tf.linalg.matvec(W_rec[:N,:], act) +\
                        tf.linalg.matvec(Wfb_to_l[:N,:], inp_fb)

    argI = gamma_ff * tf.linalg.matvec(Wff_to_l[2,:,:],inp_ff[2,:]) +\
            gamma_ff * tf.linalg.matvec(Wff_to_l[3,:,:],inp_ff[3,:]) +\
            gamma_rec * tf.linalg.matvec(W_rec[N:,:], act) +\
                        tf.linalg.matvec(Wfb_to_l[N:,:], inp_fb)
    # print("arg",arg_ff.shape,Wff_to_l.shape,argE.shape,argI.shape,act.shape)
    return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - act)

def rhs_twolayer_l23EI(act,t,**kwargs):
    N = kwargs["N"]
    inp_ff = kwargs["inp_ff"]
    gamma_ff = kwargs["gamma_FF"]
    gamma_rec = kwargs["gamma_rec"]
    Wff_to_l = kwargs["Wff_to_l"]
    W_rec = kwargs["W_rec"]
    tau = kwargs["tau"]
    nl = kwargs["nl"]

    argE = gamma_ff * tf.linalg.matvec(Wff_to_l[:N,:],inp_ff) +\
           gamma_rec * tf.linalg.matvec(W_rec[:N,:], act)
    # argI = gamma_rec * tf.linalg.matvec(W_rec[Nrec:,:], act)
    argI = gamma_ff * tf.linalg.matvec(Wff_to_l[N:,:],inp_ff) +\
           gamma_rec * tf.linalg.matvec(W_rec[N:,:], act)
    return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - act)

def rhs_l23(act,t,**kwargs):
    '''
    input:
    inp		:	input to cortical layer
    l_act		:	activity of crt layer
    '''
    N = kwargs["N"]
    inp_ff = kwargs["inp_ff"]
    gamma_ff = kwargs["gamma_FF"]
    gamma_rec = kwargs["gamma_rec"]
    Wff_to_l = kwargs["Wff_to_l"]
    W_rec = kwargs["W_rec"]
    tau = kwargs["tau"]
    nl = kwargs["nl"]

    arg = gamma_ff * tf.linalg.matvec(Wff_to_l,inp_ff) +\
          gamma_rec * tf.linalg.matvec(W_rec,l_act)
    return 1./tau * (nl(arg) - l_act)

def rhs_l4_wfb(l_act,
            t,
            inp,
            fb_act,
            gamma_FF,
            gamma_rec,
            gamma_FB,
            Wff,
            Wrec,
            Wfb,
            tau,
            nl):
    '''
    input:
    inp			:	input to cortical layer
    l_act		:	activity of crt layer
    '''
    arg = gamma_FF * tf.linalg.matvec(Wff[0,:,:],inp[0,:,:]) +\
          gamma_FF * tf.linalg.matvec(Wff[1,:,:],inp[1,:,:]) +\
          gamma_rec * tf.linalg.matvec(Wrec, l_act) +\
          gamma_FB * tf.linalg.matvec(Wfb,fb_act)
    # print("ff",np.nanmax(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)),np.nanmin(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)))
    # print("winp",np.nanstd(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)))
    # print("rec",np.nanmax(tf.linalg.matvec(W_rec_23, l23)),np.nanmin(tf.linalg.matvec(W_rec_23, l23)))
    return 1./tau * (nl(arg) - l_act)

def constrain_plasticity_update_W4to23(y,dt,dW4to23,arbor,arbor2,Wlim,params_dict,\
    c_orth,s_orth):
    """Plasticity update averaged over avg_no_inp number of input patterns"""
    ##network params
    N4 = params_dict["N4"]
    N23 = params_dict["N23"]
    Nlgn = params_dict["Nlgn"]
    Nvert = params_dict["Nvert"]
    num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

    s = num_lgn_paths*N4*N4*Nlgn*Nlgn*Nvert
    W4to23 = y[s+N4**2*2*Nvert+N23**2*2:]
    W4to23 = tf.reshape(W4to23, [N23**2*2, N4*N4*Nvert*2])
    notfrozen = tf.math.logical_and(W4to23>0, W4to23<Wlim*arbor2)
    mask = tf.math.logical_and( notfrozen, arbor2>0 )
    mask_fl = tf.cast(mask, tf.float32)

    """normalise plasticity update via orthogonalised arbor vectors"""
    dW4to23 = tf.reshape(dW4to23, [N23**2*2, N4*N4*Nvert*2])
    delta_mask = dW4to23[arbor2>0] ## complete update incl multiplied by arbor
    mask_fl = mask_fl[arbor2>0]	## boolean mask as type float

    delta_mask -= tf.reduce_sum(s_orth*tf.linalg.matvec(c_orth,delta_mask)[:,None],axis=0)
    delta_mask *= mask_fl

    delta2 = tf.scatter_nd(tf.where(arbor2>0),delta_mask,arbor2.shape)
    delta2 = tf.reshape(delta2, [N23**2*2, N4*N4*Nvert*2])

    W4to23 += dt*delta2
    return W4to23

def constrained_plasticity(t,
                l4,
                inp,
                Wff,
                beta_P,
                beta_O,
                mask,
                arbor,
                arbor2,
                mode="xalpha",
                c_orth=None,
                s_orth=None
                ):
    """
    incorporates conservation of synaptic strength, and leads to competition
    mode:
    x		:	normalize over all neurons x (normalize per on/off axon)
    alpha	:	normalize over all lgn units alhpha (normalize per crt neuron dendritic tree)
    xalpha	:	normalize first over alpha, then over x
    """
    mask_fl = tf.cast(mask, tf.float32)
    delta = unconstrained_plasticity(t,l4,inp,Wff,beta_P,beta_O) * arbor[tf.newaxis,:,:]# * mask_fl

    ### set filled = 1 since otherwise if all synapses to one cortical\
    #neuron are frozen we get RuntimeWarning due to dividing by zero.
    ### since a_frzn_on.filled(0)/a_frzn_off.filled(0) is only nonzero for\
    #active synapses, eps gets 0 nonetheless if all synapses to one cortical neuron are frozen.
    ## normalized per thalamical unit (sum over cortex units)

    constrained_delta = constrain_plasticity_update(delta,mask_fl,arbor2,mode,\
                                                    c_orth,s_orth)
    return constrained_delta

def rhs_W(t, l23, inp, Wff_to_23, beta_P, beta_post):
    # arbor = tf.where(tf.greater(Wff_to_23, 0), tf.ones(tf.shape(Wff_to_23),dtype=tf.float32),\
    #  tf.zeros(tf.shape(Wff_to_23),dtype=tf.float32) )
    arbor = tf.ones(Wff_to_23.shape)
    ## simple Hebb:
    # update = beta_P/tf.reduce_sum(arbor) * l23[:,None]*inp[None,:] * arbor
    ## Oja's rule:
    update = arbor * l23[:,None]*(beta_P*inp[None,:])# -\
        # beta_post* tf.linalg.matvec(tf.transpose(Wff_to_23),l23)[None,:])
    norm = tf.reduce_sum(update,axis=0)/tf.reduce_sum(arbor)
    update -= norm[None,:] * arbor

    ## subtractive normalization:
    ## normalization for each input loc sepearately (ala francesco)
    # update = update - tf.reduce_sum(update,axis=0)[None,:]
    # print("update",np.nanmax(update),np.nanmin(update),np.nanmin(Wff_to_23),np.nanmin(l23))


    ## normalization for each unit in layer 23 (miller 94)
    # update = update - tf.reduce_sum(update,axis=1)

    # # divisive normalization:
    # # normalised incoming connections to each cortical unit
    # norm = tf.reduce_sum(Wff_to_23 + update,axis=1)[:,None]

    # # normalised outgoing connections from each thalamic unit
    # norm = tf.reduce_sum(W_ff+update,axis=0)[None,:]
    # update = update/norm

    return update



if __name__=="__main__":
    pass