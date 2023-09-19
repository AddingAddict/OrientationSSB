import tensorflow as tf
import numpy as np
# from collections import defaultdict

# plasticity rules
def activity_based(t,postsyn_act,presyn_act,W,beta_P,Nl4=None):
    update = postsyn_act[:,None] * (beta_P * presyn_act[None,:] )
    return update

def activity_based_EI_input(t,postsyn_act,presyn_act,W,beta_P,Nl4):
    E_update = postsyn_act[None,:Nl4,None] * (beta_P * presyn_act[:2,None,:] )
    I_update = postsyn_act[None,Nl4:,None] * (beta_P * presyn_act[2:,None,:] )
    update = tf.concat([E_update,I_update],0)
    ## normalise learning rate such that learning rate is approximately indep of l4
    ## activity amplitude
    # update = update/np.nanmean(postsyn_act) * 307.
    return update

def activity_corr_EI_input(t,corr,W,beta_P):
    """
    input:
    t: timestep
    corr: activity correlation matrix N4 x N4 (assume same corr for E and I)
    W: ff weight matrix num_lgn_path x N4 x Nlgn
    """
    W_D = tf.stack([W[0,:,:],W[2,:,:]]) - tf.stack([W[1,:,:],W[3,:,:]])

    dW_D = (tf.expand_dims(tf.expand_dims(tf.reduce_sum(corr,axis=1),axis=0),axis=2)*W_D -\
            tf.matmul(corr,W_D) )* beta_P


    dW = tf.stack([dW_D[0,:,:],-dW_D[0,:,:],dW_D[1,:,:],-dW_D[1,:,:]])
    return dW

# constraints for plasticity updates
def constrain_update_x(dW,W_old,mask,A,dt):
    # sum over x
    norm = tf.reduce_sum(A,axis=1)
    norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
    eps = 1.*tf.reduce_sum(dW,axis=1)/norm
    dW_constraint = (dW - eps[:,None,:] * A) * mask
    return dW_constraint*dt+W_old

def constrain_update_alpha(dW,W_old,mask,A,dt):
    # sum over alpha and on/off
    norm = tf.reduce_sum(A,axis=(0,2))
    norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
    eps = 1.*tf.reduce_sum(dW,axis=(0,2))/norm
    dW_constraint = (dW - eps[None,:,None]*A) * mask
    return dW_constraint*dt+W_old

def constrain_update_xalpha_approx(dW,W_old,mask,A,dt):
    ## first sum over alpha and on/off, then over x
    dW_constraint = dW
    for i in range(2):
        norm = tf.reduce_sum(A,axis=(0,2))
        norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
        eps = 1.*tf.reduce_sum(dW_constraint,axis=(0,2))/norm
        dW_alpha = (dW_constraint - eps[None,:,None]*A) * mask

        norm = tf.reduce_sum(A,axis=1)
        norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
        eps = 1.*tf.reduce_sum(dW_alpha,axis=1)/norm
        dW_constraint = (dW_alpha - eps[:,None,:] * A) * mask
    return dW_constraint*dt+W_old

def constrain_update_xalpha(dW,W_old,mask,A,c_orth,s_orth,dt):
    dW_mask = dW[A>0] ## complete update incl multiplied by arbor
    mask_fl = mask[A>0] ## boolean mask as type float
    # delta_mask *= mask_fl ## inserted to see whether that incr conservation of weights

    dW_mask -= tf.reduce_sum(s_orth*tf.linalg.matvec(c_orth,dW_mask)[:,None],axis=0)
    dW_mask *= mask_fl
    dW_constraint = tf.scatter_nd(tf.where(A>0),dW_mask,A.shape)
    dW_constraint = tf.reshape(dW_constraint, tf.shape(dW))
    return dW_constraint*dt+W_old

def constrain_update_postx(dW,W_old,mask,A,dt):
    # sum over x
    norm = tf.reduce_sum(A,axis=0)
    norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
    eps = 1.*tf.reduce_sum(dW,axis=0)/norm
    dW_constraint = (dW - eps[None,:] * A) * mask
    return dW_constraint*dt+W_old

def constrain_update_prex(dW,W_old,mask,A,dt):
    # sum over alpha and on/off
    norm = tf.reduce_sum(A,axis=1)
    norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
    eps = 1.*tf.reduce_sum(dW,axis=1)/norm
    dW_constraint = (dW - eps[:,None]*A) * mask
    return dW_constraint*dt+W_old

def constrain_update_postprex_approx(dW,W_old,mask,A,dt):
    ## first sum over alpha and on/off, then over x
    norm = tf.reduce_sum(A,axis=1)
    norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
    eps = 1.*tf.reduce_sum(dW,axis=1)/norm
    dW_alpha = (dW - eps[:,None]*A) * mask

    norm = tf.reduce_sum(A,axis=0)
    norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
    eps = 1.*tf.reduce_sum(dW_alpha,axis=0)/norm
    dW_constraint = (dW_alpha - eps[None,:] * A) * mask
    return dW_constraint*dt+W_old

def constrain_update_postprex(dW,W_old,mask,A,c_orth,s_orth,dt): ##TODO, not sure if this works as intended!
    dW_mask = dW[A>0] ## complete update incl multiplied by arbor
    mask_fl = mask[A>0] ## boolean mask as type float
    # delta_mask *= mask_fl ## inserted to see whether that incr conservation of weights

    dW_mask -= tf.reduce_sum(s_orth*tf.linalg.matvec(c_orth,dW_mask)[:,None],axis=0)
    dW_mask *= mask_fl
    dW_constraint = tf.scatter_nd(tf.where(A>0),dW_mask,A.shape)
    dW_constraint = tf.reshape(dW_constraint, tf.shape(dW))
    return dW_constraint*dt+W_old

def constrain_update_divisive(dW,W_old,A,dt):
    W_new = W_old + dt*dW * A
    # print("W_new,W_old",np.nanmax(W_new),np.nanmax(W_old),np.nanmax(dW*A))
    if W_new.ndim==2:
        constraint = tf.reduce_sum(W_new,axis=-1,keepdims=True)
    elif W_new.ndim==3:
        if W_new.shape[0]==4:
            constraint1 = tf.reduce_sum(W_new[:2,:,:],axis=(0,-1),keepdims=True)
            constraint2 = tf.reduce_sum(W_new[2:,:,:],axis=(0,-1),keepdims=True)
            constraint = tf.concat([constraint1,constraint2]) * 0.5
        elif W_new.shape[0]==2:
            constraint = tf.reduce_sum(W_new,axis=(0,-1),keepdims=True) * 0.5
    W_new = W_new/np.abs(constraint)
    # print("constraint",W_new.ndim,constraint[:3],tf.reduce_sum(W_new,axis=-1)[:3],W_new.shape)
    return W_new

def clip_by_norm_arbor(W,A,Wlim):
    norm = tf.reduce_sum(A,axis=-1,keepdims=True)
    norm_A = A / norm
    return tf.clip_by_value(W,0,norm_A*Wlim)

def prune_by_norm_arbor(W,A,Wthresh):
    norm = tf.reduce_sum(A,axis=-1,keepdims=True)
    norm_A = A / norm
    thresh = norm_A*Wthresh
    return W*0.5*(1+tf.math.tanh(
        (tf.where(tf.equal(A,0),tf.ones(tf.shape(A),dtype=tf.float32),W/thresh)-1)))

# multiplicative normlaisation
def synaptic_normalization(W_clipped,H,arbor,Wlim,init_W,c_orth=None,axis=1,mode="xalpha"):
    if c_orth is None:
        frozen = tf.math.logical_or(tf.abs(W_clipped)>=(Wlim*arbor), tf.abs(W_clipped)<=0)
        frozen_fl = tf.cast(frozen,tf.float32)
        gamma = np.ones_like(W_clipped,dtype=np.float32)
        # ## norm over on/off and alpha
        # Wfrozen = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 1.),\
        #           W, tf.zeros(tf.shape(W),dtype=tf.float32)), axis=(0,2))
        # Wactive = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 0.),\
        #           W, tf.zeros(tf.shape(W),dtype=tf.float32)), axis=(0,2))
        # gamma[...] =  (-Wfrozen + tf.reduce_sum(arbor_fl,axis=1)[None,:]*2.)/Wactive
        # W = gamma * W

        if isinstance(axis,tuple):
            init_WE = tf.reduce_sum(init_W[:2,...],axis=0)
            init_WI = tf.reduce_sum(init_W[2:,...],axis=0)
            # norm over outgoing connections (sum over L4 cortical space x for conn LGN to L4)
            Wfrozen = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 1.),\
                        W_clipped, tf.zeros(tf.shape(W_clipped),dtype=tf.float32)), axis=2)
            WfrozenE = tf.reduce_sum(Wfrozen[:2,:],axis=0)
            WfrozenI = tf.reduce_sum(Wfrozen[2:,:],axis=0)
            Wactive = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 0.),\
                        W_clipped, tf.zeros(tf.shape(W_clipped),dtype=tf.float32)), axis=2)
            WactiveE = tf.reduce_sum(Wactive[:2,:],axis=0)
            WactiveI = tf.reduce_sum(Wactive[2:,:],axis=0)

            gamma[:2,...] = tf.expand_dims(tf.expand_dims((-WfrozenE + init_WE)/WactiveE,\
                            axis=axis[0]),axis=axis[1])
            gamma[2:,...] = tf.expand_dims(tf.expand_dims((-WfrozenE + init_WE)/WactiveE,\
                            axis=axis[0]),axis=axis[1])
        else:
            # norm over outgoing connections (sum over L4 cortical space x for conn LGN to L4)
            Wfrozen = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 1.),\
                        W_clipped, tf.zeros(tf.shape(W_clipped),dtype=tf.float32)), axis=axis)
            Wactive = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 0.),\
                        W_clipped, tf.zeros(tf.shape(W_clipped),dtype=tf.float32)), axis=axis)
            gamma[...] = tf.expand_dims((-Wfrozen + init_W)/Wactive,axis=axis)
        gamma[frozen.numpy()] = 1.0
        gamma = tf.clip_by_value(gamma,0.8,1.2)
        W_new = gamma * W_clipped
    else:
        num_pops = W_clipped.shape[0]//2
        W_new = 0
        n_orth = tf.cast(c_orth>0,tf.float32)
        ## do normalisation separately for E and I population
        for i in range(num_pops):
            Wpop = W_clipped[i*2:(i+1)*2,...]
            arbpop = arbor[i*2:(i+1)*2,...]
            initWpop = init_W[i,...]


            frozen = tf.math.logical_or(Wpop>=(Wlim*arbpop), Wpop<=0)
            frozen_fl = tf.cast(frozen,tf.float32)


            W_orth = Wpop[arbpop>0] - tf.linalg.matvec(c_orth,\
                                                    tf.linalg.matvec(c_orth,Wpop[arbpop>0]),\
                                                    transpose_a=True)
            Wfrozen = tf.where(tf.equal(frozen_fl, 1.),\
                                Wpop, tf.zeros(tf.shape(Wpop),dtype=tf.float32))
            Wactive = tf.where(tf.equal(frozen_fl, 0.),\
                                Wpop, tf.zeros(tf.shape(Wpop),dtype=tf.float32))
            Wfrozen_vec = tf.linalg.matvec(c_orth,Wfrozen[arbpop>0])
            Wactive_vec = tf.linalg.matvec(c_orth,Wactive[arbpop>0])
            gamma_vec = (initWpop - Wfrozen_vec)/Wactive_vec ## gamma_k
            gamma_vec = tf.where(tf.equal(Wactive_vec, 0),\
                                 tf.ones(tf.shape(Wactive_vec),dtype=tf.float32), gamma_vec)
            # #gamma = tf.clip_by_value(gamma,0.8,1.2)

            if isinstance(W_new,int):
                W_new = tf.linalg.matvec(c_orth,Wactive_vec*gamma_vec + Wfrozen_vec,\
                                        transpose_a=True) + W_orth
                W_new = tf.where(tf.equal(frozen_fl[arbpop>0], 1.), Wpop[arbpop>0], W_new)
            else:
                # W_tmp = Wactive*gamma + Wfrozen
                W_tmp = tf.linalg.matvec(c_orth,Wactive_vec*gamma_vec + Wfrozen_vec,\
                                        transpose_a=True) + W_orth
                W_tmp = tf.where(tf.equal(frozen_fl[arbpop>0], 1.), Wpop[arbpop>0], W_tmp)
                W_new = tf.concat([W_new,W_tmp], 0)

        W_new = tf.scatter_nd(tf.where(arbor>0),W_new,W_clipped.shape)

    return W_new,H


def homeostatic_normalization(W_clipped,H,running_l4_avg,l4_target,Wlim,dt):
    """
    homeostatic normalisation of weights depending on cortical activity relative
    to target activity
    """

    if isinstance(W_clipped,tuple):
        W_clipped = W_clipped[0]

    # H = tf.reshape(H,[2,-1])
    # H_new = H + 0.01 * dt * (1 - l4_avg/tf.expand_dims(l4_target,axis=1))
    H_new = 1 + 0.1 * dt * (1 - running_l4_avg/l4_target)
    # num_lgn_paths,h,w = W_clipped.shape
    W_copy = W_clipped
    # W_new = tf.concat([W_clipped[:num_lgn_paths//2,:,:] *\
    #                 tf.expand_dims(tf.expand_dims(H_new[0,:],axis=0),axis=2),\
    #                 W_clipped[num_lgn_paths//2:,:,:] *\
    #                 tf.expand_dims(tf.expand_dims(H_new[1,:],axis=0),axis=2)],axis=0)
    W_new = W_clipped * tf.expand_dims(tf.expand_dims(H_new,axis=0),axis=2)

    W_new = tf.where(tf.logical_or(W_copy<=0,W_copy>=Wlim),W_copy,W_new)

    return W_new, H_new

def convert_dW_to_pre(dW_post_dict,Nlgn,Nl4):
    # dW_on_l4 = tf.reshape(dW_post_dict["dW_on_l4"],(2*Nl4,Nlgn))
    # dW_off_l4 = tf.reshape(dW_post_dict["dW_off_l4"],(2*Nl4,Nlgn))
    # dW_e_l4 = tf.reshape(dW_post_dict["dW_e_l4"],(2*Nl4,Nl4))
    # dW_i_l4 = tf.reshape(dW_post_dict["dW_i_l4"],(2*Nl4,Nl4))
    dW_on_l4 = dW_post_dict["dW_on_l4"]
    dW_off_l4 = dW_post_dict["dW_off_l4"]
    dW_e_l4 = dW_post_dict["dW_e_l4"]
    dW_i_l4 = dW_post_dict["dW_i_l4"]

    dW_pre_dict = {}
    dW_pre_dict['dW_lgne_e'] = tf.concat([dW_on_l4[:Nl4,:], dW_off_l4[:Nl4,:],
                                          dW_e_l4[:Nl4,:]], 1)
    dW_pre_dict['dW_lgne_i'] = tf.concat([dW_on_l4[Nl4:,:], dW_off_l4[Nl4:,:],
                                          dW_e_l4[Nl4:,:]], 1)
    dW_pre_dict['dW_i_e'] = dW_i_l4[:Nl4,:]
    dW_pre_dict["dW_i_i"] = dW_i_l4[Nl4:,:]

    return dW_pre_dict

def convert_dW_to_post(dW_pre_dict,Nlgn,Nl4):
    # dW_lgne_e = tf.reshape(dW_pre_dict["dW_lgne_e"],(Nl4,2*Nlgn+Nl4))
    # dW_lgne_i = tf.reshape(dW_pre_dict["dW_lgne_i"],(Nl4,2*Nlgn+Nl4))
    # dW_i_e = tf.reshape(dW_pre_dict["dW_i_e"],(Nl4,Nl4))
    # dW_i_i = tf.reshape(dW_pre_dict["dW_i_i"],(Nl4,Nl4))
    dW_lgne_e = dW_pre_dict["dW_lgne_e"]
    dW_lgne_i = dW_pre_dict["dW_lgne_i"]
    dW_i_e = dW_pre_dict["dW_i_e"]
    dW_i_i = dW_pre_dict["dW_i_i"]

    dW_post_dict = {}
    dW_post_dict['dW_on_l4'] = tf.concat([dW_lgne_e[:,:Nlgn], dW_lgne_i[:,:Nlgn]], 0)
    dW_post_dict['dW_off_l4'] = tf.concat([dW_lgne_e[:,Nlgn:2*Nlgn], dW_lgne_i[:,Nlgn:2*Nlgn]], 0)
    dW_post_dict['dW_e_l4'] = tf.concat([dW_lgne_e[:,2*Nlgn:], dW_lgne_i[:,2*Nlgn:]], 0)
    dW_post_dict['dW_i_l4'] = tf.concat([dW_i_e,dW_i_i], 0)

    return dW_post_dict

def convert_dW_to_pre_sep(dW_post_dict,Nlgn,Nl4):
    # dW_on_l4 = tf.reshape(dW_post_dict["dW_on_l4"],(2*Nl4,Nlgn))
    # dW_off_l4 = tf.reshape(dW_post_dict["dW_off_l4"],(2*Nl4,Nlgn))
    # dW_e_l4 = tf.reshape(dW_post_dict["dW_e_l4"],(2*Nl4,Nl4))
    # dW_i_l4 = tf.reshape(dW_post_dict["dW_i_l4"],(2*Nl4,Nl4))
    dW_on_l4 = dW_post_dict["dW_on_l4"]
    dW_off_l4 = dW_post_dict["dW_off_l4"]
    dW_e_l4 = dW_post_dict["dW_e_l4"]
    dW_i_l4 = dW_post_dict["dW_i_l4"]

    dW_pre_dict = {}
    dW_pre_dict['dW_lgn_e_sep'] = tf.concat([dW_on_l4[:Nl4,:], dW_off_l4[:Nl4,:]], 1)
    dW_pre_dict['dW_lgn_i_sep'] = tf.concat([dW_on_l4[Nl4:,:], dW_off_l4[Nl4:,:]], 1)
    dW_pre_dict['dW_e_e'] = dW_e_l4[:Nl4,:]
    dW_pre_dict['dW_i_e'] = dW_i_l4[:Nl4,:]
    dW_pre_dict['dW_e_i'] = dW_e_l4[Nl4:,:]
    dW_pre_dict["dW_i_i"] = dW_i_l4[Nl4:,:]

    return dW_pre_dict

def convert_dW_to_post_sep(dW_pre_dict,Nlgn,Nl4):
    # dW_lgne_e = tf.reshape(dW_pre_dict["dW_lgne_e"],(Nl4,2*Nlgn+Nl4))
    # dW_lgne_i = tf.reshape(dW_pre_dict["dW_lgne_i"],(Nl4,2*Nlgn+Nl4))
    # dW_i_e = tf.reshape(dW_pre_dict["dW_i_e"],(Nl4,Nl4))
    # dW_i_i = tf.reshape(dW_pre_dict["dW_i_i"],(Nl4,Nl4))
    dW_lgn_e = dW_pre_dict["dW_lgn_e_sep"]
    dW_lgn_i = dW_pre_dict["dW_lgn_i_sep"]
    dW_e_e = dW_pre_dict["dW_e_e"]
    dW_i_e = dW_pre_dict["dW_i_e"]
    dW_e_i = dW_pre_dict["dW_e_i"]
    dW_i_i = dW_pre_dict["dW_i_i"]

    dW_post_dict = {}
    dW_post_dict['dW_on_l4'] = tf.concat([dW_lgn_e[:,:Nlgn], dW_lgn_i[:,:Nlgn]], 0)
    dW_post_dict['dW_off_l4'] = tf.concat([dW_lgn_e[:,Nlgn:], dW_lgn_i[:,Nlgn:]], 0)
    dW_post_dict['dW_e_l4'] = tf.concat([dW_e_e, dW_e_i], 0)
    dW_post_dict['dW_i_l4'] = tf.concat([dW_i_e,dW_i_i], 0)

    return dW_post_dict

class Plasticity:
    def __init__(self, dt, c_orth, s_orth, beta_P, plasticity_rule,\
     constraint_mode, mult_norm, clip_mode, weight_strength, Wlim=None, init_weights=None, freeze_weights=True, Wthresh=None):
        self.dt = dt
        self.plasticity_rule = plasticity_rule
        self.constraint_mode = constraint_mode
        self.multiplicative_normalisation = mult_norm
        self.clip_mode = clip_mode
        self.connectivity_type = "E"
        self.c_orth = c_orth
        self.s_orth = s_orth
        self.beta_P = beta_P
        self.Wlim = Wlim
        self.Wthresh = Wthresh
        self.init_weights = init_weights
        self.weight_strength = weight_strength
        self.freeze_weights = freeze_weights
        print("self.weight_strength",self.weight_strength)
        print("self.freeze_weights",self.freeze_weights)
        print("self.Wlim",self.Wlim)
        print("self.Wthresh",self.Wthresh)

        self._init_plasticity_rule()
        self._init_plasticity_constraint()
        self._init_pruning_nonlinearity()
        self._init_multiplicative_norm()
        self._init_clip_weights()

    def _init_plasticity_rule(self):
        """ defines synaptic plasticity rule """
        if self.plasticity_rule=="activity_based":
            if self.connectivity_type=="EI":
                self.unconstrained_update =\
                 lambda t,r,u,C,W,beta_P,Nl4: activity_based_EI_input(t,r,u,W,beta_P,Nl4)
            else:
                self.unconstrained_update =\
                 lambda t,r,u,C,W,beta_P,Nl4: activity_based(t,r,u,W,beta_P,Nl4)
        elif self.plasticity_rule=="activity_corr":
            self.unconstrained_update =\
             lambda t,r,u,C,W,beta_P,Nl4: activity_corr_EI_input(t,C,W,beta_P)

        else:
            raise Exception('_init_plasticity_rule.')

    def _init_plasticity_constraint(self):
        """ defines constraint applied to synaptic weights after plasticity update """
        if self.constraint_mode=="x":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_x(dW*A,W_old,mask,A,dt)

        elif self.constraint_mode=="alpha":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_alpha(dW*A,W_old,mask,A,dt)

        elif self.constraint_mode=="xalpha_approx":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_xalpha_approx(dW*A,W_old,\
                                                                                    mask,A,dt)
        elif self.constraint_mode=="xalpha":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_xalpha(dW*A,W_old,mask,\
                                                                            A,c_orth,s_orth,dt)
        elif self.constraint_mode=="postx":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_postx(dW*A,W_old,mask,A,dt)

        elif self.constraint_mode=="prex":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_prex(dW*A,W_old,mask,A,dt)

        elif self.constraint_mode=="postprex_approx":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_postprex_approx(dW*A,W_old,\
                                                                                    mask,A,dt)
        elif self.constraint_mode=="postprex":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_postprex(dW*A,W_old,mask,\
                                                                            A,c_orth,s_orth,dt)
        elif self.constraint_mode=="ffrec_post":
            self.constrain_update = \
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_postx(dW*A,W_old,mask,A,dt)

        elif self.constraint_mode=="ffrec_pre":
            self.constrain_update = \
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_prex(dW*A,W_old,mask,A,dt)

        elif self.constraint_mode=="None":
            self.constrain_update = lambda dW,W_old,mask,A,c_orth,s_orth,dt: dW*dt+W_old

        elif self.constraint_mode=="divisive":
            self.constrain_update =\
             lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_divisive(dW,\
                W_old/self.weight_strength,A,dt)*self.weight_strength

        else:
            raise Exception('constraint_mode.')

    def _init_clip_weights(self):
        if self.clip_mode:
            self.clip_weights = lambda W,A,Wlim: clip_by_norm_arbor(W,A,Wlim)
        else:
            self.clip_weights = lambda W,A,Wlim: W

    def _init_multiplicative_norm(self):
        """ optional multiplicative normalisation to account for loss of weight strength due
        to clipping """
        self.l4_target = None
        if self.multiplicative_normalisation=="x":
            self.mult_normalization =\
             lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,\
                                                                    self.init_weights,\
                                                                    c_orth=None,axis=1)

        elif self.multiplicative_normalisation=="alpha":
            self.mult_normalization =\
             lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,\
                                                                    self.init_weights,\
                                                                    c_orth=None,axis=(0,2))

        elif self.multiplicative_normalisation=="xalpha":
            self.mult_normalization =\
             lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,self.init_weights,\
                                                                    c_orth=self.c_orth,\
                                                                    axis=None)

        elif self.multiplicative_normalisation=="xalpha_approx":
            def xalpha_approx_mult_norm(Wnew,A,H,l4,l4_target):
                Wnew_xalpha = Wnew
                H_xalpha = H
                for i in range(2):
                    Wnew_alpha,H_alpha = synaptic_normalization(Wnew_xalpha,H_xalpha,A,self.Wlim,\
                                                                    self.init_weights[0],\
                                                                    c_orth=None,axis=(0,2))
                    Wnew_xalpha,H_xalpha = synaptic_normalization(Wnew_alpha,H_alpha,A,self.Wlim,\
                                                                    self.init_weights[1],\
                                                                    c_orth=None,axis=1)
                return Wnew_xalpha,H_xalpha
            self.mult_normalization = xalpha_approx_mult_norm

        elif self.multiplicative_normalisation=="postx":
            self.mult_normalization =\
             lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,\
                                                                    self.init_weights,\
                                                                    c_orth=None,axis=0)

        elif self.multiplicative_normalisation=="prex":
            self.mult_normalization =\
             lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,\
                                                                    self.init_weights,\
                                                                    c_orth=None,axis=1)

        elif self.multiplicative_normalisation=="postprex":
            self.mult_normalization =\
             lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,self.init_weights,\
                                                                    c_orth=self.c_orth,\
                                                                    axis=None)

        elif self.multiplicative_normalisation=="ffrec_post":
            self.mult_normalization =\
              lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,\
                                                                    self.init_weights,\
                                                                    c_orth=None,axis=0)

        elif self.multiplicative_normalisation=="ffrec_pre":
            self.mult_normalization =\
              lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,\
                                                                    self.init_weights,\
                                                                    c_orth=None,axis=1)

        elif self.multiplicative_normalisation=="homeostatic":
            self.mult_normalization =\
             lambda Wnew,A,H,l4,l4_target: homeostatic_normalization(Wnew,H,l4,l4_target,\
                                                                    self.Wlim,self.dt)

        elif self.multiplicative_normalisation=="None":
            self.mult_normalization =\
             lambda Wnew,A,H,l4,l4_target: (Wnew,H)

        else:
            raise Exception('multiplicative_normalisation not defined.\
                            Choose either "x", "alpha", "xalpha" "homeostatic", "divisive".')

    def _init_pruning_nonlinearity(self):
        if self.Wthresh is not None:
            self.prune_weights = lambda W,A,Wthresh: prune_by_norm_arbor(W,A,Wthresh)
        else:
            self.prune_weights = lambda W,A,Wthresh: W


def unconstrained_plasticity_wrapper(p_dict, l4, l23, lgn, Wlgn_to_4, W4to4, W4to23, W23to23, t):
    """ apply plasticity update to all selected connections """
    dW_dict = {}

    if p_dict["p_lgn_e"] is not None:
        pop_size = Wlgn_to_4.shape[1]
        l4_e = l4[:pop_size]
        # Wlgn_to_4_e = tf.reshape(Wlgn_to_4[:2,:,:],[-1,Wlgn_to_4.shape[2]])
        # print("Wlgn_to_4_e",t,Wlgn_to_4_e.shape)
        dW = p_dict["p_lgn_e"].unconstrained_update(t,l4_e,tf.reshape(lgn[:2,:],[-1]),None,\
                None,p_dict["p_lgn_e"].beta_P,None)
        # print("Wlgn_to_4",np.sum(Wlgn_to_4[:2,:,:],axis=(0,2)))
        dW = tf.transpose(tf.reshape(dW,[dW.shape[0],2,-1]),perm=[1,0,2])
        dW_dict["dW_lgn_e"] = tf.reshape(dW, [-1])

    if p_dict["p_lgn_i"] is not None:
        pop_size = Wlgn_to_4.shape[1]
        l4_i = l4[pop_size:]
        # Wlgn_to_4_i = tf.reshape(Wlgn_to_4[2:,:,:],[-1,Wlgn_to_4.shape[2]])
        dW = p_dict["p_lgn_i"].unconstrained_update(t,l4_i,tf.reshape(lgn[2:,:],[-1]),None,\
                None,p_dict["p_lgn_i"].beta_P,None)
        dW = tf.transpose(tf.reshape(dW,[dW.shape[0],2,-1]),perm=[1,0,2])
        dW_dict["dW_lgn_i"] = tf.reshape(dW, [-1])

    if p_dict["p_4to23_e"] is not None:
        l4_e = l4[:tf.size(l4)//2]
        l23_e = l23[:tf.size(l23)//2]
        W4to23_e = W4to23[:tf.size(l23)//2,:tf.size(l23)//2]
        dW = p_dict["p_4to23_e"].unconstrained_update(t,l4_e,l23_e,None,W4to23_e,\
             p_dict["p_4to23_e"].beta_P,None)
        dW_dict["dW_4to23_e"] = tf.reshape(dW, [-1])

    if p_dict["p_4to23_i"] is not None:
        l4_i = l4[:tf.size(l4)//2]
        l23_i = l23[:tf.size(l23)//2]
        W4to23_i = W4to23[tf.size(l23)//2:,:tf.size(l23)//2]
        dW = p_dict["p_4to23_i"].unconstrained_update(t,l4_i,l23_i,None,W4to23_i,\
             p_dict["p_4to23_i"].beta_P,None)
        dW_dict["dW_4to23_i"] = tf.reshape(dW, [-1])

    if p_dict["p_rec4_ee"] is not None:
        l4_e = l4[:tf.size(l4)//2]
        l4_i = l4[tf.size(l4)//2:]
        W4to4_ee = W4to4[:tf.size(l4)//2,:tf.size(l4)//2]
        dW = p_dict["p_rec4_ee"].unconstrained_update(t,l4_e,l4_e,None,W4to4_ee,\
             p_dict["p_rec4_ee"].beta_P,None)
        print("dW p_rec4_ee",np.nanmax(dW),np.nanmin(dW))
        dW_dict["dW_rec4_ee"] = tf.reshape(dW, [-1])

    if p_dict["p_rec4_ie"] is not None:
        l4_i = l4[tf.size(l4)//2:]
        W4to4_ie = W4to4[tf.size(l4)//2:,:tf.size(l4)//2]
        dW = p_dict["p_rec4_ie"].unconstrained_update(t,l4_e,l4_i,None,W4to4_ie,\
             p_dict["p_rec4_ie"].beta_P,None)
        dW_dict["dW_rec4_ie"] = tf.reshape(dW, [-1])

    if p_dict["p_rec4_ei"] is not None:
        l4_e = l4[:tf.size(l4)//2]
        l4_i = l4[tf.size(l4)//2:]
        W4to4_ei = W4to4[:tf.size(l4)//2,tf.size(l4)//2:]
        dW = p_dict["p_rec4_ei"].unconstrained_update(t,l4_i,l4_e,None,W4to4_ei,\
             p_dict["p_rec4_ei"].beta_P,None)
        print("dW p_rec4_ei",np.nanmax(dW),np.nanmin(dW))
        dW_dict["dW_rec4_ei"] = -tf.reshape(dW, [-1])

    if p_dict["p_rec4_ii"] is not None:
        l4_i = l4[tf.size(l4)//2:]
        W4to4_ii = W4to4[tf.size(l4)//2:,tf.size(l4)//2:]
        dW = p_dict["p_rec4_ii"].unconstrained_update(t,l4_i,l4_i,None,W4to4_ii,\
             p_dict["p_rec4_ii"].beta_P,None)
        dW_dict["dW_rec4_ii"] = -tf.reshape(dW, [-1])

    if p_dict["p_rec23_ei"] is not None:
        l23_e = l23[:tf.size(l23)//2]
        l23_i = l23[tf.size(l23)//2:]
        W23to23_ei = W23to23[:tf.size(l23)//2,tf.size(l23)//2:]
        # print("W23to23_ei",t,np.sum(W23to23_ei,axis=1),W23to23_ei.shape)
        dW = p_dict["p_rec23_ei"].unconstrained_update(t,l23_i,l23_e,None,W23to23_ei,\
             p_dict["p_rec23_ei"].beta_P,None)
        dW_dict["dW_rec23_ei"] = -tf.reshape(dW, [-1])

    if p_dict["p_rec23_ii"] is not None:
        l23_i = l23[tf.size(l23)//2:]
        W23to23_ii = W23to23[tf.size(l23)//2:,tf.size(l23)//2:]
        dW = p_dict["p_rec23_ii"].unconstrained_update(t,l23_i,l23_i,None,W23to23_ii,\
             p_dict["p_rec23_ii"].beta_P,None)
        dW_dict["dW_rec23_ii"] = -tf.reshape(dW, [-1])

    # ========= Q_dict options ==============
    # this is for all constraints equal to ffpostx
    if p_dict["p_on_l4"] is not None:
        l4_e = l4[:tf.size(l4)//2]
        l4_i = l4[tf.size(l4)//2:]
        dW = tf.concat([ p_dict["p_on_l4"].unconstrained_update(t,l4_e,lgn[0,:],None,\
                None,p_dict["p_on_l4"].beta_P,None)   ,\
                    p_dict["p_on_l4"].unconstrained_update(t,l4_i,lgn[2,:],None,\
                None,p_dict["p_on_l4"].beta_P,None)], 0)
        dW_dict["dW_on_l4"] = tf.reshape(dW, [-1])

    if p_dict["p_off_l4"] is not None:
        l4_e = l4[:tf.size(l4)//2]
        l4_i = l4[tf.size(l4)//2:]
        dW = tf.concat([ p_dict["p_off_l4"].unconstrained_update(t,l4_e,lgn[1,:],None,\
                None,p_dict["p_off_l4"].beta_P,None)   ,\
                    p_dict["p_off_l4"].unconstrained_update(t,l4_i,lgn[3,:],None,\
                None,p_dict["p_off_l4"].beta_P,None)], 0)
        dW_dict["dW_off_l4"] = tf.reshape(dW, [-1])

    if p_dict["p_e_l4"] is not None:
        pop_size = Wlgn_to_4.shape[1]
        l4_e = l4[:pop_size]
        dW = p_dict["p_e_l4"].unconstrained_update(t,l4,l4_e,None,\
                None,p_dict["p_e_l4"].beta_P,None)
        dW_dict["dW_e_l4"] = tf.reshape(dW, [-1])

    if p_dict["p_i_l4"] is not None:
        pop_size = Wlgn_to_4.shape[1]
        l4_i = l4[pop_size:]
        dW = p_dict["p_i_l4"].unconstrained_update(t,l4,l4_i,None,\
                None,p_dict["p_i_l4"].beta_P,None)
        dW_dict["dW_i_l4"] = -tf.reshape(dW, [-1]) #negative because of it's inhibitory activity
    # =======================================

    # ========== Q_dict cases for pre synpatic normalisation ================
    #if p_dict["p_lgne_e"] is not None:

    #if p_dict["p_lgne_i"] is not None:

    #if p_dict["p_i_e"] is not None:

    #if p_dict["p_i_i"] is not None:
    # =======================================================================

    if p_dict["p_ffrec"] is not None:
        return unconstrained_plasticity_wrapper(p_dict["p_ffrec"][0],l4,l23,lgn,Wlgn_to_4,W4to4,W4to23,W23to23,t)
    
    if p_dict["p_ffrec_sep"] is not None:
        return unconstrained_plasticity_wrapper(p_dict["p_ffrec_sep"][0],l4,l23,lgn,Wlgn_to_4,W4to4,W4to23,W23to23,t)

    return dW_dict

def constraint_update_wrapper(dW_dict,p_dict,Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
    W4to23,arbor4to23,W23to23,arbor23to23,dt,params_dict):

    if p_dict["p_lgn_e"] is not None:
        Wlgn_to_4_e = Wlgn_to_4[:2,:,:]
        dW = tf.reshape(dW_dict["dW_lgn_e"],arbor_lgn[:2,:,:].shape)
        if p_dict["p_lgn_e"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgn_to_4_e>0, Wlgn_to_4_e<(p_dict["p_lgn_e"].Wlim*arbor_lgn[:2,:,:]))
        else:
            notfrozen = tf.ones_like(Wlgn_to_4_e,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn[:2,:,:]>0 )
        mask_fl = tf.cast(mask, tf.float32)
        W_new = p_dict["p_lgn_e"].constrain_update(dW,Wlgn_to_4[:2,:,:],mask_fl,\
                    arbor_lgn[:2,:,:],p_dict["p_lgn_e"].c_orth,p_dict["p_lgn_e"].s_orth,dt)
        # print("Wlgn_to_4 before",np.sum(Wlgn_to_4[2:,:,:],axis=(0,2)),Wlgn_to_4.shape)
        Wlgn_to_4 = tf.concat([W_new,Wlgn_to_4[2:,:,:]],0)
        # print("Wlgn_to_4 after",np.sum(Wlgn_to_4[2:,:,:],axis=(0,2)),Wlgn_to_4.shape)

    if p_dict["p_lgn_i"] is not None:
        Wlgn_to_4_i = Wlgn_to_4[2:,:,:]
        dW = tf.reshape(dW_dict["dW_lgn_i"],arbor_lgn[2:,:,:].shape)
        if p_dict["p_lgn_i"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgn_to_4_i>0, Wlgn_to_4_i<(p_dict["p_lgn_i"].Wlim*arbor_lgn[2:,:,:]))
        else:
            notfrozen = tf.ones_like(Wlgn_to_4_e,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn[2:,:,:]>0 )
        mask_fl = tf.cast(mask, tf.float32)
        W_new = p_dict["p_lgn_i"].constrain_update(dW,Wlgn_to_4[2:,:,:],\
            mask_fl,arbor_lgn[2:,:,:],p_dict["p_lgn_i"].c_orth,p_dict["p_lgn_i"].s_orth,dt)
        Wlgn_to_4 = tf.concat([Wlgn_to_4[:2,:,:],W_new],0)

    if p_dict["p_4to23_e"] is not None:
        N = W4to23.shape[0]//2
        W4to23_e = W4to23[:N,:N]
        if arbor4to23 is None:
            A = 1
        else:
            A = arbor4to23[:N,:N]
        dW = tf.reshape(dW_dict["dW_4to23_e"],W4to23_e.shape)
        W_new = p_dict["p_4to23_e"].constrain_update(dW,W4to23_e,None,A,\
            p_dict["p_4to23_e"].c_orth,p_dict["p_4to23_e"].s_orth,dt)
        W4to23 = tf.concat([tf.concat([W_new,W4to23[N:,:N]],0),W4to23[:,N:]],1)
        params_dict["W4to23"] = W4to23

    if p_dict["p_4to23_i"] is not None:
        N = W4to23.shape[0]//2
        W4to23_i = W4to23[N:,:N]
        print("arbor4to23",arbor4to23.shape,W4to23_i.shape)
        if arbor4to23 is None:
            A = 1
        else:
            A = arbor4to23[N:,:N]
        dW = tf.reshape(dW_dict["dW_4to23_i"],W4to23_i.shape)
        print("dW",dW.shape,A.shape,N)
        W_new = p_dict["p_4to23_i"].constrain_update(dW,W4to23_i,None,A,\
                p_dict["p_4to23_i"].c_orth,p_dict["p_4to23_i"].s_orth,dt)
        W4to23 = tf.concat([tf.concat([W4to23[:N,:N],W_new],0),W4to23[:,N:]],1)
        params_dict["W4to23"] = W4to23

    if p_dict["p_rec4_ee"] is not None:
        Nl4 = W4to4.shape[0]//2
        W4to4_ee = W4to4[:Nl4,:Nl4]
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[:Nl4,:Nl4]
        dW = tf.reshape(dW_dict["dW_rec4_ee"],W4to4_ee.shape)
        if p_dict["p_rec4_ee"].freeze_weights:
            notfrozen = tf.math.logical_and(W4to4_ee>0, W4to4_ee<(p_dict["p_rec4_ee"].Wlim*A))
        else:
            notfrozen = tf.ones_like(W4to4_ee,dtype=bool)
        mask = tf.math.logical_and( notfrozen, A>0 )
        mask_fl = tf.cast(mask, tf.float32)
        W_new = p_dict["p_rec4_ee"].constrain_update(dW,W4to4_ee,mask_fl,A,\
                p_dict["p_rec4_ee"].c_orth,p_dict["p_rec4_ee"].s_orth,dt)
        W4to4 = tf.concat([tf.concat([W_new,W4to4[Nl4:,:Nl4]],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_rec4_ie"] is not None:
        Nl4 = W4to4.shape[0]//2
        W4to4_ie = W4to4[Nl4:,:Nl4]
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[Nl4:,:Nl4]
        dW = tf.reshape(dW_dict["dW_rec4_ie"],W4to4_ie.shape)
        if p_dict["p_rec4_ie"].freeze_weights:
            notfrozen = tf.math.logical_and(W4to4_ie>0, W4to4_ie<(p_dict["p_rec4_ie"].Wlim*A))
        else:
            notfrozen = tf.ones_like(W4to4_ie,dtype=bool)
        mask = tf.math.logical_and( notfrozen, A>0 )
        mask_fl = tf.cast(mask, tf.float32)
        W_new = p_dict["p_rec4_ie"].constrain_update(dW,W4to4_ie,None,A,\
                p_dict["p_rec4_ie"].c_orth,p_dict["p_rec4_ie"].s_orth,dt)
        W4to4 = tf.concat([tf.concat([W4to4[:Nl4,:Nl4],W_new],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_rec4_ei"] is not None:
        Nl4 = W4to4.shape[0]//2
        W4to4_ei = W4to4[:Nl4,Nl4:]
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[:Nl4,Nl4:]
        dW = tf.reshape(dW_dict["dW_rec4_ei"],W4to4_ei.shape)
        if p_dict["p_rec4_ei"].freeze_weights:
            notfrozen = tf.math.logical_and(W4to4_ei>0, W4to4_ei<(p_dict["p_rec4_ei"].Wlim*A))
        else:
            notfrozen = tf.ones_like(W4to4_ei,dtype=bool)
        mask = tf.math.logical_and( notfrozen, A>0 )
        mask_fl = tf.cast(mask, tf.float32)
        W_new = p_dict["p_rec4_ei"].constrain_update(dW,W4to4_ei,None,A,\
                p_dict["p_rec4_ei"].c_orth,p_dict["p_rec4_ei"].s_orth,dt)
        W4to4 = tf.concat([W4to4[:,:Nl4],tf.concat([W_new,W4to4[Nl4:,Nl4:]],0)],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_rec4_ii"] is not None:
        Nl4 = W4to4.shape[0]//2
        W4to4_ii = W4to4[Nl4:,Nl4:]
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[Nl4:,Nl4:]
        dW = tf.reshape(dW_dict["dW_rec4_ii"],W4to4_ii.shape)
        if p_dict["p_rec4_ii"].freeze_weights:
            notfrozen = tf.math.logical_and(W4to4_ii>0, W4to4_ii<(p_dict["p_rec4_ii"].Wlim*A))
        else:
            notfrozen = tf.ones_like(W4to4_ii,dtype=bool)
        mask = tf.math.logical_and( notfrozen, A>0 )
        mask_fl = tf.cast(mask, tf.float32)
        W_new = p_dict["p_rec4_ii"].constrain_update(dW,W4to4_ii,None,A,\
                p_dict["p_rec4_ii"].c_orth,p_dict["p_rec4_ii"].s_orth,dt)
        W4to4 = tf.concat([W4to4[:,:Nl4],tf.concat([W4to4[:Nl4,Nl4:],W_new],0)],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_rec23_ei"] is not None:
        N = W23to23.shape[0]//2
        W23to23_ei = W23to23[:N,N:]
        if arbor23to23 is None:
            A = 1
        else:
            A = arbor23to23[:N,N:]
        dW = tf.reshape(dW_dict["dW_rec23_ei"],W23to23_ei.shape)
        W_new = p_dict["p_rec23_ei"].constrain_update(dW,W23to23_ei,None,A,\
                p_dict["p_rec23_ei"].c_orth,p_dict["p_rec23_ei"].s_orth,dt)
        # print("W23to23 after",np.sum(W23to23[:,Nl4:],axis=1),W23to23.shape)
        W23to23 = tf.concat([W23to23[:,:N],tf.concat([W_new,W23to23[N:,N:]],0)],1)
        # print("W23to23 after",np.sum(W23to23[:,Nl4:],axis=1),W23to23.shape)
        params_dict["W23to23"] = W23to23

    if p_dict["p_rec23_ii"] is not None:
        N = W23to23.shape[0]//2
        W23to23_ii = W23to23[N:,N:]
        if arbor23to23 is None:
            A = 1
        else:
            A = arbor23to23[N:,N:]
        dW = tf.reshape(dW_dict["dW_rec23_ii"],W23to23_ii.shape)
        W_new = p_dict["p_rec23_ii"].constrain_update(dW,W23to23_ii,None,A,\
                p_dict["p_rec23_ii"].c_orth,p_dict["p_rec23_ii"].s_orth,dt)
        W23to23 = tf.concat([W23to23[:,:N],tf.concat([W23to23[:N,N:],W_new],0)],1)
        params_dict["W23to23"] = W23to23

    # ========== Q_dict cases for post synaptic normalisation ===============
    if p_dict["p_on_l4"] is not None:
        Won_l4 = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[2,:,:]],0)
        arbor_lgn_on = tf.concat([arbor_lgn[0,:,:],arbor_lgn[2,:,:]],0)

        dW = tf.reshape(dW_dict["dW_on_l4"],Won_l4.shape)
        if p_dict["p_on_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(Won_l4>0, Won_l4< (p_dict["p_on_l4"].Wlim*arbor_lgn_on))
        else:
            notfrozen = tf.ones_like(Won_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_on>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_on_l4"].constrain_update(dW,Won_l4,mask_fl,arbor_lgn_on,\
                                        p_dict["p_on_l4"].c_orth,p_dict["p_on_l4"].s_orth,dt)
        Wlgn_to_4 = tf.concat([ W_new[None,:W_new.shape[0]//2,:], Wlgn_to_4[1:2,:,:],\
                                        W_new[None,W_new.shape[0]//2:,:], Wlgn_to_4[3:4,:,:] ], 0)

    if p_dict["p_off_l4"] is not None:
        Woff_l4 = tf.concat([Wlgn_to_4[1,:,:],Wlgn_to_4[3,:,:]],0)
        arbor_lgn_off = tf.concat([arbor_lgn[1,:,:],arbor_lgn[3,:,:]],0)

        dW = tf.reshape(dW_dict["dW_off_l4"],Woff_l4.shape)
        if p_dict["p_off_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(Woff_l4>0, Woff_l4< (p_dict["p_off_l4"].Wlim*arbor_lgn_off))
        else:
            notfrozen = tf.ones_like(Woff_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_off>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_off_l4"].constrain_update(dW,Woff_l4,mask_fl,arbor_lgn_off,\
                                        p_dict["p_off_l4"].c_orth,p_dict["p_off_l4"].s_orth,dt)
        Wlgn_to_4 = tf.concat([ Wlgn_to_4[0:1,:,:], W_new[None,:W_new.shape[0]//2,:],\
                                        Wlgn_to_4[2:3,:,:], W_new[None,W_new.shape[0]//2:,:] ], 0)

    if p_dict["p_e_l4"] is not None:
        Nl4 = W4to4.shape[0]//2
        We_to_l4 = W4to4[:,:Nl4]
        arbor_e_to_l4 = arbor4to4[:,:Nl4]
        dW = tf.reshape(dW_dict["dW_e_l4"],We_to_l4.shape)
        if p_dict["p_e_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(We_to_l4>0, We_to_l4<(p_dict["p_e_l4"].Wlim*arbor_e_to_l4))
        else:
            notfrozen = tf.ones_like(We_to_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_e_to_l4>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_e_l4"].constrain_update(dW,We_to_l4,mask_fl,arbor_e_to_l4,\
                p_dict["p_e_l4"].c_orth,p_dict["p_e_l4"].s_orth,dt)
        W4to4 = tf.concat([W_new,W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_i_l4"] is not None:
        Nl4 = W4to4.shape[0]//2
        Wi_to_l4 = W4to4[:,Nl4:]
        arbor_i_to_l4 = arbor4to4[:,Nl4:]
        dW = tf.reshape(dW_dict["dW_i_l4"],Wi_to_l4.shape)
        if p_dict["p_i_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(Wi_to_l4)>0, tf.abs(Wi_to_l4)<(p_dict["p_i_l4"].Wlim*arbor_i_to_l4))
        else:
            notfrozen = tf.ones_like(Wi_to_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_i_to_l4>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_i_l4"].constrain_update(dW,Wi_to_l4,mask_fl,arbor_i_to_l4,\
                p_dict["p_i_l4"].c_orth,p_dict["p_i_l4"].s_orth,dt)
        W4to4 = tf.concat([W4to4[:,:Nl4],W_new],1)
        params_dict["W4to4"] = W4to4
    # =======================================================================

    # ========== Q_dict cases for prepost synpatic normalisation (including for _sep)================
    if p_dict["p_lgne_e"] is not None:
        #assume dWpre["dW_lgne_e"]
        #(constrain_update_prex = constrain_update)
        Nl4 = W4to4.shape[0]//2
        Nlgn = Wlgn_to_4.shape[2]
        Wlgne_to_e = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[1,:,:],W4to4[:Nl4,:Nl4]],1)
        arbor_lgne_to_e = tf.concat([arbor_lgn[0,:,:],arbor_lgn[1,:,:],arbor4to4[:Nl4,:Nl4]],1)
        dW = tf.reshape(dW_dict["dW_lgne_e"],Wlgne_to_e.shape)
        if p_dict["p_lgne_e"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgne_to_e>0, Wlgne_to_e< (p_dict["p_lgne_e"].Wlim*arbor_lgne_to_e))
        else:
            notfrozen = tf.ones_like(Wlgne_to_e,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgne_to_e>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_lgne_e"].constrain_update(dW,Wlgne_to_e,mask_fl,arbor_lgne_to_e,\
                                        p_dict["p_lgne_e"].c_orth,p_dict["p_lgne_e"].s_orth,dt)

        Wlgn_to_4 = tf.concat( [W_new[None,:,:Nlgn], W_new[None, :,Nlgn:2*Nlgn], \
                                            Wlgn_to_4[2:,:,:]], 0)
        W4to4 = tf.concat([tf.concat([W_new[:,2*Nlgn:], W4to4[Nl4:,:Nl4]],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_lgne_i"] is not None:

        Nl4 = W4to4.shape[0]//2
        Nlgn = Wlgn_to_4.shape[2]
        Wlgne_to_i = tf.concat([Wlgn_to_4[2,:,:],Wlgn_to_4[3,:,:],W4to4[Nl4:,:Nl4]],1)
        arbor_lgne_to_i = tf.concat([arbor_lgn[2,:,:],arbor_lgn[3,:,:],arbor4to4[Nl4:,:Nl4]],1)
        dW = tf.reshape(dW_dict["dW_lgne_i"],Wlgne_to_i.shape)
        if p_dict["p_lgne_i"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgne_to_i>0, Wlgne_to_i< (p_dict["p_lgne_i"].Wlim*arbor_lgne_to_i))
        else:
            notfrozen = tf.ones_like(Wlgne_to_i,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgne_to_i>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_lgne_i"].constrain_update(dW,Wlgne_to_i,mask_fl,arbor_lgne_to_i,\
                                        p_dict["p_lgne_i"].c_orth,p_dict["p_lgne_i"].s_orth,dt)

        Wlgn_to_4 = tf.concat( [ Wlgn_to_4[:2,:,:], \
                                        W_new[None,:,:Nlgn], W_new[None,:,Nlgn:2*Nlgn] ], 0)
        W4to4 = tf.concat([tf.concat([W4to4[:Nl4,:Nl4], W_new[:,2*Nlgn:]],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_lgn_e_sep"] is not None:
        Nlgn = Wlgn_to_4.shape[2]
        Wlgn_to_e_sep = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[1,:,:]],1)
        arbor_lgn_to_e_sep = tf.concat([arbor_lgn[0,:,:],arbor_lgn[1,:,:]],1)
        dW = tf.reshape(dW_dict["dW_lgn_e_sep"],Wlgn_to_e_sep.shape)
        if p_dict["p_lgne_e_sep"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgn_to_e_sep>0, Wlgn_to_e_sep< (p_dict["p_lgne_e_sep"].Wlim*arbor_lgn_to_e_sep))
        else:
            notfrozen = tf.ones_like(Wlgn_to_e_sep,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_to_e_sep>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_lgne_e_sep"].constrain_update(dW,Wlgn_to_e_sep,mask_fl,arbor_lgn_to_e_sep,\
                                        p_dict["p_lgne_e_sep"].c_orth,p_dict["p_lgne_e_sep"].s_orth,dt)

        Wlgn_to_4 = tf.concat( [W_new[None,:,:Nlgn], W_new[None, :,Nlgn:2*Nlgn], \
                                            Wlgn_to_4[2:,:,:]], 0)

    if p_dict["p_lgn_i_sep"] is not None:
        Nlgn = Wlgn_to_4.shape[2]
        Wlgn_to_i_sep = tf.concat([Wlgn_to_4[2,:,:],Wlgn_to_4[3,:,:]],1)
        arbor_lgn_to_i_sep = tf.concat([arbor_lgn[2,:,:],arbor_lgn[3,:,:]],1)
        dW = tf.reshape(dW_dict["dW_lgn_i_sep"],Wlgn_to_i_sep.shape)
        if p_dict["p_lgne_i"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgn_to_i_sep>0, Wlgn_to_i_sep< (p_dict["p_lgn_i_sep"].Wlim*arbor_lgn_to_i_sep))
        else:
            notfrozen = tf.ones_like(Wlgn_to_i_sep,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_to_i_sep>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_lgn_i_sep"].constrain_update(dW,Wlgn_to_i_sep,mask_fl,arbor_lgn_to_i_sep,\
                                        p_dict["p_lgn_i_sep"].c_orth,p_dict["p_lgn_i_sep"].s_orth,dt)

        Wlgn_to_4 = tf.concat( [ Wlgn_to_4[:2,:,:], \
                                        W_new[None,:,:Nlgn], W_new[None,:,Nlgn:2*Nlgn] ], 0)

    if p_dict["p_e_e"] is not None:

        Nl4 = W4to4.shape[0]//2
        WE_to_E = W4to4[:Nl4,:Nl4]
        arbor_E_to_E = arbor4to4[:Nl4,:Nl4]
        dW = tf.reshape(dW_dict["dW_e_e"],WE_to_E.shape)
        if p_dict["p_e_e"].freeze_weights:
            notfrozen = tf.math.logical_and(WE_to_E>0, WE_to_E< (p_dict["p_e_e"].Wlim*arbor_E_to_E))
        else:
            notfrozen = tf.ones_like(WE_to_E,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_E_to_E>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_e_e"].constrain_update(dW,WE_to_E,mask_fl,arbor_E_to_E,\
                                        p_dict["p_e_e"].c_orth,p_dict["p_e_e"].s_orth,dt)
        W4to4 = tf.concat([tf.concat([W_new, W4to4[:Nl4,Nl4:]],1),W4to4[Nl4:,:]],0)
        params_dict["W4to4"] = W4to4

    if p_dict["p_i_e"] is not None:

        Nl4 = W4to4.shape[0]//2
        WI_to_E = W4to4[:Nl4,Nl4:]
        arbor_I_to_E = arbor4to4[:Nl4,Nl4:]
        dW = tf.reshape(dW_dict["dW_i_e"],WI_to_E.shape)
        if p_dict["p_i_e"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(WI_to_E)>0, tf.abs(WI_to_E)< (p_dict["p_i_e"].Wlim*arbor_I_to_E))
        else:
            notfrozen = tf.ones_like(WI_to_E,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_I_to_E>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_i_e"].constrain_update(dW,WI_to_E,mask_fl,arbor_I_to_E,\
                                        p_dict["p_i_e"].c_orth,p_dict["p_i_e"].s_orth,dt)
        W4to4 = tf.concat([tf.concat([W4to4[:Nl4,:Nl4], W_new],1),W4to4[Nl4:,:]],0)
        params_dict["W4to4"] = W4to4

    if p_dict["p_e_i"] is not None:

        Nl4 = W4to4.shape[0]//2
        WE_to_I = W4to4[Nl4:,:Nl4]
        arbor_E_to_I = arbor4to4[Nl4:,:Nl4]
        dW = tf.reshape(dW_dict["dW_e_i"],WE_to_I.shape)
        if p_dict["p_e_i"].freeze_weights:
            notfrozen = tf.math.logical_and(WE_to_I>0, WE_to_I< (p_dict["p_e_i"].Wlim*arbor_E_to_I))
        else:
            notfrozen = tf.ones_like(WE_to_I,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_E_to_I>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_e_i"].constrain_update(dW,WE_to_I,mask_fl,arbor_E_to_I,\
                                        p_dict["p_e_i"].c_orth,p_dict["p_e_i"].s_orth,dt)
        W4to4 = tf.concat([W4to4[:Nl4,:],tf.concat([W_new,W4to4[Nl4:,Nl4:]],1)],0)
        params_dict["W4to4"] = W4to4

    if p_dict["p_i_i"] is not None:

        Nl4 = W4to4.shape[0]//2
        WI_to_I = W4to4[Nl4:,Nl4:]
        arbor_I_to_I = arbor4to4[Nl4:,Nl4:]
        dW = tf.reshape(dW_dict["dW_i_i"],WI_to_I.shape)
        if p_dict["p_i_i"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(WI_to_I)>0, tf.abs(WI_to_I)< (p_dict["p_i_i"].Wlim*arbor_I_to_I))
        else:
            notfrozen = tf.ones_like(WI_to_I,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_I_to_I>0 )
        mask_fl = tf.cast(mask,tf.float32)
        W_new = p_dict["p_i_i"].constrain_update(dW,WI_to_I,mask_fl,arbor_I_to_I,\
                                        p_dict["p_i_i"].c_orth,p_dict["p_i_i"].s_orth,dt)
        W4to4 = tf.concat([W4to4[:Nl4,:],tf.concat([W4to4[Nl4:,:Nl4],W_new],1)],0)
        params_dict["W4to4"] = W4to4

    # =======================================================================

    if p_dict["p_ffrec"] is not None:
        Nlgn = Wlgn_to_4.shape[2]
        Nl4 = Wlgn_to_4.shape[1]

        p_post_dict = p_dict["p_ffrec"][0]
        p_pre_dict = p_dict["p_ffrec"][1]

        # extract post W and arbors
        Won_l4 = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[2,:,:]],0)
        Woff_l4 = tf.concat([Wlgn_to_4[1,:,:],Wlgn_to_4[3,:,:]],0)
        We_to_l4 = W4to4[:,:Nl4]
        Wi_to_l4 = W4to4[:,Nl4:]
        arbor_lgn_on = tf.concat([arbor_lgn[0,:,:],arbor_lgn[2,:,:]],0)
        arbor_lgn_off = tf.concat([arbor_lgn[1,:,:],arbor_lgn[3,:,:]],0)
        arbor_e_to_l4 = arbor4to4[:,:Nl4]
        arbor_i_to_l4 = arbor4to4[:,Nl4:]

        # calculate post inverse arbors
        invarb_lgn_on = tf.where(tf.equal(arbor_lgn_on, 0),
                            tf.ones(tf.shape(arbor_lgn_on),dtype=tf.float32), tf.math.reciprocal(arbor_lgn_on))
        invarb_lgn_off = tf.where(tf.equal(arbor_lgn_off, 0),
                            tf.ones(tf.shape(arbor_lgn_off),dtype=tf.float32), tf.math.reciprocal(arbor_lgn_off))
        invarb_e_to_l4 = tf.where(tf.equal(arbor_e_to_l4, 0),
                            tf.ones(tf.shape(arbor_e_to_l4),dtype=tf.float32), tf.math.reciprocal(arbor_e_to_l4))
        invarb_i_to_l4 = tf.where(tf.equal(arbor_i_to_l4, 0),
                            tf.ones(tf.shape(arbor_i_to_l4),dtype=tf.float32), tf.math.reciprocal(arbor_i_to_l4))

        # calculate post masks
        if p_post_dict["p_on_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(Won_l4>0, Won_l4< (p_post_dict["p_on_l4"].Wlim*arbor_lgn_on))
        else:
            notfrozen = tf.ones_like(Won_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_on>0 )
        mask_fl_on_l4 = tf.cast(mask,tf.float32)

        if p_post_dict["p_off_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(Woff_l4>0, Woff_l4< (p_post_dict["p_off_l4"].Wlim*arbor_lgn_off))
        else:
            notfrozen = tf.ones_like(Woff_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_off>0 )
        mask_fl_off_l4 = tf.cast(mask,tf.float32)

        if p_post_dict["p_e_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(We_to_l4>0, We_to_l4<(p_post_dict["p_e_l4"].Wlim*arbor_e_to_l4))
        else:
            notfrozen = tf.ones_like(We_to_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_e_to_l4>0 )
        mask_fl_e_l4 = tf.cast(mask,tf.float32)

        if p_post_dict["p_i_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(Wi_to_l4)>0,
                                            tf.abs(Wi_to_l4)<(p_post_dict["p_i_l4"].Wlim*arbor_i_to_l4))
        else:
            notfrozen = tf.ones_like(Wi_to_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_i_to_l4>0 )
        mask_fl_i_l4 = tf.cast(mask,tf.float32)

        # extract pre W and arbors
        Wlgne_to_e = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[1,:,:],W4to4[:Nl4,:Nl4]],1)
        Wlgne_to_i = tf.concat([Wlgn_to_4[2,:,:],Wlgn_to_4[3,:,:],W4to4[Nl4:,:Nl4]],1)
        WI_to_E = W4to4[:Nl4,Nl4:]
        WI_to_I = W4to4[Nl4:,Nl4:]
        arbor_lgne_to_e = tf.concat([arbor_lgn[0,:,:],arbor_lgn[1,:,:],arbor4to4[:Nl4,:Nl4]],1)
        arbor_lgne_to_i = tf.concat([arbor_lgn[2,:,:],arbor_lgn[3,:,:],arbor4to4[Nl4:,:Nl4]],1)
        arbor_I_to_E = arbor4to4[:Nl4,Nl4:]
        arbor_I_to_I = arbor4to4[Nl4:,Nl4:]

        # calculate pre inverse arbors
        invarb_lgne_to_e = tf.where(tf.equal(arbor_lgne_to_e, 0),
                            tf.ones(tf.shape(arbor_lgne_to_e),dtype=tf.float32), tf.math.reciprocal(arbor_lgne_to_e))
        invarb_lgne_to_i = tf.where(tf.equal(arbor_lgne_to_i, 0),
                            tf.ones(tf.shape(arbor_lgne_to_i),dtype=tf.float32), tf.math.reciprocal(arbor_lgne_to_i))
        invarb_I_to_E = tf.where(tf.equal(arbor_I_to_E, 0),
                            tf.ones(tf.shape(arbor_I_to_E),dtype=tf.float32), tf.math.reciprocal(arbor_I_to_E))
        invarb_I_to_I = tf.where(tf.equal(arbor_I_to_I, 0),
                            tf.ones(tf.shape(arbor_I_to_I),dtype=tf.float32), tf.math.reciprocal(arbor_I_to_I))

        # calculate pre masks
        if p_pre_dict["p_lgne_e"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgne_to_e>0, Wlgne_to_e< (p_pre_dict["p_lgne_e"].Wlim*arbor_lgne_to_e))
        else:
            notfrozen = tf.ones_like(Wlgne_to_e,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgne_to_e>0 )
        mask_fl_lgne_e = tf.cast(mask,tf.float32)

        if p_pre_dict["p_lgne_i"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgne_to_i>0, Wlgne_to_i< (p_pre_dict["p_lgne_i"].Wlim*arbor_lgne_to_i))
        else:
            notfrozen = tf.ones_like(Wlgne_to_i,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgne_to_i>0 )
        mask_fl_lgne_i = tf.cast(mask,tf.float32)

        if p_pre_dict["p_i_e"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(WI_to_E)>0, tf.abs(WI_to_E)< (p_pre_dict["p_i_e"].Wlim*arbor_I_to_E))
        else:
            notfrozen = tf.ones_like(WI_to_E,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_I_to_E>0 )
        mask_fl_i_e = tf.cast(mask,tf.float32)

        if p_pre_dict["p_i_i"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(WI_to_I)>0, tf.abs(WI_to_I)< (p_pre_dict["p_i_i"].Wlim*arbor_I_to_I))
        else:
            notfrozen = tf.ones_like(WI_to_I,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_I_to_I>0 )
        mask_fl_i_i = tf.cast(mask,tf.float32)

        # Extract unconstrained Hebbian weight changes
        dW_post_dict = {}
        dW_post_dict['dW_on_l4'] = tf.reshape(dW_dict["dW_on_l4"],(2*Nl4,Nlgn)) * arbor_lgn_on
        dW_post_dict['dW_off_l4'] = tf.reshape(dW_dict["dW_off_l4"],(2*Nl4,Nlgn)) * arbor_lgn_off
        dW_post_dict['dW_e_l4'] = tf.reshape(dW_dict["dW_e_l4"],(2*Nl4,Nl4)) * arbor_e_to_l4
        dW_post_dict['dW_i_l4'] = tf.reshape(dW_dict["dW_i_l4"],(2*Nl4,Nl4)) * arbor_i_to_l4

        for i in range(8):
            # pre
            dW_pre_dict = convert_dW_to_pre(dW_post_dict,Nlgn,Nl4)
            dW_pre_dict['dW_lgne_e'] = p_pre_dict["p_lgne_e"].constrain_update(
                                            dW_pre_dict['dW_lgne_e']*invarb_lgne_to_e,\
                                            tf.zeros_like(Wlgne_to_e),mask_fl_lgne_e,arbor_lgne_to_e,\
                                            p_pre_dict["p_lgne_e"].c_orth,p_pre_dict["p_lgne_e"].s_orth,1)
            dW_pre_dict['dW_lgne_i'] = p_pre_dict["p_lgne_i"].constrain_update(
                                            dW_pre_dict['dW_lgne_i']*invarb_lgne_to_i,\
                                            tf.zeros_like(Wlgne_to_i),mask_fl_lgne_i,arbor_lgne_to_i,\
                                            p_pre_dict["p_lgne_i"].c_orth,p_pre_dict["p_lgne_i"].s_orth,1)
            dW_pre_dict['dW_i_e'] = p_pre_dict["p_i_e"].constrain_update(
                                            dW_pre_dict['dW_i_e']*invarb_I_to_E,\
                                            tf.zeros_like(WI_to_E),mask_fl_i_e,arbor_I_to_E,\
                                            p_pre_dict["p_i_e"].c_orth,p_pre_dict["p_i_e"].s_orth,1)
            dW_pre_dict['dW_i_i'] = p_pre_dict["p_i_i"].constrain_update(
                                            dW_pre_dict['dW_i_i']*invarb_I_to_I,\
                                            tf.zeros_like(WI_to_I),mask_fl_i_i,arbor_I_to_I,\
                                            p_pre_dict["p_i_i"].c_orth,p_pre_dict["p_i_i"].s_orth,1)

            # post
            dW_post_dict = convert_dW_to_post(dW_pre_dict,Nlgn,Nl4)
            dW_post_dict['dW_on_l4'] =  p_post_dict["p_on_l4"].constrain_update(
                                            dW_post_dict['dW_on_l4']*invarb_lgn_on,\
                                            tf.zeros_like(Won_l4),mask_fl_on_l4,arbor_lgn_on,\
                                            p_post_dict["p_on_l4"].c_orth,p_post_dict["p_on_l4"].s_orth,1)
            dW_post_dict['dW_off_l4'] =  p_post_dict["p_off_l4"].constrain_update(
                                            dW_post_dict['dW_off_l4']*invarb_lgn_off,\
                                            tf.zeros_like(Woff_l4),mask_fl_off_l4,arbor_lgn_off,\
                                            p_post_dict["p_off_l4"].c_orth,p_post_dict["p_off_l4"].s_orth,1)
            dW_post_dict['dW_e_l4'] =  p_post_dict["p_e_l4"].constrain_update(
                                            dW_post_dict['dW_e_l4']*invarb_e_to_l4,\
                                            tf.zeros_like(We_to_l4),mask_fl_e_l4,arbor_e_to_l4,\
                                            p_post_dict["p_e_l4"].c_orth,p_post_dict["p_e_l4"].s_orth,1)
            dW_post_dict['dW_i_l4'] =  p_post_dict["p_i_l4"].constrain_update(
                                            dW_post_dict['dW_i_l4']*invarb_i_to_l4,\
                                            tf.zeros_like(Wi_to_l4),mask_fl_i_l4,arbor_i_to_l4,\
                                            p_post_dict["p_i_l4"].c_orth,p_post_dict["p_i_l4"].s_orth,1)

        # update weights
        Wlgn_to_4 = Wlgn_to_4 + dt*tf.concat([dW_post_dict['dW_on_l4'][None,:Nl4,:],
            dW_post_dict['dW_off_l4'][None,:Nl4,:],dW_post_dict['dW_on_l4'][None,Nl4:,:],
            dW_post_dict['dW_off_l4'][None,Nl4:,:]], 0)
        W4to4 = W4to4 + dt*tf.concat([dW_post_dict['dW_e_l4'],dW_post_dict['dW_i_l4']], 1)
        params_dict["W4to4"] = W4to4
    
    if p_dict["p_ffrec_sep"] is not None:
        Nlgn = Wlgn_to_4.shape[2]
        Nl4 = Wlgn_to_4.shape[1]

        p_post_dict = p_dict["p_ffrec_sep"][0]
        p_pre_dict = p_dict["p_ffrec_sep"][1]

        # extract post W and arbors
        Won_l4 = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[2,:,:]],0)
        Woff_l4 = tf.concat([Wlgn_to_4[1,:,:],Wlgn_to_4[3,:,:]],0)
        We_to_l4 = W4to4[:,:Nl4]
        Wi_to_l4 = W4to4[:,Nl4:]
        arbor_lgn_on = tf.concat([arbor_lgn[0,:,:],arbor_lgn[2,:,:]],0)
        arbor_lgn_off = tf.concat([arbor_lgn[1,:,:],arbor_lgn[3,:,:]],0)
        arbor_e_to_l4 = arbor4to4[:,:Nl4]
        arbor_i_to_l4 = arbor4to4[:,Nl4:]

        # calculate post inverse arbors
        invarb_lgn_on = tf.where(tf.equal(arbor_lgn_on, 0),
                            tf.ones(tf.shape(arbor_lgn_on),dtype=tf.float32), tf.math.reciprocal(arbor_lgn_on))
        invarb_lgn_off = tf.where(tf.equal(arbor_lgn_off, 0),
                            tf.ones(tf.shape(arbor_lgn_off),dtype=tf.float32), tf.math.reciprocal(arbor_lgn_off))
        invarb_e_to_l4 = tf.where(tf.equal(arbor_e_to_l4, 0),
                            tf.ones(tf.shape(arbor_e_to_l4),dtype=tf.float32), tf.math.reciprocal(arbor_e_to_l4))
        invarb_i_to_l4 = tf.where(tf.equal(arbor_i_to_l4, 0),
                            tf.ones(tf.shape(arbor_i_to_l4),dtype=tf.float32), tf.math.reciprocal(arbor_i_to_l4))

        # calculate post masks
        if p_post_dict["p_on_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(Won_l4>0, Won_l4< (p_post_dict["p_on_l4"].Wlim*arbor_lgn_on))
        else:
            notfrozen = tf.ones_like(Won_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_on>0 )
        mask_fl_on_l4 = tf.cast(mask,tf.float32)

        if p_post_dict["p_off_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(Woff_l4>0, Woff_l4< (p_post_dict["p_off_l4"].Wlim*arbor_lgn_off))
        else:
            notfrozen = tf.ones_like(Woff_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_off>0 )
        mask_fl_off_l4 = tf.cast(mask,tf.float32)

        if p_post_dict["p_e_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(We_to_l4>0, We_to_l4<(p_post_dict["p_e_l4"].Wlim*arbor_e_to_l4))
        else:
            notfrozen = tf.ones_like(We_to_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_e_to_l4>0 )
        mask_fl_e_l4 = tf.cast(mask,tf.float32)

        if p_post_dict["p_i_l4"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(Wi_to_l4)>0,
                                            tf.abs(Wi_to_l4)<(p_post_dict["p_i_l4"].Wlim*arbor_i_to_l4))
        else:
            notfrozen = tf.ones_like(Wi_to_l4,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_i_to_l4>0 )
        mask_fl_i_l4 = tf.cast(mask,tf.float32)

        # extract pre W and arbors
        Wlgn_to_e = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[1,:,:]],1)
        Wlgn_to_i = tf.concat([Wlgn_to_4[2,:,:],Wlgn_to_4[3,:,:]],1)
        WE_to_E = W4to4[:Nl4,:Nl4]
        WI_to_E = W4to4[:Nl4,Nl4:]
        WE_to_I = W4to4[Nl4:,:Nl4]
        WI_to_I = W4to4[Nl4:,Nl4:]
        arbor_lgn_to_e = tf.concat([arbor_lgn[0,:,:],arbor_lgn[1,:,:]],1)
        arbor_lgn_to_i = tf.concat([arbor_lgn[2,:,:],arbor_lgn[3,:,:]],1)
        arbor_E_to_E = arbor4to4[:Nl4,:Nl4]
        arbor_I_to_E = arbor4to4[:Nl4,Nl4:]
        arbor_E_to_I = arbor4to4[Nl4:,:Nl4]
        arbor_I_to_I = arbor4to4[Nl4:,Nl4:]

        # calculate pre inverse arbors
        invarb_lgn_to_e = tf.where(tf.equal(arbor_lgn_to_e, 0),
                            tf.ones(tf.shape(arbor_lgn_to_e),dtype=tf.float32), tf.math.reciprocal(arbor_lgn_to_e))
        invarb_lgn_to_i = tf.where(tf.equal(arbor_lgn_to_i, 0),
                            tf.ones(tf.shape(arbor_lgn_to_i),dtype=tf.float32), tf.math.reciprocal(arbor_lgn_to_i))
        invarb_E_to_E = tf.where(tf.equal(arbor_E_to_E, 0),
                            tf.ones(tf.shape(arbor_E_to_E),dtype=tf.float32), tf.math.reciprocal(arbor_E_to_E))
        invarb_I_to_E = tf.where(tf.equal(arbor_I_to_E, 0),
                            tf.ones(tf.shape(arbor_I_to_E),dtype=tf.float32), tf.math.reciprocal(arbor_I_to_E))
        invarb_E_to_I = tf.where(tf.equal(arbor_E_to_I, 0),
                            tf.ones(tf.shape(arbor_E_to_I),dtype=tf.float32), tf.math.reciprocal(arbor_E_to_I))
        invarb_I_to_I = tf.where(tf.equal(arbor_I_to_I, 0),
                            tf.ones(tf.shape(arbor_I_to_I),dtype=tf.float32), tf.math.reciprocal(arbor_I_to_I))

        # calculate pre masks
        if p_pre_dict["p_lgn_e_sep"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgn_to_e>0, Wlgn_to_e< (p_pre_dict["p_lgn_e_sep"].Wlim*arbor_lgn_to_e))
        else:
            notfrozen = tf.ones_like(Wlgn_to_e,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_to_e>0 )
        mask_fl_lgn_e = tf.cast(mask,tf.float32)

        if p_pre_dict["p_lgn_i_sep"].freeze_weights:
            notfrozen = tf.math.logical_and(Wlgn_to_i>0, Wlgn_to_i< (p_pre_dict["p_lgn_i_sep"].Wlim*arbor_lgn_to_i))
        else:
            notfrozen = tf.ones_like(Wlgn_to_i,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_lgn_to_i>0 )
        mask_fl_lgn_i = tf.cast(mask,tf.float32)

        if p_pre_dict["p_e_e"].freeze_weights:
            notfrozen = tf.math.logical_and(WE_to_E>0, WE_to_E< (p_pre_dict["p_e_e"].Wlim*arbor_E_to_E))
        else:
            notfrozen = tf.ones_like(WE_to_E,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_E_to_E>0 )
        mask_fl_e_e = tf.cast(mask,tf.float32)

        if p_pre_dict["p_i_e"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(WI_to_E)>0, tf.abs(WI_to_E)< (p_pre_dict["p_i_e"].Wlim*arbor_I_to_E))
        else:
            notfrozen = tf.ones_like(WI_to_E,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_I_to_E>0 )
        mask_fl_i_e = tf.cast(mask,tf.float32)

        if p_pre_dict["p_e_i"].freeze_weights:
            notfrozen = tf.math.logical_and(WE_to_I>0, WE_to_I< (p_pre_dict["p_e_i"].Wlim*arbor_E_to_I))
        else:
            notfrozen = tf.ones_like(WE_to_I,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_E_to_I>0 )
        mask_fl_e_i = tf.cast(mask,tf.float32)

        if p_pre_dict["p_i_i"].freeze_weights:
            notfrozen = tf.math.logical_and(tf.abs(WI_to_I)>0, tf.abs(WI_to_I)< (p_pre_dict["p_i_i"].Wlim*arbor_I_to_I))
        else:
            notfrozen = tf.ones_like(WI_to_I,dtype=bool)
        mask = tf.math.logical_and( notfrozen, arbor_I_to_I>0 )
        mask_fl_i_i = tf.cast(mask,tf.float32)

        # Extract unconstrained Hebbian weight changes
        dW_post_dict = {}
        dW_post_dict['dW_on_l4'] = tf.reshape(dW_dict["dW_on_l4"],(2*Nl4,Nlgn)) * arbor_lgn_on
        dW_post_dict['dW_off_l4'] = tf.reshape(dW_dict["dW_off_l4"],(2*Nl4,Nlgn)) * arbor_lgn_off
        dW_post_dict['dW_e_l4'] = tf.reshape(dW_dict["dW_e_l4"],(2*Nl4,Nl4)) * arbor_e_to_l4
        dW_post_dict['dW_i_l4'] = tf.reshape(dW_dict["dW_i_l4"],(2*Nl4,Nl4)) * arbor_i_to_l4

        for i in range(8):
            # pre
            dW_pre_dict = convert_dW_to_pre_sep(dW_post_dict,Nlgn,Nl4)
            dW_pre_dict['dW_lgn_e_sep'] = p_pre_dict["p_lgn_e_sep"].constrain_update(
                                            dW_pre_dict['dW_lgn_e_sep']*invarb_lgn_to_e,\
                                            tf.zeros_like(Wlgn_to_e),mask_fl_lgn_e,arbor_lgn_to_e,\
                                            p_pre_dict["p_lgn_e_sep"].c_orth,p_pre_dict["p_lgn_e_sep"].s_orth,1)
            dW_pre_dict['dW_lgn_i_sep'] = p_pre_dict["p_lgn_i_sep"].constrain_update(
                                            dW_pre_dict['dW_lgn_i_sep']*invarb_lgn_to_i,\
                                            tf.zeros_like(Wlgn_to_i),mask_fl_lgn_i,arbor_lgn_to_i,\
                                            p_pre_dict["p_lgn_i_sep"].c_orth,p_pre_dict["p_lgn_i_sep"].s_orth,1)
            dW_pre_dict['dW_e_e'] = p_pre_dict["p_e_e"].constrain_update(
                                            dW_pre_dict['dW_e_e']*invarb_E_to_E,\
                                            tf.zeros_like(WE_to_E),mask_fl_e_e,arbor_E_to_E,\
                                            p_pre_dict["p_e_e"].c_orth,p_pre_dict["p_e_e"].s_orth,1)
            dW_pre_dict['dW_i_e'] = p_pre_dict["p_i_e"].constrain_update(
                                            dW_pre_dict['dW_i_e']*invarb_I_to_E,\
                                            tf.zeros_like(WI_to_E),mask_fl_i_e,arbor_I_to_E,\
                                            p_pre_dict["p_i_e"].c_orth,p_pre_dict["p_i_e"].s_orth,1)
            dW_pre_dict['dW_e_i'] = p_pre_dict["p_e_i"].constrain_update(
                                            dW_pre_dict['dW_e_i']*invarb_E_to_I,\
                                            tf.zeros_like(WE_to_I),mask_fl_e_i,arbor_E_to_I,\
                                            p_pre_dict["p_e_i"].c_orth,p_pre_dict["p_e_i"].s_orth,1)
            dW_pre_dict['dW_i_i'] = p_pre_dict["p_i_i"].constrain_update(
                                            dW_pre_dict['dW_i_i']*invarb_I_to_I,\
                                            tf.zeros_like(WI_to_I),mask_fl_i_i,arbor_I_to_I,\
                                            p_pre_dict["p_i_i"].c_orth,p_pre_dict["p_i_i"].s_orth,1)

            # post
            dW_post_dict = convert_dW_to_post_sep(dW_pre_dict,Nlgn,Nl4)
            dW_post_dict['dW_on_l4'] =  p_post_dict["p_on_l4"].constrain_update(
                                            dW_post_dict['dW_on_l4']*invarb_lgn_on,\
                                            tf.zeros_like(Won_l4),mask_fl_on_l4,arbor_lgn_on,\
                                            p_post_dict["p_on_l4"].c_orth,p_post_dict["p_on_l4"].s_orth,1)
            dW_post_dict['dW_off_l4'] =  p_post_dict["p_off_l4"].constrain_update(
                                            dW_post_dict['dW_off_l4']*invarb_lgn_off,\
                                            tf.zeros_like(Woff_l4),mask_fl_off_l4,arbor_lgn_off,\
                                            p_post_dict["p_off_l4"].c_orth,p_post_dict["p_off_l4"].s_orth,1)
            dW_post_dict['dW_e_l4'] =  p_post_dict["p_e_l4"].constrain_update(
                                            dW_post_dict['dW_e_l4']*invarb_e_to_l4,\
                                            tf.zeros_like(We_to_l4),mask_fl_e_l4,arbor_e_to_l4,\
                                            p_post_dict["p_e_l4"].c_orth,p_post_dict["p_e_l4"].s_orth,1)
            dW_post_dict['dW_i_l4'] =  p_post_dict["p_i_l4"].constrain_update(
                                            dW_post_dict['dW_i_l4']*invarb_i_to_l4,\
                                            tf.zeros_like(Wi_to_l4),mask_fl_i_l4,arbor_i_to_l4,\
                                            p_post_dict["p_i_l4"].c_orth,p_post_dict["p_i_l4"].s_orth,1)

        # update weights
        Wlgn_to_4 = Wlgn_to_4 + dt*tf.concat([dW_post_dict['dW_on_l4'][None,:Nl4,:],
            dW_post_dict['dW_off_l4'][None,:Nl4,:],dW_post_dict['dW_on_l4'][None,Nl4:,:],
            dW_post_dict['dW_off_l4'][None,Nl4:,:]], 0)
        W4to4 = W4to4 + dt*tf.concat([dW_post_dict['dW_e_l4'],dW_post_dict['dW_i_l4']], 1)
        params_dict["W4to4"] = W4to4

    return Wlgn_to_4,W4to4,W4to23,W23to23

def clip_weights_wrapper(p_dict,Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
    W4to23,arbor4to23,W23to23,arbor23to23,params_dict):

    if (p_dict["p_lgn_e"] is not None and p_dict["p_lgn_e"].Wlim is not None):
        Wlgn_to_4_e = Wlgn_to_4[:2,:,:]
        Wlgn_to_4_e = p_dict["p_lgn_e"].clip_weights(Wlgn_to_4_e,arbor_lgn[:2,:,:],p_dict["p_lgn_e"].Wlim)
        Wlgn_to_4 = tf.concat([Wlgn_to_4_e,Wlgn_to_4[2:,:,:]],0)

    if (p_dict["p_lgn_i"] is not None and p_dict["p_lgn_i"].Wlim is not None):
        Wlgn_to_4_i = Wlgn_to_4[2:,:,:]
        Wlgn_to_4_i = p_dict["p_lgn_i"].clip_weights(Wlgn_to_4_e,arbor_lgn[:2,:,:],p_dict["p_lgn_i"].Wlim)
        Wlgn_to_4 = tf.concat([Wlgn_to_4[:2,:,:],Wlgn_to_4_i],0)

    if (p_dict["p_rec4_ee"] is not None and p_dict["p_rec4_ee"].Wlim is not None):
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[:Nl4,:Nl4]
        W4to4_ee = W4to4[:Nl4,:Nl4]
        W4to4_ee = p_dict["p_rec4_ee"].clip_weights(W4to4_ee,A,p_dict["p_rec4_ee"].Wlim)
        W4to4 = tf.concat([tf.concat([W4to4_ee,W4to4[Nl4:,:Nl4]],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_rec4_ie"] is not None and p_dict["p_rec4_ie"].Wlim is not None):
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[Nl4:,:Nl4]
        W4to4_ie = W4to4[Nl4:,:Nl4]
        W4to4_ie = p_dict["p_rec4_ie"].clip_weights(W4to4_ie,A,p_dict["p_rec4_ie"].Wlim)
        W4to4 = tf.concat([tf.concat([W4to4[:Nl4,:Nl4],W4to4_ie],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_rec4_ei"] is not None and p_dict["p_rec4_ei"].Wlim is not None):
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[:Nl4,Nl4:]
        W4to4_ei = W4to4[:Nl4,Nl4:]
        W4to4_ei = -p_dict["p_rec4_ei"].clip_weights(-W4to4_ei,A,p_dict["p_rec4_ei"].Wlim)
        W4to4 = tf.concat([W4to4[:,:Nl4],tf.concat([W4to4_ei,W4to4[Nl4:,Nl4:]],0)],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_rec4_ii"] is not None and p_dict["p_rec4_ii"].Wlim is not None):
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[Nl4:,Nl4:]
        W4to4_ii = W4to4[Nl4:,Nl4:]
        W4to4_ii = -p_dict["p_rec4_ii"].clip_weights(-W4to4_ii,A,p_dict["p_rec4_ii"].Wlim)
        W4to4 = tf.concat([W4to4[:,:Nl4],tf.concat([W4to4[:Nl4,Nl4:],W4to4_ii],0)],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_on_l4"] is not None and p_dict["p_on_l4"].Wlim is not None):
        Won_l4 = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[2,:,:]],0)
        arbor_lgn_on = tf.concat([arbor_lgn[0,:,:],arbor_lgn[2,:,:]],0)
        Won_l4 = p_dict["p_on_l4"].clip_weights(Won_l4,arbor_lgn_on,p_dict["p_on_l4"].Wlim)
        Wlgn_to_4 = tf.concat([ Won_l4[None,:Won_l4.shape[0]//2,:], Wlgn_to_4[1:2,:,:],\
                                        Won_l4[None,Won_l4.shape[0]//2:,:], Wlgn_to_4[3:4,:,:] ], 0)

    if (p_dict["p_off_l4"] is not None and p_dict["p_off_l4"].Wlim is not None):
        Woff_l4 = tf.concat([Wlgn_to_4[1,:,:],Wlgn_to_4[3,:,:]],0)
        arbor_lgn_off = tf.concat([arbor_lgn[1,:,:],arbor_lgn[3,:,:]],0)
        Woff_l4 = p_dict["p_off_l4"].clip_weights(Woff_l4,arbor_lgn_off,p_dict["p_off_l4"].Wlim)
        Wlgn_to_4 = tf.concat([ Wlgn_to_4[0:1,:,:], Woff_l4[None,:Woff_l4.shape[0]//2,:],\
                                        Wlgn_to_4[2:3,:,:], Woff_l4[None,Woff_l4.shape[0]//2:,:] ], 0)

    if (p_dict["p_e_l4"] is not None and p_dict["p_e_l4"].Wlim is not None):
        Nl4 = W4to4.shape[0]//2
        We_to_l4 = W4to4[:,:Nl4]
        arbor_e_to_l4 = arbor4to4[:,:Nl4]
        We_to_l4 = p_dict["p_e_l4"].clip_weights(We_to_l4,arbor_e_to_l4,p_dict["p_e_l4"].Wlim)
        W4to4 = tf.concat([We_to_l4,W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_i_l4"] is not None and p_dict["p_i_l4"].Wlim is not None):
        Nl4 = W4to4.shape[0]//2
        Wi_to_l4 = W4to4[:,Nl4:]
        arbor_i_to_l4 = arbor4to4[:,Nl4:]
        Wi_to_l4 = -p_dict["p_i_l4"].clip_weights(-Wi_to_l4,arbor_i_to_l4,p_dict["p_i_l4"].Wlim)
        W4to4 = tf.concat([W4to4[:,:Nl4],Wi_to_l4],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_ffrec"] is not None:
        return clip_weights_wrapper(p_dict["p_ffrec"][0],Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
            W4to23,arbor4to23,W23to23,arbor23to23,params_dict)

    if p_dict["p_ffrec_sep"] is not None:
        return clip_weights_wrapper(p_dict["p_ffrec_sep"][0],Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
            W4to23,arbor4to23,W23to23,arbor23to23,params_dict)

    return Wlgn_to_4,W4to4,W4to23,W23to23

def mult_norm_wrapper(p_dict,Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
    W4to23,arbor4to23,W23to23,arbor23to23,H,running_l4_avg,l4_target,params_dict):

    if p_dict["p_lgn_e"] is not None:
        Wlgn_to_4_e = Wlgn_to_4[:2,:,:]
        A = arbor_lgn[:2,:,:]
        Wlgn_to_4_e,H_new = p_dict["p_lgn_e"].mult_normalization(Wlgn_to_4_e,A,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        if H.shape[0]==2:
            H = tf.stack([H_new,H[1,:]])
        else:
            H = tf.reshape(H_new,[1,H_new.shape[0]])
        print("H1",H.shape)
        Wlgn_to_4 = tf.concat([Wlgn_to_4_e,Wlgn_to_4[2:,:,:]],0)

    if p_dict["p_lgn_i"] is not None:
        Wlgn_to_4_i = Wlgn_to_4[2:,:,:]
        A = arbor_lgn[2:,:,:]
        Wlgn_to_4_i,H_new = p_dict["p_lgn_e"].mult_normalization(Wlgn_to_4_e,A,H[1,:],\
                                                                running_l4_avg[1,:],\
                                                                l4_target[1])
        H = tf.stack([H[0,:],H_new])
        Wlgn_to_4 = tf.concat([Wlgn_to_4[:2,:,:],Wlgn_to_4_i],0)

    if p_dict["p_rec4_ee"] is not None:
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[:Nl4,:Nl4]
        W4to4_ee = W4to4[:Nl4,:Nl4]
        W4to4_ee,_ = p_dict["p_rec4_ee"].mult_normalization(W4to4_ee,A,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        W4to4 = tf.concat([tf.concat([W4to4_ee,W4to4[Nl4:,:Nl4]],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_rec4_ie"] is not None:
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[Nl4:,:Nl4]
        W4to4_ie = W4to4[Nl4:,:Nl4]
        W4to4_ie,_ = p_dict["p_rec4_ie"].mult_normalization(W4to4_ie,A,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        W4to4 = tf.concat([tf.concat([W4to4[:Nl4,:Nl4],W4to4_ie],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_rec4_ei"] is not None:
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[:Nl4,Nl4:]
        W4to4_ei = W4to4[:Nl4,Nl4:]
        W4to4_ei,_ = p_dict["p_rec4_ei"].mult_normalization(W4to4_ei,A,H[1,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        W4to4 = tf.concat([W4to4[:,:Nl4],tf.concat([W4to4_ei,W4to4[Nl4:,Nl4:]],0)],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_rec4_ii"] is not None:
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[Nl4:,Nl4:]
        W4to4_ii = W4to4[Nl4:,Nl4:]
        W4to4_ii,_ = p_dict["p_rec4_ii"].mult_normalization(W4to4_ii,A,H[1,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        W4to4 = tf.concat([W4to4[:,:Nl4],tf.concat([W4to4[:Nl4,Nl4:],W4to4_ii],0)],1)
        params_dict["W4to4"] = W4to4

    # =================== Q_dict pre ===================
    if p_dict["p_on_l4"] is not None:
        Won_l4 = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[2,:,:]],0)
        arbor_lgn_on = tf.concat([arbor_lgn[0,:,:],arbor_lgn[2,:,:]],0)
        Won_l4,H_new = p_dict["p_on_l4"].mult_normalization(Won_l4,arbor_lgn_on,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        Wlgn_to_4 = tf.concat([ Won_l4[None,:Won_l4.shape[0]//2,:], Wlgn_to_4[1:2,:,:],\
                                        Won_l4[None,Won_l4.shape[0]//2:,:], Wlgn_to_4[3:4,:,:] ], 0)

    if p_dict["p_off_l4"] is not None:
        Woff_l4 = tf.concat([Wlgn_to_4[1,:,:],Wlgn_to_4[3,:,:]],0)
        arbor_lgn_off = tf.concat([arbor_lgn[1,:,:],arbor_lgn[3,:,:]],0)
        Woff_l4,H_new = p_dict["p_on_l4"].mult_normalization(Woff_l4,arbor_lgn_off,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        Wlgn_to_4 = tf.concat([ Wlgn_to_4[0:1,:,:], Woff_l4[None,:Woff_l4.shape[0]//2,:],\
                                        Wlgn_to_4[2:3,:,:], Woff_l4[None,Woff_l4.shape[0]//2:,:] ], 0)
        
    if p_dict["p_e_l4"] is not None:
        Nl4 = W4to4.shape[0]//2
        We_to_l4 = W4to4[:,:Nl4]
        arbor_e_to_l4 = arbor4to4[:,:Nl4]
        We_to_l4, H_new = p_dict["p_e_l4"].mult_normalization(We_to_l4,arbor_e_to_l4,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        W4to4 = tf.concat([We_to_l4,W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_i_l4"] is not None:
        Nl4 = W4to4.shape[0]//2
        Wi_to_l4 = W4to4[:,Nl4:]
        arbor_i_to_l4 = arbor4to4[:,Nl4:]
        Wi_to_l4, H_new = p_dict["p_i_l4"].mult_normalization(Wi_to_l4,arbor_i_to_l4,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        W4to4 = tf.concat([W4to4[:,:Nl4],Wi_to_l4],1)
        params_dict["W4to4"] = W4to4
    # ===================================================
    # ================= Q_dict post =====================

    if p_dict["p_lgne_e"] is not None:
        Nl4 = W4to4.shape[0]//2
        Nlgn = Wlgn_to_4.shape[2]
        Wlgne_to_e = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[1,:,:],W4to4[:Nl4,:Nl4]],1)
        arbor_lgne_to_e = tf.concat([arbor_lgn[0,:,:],arbor_lgn[1,:,:],arbor4to4[:Nl4,:Nl4]],1)
        W_new, H_new = p_dict["p_lgne_e"].mult_normalization(Wlgne_to_e,arbor_lgne_to_e,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])

        Wlgn_to_4 = tf.concat( [W_new[None,:,:Nlgn], W_new[None, :,Nlgn:2*Nlgn], \
                                            Wlgn_to_4[2:,:,:]], 0)
        W4to4 = tf.concat([tf.concat([W_new[:,2*Nlgn:], W4to4[Nl4:,:Nl4]],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_lgne_i"] is not None:
        Nl4 = W4to4.shape[0]//2
        Nlgn = Wlgn_to_4.shape[2]
        Wlgne_to_i = tf.concat([Wlgn_to_4[2,:,:],Wlgn_to_4[3,:,:],W4to4[:Nl4,:Nl4]],1)
        arbor_lgne_to_i = tf.concat([arbor_lgn[2,:,:],arbor_lgn[3,:,:],arbor4to4[:Nl4,:Nl4]],1)
        W_new, H_new = p_dict["p_lgne_i"].mult_normalization(Wlgne_to_i,arbor_lgne_to_i,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        Wlgn_to_4 = tf.concat( [ Wlgn_to_4[:2,:,:], \
                                        W_new[None,:,:Nlgn], W_new[None,:,Nlgn:2*Nlgn] ], 0)
        W4to4 = tf.concat([tf.concat([W4to4[:Nl4,:Nl4], W_new[:,2*Nlgn:]],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_i_e"] is not None:
        Nl4 = W4to4.shape[0]//2
        WI_to_E = W4to4[:Nl4,Nl4:]
        arbor_I_to_E = arbor4to4[:Nl4,Nl4:]
        WI_to_E, H_new = p_dict["p_i_e"].mult_normalization(WI_to_E,arbor_I_to_E,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        W4to4 = tf.concat([tf.concat([W4to4[:Nl4,:Nl4], WI_to_E],1),W4to4[Nl4:,:]],0)
        params_dict["W4to4"] = W4to4

    if p_dict["p_i_i"] is not None:
        Nl4 = W4to4.shape[0]//2
        WI_to_I = W4to4[Nl4:,Nl4:]
        arbor_I_to_I = arbor4to4[Nl4:,Nl4:]
        WI_to_I, H_new = p_dict["p_i_i"].mult_normalization(WI_to_I,arbor_I_to_I,H[0,:],\
                                                                running_l4_avg[0,:],\
                                                                l4_target[0])
        W4to4 = tf.concat([W4to4[:Nl4,:],tf.concat([W4to4[Nl4:,:Nl4],WI_to_I],1)],0)
        params_dict["W4to4"] = W4to4

    if p_dict["p_ffrec"] is not None:
        Nlgn = Wlgn_to_4.shape[2]
        Nl4 = Wlgn_to_4.shape[1]

        p_post_dict = p_dict["p_ffrec"][0]
        p_pre_dict = p_dict["p_ffrec"][1]

        for i in range(8):
            # extract pre W and arbors
            Wlgne_to_e = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[1,:,:],W4to4[:Nl4,:Nl4]],1)
            Wlgne_to_i = tf.concat([Wlgn_to_4[2,:,:],Wlgn_to_4[3,:,:],W4to4[Nl4:,:Nl4]],1)
            WI_to_E = W4to4[:Nl4,Nl4:]
            WI_to_I = W4to4[Nl4:,Nl4:]
            arbor_lgne_to_e = tf.concat([arbor_lgn[0,:,:],arbor_lgn[1,:,:],arbor4to4[:Nl4,:Nl4]],1)
            arbor_lgne_to_i = tf.concat([arbor_lgn[2,:,:],arbor_lgn[3,:,:],arbor4to4[Nl4:,:Nl4]],1)
            arbor_I_to_E = arbor4to4[:Nl4,Nl4:]
            arbor_I_to_I = arbor4to4[Nl4:,Nl4:]

            # normalize pre weights
            Wlgne_to_e_new,_ = p_pre_dict["p_lgne_e"].mult_normalization(Wlgne_to_e,arbor_lgne_to_e,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            Wlgne_to_i_new,_ = p_pre_dict["p_lgne_i"].mult_normalization(Wlgne_to_i,arbor_lgne_to_i,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            WI_to_E_new,_ = p_pre_dict["p_i_e"].mult_normalization(WI_to_E,arbor_I_to_E,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            WI_to_I_new,_ = p_pre_dict["p_i_i"].mult_normalization(WI_to_I,arbor_I_to_I,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])

            # update weights
            Wlgn_to_4 = tf.concat([Wlgne_to_e_new[None,:,:Nlgn],Wlgne_to_e_new[None,:,Nlgn:2*Nlgn],
                                   Wlgne_to_i_new[None,:,:Nlgn],Wlgne_to_i_new[None,:,Nlgn:2*Nlgn]], 0)
            W4to4 = tf.concat([tf.concat([Wlgne_to_e_new[:,2*Nlgn:],Wlgne_to_i_new[:,2*Nlgn:]], 0),
                               tf.concat([WI_to_E_new,WI_to_I_new], 0)], 1)

            # extract post W and arbors
            Won_l4 = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[2,:,:]],0)
            Woff_l4 = tf.concat([Wlgn_to_4[1,:,:],Wlgn_to_4[3,:,:]],0)
            We_to_l4 = W4to4[:,:Nl4]
            Wi_to_l4 = W4to4[:,Nl4:]
            arbor_lgn_on = tf.concat([arbor_lgn[0,:,:],arbor_lgn[2,:,:]],0)
            arbor_lgn_off = tf.concat([arbor_lgn[1,:,:],arbor_lgn[3,:,:]],0)
            arbor_e_to_l4 = arbor4to4[:,:Nl4]
            arbor_i_to_l4 = arbor4to4[:,Nl4:]

            # normalize post weights
            Won_l4_new,_ = p_post_dict["p_on_l4"].mult_normalization(Won_l4,arbor_lgn_on,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            Woff_l4_new,_ = p_post_dict["p_on_l4"].mult_normalization(Woff_l4,arbor_lgn_off,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            We_to_l4_new,_ = p_post_dict["p_e_l4"].mult_normalization(We_to_l4,arbor_e_to_l4,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            Wi_to_l4_new,_ = p_post_dict["p_i_l4"].mult_normalization(Wi_to_l4,arbor_i_to_l4,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])

            # update weights
            Wlgn_to_4 = tf.concat([Won_l4_new[None,:Nl4,:],Woff_l4_new[None,:Nl4,:],
                                   Won_l4_new[None,Nl4:,:],Woff_l4_new[None,Nl4:,:]], 0)
            W4to4 = tf.concat([We_to_l4_new,Wi_to_l4_new], 1)

        params_dict["W4to4"] = W4to4

    if p_dict["p_ffrec_sep"] is not None:
        Nlgn = Wlgn_to_4.shape[2]
        Nl4 = Wlgn_to_4.shape[1]

        p_post_dict = p_dict["p_ffrec_sep"][0]
        p_pre_dict = p_dict["p_ffrec_sep"][1]

        for i in range(8):
            # extract pre W and arbors
            Wlgn_to_e = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[1,:,:]],1)
            Wlgn_to_i = tf.concat([Wlgn_to_4[2,:,:],Wlgn_to_4[3,:,:]],1)
            WE_to_E = W4to4[:Nl4,:Nl4]
            WI_to_E = W4to4[:Nl4,Nl4:]
            WE_to_I = W4to4[Nl4:,:Nl4]
            WI_to_I = W4to4[Nl4:,Nl4:]
            arbor_lgn_to_e = tf.concat([arbor_lgn[0,:,:],arbor_lgn[1,:,:]],1)
            arbor_lgn_to_i = tf.concat([arbor_lgn[2,:,:],arbor_lgn[3,:,:]],1)
            arbor_E_to_E = arbor4to4[:Nl4,:Nl4]
            arbor_I_to_E = arbor4to4[:Nl4,Nl4:]
            arbor_E_to_I = arbor4to4[Nl4:,:Nl4]
            arbor_I_to_I = arbor4to4[Nl4:,Nl4:]

            # normalize pre weights
            Wlgn_to_e_new,_ = p_pre_dict["p_lgn_e_sep"].mult_normalization(Wlgn_to_e,arbor_lgn_to_e,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            Wlgn_to_i_new,_ = p_pre_dict["p_lgn_i_sep"].mult_normalization(Wlgn_to_i,arbor_lgn_to_i,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            WE_to_E_new,_ = p_pre_dict["p_e_e"].mult_normalization(WE_to_E,arbor_E_to_E,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            WI_to_E_new,_ = p_pre_dict["p_i_e"].mult_normalization(WI_to_E,arbor_I_to_E,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            WE_to_I_new,_ = p_pre_dict["p_e_i"].mult_normalization(WE_to_I,arbor_E_to_I,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            WI_to_I_new,_ = p_pre_dict["p_i_i"].mult_normalization(WI_to_I,arbor_I_to_I,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])

            # update weights
            Wlgn_to_4 = tf.concat([Wlgn_to_e_new[None,:,:Nlgn],Wlgn_to_e_new[None,:,Nlgn:],
                                   Wlgn_to_i_new[None,:,:Nlgn],Wlgn_to_i_new[None,:,Nlgn:]], 0)
            W4to4 = tf.concat([tf.concat([WE_to_E_new,WE_to_I_new], 0),
                               tf.concat([WI_to_E_new,WI_to_I_new], 0)], 1)

            # extract post W and arbors
            Won_l4 = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[2,:,:]],0)
            Woff_l4 = tf.concat([Wlgn_to_4[1,:,:],Wlgn_to_4[3,:,:]],0)
            We_to_l4 = W4to4[:,:Nl4]
            Wi_to_l4 = W4to4[:,Nl4:]
            arbor_lgn_on = tf.concat([arbor_lgn[0,:,:],arbor_lgn[2,:,:]],0)
            arbor_lgn_off = tf.concat([arbor_lgn[1,:,:],arbor_lgn[3,:,:]],0)
            arbor_e_to_l4 = arbor4to4[:,:Nl4]
            arbor_i_to_l4 = arbor4to4[:,Nl4:]

            # normalize post weights
            Won_l4_new,_ = p_post_dict["p_on_l4"].mult_normalization(Won_l4,arbor_lgn_on,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            Woff_l4_new,_ = p_post_dict["p_on_l4"].mult_normalization(Woff_l4,arbor_lgn_off,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            We_to_l4_new,_ = p_post_dict["p_e_l4"].mult_normalization(We_to_l4,arbor_e_to_l4,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])
            Wi_to_l4_new,_ = p_post_dict["p_i_l4"].mult_normalization(Wi_to_l4,arbor_i_to_l4,H[0,:],\
                                                                    running_l4_avg[0,:],\
                                                                    l4_target[0])

            # update weights
            Wlgn_to_4 = tf.concat([Won_l4_new[None,:Nl4,:],Woff_l4_new[None,:Nl4,:],
                                   Won_l4_new[None,Nl4:,:],Woff_l4_new[None,Nl4:,:]], 0)
            W4to4 = tf.concat([We_to_l4_new,Wi_to_l4_new], 1)

        params_dict["W4to4"] = W4to4

    return Wlgn_to_4,W4to4,W4to23,W23to23,H

def prune_weights_wrapper(p_dict,Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
    W4to23,arbor4to23,W23to23,arbor23to23,params_dict):

    if (p_dict["p_lgn_e"] is not None and p_dict["p_lgn_e"].Wthresh is not None):
        Wlgn_to_4_e = Wlgn_to_4[:2,:,:]
        Wlgn_to_4_e = p_dict["p_lgn_e"].prune_weights(Wlgn_to_4_e,arbor_lgn[:2,:,:],p_dict["p_lgn_e"].Wthresh)
        Wlgn_to_4 = tf.concat([Wlgn_to_4_e,Wlgn_to_4[2:,:,:]],0)

    if (p_dict["p_lgn_i"] is not None and p_dict["p_lgn_i"].Wthresh is not None):
        Wlgn_to_4_i = Wlgn_to_4[2:,:,:]
        Wlgn_to_4_i = p_dict["p_lgn_i"].prune_weights(Wlgn_to_4_e,arbor_lgn[:2,:,:],p_dict["p_lgn_i"].Wthresh)
        Wlgn_to_4 = tf.concat([Wlgn_to_4[:2,:,:],Wlgn_to_4_i],0)

    if (p_dict["p_rec4_ee"] is not None and p_dict["p_rec4_ee"].Wthresh is not None):
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[:Nl4,:Nl4]
        W4to4_ee = W4to4[:Nl4,:Nl4]
        W4to4_ee = p_dict["p_rec4_ee"].prune_weights(W4to4_ee,A,p_dict["p_rec4_ee"].Wthresh)
        W4to4 = tf.concat([tf.concat([W4to4_ee,W4to4[Nl4:,:Nl4]],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_rec4_ie"] is not None and p_dict["p_rec4_ie"].Wthresh is not None):
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[Nl4:,:Nl4]
        W4to4_ie = W4to4[Nl4:,:Nl4]
        W4to4_ie = p_dict["p_rec4_ie"].prune_weights(W4to4_ie,A,p_dict["p_rec4_ie"].Wthresh)
        W4to4 = tf.concat([tf.concat([W4to4[:Nl4,:Nl4],W4to4_ie],0),W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_rec4_ei"] is not None and p_dict["p_rec4_ei"].Wthresh is not None):
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[:Nl4,Nl4:]
        W4to4_ei = W4to4[:Nl4,Nl4:]
        W4to4_ei = -p_dict["p_rec4_ei"].prune_weights(-W4to4_ei,A,p_dict["p_rec4_ei"].Wthresh)
        W4to4 = tf.concat([W4to4[:,:Nl4],tf.concat([W4to4_ei,W4to4[Nl4:,Nl4:]],0)],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_rec4_ii"] is not None and p_dict["p_rec4_ii"].Wthresh is not None):
        Nl4 = W4to4.shape[0]//2
        if arbor4to4 is None:
            A = 1
        else:
            A = arbor4to4[Nl4:,Nl4:]
        W4to4_ii = W4to4[Nl4:,Nl4:]
        W4to4_ii = -p_dict["p_rec4_ii"].prune_weights(-W4to4_ii,A,p_dict["p_rec4_ii"].Wthresh)
        W4to4 = tf.concat([W4to4[:,:Nl4],tf.concat([W4to4[:Nl4,Nl4:],W4to4_ii],0)],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_on_l4"] is not None and p_dict["p_on_l4"].Wthresh is not None):
        Won_l4 = tf.concat([Wlgn_to_4[0,:,:],Wlgn_to_4[2,:,:]],0)
        arbor_lgn_on = tf.concat([arbor_lgn[0,:,:],arbor_lgn[2,:,:]],0)
        Won_l4 = p_dict["p_on_l4"].prune_weights(Won_l4,arbor_lgn_on,p_dict["p_on_l4"].Wthresh)
        Wlgn_to_4 = tf.concat([ Won_l4[None,:Won_l4.shape[0]//2,:], Wlgn_to_4[1:2,:,:],\
                                        Won_l4[None,Won_l4.shape[0]//2:,:], Wlgn_to_4[3:4,:,:] ], 0)

    if (p_dict["p_off_l4"] is not None and p_dict["p_off_l4"].Wthresh is not None):
        Woff_l4 = tf.concat([Wlgn_to_4[1,:,:],Wlgn_to_4[3,:,:]],0)
        arbor_lgn_off = tf.concat([arbor_lgn[1,:,:],arbor_lgn[3,:,:]],0)
        Woff_l4 = p_dict["p_off_l4"].prune_weights(Woff_l4,arbor_lgn_off,p_dict["p_off_l4"].Wthresh)
        Wlgn_to_4 = tf.concat([ Wlgn_to_4[0:1,:,:], Woff_l4[None,:Woff_l4.shape[0]//2,:],\
                                        Wlgn_to_4[2:3,:,:], Woff_l4[None,Woff_l4.shape[0]//2:,:] ], 0)

    if (p_dict["p_e_l4"] is not None and p_dict["p_e_l4"].Wthresh is not None):
        Nl4 = W4to4.shape[0]//2
        We_to_l4 = W4to4[:,:Nl4]
        arbor_e_to_l4 = arbor4to4[:,:Nl4]
        We_to_l4 = p_dict["p_e_l4"].prune_weights(We_to_l4,arbor_e_to_l4,p_dict["p_e_l4"].Wthresh)
        W4to4 = tf.concat([We_to_l4,W4to4[:,Nl4:]],1)
        params_dict["W4to4"] = W4to4

    if (p_dict["p_i_l4"] is not None and p_dict["p_i_l4"].Wthresh is not None):
        Nl4 = W4to4.shape[0]//2
        Wi_to_l4 = W4to4[:,Nl4:]
        arbor_i_to_l4 = arbor4to4[:,Nl4:]
        Wi_to_l4 = -p_dict["p_i_l4"].prune_weights(-Wi_to_l4,arbor_i_to_l4,p_dict["p_i_l4"].Wthresh)
        W4to4 = tf.concat([W4to4[:,:Nl4],Wi_to_l4],1)
        params_dict["W4to4"] = W4to4

    if p_dict["p_ffrec"] is not None:
        return prune_weights_wrapper(p_dict["p_ffrec"][0],Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
            W4to23,arbor4to23,W23to23,arbor23to23,params_dict)

    if p_dict["p_ffrec_sep"] is not None:
        return prune_weights_wrapper(p_dict["p_ffrec_sep"][0],Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
            W4to23,arbor4to23,W23to23,arbor23to23,params_dict)

    return Wlgn_to_4,W4to4,W4to23,W23to23