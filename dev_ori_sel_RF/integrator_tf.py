import sys
import tensorflow as tf
import numpy as np
from collections import defaultdict


from dev_ori_sel_RF import inputs,network_tools,dynamics,\
normalisation_constraints,plasticity_dynamics
from dev_ori_sel_RF.tools import calc_tools,\
get_spatial_wavelength,gen_lowd_activity_patterns


def tf_check_type(t, y0):
    # Ensure Input type is Correct
    if not (y0.dtype.is_floating and t.dtype.is_floating):
        raise TypeError('Error in Datatype')


# def clip_weights(Wlgn_to_4,Wlim,arbor):
# 	return tf.clip_by_value(Wlgn_to_4,0,arbor*Wlim)

def check_for_frozen_weights(Wlgn_to_4,Wlim,arbor):
    frozen = tf.math.logical_or(Wlgn_to_4[arbor>0]>=(Wlim*arbor[arbor>0]), Wlgn_to_4[arbor>0]<=0)
    return np.sum(frozen)>0



class Tf_integrator_new:
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.scheme = params_dict["config_dict"]["integrator"]
        self.avg_no_inp = params_dict["config_dict"]["Inp_params"]["avg_no_inp"]
        self.num_inputs = int(self.params_dict["config_dict"]["runtime"]/\
                            self.params_dict["config_dict"]["dt"]/\
                            self.params_dict["config_dict"]["Inp_params"]["pattern_duration"])
        self.num_plasticity_steps = int(self.num_inputs/self.avg_no_inp)
        self.expanse_steps = int(self.params_dict["config_dict"]["Inp_params"]["expanse_time"])
        self.frozen = False

        self.nonlinearity_rule_l4 = params_dict["config_dict"]["nonlinearity_l4"]
        self.nonlinearity_rule_l23 = params_dict["config_dict"]["nonlinearity_l23"]

        self.num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]
        self.saving_stepsize = params_dict["config_dict"]["saving_stepsize"]
        self.system = params_dict["config_dict"]["system"]
        self.connectivity_type = params_dict["config_dict"]["Wlgn_to4_params"]["connectivity_type"]

        self.num_pop = 2 if "2pop" in params_dict["config_dict"]["W4to4_params"]["Wrec_mode"] else 1

        self._init_inverse_cortical_interaction_function() #I_xy
        nl = self._init_nonlinearity(self.nonlinearity_rule_l4) #f is rectified, linear, etc for both l4 and l23
        self.params_dict.update({"nonlinearity_l4" : nl})
        nl = self._init_nonlinearity(self.nonlinearity_rule_l23)
        self.params_dict.update({"nonlinearity_l23" : nl})

        self.l4e_target = params_dict["config_dict"]["W4to4_params"].get("l4e_target",None)
        self.l4i_target = params_dict["config_dict"]["W4to4_params"].get("l4i_target",None)
        self.l4_target = [self.l4e_target,self.l4i_target]
        self.adaptive = params_dict["config_dict"]["Inp_params"]["simulate_activity"] in (
            "dynamics_adaptive","stevens_etal","antolik_etal")
        self.l4_avg = params_dict["l4_avg"]
        self.theta_4 = params_dict["theta_4"]
        self._init_cortical_layer_dynamics()

        tf.random.set_seed(self.params_dict["config_dict"]["random_seed"]*114)
        self.divergence_limit = 10000000.

        self._init_running_avg()

        self.estimate_spatial_wavelength_activity_from_conn()

        # self.plastic = plasticity_dynamics.Plasticity(self.params_dict)
        self._init_plasticity_dynamics()


    def _init_nonlinearity(self,nonlinearity_rule):
        # TODO: add two different nonlinearities for L 4 and 2/3
        if nonlinearity_rule == 'rectifier':
            tf_nonlinearity = network_tools.nl_rect
        elif nonlinearity_rule == 'linear':
            tf_nonlinearity = network_tools.nl_linear
        elif nonlinearity_rule == "powerlaw":
            tf_nonlinearity = network_tools.nl_powerlaw
        else:
            raise Exception('Unknown nonlinearity rule')
        return tf_nonlinearity


    def _init_cortical_layer_dynamics(self):
        if self.system=="one_layer":
            self.rhs_l23 = None
            if self.connectivity_type=="E":
                self.rhs_l4 = dynamics.rhs_l4EI

            elif self.connectivity_type=="EI":
                self.rhs_l4 = dynamics.rhs_l4EI_full_LGN_input

            else:
                raise Exception('Number of LGN paths not defined. Choose either 2 or 4.')

        elif self.system=="two_layer":
            self.rhs_l23 = dynamics.rhs_twolayer_l23EI
            if self.num_lgn_paths==2:
                self.rhs_l4 = dynamics.rhs_twolayer_l4EI

            elif self.num_lgn_paths==4:
                self.rhs_l4 = dynamics.rhs_twolayer_l4EI_full_LGN_input

            else:
                raise Exception('Number of LGN paths not defined. Choose either 2 or 4.')

        else:
            raise Exception('System not defined. Choose either one_layer or two_layer.')


    def yield_init_conditions(self,y):
        """initiate conditions for cortical layer(s)"""
        if self.params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="dynamics":
            N4 = self.params_dict["N4"]
            Nvert = self.params_dict["Nvert"]
            Nlgn = self.params_dict["Nlgn"]
            lim = Nlgn*Nlgn*N4*N4*Nvert

            if self.system=="one_layer":
                if "2pop" in self.params_dict["config_dict"]["W4to4_params"]["Wrec_mode"]:
                    l40 = tf.random.uniform([N4**2*2*Nvert],minval=0,maxval=0.01,dtype=tf.float32)
                else:
                    l40 = tf.random.uniform([N4**2*Nvert],minval=0,maxval=0.01,dtype=tf.float32)
                y0 = tf.concat([y[:self.num_lgn_paths*lim], l40], axis=0)

            elif self.system=="two_layer":
                N23 = self.params_dict["N23"]
                l40 = tf.random.uniform([N4*N4*2*Nvert], minval=0, maxval=0.01, dtype=tf.float32)
                l230 = tf.random.uniform([N23*N23*2], minval=0, maxval=0.01, dtype=tf.float32)

                if self.params_dict["config_dict"]["W4to23_params"]["plasticity_rule"]!="None":
                    y0 = tf.concat([y[:self.num_lgn_paths*lim], l40, l230,\
                                    y[self.num_lgn_paths*lim+N4*N4*2*Nvert+N23*N23*2:]], axis=0)
                else:
                    y0 = tf.concat([y[:self.num_lgn_paths*lim], l40, l230], axis=0)

            else:
                raise Exception('System not defined. Choose either one_layer or two_layer.')
        else:
            y0 = y
        return y0


    def _init_inverse_cortical_interaction_function(self):
        if True:#self.params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="steady_state":

            if self.system=="one_layer":
                # N4 = self.params_dict["N4"]
                # Nvert = self.params_dict["Nvert"]
                W4to4 = self.params_dict["W4to4"]
                N = W4to4.shape[0]
                if self.params_dict.get("invert_rec",False):
                    I_crt = np.linalg.inv(np.diagflat(np.ones(N)) - W4to4) #
                else:
                    I_crt = W4to4
                self.I_crt = tf.convert_to_tensor(I_crt, dtype=tf.float32)
                self.I_crt23 = None

            elif self.system=="two_layer":
                N4 = self.params_dict["N4"]
                N23 = self.params_dict["N23"]
                Nvert = self.params_dict["Nvert"]
                W4to4 = self.params_dict["W4to4"]
                W4to23 = self.params_dict["W4to23"]
                W23to4 = self.params_dict["W23to4"]
                W23to23 = self.params_dict["W23to23"]

                I = np.diagflat(np.ones(N23**2*2))
                Ir = np.linalg.inv(I - W4to4)
                Wtmp = np.dot(W4to23,Ir)
                self.I_crt23 = np.dot(np.linalg.inv(I - W23to23 - np.dot(Wtmp,W23to4)),Wtmp)

                I = np.diagflat(np.ones(N4**2*2*Nvert))
                Ix = np.linalg.inv(I - W23to23)
                self.I_crt = np.linalg.inv(I - W4to4 - np.dot(np.dot(W23to4,Ix),W4to23))

                ew23,_ = np.linalg.eig(W23to23 + np.dot(Wtmp,W23to4))
                ew4,_ = np.linalg.eig(W4to4 + np.dot(np.dot(W23to4,Ix),W4to23))
                print("Eigvals, L23: {}".format(np.nanmax(np.real(ew23))))
                print("Eigvals, L4: {}".format(np.nanmax(np.real(ew4))))
                sys.stdout.flush()

        else:
            self.I_crt = None
            self.I_crt23 = None


    def _init_running_avg(self):
        self.count = 1.
        self.avg = None
        self.tau = 1.
        self.len_running_avg = 2
        self.avg_ampl = 1.1
        N4 = self.params_dict["N4"]
        Nvert = self.params_dict["Nvert"]
        l4_size = N4**2*Nvert
        self.running_l4_avg = None


    def running_avg(self,new_value):
        new_value = tf.reshape(new_value,[self.num_pop,-1])
        if self.running_l4_avg is None:
            self.running_l4_avg = new_value
            self.avg = new_value
        else:
            self.count =  (self.count * np.exp(-1./self.tau) + 1)
            self.avg = (self.avg * np.exp(-1./self.tau) + new_value)
            self.running_l4_avg = self.avg/self.count
        # self.running_l4_avg = tf.roll(self.running_l4_avg,shift=-1,axis=0)
        # indices = tf.constant([[self.len_running_avg-1],])
        # updates = [y[lim*num_lgn_paths:num_lgn_paths*lim+l4_size],]
        # self.running_l4_avg = tf.tensor_scatter_nd_update(self.running_l4_avg, indices, updates)


    def estimate_spatial_wavelength_activity_from_conn(self):
        # estimate_spatial_wavelength_activity_from_conn
        k = np.arange(0,60,0.1)
        aee = self.params_dict["config_dict"]["W4to4_params"]["aEE"]
        aie = self.params_dict["config_dict"]["W4to4_params"]["aIE"]
        aei = self.params_dict["config_dict"]["W4to4_params"]["aEI"]
        aii = self.params_dict["config_dict"]["W4to4_params"]["aII"]
        sigma_factor = self.params_dict["config_dict"]["W4to4_params"]["sigma_factor"]
        see = self.params_dict["config_dict"]["W4to4_params"]["sigma_EE"] * sigma_factor
        sei = self.params_dict["config_dict"]["W4to4_params"]["sigma_EI"] * sigma_factor
        sii = self.params_dict["config_dict"]["W4to4_params"]["sigma_II"] * sigma_factor
        eigvals = get_spatial_wavelength.eigval_max(k,aee,aei*aie,aii,see,sei,sii,tau=1,alpha=0)
        max_freq = k[np.nanargmax(eigvals)]
        print("max_freq",max_freq,1./max_freq,2*np.pi/max_freq)

        ndim = 4
        width = 0.2
        size = self.params_dict["config_dict"]["N4"]
        freq = max_freq
        lowD_subset = gen_lowd_activity_patterns.create_activity_patterns(freq,size,ndim,width)
        cc = np.corrcoef(lowD_subset)
        lowD_subset = np.clip(1000 * lowD_subset,0,100000)
        lowD_subset = tf.convert_to_tensor(lowD_subset, dtype=tf.float32)
        lowD_subset = tf.concat([lowD_subset,lowD_subset],0)
        self.params_dict["lowD_subset"] = lowD_subset

        self.params_dict["Corr"] = tf.convert_to_tensor(cc,dtype=tf.float32)


    def _init_plasticity_dynamics(self): #p_dict here
        if self.params_dict["config_dict"]["Wlgn_to4_params"]["mult_norm"] in (
                "ffrec_post","ffrec_pre","ffrec_postpre_approx","ffrec_postpre_approx_sep"):
            # ======= Q_dict =================
            q_dict = defaultdict(lambda:None)
            dt = self.params_dict["config_dict"]["dt"]

            c_orth = self.params_dict["c_orth"]
            s_orth = self.params_dict["s_orth"]
            Wlim = self.params_dict["config_dict"]["Wlgn_to4_params"]["Wlim"]
            Wthresh = self.params_dict["config_dict"]["Wlgn_to4_params"].get("Wthresh",None)
            plasticity_rule = self.params_dict["config_dict"]["Wlgn_to4_params"]["plasticity_rule"]
            constraint_mode = self.params_dict["config_dict"]["Wlgn_to4_params"]["constraint_mode"]
            mult_norm = self.params_dict["config_dict"]["Wlgn_to4_params"]["mult_norm"]
            clip_mode = self.params_dict["config_dict"]["Wlgn_to4_params"]["clip_weights"]
            weight_strength = 1.

            init_weights = None
            if mult_norm=="ffrec_post":
                #ON_to_l4 and Off_to_l4 first and then l4e_to_l4 and l4i_to_l4)
                init_weights = [ self.params_dict["init_weights"], \
                                          self.params_dict["init_weights_4to4"] ]
                freeze_weights = self.params_dict["config_dict"]["Wlgn_to4_params"].get("freeze_weights",True)

                beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
                if self.params_dict["config_dict"]["Wlgn_to4_params"].get("scale_beta",False):
                    beta_P /= tf.reduce_sum(self.params_dict["arbor2"][:,0,:])
                q_dict["p_on_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][0,:],freeze_weights,Wthresh)
                q_dict["p_off_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][1,:],freeze_weights,Wthresh)

                beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"]
                if self.params_dict["config_dict"]["Wlgn_to4_params"].get("scale_beta",False):
                    beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"] /\
                            tf.reduce_sum(self.params_dict["arbor2"][:,0,:])
                Wthresh = self.params_dict["config_dict"]["W4to4_params"].get("Wthresh",None)
                q_dict["p_e_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][0,:],freeze_weights,Wthresh)
                q_dict["p_i_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][1,:],freeze_weights,Wthresh)

            elif mult_norm=="ffrec_pre":
                init_weights = [ self.params_dict["init_weights"], \
                                          self.params_dict["init_weights_4to4"] ] #[  arr[lgne_e,lgn_i], arr[i_e,i_i]   ]
                freeze_weights = self.params_dict["config_dict"]["Wlgn_to4_params"].get("freeze_weights",True)

                ff_beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
                if self.params_dict["config_dict"]["Wlgn_to4_params"].get("scale_beta",False):
                    ff_beta_P /= tf.reduce_sum(self.params_dict["arbor2"][:,0,:])
                rec_beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
                beta_P = [ff_beta_P,rec_beta_P]

                q_dict["p_lgne_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][0,:],freeze_weights)
                q_dict["p_lgne_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][1,:],freeze_weights)
                q_dict["p_i_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][0,:],freeze_weights)
                q_dict["p_i_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][1,:],freeze_weights)

            elif mult_norm=="ffrec_postpre_approx":
                q_post_dict = defaultdict(lambda:None)
                q_pre_dict = defaultdict(lambda:None)

                # Build q_post_dict
                constraint_mode = "ffrec_post"
                mult_norm = "ffrec_post"

                init_weights = [ self.params_dict["init_weights"][0], \
                                          self.params_dict["init_weights_4to4"][0] ]
                freeze_weights = self.params_dict["config_dict"]["Wlgn_to4_params"].get("freeze_weights",True)

                beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
                if self.params_dict["config_dict"]["Wlgn_to4_params"].get("scale_beta",False):
                    beta_P /= tf.reduce_sum(self.params_dict["arbor2"][:,0,:])
                q_post_dict["p_on_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][0,:],freeze_weights)
                q_post_dict["p_off_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][1,:],freeze_weights)

                beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"]
                q_post_dict["p_e_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][0,:],freeze_weights)
                q_post_dict["p_i_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][1,:],freeze_weights)

                init_weights = [ self.params_dict["init_weights"][1], \
                                          self.params_dict["init_weights_4to4"][1] ] #[  arr[lgne_e,lgn_i], arr[i_e,i_i]   ]
                freeze_weights = self.params_dict["config_dict"]["Wlgn_to4_params"].get("freeze_weights",True)

                # Build q_pre_dict
                constraint_mode = "ffrec_pre"
                mult_norm = "ffrec_pre"

                ff_beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
                if self.params_dict["config_dict"]["Wlgn_to4_params"].get("scale_beta",False):
                    ff_beta_P /= tf.reduce_sum(self.params_dict["arbor2"][:,0,:])
                rec_beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"]
                beta_P = [ff_beta_P,rec_beta_P]

                q_pre_dict["p_lgne_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][0,:],freeze_weights)
                q_pre_dict["p_lgne_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][1,:],freeze_weights)
                q_pre_dict["p_i_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][0,:],freeze_weights)
                q_pre_dict["p_i_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][1,:],freeze_weights)

                q_dict['p_ffrec'] = (q_post_dict,q_pre_dict)

            elif mult_norm=="ffrec_postpre_approx_sep":
                q_post_dict = defaultdict(lambda:None)
                q_pre_dict = defaultdict(lambda:None)

                # Build q_post_dict
                constraint_mode = "ffrec_post"
                mult_norm = "ffrec_post"

                init_weights = [ self.params_dict["init_weights"][0], \
                                          self.params_dict["init_weights_4to4"][0] ]
                freeze_weights = self.params_dict["config_dict"]["Wlgn_to4_params"].get("freeze_weights",True)

                beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
                if self.params_dict["config_dict"]["Wlgn_to4_params"].get("scale_beta",False):
                    beta_P /= tf.reduce_sum(self.params_dict["arbor2"][:,0,:])
                q_post_dict["p_on_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][0,:],freeze_weights,Wthresh)
                q_post_dict["p_off_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][1,:],freeze_weights,Wthresh)

                beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"]
                Wthresh = self.params_dict["config_dict"]["W4to4_params"].get("Wthresh",None)
                q_post_dict["p_e_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][0,:],freeze_weights,Wthresh)
                q_post_dict["p_i_l4"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][1,:],freeze_weights,Wthresh)

                init_weights = [ self.params_dict["init_weights"][1], \
                                          self.params_dict["init_weights_4to4"][1] ] #[  arr[lgne_e,lgn_i], arr[i_e,i_i]   ]
                freeze_weights = self.params_dict["config_dict"]["Wlgn_to4_params"].get("freeze_weights",True)

                # Build q_pre_dict
                constraint_mode = "ffrec_pre"
                mult_norm = "ffrec_pre"

                ff_beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
                if self.params_dict["config_dict"]["Wlgn_to4_params"].get("scale_beta",False):
                    ff_beta_P /= tf.reduce_sum(self.params_dict["arbor2"][:,0,:])
                rec_beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"]
                ff_Wthresh = self.params_dict["config_dict"]["Wlgn_to4_params"].get("Wthresh",None)
                rec_Wthresh = self.params_dict["config_dict"]["W4to4_params"].get("Wthresh",None)

                q_pre_dict["p_lgn_e_sep"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,ff_beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][0,:],freeze_weights,ff_Wthresh)
                q_pre_dict["p_lgn_i_sep"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,ff_beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[0][1,:],freeze_weights,ff_Wthresh)
                q_pre_dict["p_e_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,rec_beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][0,:],freeze_weights,rec_Wthresh)
                q_pre_dict["p_i_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,rec_beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][1,:],freeze_weights,rec_Wthresh)
                q_pre_dict["p_e_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,rec_beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][2,:],freeze_weights,rec_Wthresh)
                q_pre_dict["p_i_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,rec_beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                            Wlim,init_weights[1][3,:],freeze_weights,rec_Wthresh)

                q_dict['p_ffrec_sep'] = (q_post_dict,q_pre_dict)

            else:
                raise("q_dict mult_norm not implemented")

            self.p_dict = q_dict
            # ================================

        else:
            # ======== P_dict ================
            p_dict = defaultdict(lambda:None)

            dt = self.params_dict["config_dict"]["dt"]
            c_orth = self.params_dict["c_orth"]
            s_orth = self.params_dict["s_orth"]
            beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
            if self.params_dict["config_dict"]["Wlgn_to4_params"].get("scale_beta",False):
                beta_P /= tf.reduce_sum(self.params_dict["arbor2"][:2,0,:])
            Wlim = self.params_dict["config_dict"]["Wlgn_to4_params"]["Wlim"]
            Wthresh = self.params_dict["config_dict"]["Wlgn_to4_params"].get("Wthresh",None)
            plasticity_rule = self.params_dict["config_dict"]["Wlgn_to4_params"]["plasticity_rule"]
            constraint_mode = self.params_dict["config_dict"]["Wlgn_to4_params"]["constraint_mode"]
            mult_norm = self.params_dict["config_dict"]["Wlgn_to4_params"]["mult_norm"]
            clip_mode = self.params_dict["config_dict"]["Wlgn_to4_params"]["clip_weights"]
            weight_strength = self.params_dict["config_dict"]["Wlgn_to4_params"]["ampl"]
            init_weights = None

            if mult_norm in ("x","alpha","xalpha"):
                init_weights = self.params_dict["init_weights"][:2,:]
            elif mult_norm in ("xalpha_approx"):
                init_weights = [self.params_dict["init_weights"][0][:2,:],self.params_dict["init_weights"][1][:2,:]]
            freeze_weights = self.params_dict["config_dict"]["Wlgn_to4_params"].get("freeze_weights",True)

            p_dict["p_lgn_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                                plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength,\
                                Wlim,init_weights,freeze_weights,Wthresh)

            if self.connectivity_type=="EI":
                init_weights = None
                if mult_norm in ("x","alpha","xalpha"):
                    init_weights = self.params_dict["init_weights"][2:,:]
                elif mult_norm in ("xalpha_approx"):
                    init_weights = [self.params_dict["init_weights"][0][2:,:],self.params_dict["init_weights"][1][2:,:]]
                p_dict["p_lgn_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                                    plasticity_rule,constraint_mode,mult_norm,clip_mode,\
                                    weight_strength,Wlim,init_weights,freeze_weights,Wthresh)

            if self.params_dict["config_dict"]["W4to23_params"]["plasticity_rule"]!="None":
                c_orth = self.params_dict["c_orth_4to23"]
                s_orth = self.params_dict["s_orth_4to23"]
                beta_P = self.params_dict["config_dict"]["W4to23_params"]["beta_P"]
                weight_strength_e = self.params_dict["config_dict"]["W4to23_params"]["aEE"]
                weight_strength_i = self.params_dict["config_dict"]["W4to23_params"]["aEI"]
                Wlim = self.params_dict["config_dict"]["W4to23_params"]["Wlim"]
                Wthresh = self.params_dict["config_dict"]["W4to23_params"].get("Wthresh",None)
                init_weights = self.params_dict["init_weights_4to23"]
                freeze_weights = self.params_dict["config_dict"]["W4to23_params"].get("freeze_weights",True)
                N = init_weights.shape[0]//2

                plasticity_rule = self.params_dict["config_dict"]["W4to23_params"]["plasticity_rule"]
                constraint_mode = self.params_dict["config_dict"]["W4to23_params"]["constraint_mode"]
                mult_norm = self.params_dict["config_dict"]["W4to23_params"]["mult_norm"]
                clip_mode = self.params_dict["config_dict"]["W4to23_params"]["clip_weights"]
                p_dict["p_4to23_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                                    plasticity_rule,constraint_mode,mult_norm,clip_mode,\
                                    weight_strength_e,Wlim,init_weights[:N],freeze_weights,Wthresh)
                p_dict["p_4to23_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                                    plasticity_rule,constraint_mode,mult_norm,clip_mode,\
                                    weight_strength_i,Wlim,init_weights[N:],freeze_weights,Wthresh)

            if self.params_dict["config_dict"]["W4to4_params"]["plasticity_rule"]!="None":
                Nl4 = tf.shape(self.params_dict["arbor4to4_full"])[0] // 2
                c_orth = self.params_dict["c_orth_4to4"]
                s_orth = self.params_dict["s_orth_4to4"]
                beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"]
                plasticity_rule = self.params_dict["config_dict"]["W4to4_params"]["plasticity_rule"]
                constraint_mode = self.params_dict["config_dict"]["W4to4_params"]["constraint_mode"]
                mult_norm = self.params_dict["config_dict"]["W4to4_params"]["mult_norm"]
                clip_mode = self.params_dict["config_dict"]["W4to4_params"]["clip_weights"]
                init_weights = None
                if "init_weights_4to4" in self.params_dict:
                    if mult_norm in ("postx","prex","postprex"):
                        init_weights = self.params_dict["init_weights_4to4"]
                    init_weights_ee = init_weights[0]
                    init_weights_ie = init_weights[2]
                    init_weights_ei = init_weights[1]
                    init_weights_ii = init_weights[3]
                weight_strength_ei = self.params_dict["config_dict"]["W4to4_params"]["aEI"]
                weight_strength_ii = self.params_dict["config_dict"]["W4to4_params"]["aII"]
                Wlim = self.params_dict["config_dict"]["W4to4_params"]["Wlim"]
                Wthresh = self.params_dict["config_dict"]["W4to4_params"].get("Wthresh",None)
                freeze_weights = self.params_dict["config_dict"]["W4to4_params"].get("freeze_weights",True)
                if self.params_dict["config_dict"]["W4to4_params"].get("scale_beta",False):
                    beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"] /\
                            tf.reduce_sum(self.params_dict["arbor4to4_full"][0,Nl4:])
                p_dict["p_rec4_ei"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength_ei,\
                            Wlim,init_weights_ei,freeze_weights,Wthresh)

                if self.connectivity_type=="EI":
                    if self.params_dict["config_dict"]["W4to4_params"].get("scale_beta",False):
                        beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"] /\
                                tf.reduce_sum(self.params_dict["arbor4to4_full"][Nl4,Nl4:])
                    p_dict["p_rec4_ii"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                                plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength_ii,\
                                Wlim,init_weights_ii,freeze_weights,Wthresh)
                if self.params_dict["config_dict"]["W4to4_params"].get("e_plasticity",False):
                    weight_strength_ee = self.params_dict["config_dict"]["W4to4_params"]["aEE"]
                    weight_strength_ie = self.params_dict["config_dict"]["W4to4_params"]["aIE"]
                    if self.params_dict["config_dict"]["W4to4_params"].get("scale_beta",False):
                        beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"] /\
                                tf.reduce_sum(self.params_dict["arbor4to4_full"][0,:Nl4])
                    p_dict["p_rec4_ee"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                                plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength_ee,\
                                Wlim,init_weights_ee,freeze_weights,Wthresh)
                    if self.connectivity_type=="EI":
                        if self.params_dict["config_dict"]["W4to4_params"].get("scale_beta",False):
                            beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"] /\
                                    tf.reduce_sum(self.params_dict["arbor4to4_full"][Nl4,:Nl4])
                        p_dict["p_rec4_ie"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                                    plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength_ie,\
                                    Wlim,init_weights_ie,freeze_weights,Wthresh)

            if self.params_dict["config_dict"]["W23_params"]["plasticity_rule"]!="None":
                c_orth = None
                s_orth = None
                beta_P = self.params_dict["config_dict"]["W23_params"]["beta_P"]
                plasticity_rule = self.params_dict["config_dict"]["W23_params"]["plasticity_rule"]
                constraint_mode = self.params_dict["config_dict"]["W23_params"]["constraint_mode"]
                mult_norm = self.params_dict["config_dict"]["W23_params"]["mult_norm"]
                clip_mode = self.params_dict["config_dict"]["W4to4_params"]["clip_weights"]
                weight_strength_ei = self.params_dict["config_dict"]["W23_params"]["aEI"]
                weight_strength_ii = self.params_dict["config_dict"]["W23_params"]["aII"]
                freeze_weights = self.params_dict["config_dict"]["W23_params"].get("freeze_weights",True)
                p_dict["p_rec23_ei"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength_ei,freeze_weights)
                p_dict["p_rec23_ii"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
                            plasticity_rule,constraint_mode,mult_norm,clip_mode,weight_strength_ii,freeze_weights)
            self.p_dict = p_dict
            # ===============================================

            # =============== Q_dict ========================

            # ===============================================


    def update_developing_parameters(self,timestep):
        """
        update before (!) each plasticity update step all parameters
        that are changing during development,
        e.g. beta, off_bias_strength, off_bias_region_size,onoff_rel_weight
        arbor parameters,
        """

        for item in self.params_dict["config_dict"]["developing_params"]:

            if item=="learning_rate":
                self.params_dict["config_dict"]["W4to4_params"]["beta_P"] *=\
                 np.exp(-1./self.params_dict["config_dict"]["learning_rate_decay_const"])

                self.params_dict["config_dict"]["W23_params"]["beta_P"] *=\
                 np.exp(-1./self.params_dict["config_dict"]["learning_rate_decay_const"])

                self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"] *=\
                 np.exp(-1./self.params_dict["config_dict"]["learning_rate_decay_const"])


    def integrate(self, l4_dynamics, y0, t, dt):
        ## NETWORK PARAMETERS
        Nret = self.params_dict["Nret"]
        N4 = self.params_dict["N4"]
        Nlgn = self.params_dict["Nlgn"]
        Nvert = self.params_dict["Nvert"]
        lim = Nlgn*Nlgn*N4*N4*Nvert
        l4_size = N4**2*Nvert*self.num_pop
        pop_size = N4**2*Nvert
        crt_size = N4**2*Nvert*self.num_pop
        if self.system=="two_layer":
            crt_size = N4**2*Nvert*self.num_pop + self.params_dict["N23"]**2*self.num_pop

        arbor2 = self.params_dict["arbor2"]
        arbor4to23 = self.params_dict["arbor4to23_full"]
        # arbor2 = tf.stack([arbor]*2)

        init_weights = self.params_dict["init_weights"]
        Wlim = self.params_dict["config_dict"]["Wlgn_to4_params"]["Wlim"]
        num_lgn_paths = self.params_dict["config_dict"]["num_lgn_paths"]

        T_pd = self.params_dict["config_dict"]["Inp_params"]["pattern_duration"]
        # mode = self.params_dict["normalisation_mode"]
        c_orth = self.params_dict["c_orth"]
        s_orth = self.params_dict["s_orth"]
        c_orth_4to23 = self.params_dict["c_orth_4to23"]
        s_orth_4to23 = self.params_dict["s_orth_4to23"]


        def scan_func(func, y, t_dt, **kwargs):
            t, dt = t_dt
            dy = self._step_func(func, y, t, dt, **kwargs)
            out = y + dy
            return out

        ## init norm
        Wlgn_to_4 = y0[:lim*num_lgn_paths].numpy().reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
        init_norm_alpha = np.nansum(Wlgn_to_4,axis=(0,2))
        init_norm_x = np.nansum(Wlgn_to_4,axis=1)
        # print("init_sum",init_norm_alpha[:5],init_norm_x[0,:5])
        H = tf.convert_to_tensor(np.ones((self.num_pop,N4**2)),dtype=tf.float32)


        ## init activity
        y = y0
        yt = tf.convert_to_tensor([y0])
        time_dep_dict = defaultdict(list)
        time_dep_dict["l4t"].append(y0[num_lgn_paths*lim:num_lgn_paths*lim+crt_size])
        if self.p_dict["p_rec4_ii"] is not None or self.p_dict["p_on_l4"] is not None or\
                self.p_dict["p_lgne_e"] is not None or self.p_dict["p_ffrec"] is not None or\
                self.p_dict["p_ffrec_sep"] is not None:
            time_dep_dict["W4to4t"].append(self.params_dict["W4to4"].numpy())
        if self.p_dict["p_rec23_ii"] is not None:
            time_dep_dict["W23to23t"].append(self.params_dict["W23to23"].numpy())
        if self.adaptive:
            time_dep_dict["l4_avgt"].append(self.params_dict["l4_avg"].numpy())
            time_dep_dict["theta_4t"].append(self.params_dict["theta_4"].numpy())

        tf.random.set_seed(self.params_dict["config_dict"]["random_seed"]*113)

        print("Number of plasticity steps: {}".format(self.num_plasticity_steps))
        for istep in range(self.num_plasticity_steps):
            dW = defaultdict(lambda:0)

            self.update_developing_parameters(istep)
            y = self.yield_init_conditions(y)

            # shift = tf.random.categorical(tf.math.log([[0.5,0.5]]),2) * 2 - 1
            # shift_y,shift_x = shift[0,0],shift[0,1]
            for jexp in range(self.expanse_steps):

                # average plasticity step over avg_no_inp input patterns
                for kinput in range(self.avg_no_inp):

                    ####### generate LGN input per L4 simulation #######
                    rng_seed = self.params_dict["config_dict"]["random_seed"]*1000 +\
                               kinput + istep*self.avg_no_inp
                    inp = inputs.Inputs_lgn((Nret,Nret),self.params_dict["Version"].numpy(),\
                            rng_seed)
                    lgn = inp.create_lgn_input(\
                          self.params_dict["config_dict"]["Inp_params"],\
                          self.params_dict["config_dict"]["Inp_params"]["input_type"],\
                          self.params_dict["Wret_to_lgn"].numpy(),\
                          expansion_timestep = jexp,
                          )
                    lgn = inp.apply_ONOFF_bias(lgn,self.params_dict["config_dict"]["Inp_params"])
                    ## use same inputs for E and I units
                    if num_lgn_paths==4:
                        lgn = np.concatenate([lgn,lgn])
                    lgn = tf.convert_to_tensor(lgn.reshape(num_lgn_paths,Nlgn**2),dtype=tf.float32)
                    print("lgn",lgn.shape,lgn[0,0])

                    print("")
                    print("Plasticity step: {}, exp: {}, input: {}".format(istep,jexp,kinput))
                    sys.stdout.flush()

                    # shift input pattern
                    self.params_dict.update({"lgn" : lgn})
                    # lgn = tf.reshape(self.params_dict["lgn"],[-1,Nlgn,Nlgn])
                    # lgn = tf.roll(tf.roll(lgn,shift_y,axis=1),shift_x,axis=2)
                    # self.params_dict.update({"lgn" : tf.reshape(lgn,[-1,Nlgn**2])})

                    print("Comment out next line if running with antolik_etal settings")
                    if self.expanse_steps<2:
                        y = self.yield_init_conditions(y)


                    # L4 activity dynamics
                    y,l4_avg,theta_4 = l4_dynamics(scan_func,y,t,dt,self.params_dict,\
                                        rhs_l4=self.rhs_l4,rhs_l23=self.rhs_l23,\
                                        I_crt=self.I_crt,I_crt23=self.I_crt23)
                    self.params_dict["theta_4"] = theta_4
                    self.params_dict["l4_avg"] = l4_avg
                    l4 = y[lim*num_lgn_paths:num_lgn_paths*lim+crt_size]
                    if self.connectivity_type=="EI":
                        print("L4",istep,np.nanmin(l4[:crt_size//2]),np.nanmax(l4[:crt_size//2]),\
                            np.nanmin(l4[crt_size//2:]),np.nanmax(l4[crt_size//2:]),\
                            np.nanmin(theta_4[:crt_size//2]),np.nanmax(theta_4[:crt_size//2]),\
                            np.nanmin(theta_4[crt_size//2:]),np.nanmax(theta_4[crt_size//2:]),\
                            np.nanmin(l4_avg[:crt_size//2]),np.nanmax(l4_avg[:crt_size//2]),\
                            np.nanmin(l4_avg[crt_size//2:]),np.nanmax(l4_avg[crt_size//2:]))
                    else:
                        print("L4",istep,np.nanmin(l4),np.nanmax(l4),\
                             np.nanmin(theta_4),np.nanmax(theta_4),np.nanmin(l4_avg),np.nanmax(l4_avg))

                    # compute running average of l4 activity
                    self.running_avg(y[lim*num_lgn_paths:num_lgn_paths*lim+l4_size])

                    # save timestep of cortical activity
                    if ((istep%self.saving_stepsize)==0 or istep==(self.num_plasticity_steps-1)):
                        time_dep_dict["l4t"].append(y[lim*num_lgn_paths:num_lgn_paths*lim+crt_size].numpy())

                    # check whether cortical activity has diverged, if so break
                    if np.nanmean(y[lim*num_lgn_paths:])>self.divergence_limit:
                        print("break")
                        break

                    # compute target activity for homeostatic normalisation
                    if self.l4_target is None:
                        self.l4_target = self.avg_ampl*tf.reduce_mean(tf.reshape(\
                                         y[lim*num_lgn_paths:num_lgn_paths*lim+l4_size],[self.num_pop,-1]),\
                                         axis=1)

                    # ff plasticity update
                    # dW += self.plastic.plasticity_wrapper(y,0)
                    # dW = tf.reshape(dW, [s])
                    timestep = istep * self.expanse_steps + jexp
                    l4 = y[lim*num_lgn_paths:num_lgn_paths*lim+l4_size]
                    l23 = y[lim*num_lgn_paths+l4_size:num_lgn_paths*lim+crt_size]
                    Wlgn_to_4 = tf.reshape(y[:lim*num_lgn_paths],[num_lgn_paths,pop_size,-1])
                    dW_dict = plasticity_dynamics.unconstrained_plasticity_wrapper(self.p_dict,\
                                l4, l23, lgn, Wlgn_to_4, self.params_dict["W4to4"],\
                                self.params_dict["W4to23"], self.params_dict["W23to23"], timestep)
                    for key,value in dW_dict.items():
                        dW[key] += value
                        pre_size = value.shape[-1]
                        if self.connectivity_type=="EI":
                            print("key",key,istep,np.nanmin(value[...,:pre_size//2]),np.nanmax(value[...,:pre_size//2]),
                                 np.nanmin(value[...,pre_size//2:]),np.nanmax(value[...,pre_size//2:]))
                        else:
                            print("key",key,istep,np.nanmin(value),np.nanmax(value))

                try:
                    print("CHECK SHAPE2",dW["dW_lgn_e"].shape)
                    if isinstance(dW["dW_lgn_e"],int):
                        print("int break")
                        break
                    elif np.isnan(np.nanmedian(dW["dW_lgn_e"][tf.reshape(arbor2[:2,:,:],[-1])>0])):
                        print("Median plasticity update is NaN at t={}, exit!".format(kinput))
                        break
                except:
                    pass


                # Plasticity update averaged over avg_no_inp number of input patterns
                s = num_lgn_paths*lim
                Wlgn_to_4 = tf.reshape(y[:s],[num_lgn_paths,pop_size,Nlgn**2])
                Wlgn_to_4,W4to4,W4to23,W23to23 =\
                 plasticity_dynamics.constraint_update_wrapper(dW,self.p_dict,Wlgn_to_4,\
                                                            self.params_dict["arbor2"],\
                                                            self.params_dict["W4to4"],#change to updated W4to4\
                                                            self.params_dict["arbor4to4_full"],\
                                                            self.params_dict["W4to23"],\
                                                            self.params_dict["arbor4to23_full"],\
                                                            self.params_dict["W23to23"],\
                                                            self.params_dict["arbor23to23_full"],\
                                                            dt,self.params_dict)
                
                # synaptic normalisation
                Wlgn_to_4,W4to4,W4to23,W23to23 =\
                    plasticity_dynamics.prune_weights_wrapper(self.p_dict,Wlgn_to_4,\
                                                        self.params_dict["arbor2"],\
                                                        self.params_dict["W4to4"],\
                                                        self.params_dict["arbor4to4_full"],\
                                                        self.params_dict["W4to23"],\
                                                        self.params_dict["arbor4to23_full"],\
                                                        self.params_dict["W23to23"],\
                                                        self.params_dict["arbor23to23_full"],\
                                                        self.params_dict)

                for rep in range(4):
                    # synaptic normalisation
                    Wlgn_to_4,W4to4,W4to23,W23to23 =\
                     plasticity_dynamics.clip_weights_wrapper(self.p_dict,Wlgn_to_4,\
                                                            self.params_dict["arbor2"],\
                                                            self.params_dict["W4to4"],\
                                                            self.params_dict["arbor4to4_full"],\
                                                            self.params_dict["W4to23"],\
                                                            self.params_dict["arbor4to23_full"],\
                                                            self.params_dict["W23to23"],\
                                                            self.params_dict["arbor23to23_full"],\
                                                            self.params_dict)

                    if True:#check_for_frozen_weights(Wlgn_to_4,Wlim,arbor2):
                        Wlgn_to_4,W4to4,W4to23,W23to23,H =\
                         plasticity_dynamics.mult_norm_wrapper(self.p_dict,Wlgn_to_4,\
                                                            self.params_dict["arbor2"],\
                                                            self.params_dict["W4to4"],\
                                                            self.params_dict["arbor4to4_full"],\
                                                            self.params_dict["W4to23"],\
                                                            self.params_dict["arbor4to23_full"],\
                                                            self.params_dict["W23to23"],\
                                                            self.params_dict["arbor23to23_full"],\
                                                            H,self.running_l4_avg,self.l4_target,\
                                                            self.params_dict)

                # for i in range(20):
                # 	norm = tf.reduce_sum(Wlgn_to_4,axis=(0,2))
                # 	norm = tf.where(tf.equal(norm, 0), tf.ones_like(norm), norm)
                # 	Wlgn_to_4 /= norm[None,:,None]
                # 	norm = tf.reduce_sum(Wlgn_to_4,axis=1)
                # 	norm = tf.where(tf.equal(norm, 0), tf.ones_like(norm), norm)
                # 	Wlgn_to_4 /= norm[:,None,:]
                # 	Wlgn_to_4 = tf.clip_by_value(Wlgn_to_4,0,Wlim*arbor2)

                # check fraction of synaptic ff weights are frozen
                frozen = np.sum(tf.boolean_mask(Wlgn_to_4,arbor2)<=0) + \
                         np.sum(tf.boolean_mask(Wlgn_to_4,arbor2)>=Wlim*(arbor2[arbor2>0]))
                if frozen>(0.9*np.sum(arbor2.numpy())):
                    self.frozen = True
                    print("0.9 of ff weights are frozen at t={}, exit integration!".format(kinput))
                    break
                Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim*num_lgn_paths])


                ## if w4to23 is plastic do multiplicative normalisation
                # if self.params_dict["config_dict"]["W4to23_params"]["plastic"]:
                # 	l23_size = self.params_dict["N23"]**2*2
                # 	W4to23 = y[s+crt_size:]
                # 	arbor4to23_full = self.params_dict["arbor4to23_full"]
                # 	W4to23 = dynamics.constrain_plasticity_update_W4to23(y,dt,dW[s:],arbor4to23,\
                # 													arbor4to23_full,Wlim,\
                # 													self.params_dict,c_orth_4to23,\
                # 													s_orth_4to23)
                # 	W4to23 = self.plastic.clip_weights(W4to23,Wlim,arbor4to23_full)
                # 	init_weights_4to23 = self.params_dict["init_weights_4to23"]
                # 	W4to23,_ = synaptic_normalization(W4to23,arbor4to23_full,Wlim,init_weights_4to23,\
                # 									axis=0)
                # 	W4to23 = tf.reshape(W4to23,[l23_size*l4_size])
                # 	####### concatenate updated connectivity to y vector #######
                # 	y = tf.concat([Wlgn_to_4,y[lim*num_lgn_paths:lim*num_lgn_paths+crt_size],\
                # 					W4to23],0)
                # else:
                y = tf.concat([Wlgn_to_4,y[lim*num_lgn_paths:]],0)
                # if self.params_dict["config_dict"]["W4to23_params"]["plasticity_rule"]!="None":
                # 	y = tf.concat([y[:lim*num_lgn_paths+crt_size],\
                # 		tf.reshape(self.params_dict["W4to23"],[-1])],0)


                if ((istep%self.saving_stepsize)==0 or istep==(self.num_plasticity_steps-1)):
                    yt = tf.concat([yt,[y]], 0)
                    if self.p_dict["p_rec4_ii"] is not None or self.p_dict["p_on_l4"] is not None or\
                            self.p_dict["p_lgne_e"] is not None or self.p_dict["p_ffrec"] is not None or\
                            self.p_dict["p_ffrec_sep"] is not None:
                        time_dep_dict["W4to4t"].append(self.params_dict["W4to4"])
                    if self.p_dict["p_rec23_ii"] is not None:
                        time_dep_dict["W23to23t"].append(self.params_dict["W23to23"])
                    if self.adaptive:
                        time_dep_dict["l4_avgt"].append(self.params_dict["l4_avg"].numpy())
                        time_dep_dict["theta_4t"].append(self.params_dict["theta_4"].numpy())

        return yt,time_dep_dict


    def _step_func(self, func, y, t, dt, **kwargs):
        if self.scheme=="Euler":
            dt_cast = tf.cast(dt, y.dtype)
            dy = dt_cast*func(y, t, **kwargs)
        elif self.scheme=="RK4":
            k1 = func(y, t, **kwargs)
            half_step = t + dt/2.
            dt_cast = tf.cast(dt, y.dtype)
            k2 = func(y + dt_cast * k1 / 2., half_step, **kwargs)
            k3 = func(y + dt_cast * k2 / 2., half_step, **kwargs)
            k4 = func(y + dt_cast * k3, t + dt, self.params_dict, **kwargs)
            dy = tf.add_n([k1, 2*k2, 2*k3, k4]) * (dt_cast / 6.)
        return dy


def odeint_new(func1, y0, t, dt, params_dict, mode="dynamic"):
    t = tf.convert_to_tensor(t, name='t', dtype=tf.float32)
    y0 = tf.convert_to_tensor(y0, name='y0', dtype=tf.float32)
    dt = tf.convert_to_tensor(dt,name="dt", dtype=tf.float32)
    tf_check_type(t, y0)
    if mode=="dynamic":
        return Tf_integrator_new(params_dict).integrate(func1,y0,t,dt)
    elif mode=="static":
        return Tf_integrator_new(params_dict).integrate(func1,y0,t,dt)
    elif mode=="single_stim_update":
        print("mode single_stim_update not implemented yet")
        return None




class Tf_integrator:
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.scheme = params_dict["integrator"]
        self.avg_no_inp = params_dict["avg_no_inp"]
        self.frozen = False

    def integrate(self, func, y0, t, dt):
        # time_delta_grid = dt*tf.ones(t.shape,dtype=tf.float32)

        def scan_func(y, t_dt, yt):
            t, dt = t_dt
            dy = self._step_func(func, y, t, dt, yt)
            out = y + dy

            T_pd = self.params_dict["pattern_duration"]


            ## update ff weights after activity has converged for specific input
            # if (t%T_pd)==(T_pd-1):
            if ((((t+1)/T_pd)%self.avg_no_inp)==1 and t>0):
                N4 = self.params_dict["N4"]
                Nlgn = self.params_dict["Nlgn"]
                Nvert = self.params_dict["Nvert"]
                arbor = self.params_dict["arbor"]
                lim = 2*Nlgn*Nlgn*N4*N4*Nvert

                weight_change = dy[:lim].numpy()
                arbor2 = np.stack([arbor,arbor]).flatten()
                init_weights = self.params_dict["init_weights"]
                Wlim = self.params_dict["Wlim"]
                Wlgn_to_4 = out[:lim]
                Wlgn_to_4 = tf.reshape(Wlgn_to_4,[2,N4*N4*Nvert,Nlgn*Nlgn])
                Wlgn_to_4 = clip_weights(Wlgn_to_4,Wlim,arbor)

                Wlgn_to_4 = synaptic_normalization(Wlgn_to_4,arbor,Wlim,init_weights)
                Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim])
                if (np.sum(Wlgn_to_4[arbor2>0]==0)+np.sum(Wlgn_to_4[arbor2>0]==Wlim))>\
                   (1.8*np.sum(arbor.numpy())):
                    self.frozen = True
                out = tf.concat([Wlgn_to_4,out[lim:]],0)
            # if (t%T_pd)==(T_pd-1):
            # 	out = tf.concat([out[:lim],0*out[lim:]],0)

            return out

        # y = tf.scan(scan_func, (t, time_delta_grid), y0, swap_memory=True)
        # y = y[::100,:]

        y = y0
        yt = tf.convert_to_tensor([y0])
        N4 = self.params_dict["N4"]
        Nlgn = self.params_dict["Nlgn"]
        Nvert = self.params_dict["Nvert"]
        lim = 2*Nlgn*Nlgn*N4*N4*Nvert
        l4t = tf.convert_to_tensor([y0[lim:]])
        # cc_lgn_l4_t = []
        T_pd = self.params_dict["pattern_duration"]
        for it in t:
            if self.frozen:
                print("0.9 of all ff weights are frozen at t={}, exit integration!".format(it))
                break
            y = scan_func(y, (it, dt), yt)

            ## set yt to y again to only store values since last W update
            if ((((it+1)/T_pd)%self.avg_no_inp)==0 and it>0):
                no_update_step = int(((it+1)/T_pd) / self.avg_no_inp)
                print(no_update_step,int(tf.math.floor(it/T_pd)));sys.stdout.flush()
                # yt = tf.convert_to_tensor([y0])
                # for jt in range(no_update_step):
                # 	print("jt",jt,yt[jt,10])
                # 	yt = tf.concat([yt,[yt[jt,:]]], 0)
                yt = yt[:no_update_step,:]

                # if False:
                # 	print("l4t",(l4t.numpy()).shape)
                # 	import matplotlib.pyplot as plt
                # 	fig = plt.figure()
                # 	ax = fig.add_subplot(121)
                # 	ax.plot(np.nanmean(l4t,axis=1),'-k')
                # 	ax.plot(l4t[:,100],'-m')
                # 	ax = fig.add_subplot(122)
                # 	im=ax.imshow((l4t[-500,:N4**2].numpy()).reshape(N4,N4),interpolation="nearest",cmap="binary")
                # 	plt.colorbar(im,ax=ax)
                # 	plt.savefig(image_dir + \
                # 		"layer4/tmp/v96_rA2.0_srec1.00_scc0.04_xalpha_AVG_test/activity_pattern/act_{}_ev95_t-1.pdf".format(it))
                # 	plt.close()

                # print("set yt to zero again",yt.shape)

            ## store values of l4 act and W as fct of timesteps
            ## TODO: only keep values of l4 act (needed for plasticity step)
            if (it%T_pd)==(T_pd-1):
                # print("append yt",it,yt.shape)
                yt = tf.concat([yt,[y]], 0)
                l4t = tf.concat([l4t,[y[lim:]]],0)

        no_update_step = int(np.ceil(((it+1)/T_pd) / self.avg_no_inp))
        yt = yt[:no_update_step+1,:]

        return yt,l4t

    def _step_func(self, func, y, t, dt, yt):
        if self.scheme=="Euler":
            dy = dt*func(y, t, self.params_dict, yt)
        elif self.scheme=="RK4":
            k1 = func(y, t, self.params_dict)
            half_step = t + dt/2.
            dt_cast = tf.cast(dt, y.dtype)
            k2 = func(y + dt_cast * k1 / 2., half_step, self.params_dict)
            k3 = func(y + dt_cast * k2 / 2., half_step, self.params_dict)
            k4 = func(y + dt_cast * k3, t + dt, self.params_dict)
            dy = tf.add_n([k1, 2*k2, 2*k3, k4]) * (dt_cast / 6.)
        return dy


    def integrate_sgl(self,func,y0,t,dt):
        """use this function if plasticity update is done after each new stimulus,
           and time dependence of W and L4 is not saved"""
        def scan_func_sgl(y, t_dt):
            t, dt = t_dt
            dy = self._step_func(func, y, t, dt, None)
            # print("out",y.shape,dy.shape)
            out = y + dy

            T_pd = self.params_dict["pattern_duration"]
            ## update ff weights after activity has converged for specific input
            if ((((t+1)/T_pd)%self.avg_no_inp)==1 and t>0):
                N4 = self.params_dict["N4"]
                Nlgn = self.params_dict["Nlgn"]
                Nvert = self.params_dict["Nvert"]
                arbor = self.params_dict["arbor"]
                lim = 2*Nlgn*Nlgn*N4*N4*Nvert

                weight_change = dy[:lim].numpy()
                arbor2 = np.stack([arbor,arbor]).flatten()
                init_weights = self.params_dict["init_weights"]
                Wlim = self.params_dict["Wlim"]
                Wlgn_to_4 = out[:lim]
                Wlgn_to_4 = tf.reshape(Wlgn_to_4,[2,N4*N4*Nvert,Nlgn*Nlgn])
                Wlgn_to_4 = clip_weights(Wlgn_to_4,Wlim,arbor)

                Wlgn_to_4 = synaptic_normalization(Wlgn_to_4,arbor,Wlim,init_weights)
                Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim])
                if (np.sum(Wlgn_to_4[arbor2>0]==0)+np.sum(Wlgn_to_4[arbor2>0]==Wlim))>\
                   (1.8*np.sum(arbor.numpy())):
                    self.frozen = True
                out = tf.concat([Wlgn_to_4,out[lim:]],0)
            return out

        y = y0
        N4 = self.params_dict["N4"]
        Nlgn = self.params_dict["Nlgn"]
        Nvert = self.params_dict["Nvert"]
        lim = 2*Nlgn*Nlgn*N4*N4*Nvert
        T_pd = self.params_dict["pattern_duration"]
        for it in t:
            if self.frozen:
                print("0.9 of all ff weights are frozen at t={}, exit integration!".format(it))
                break
            y = scan_func_sgl(y, (it, dt))

        return y

    def integrate_static(self, func, y0, t, dt):
        """use this function if layer 4 activity is not simulated"""
        def scan_func_static(y, t_dt, yt):
            t, dt = t_dt
            dy = self._step_func(func, y, t, dt, yt)
            # print("out",y.shape,dy.shape)
            out = y + dy

            ## update ff weights after activity has converged for specific input
            if (((t+1)%self.avg_no_inp)==0 and t>0):
                N4 = self.params_dict["N4"]
                Nlgn = self.params_dict["Nlgn"]
                Nvert = self.params_dict["Nvert"]
                arbor = self.params_dict["arbor"]
                lim = 2*Nlgn*Nlgn*N4*N4*Nvert

                init_weights = self.params_dict["init_weights"]
                Wlim = self.params_dict["Wlim"]
                Wlgn_to_4 = out
                Wlgn_to_4 = tf.reshape(Wlgn_to_4,[2,N4*N4*Nvert,Nlgn*Nlgn])
                Wlgn_to_4 = clip_weights(Wlgn_to_4,Wlim,arbor)

                Wlgn_to_4 = synaptic_normalization(Wlgn_to_4,arbor,Wlim,init_weights)
                Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim])
                arbor2 = np.stack([arbor,arbor]).flatten()
                if (np.sum(Wlgn_to_4[arbor2>0]==0)+np.sum(Wlgn_to_4[arbor2>0]==Wlim))>\
                    (1.8*np.sum(arbor.numpy())):
                    self.frozen = True
                out = Wlgn_to_4

            return out

        y = y0
        yt = tf.convert_to_tensor([y0])
        # N4 = self.params_dict["N4"]
        # Nlgn = self.params_dict["Nlgn"]
        # Nvert = self.params_dict["Nvert"]
        # lim = 2*Nlgn*Nlgn*N4*N4*Nvert
        for it in t:
            if self.frozen:
                print("0.9 of all ff weights are frozen at t={}, exit integration!".format(it))
                break
            y = scan_func_static(y, (it, dt), yt)

            ## set yt to y again to only store values since last W update
            if (((it+1)%self.avg_no_inp)==0 and it>0):
                print("it","append",yt.shape);sys.stdout.flush()
                yt = tf.concat([yt,[y]], 0)

        return yt


def odeint(func, y0, t, dt, params_dict, mode="dynamic"):
    t = tf.convert_to_tensor(t, name='t')
    y0 = tf.convert_to_tensor(y0, name='y0')
    dt = tf.convert_to_tensor(dt,name="dt")
    tf_check_type(t, y0)
    if mode=="dynamic":
        return Tf_integrator(params_dict).integrate(func,y0,t,dt)
    elif mode=="static":
        return Tf_integrator(params_dict).integrate_static(func,y0,t,dt)
    elif mode=="single_stim_update":
        return Tf_integrator(params_dict).integrate_sgl(func,y0,t,dt)

    #tf.placeholder(tf.float32)

