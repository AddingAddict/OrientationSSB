import numpy as np
from scipy import linalg
import os

from dev_ori_sel_RF import connectivity, inputs, data_dir

class Network:

    def __init__(self, Version, config_dict, **kwargs):
        self.Version = Version
        self.config_dict = config_dict
        self.kwargs = kwargs
        try:
            self.verbose = kwargs['verbose']
        except:
            self.verbose = True

        self.Nret = config_dict["Nret"]
        self.Nlgn = config_dict["Nlgn"]
        self.N4 = config_dict["N4"]
        self.N23 = config_dict["N23"]
        self.Nvert = config_dict["Nvert"]
        self._init_connectivity()


    def _init_connectivity(self):
        ## retina to lgn connectivity
        ## moving sinusoidal input of varying orientation and spatial frequency
        self.Wret_to_lgn,_ = connectivity.Connectivity((self.Nret,self.Nret),(self.Nlgn,self.Nlgn),\
                          random_seed=self.config_dict["random_seed"],Nvert=1, verbose=self.verbose).create_matrix(\
                          self.config_dict["Wret_to_lgn_params"],\
                          self.config_dict["Wret_to_lgn_params"]["profile"])
        self.Wret_to_lgn *= self.config_dict["Wret_to_lgn_params"]["gamma_ret"]

        if self.config_dict["Wret_to_lgn_params"]["profile"]=="Gaussian_broadOFF":
            Wret_to_lgn_OFF,_ = connectivity.Connectivity((self.Nret,self.Nret),\
                                (self.Nlgn,self.Nlgn),\
                                random_seed=self.config_dict["random_seed"],Nvert=1, verbose=self.verbose).create_matrix(\
                                {"ampl" : self.config_dict["Wret_to_lgn_params"]["ampl"],\
                                "noise" : self.config_dict["Wret_to_lgn_params"]["noise"],\
                                "sigma" : 2*self.config_dict["Wret_to_lgn_params"]["sigma"]},\
                                "Gaussian")
            Wret_to_lgn_OFF *= self.config_dict["Wret_to_lgn_params"]["gamma_ret"]
            self.Wret_to_lgn = np.stack([self.Wret_to_lgn,Wret_to_lgn_OFF])


        ## lgn to l4 connectivity
        Wlgn4 = connectivity.Connectivity((self.Nlgn,self.Nlgn), (self.N4,self.N4),\
                                           random_seed=self.config_dict["random_seed"],\
                                           Nvert=(1,self.Nvert), verbose=self.verbose)
        self.Wlgnto4,self.arbor_on,self.arbor_off,self.arbor2 = \
            self.get_RFs(self.config_dict["Wlgn_to4_params"]["W_mode"],Wlgn4=Wlgn4,\
                                      system_mode=self.config_dict["system"],**self.kwargs)
        self.arbor2 = np.concatenate([self.arbor2]*(self.config_dict["num_lgn_paths"]//2))
        if (self.config_dict["Wlgn_to4_params"]["connectivity_type"]=="EI" and\
            self.config_dict["Wlgn_to4_params"]["W_mode"]!="load_from_external"):
            Wlgn4_I = connectivity.Connectivity((self.Nlgn,self.Nlgn), (self.N4,self.N4),\
                                                 random_seed=self.config_dict["random_seed"]+1,\
                                                 Nvert=(1,self.Nvert), verbose=self.verbose)
            Wlgnto4_I,_,_,_ = self.get_RFs(self.config_dict["Wlgn_to4_params"]["W_mode"],Wlgn4=Wlgn4_I,\
                                        system_mode=self.config_dict["system"], **self.kwargs)
            self.Wlgnto4 = np.concatenate([self.Wlgnto4,Wlgnto4_I])

        # recurrent connectivity
        W4 = connectivity.Connectivity((self.N4,self.N4), (self.N4,self.N4),\
                                        random_seed=self.config_dict["random_seed"],Nvert=self.Nvert,verbose=self.verbose)
        if "2pop" in self.config_dict["W4to4_params"]["Wrec_mode"]:
            if "load_from_external" in self.config_dict["W4to4_params"]["Wrec_mode"]:
                self.W4to4,self.arbor4to4 = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"],W4=W4,
                                            conn_type="EI_all",system_mode=self.config_dict["system"],**self.kwargs)
            else:
                W4to4_EE,arbor4to4_EE = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"].replace("2pop",""),
                                            W4=W4,conn_type="EE",system_mode=self.config_dict["system"],**self.kwargs)
                W4.rng = np.random.RandomState((self.config_dict["random_seed"]+1)*90)
                W4to4_IE,arbor4to4_IE = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"].replace("2pop",""),
                                            W4=W4,conn_type="IE",system_mode=self.config_dict["system"],**self.kwargs)
                W4.rng = np.random.RandomState((self.config_dict["random_seed"]+2)*90)
                W4to4_EI,arbor4to4_EI = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"].replace("2pop",""),
                                            W4=W4,conn_type="EI",system_mode=self.config_dict["system"],**self.kwargs)
                W4.rng = np.random.RandomState((self.config_dict["random_seed"]+3)*90)
                W4to4_II,arbor4to4_II = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"].replace("2pop",""),
                                            W4=W4,conn_type="II",system_mode=self.config_dict["system"],**self.kwargs)
                self.W4to4 = np.block([[W4to4_EE,-W4to4_EI],[W4to4_IE,-W4to4_II]])
                self.arbor4to4 = np.block([[arbor4to4_EE,arbor4to4_EI],[arbor4to4_IE,arbor4to4_II]])
        elif "EI1pop" in self.config_dict["W4to4_params"]["Wrec_mode"]:
            if "load_from_external" in self.config_dict["W4to4_params"]["Wrec_mode"]:
                self.W4to4,self.arbor4to4 = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"],W4=W4,
                                            conn_type="EI1pop_all",system_mode=self.config_dict["system"],**self.kwargs)
            else:
                W4to4_EE,arbor4to4_EE = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"].replace("2pop",""),
                                            W4=W4,conn_type="EE",system_mode=self.config_dict["system"],**self.kwargs)
                W4.rng = np.random.RandomState((self.config_dict["random_seed"]+1)*90)
                W4to4_EI,arbor4to4_EI = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"].replace("2pop",""),
                                            W4=W4,conn_type="EI",system_mode=self.config_dict["system"],**self.kwargs)
        else:
            self.W4to4,self.arbor4to4 = self.get_Wrec4(self.config_dict["W4to4_params"]["Wrec_mode"],W4=W4,
                                          conn_type=None,system_mode=self.config_dict["system"],**self.kwargs)
        if self.config_dict["Wlgn_to4_params"]["connectivity_type"]=="EI":
            num_pops = 2
        else:
            num_pops = 1

        # init normalization
        # syn norm over x
        if self.config_dict["Wlgn_to4_params"]["mult_norm"]=="x":
            self.init_weights = np.sum(self.Wlgnto4,axis=1)
        # syn norm over alpha
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="alpha":
            self.init_weights = np.sum(self.Wlgnto4,axis=2)
        # syn norm over x and alpha
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="xalpha":
            self.init_weights = None ## create in script, needs orth norm vectors
        # == ff and rec plasticity
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="ffrec_post":
            self.init_weights = np.stack( \
                ( np.sum(np.concatenate( [self.Wlgnto4[0,:,:],self.Wlgnto4[2,:,:]] ,0) , axis=0) , \
                  np.sum(np.concatenate( [self.Wlgnto4[1,:,:],self.Wlgnto4[3,:,:]] ,0), axis=0) )    )
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="ffrec_pre":
            N = self.W4to4.shape[1]//2
            self.init_weights = np.stack( \
                ( np.sum(self.Wlgnto4[:2,:,:],axis=(0,2)) + np.sum(self.W4to4[:N,:N],axis=1) , \
                    np.sum(self.Wlgnto4[2:,:,:],axis=(0,2)) + np.sum(self.W4to4[N:,:N],axis=1) )    )
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="ffrec_postpre_approx":
            N = self.W4to4.shape[1]//2
            self.init_weights = [np.stack( \
                ( np.sum(np.concatenate( [self.Wlgnto4[0,:,:],self.Wlgnto4[2,:,:]] ,0) , axis=0) , \
                  np.sum(np.concatenate( [self.Wlgnto4[1,:,:],self.Wlgnto4[3,:,:]] ,0), axis=0) )    ),
                  np.stack( \
                ( np.sum(self.Wlgnto4[:2,:,:],axis=(0,2)) + np.sum(self.W4to4[:N,:N],axis=1) , \
                    np.sum(self.Wlgnto4[2:,:,:],axis=(0,2)) + np.sum(self.W4to4[N:,:N],axis=1) )    )]
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="ffrec_postpre_approx_sep":
            N = self.W4to4.shape[1]//2
            self.init_weights = [np.stack( \
                ( np.sum(np.concatenate( [self.Wlgnto4[0,:,:],self.Wlgnto4[2,:,:]] ,0) , axis=0) , \
                  np.sum(np.concatenate( [self.Wlgnto4[1,:,:],self.Wlgnto4[3,:,:]] ,0), axis=0) )    ),
                  np.stack( \
                ( np.sum(self.Wlgnto4[:2,:,:],axis=(0,2)) , \
                    np.sum(self.Wlgnto4[2:,:,:],axis=(0,2))  )    )]
            
        # ========================
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="homeostatic":
            self.init_weights = np.array([]) ## not needed
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="divisive":
            self.init_weights = np.array([]) ## not needed
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"]=="None":
            self.init_weights = np.array([]) ## not needed


        # init normalization
        # syn norm over x
        if self.config_dict["W4to4_params"]["mult_norm"]=="postx":
            self.init_weights_4to4 = np.sum(self.W4to4.reshape(num_pops,self.N4**2*self.Nvert,
                num_pops,self.N4**2*self.Nvert).transpose(0,2,1,3).reshape(num_pops**2,self.N4**2*self.Nvert,self.N4**2*self.Nvert),axis=1)
        # syn norm over alpha
        elif self.config_dict["W4to4_params"]["mult_norm"]=="prex":
            self.init_weights_4to4 = np.sum(self.W4to4.reshape(num_pops,self.N4**2*self.Nvert,
                num_pops,self.N4**2*self.Nvert).transpose(0,2,1,3).reshape(num_pops**2,self.N4**2*self.Nvert,self.N4**2*self.Nvert),axis=2)
        # syn norm over x and alpha
        elif self.config_dict["W4to4_params"]["mult_norm"]=="postprex":
            self.init_weights_4to4 = None ## create in script, needs orth norm vectors
        elif self.config_dict["W4to4_params"]["mult_norm"]=="homeostatic":
            self.init_weights_4to4 = np.array([]) ## not needed
        elif self.config_dict["W4to4_params"]["mult_norm"]=="divisive":
            self.init_weights_4to4 = np.array([]) ## not needed
        # === ff and rec platicity
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"] =="ffrec_post":
            self.init_weights_4to4 = np.reshape(np.sum(self.W4to4,axis=0),(2,-1))
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"] =="ffrec_pre":
            self.init_weights_4to4 = np.reshape(np.sum(self.W4to4[:,N:],axis=1),(2,-1)) #[0,:] I to E and [1,:] I to I
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"] =="ffrec_postpre_approx":
            self.init_weights_4to4 = [np.reshape(np.sum(self.W4to4,axis=0),(2,-1)),
                                      np.reshape(np.sum(self.W4to4[:,N:],axis=1),(2,-1))]
        elif self.config_dict["Wlgn_to4_params"]["mult_norm"] =="ffrec_postpre_approx_sep":
            self.init_weights_4to4 = [np.reshape(np.sum(self.W4to4,axis=0),(2,-1)),
                                      np.stack( ( np.sum(self.W4to4[:N,:N],axis=1), \
                                                  np.sum(self.W4to4[:N,N:],axis=1), \
                                                  np.sum(self.W4to4[N:,:N],axis=1), \
                                                  np.sum(self.W4to4[N:,N:],axis=1) ) )  ] # EtoE, ItoE, EtoI, ItoI
        # =======================
        elif self.config_dict["W4to4_params"]["mult_norm"]=="None":
            self.init_weights_4to4 = np.array([]) ## not needed

        if self.config_dict["system"]=="one_layer":
            self.system = (self.Wret_to_lgn,self.Wlgnto4,self.arbor_on,self.arbor_off,self.arbor2,self.init_weights,
                self.W4to4,self.arbor4to4,self.init_weights_4to4)

        elif self.config_dict["system"]=="two_layer":
            N23 = self.config_dict["N23"]
            W4 = connectivity.Connectivity_2pop((self.N23,self.N23),(self.N23,self.N23),
                                                (self.N23,self.N23),(self.N23,self.N23),\
                                                random_seed=self.config_dict["random_seed"], verbose=self.verbose)
            Wrec_mode = self.config_dict["W23_params"]["Wrec_mode"]
            self.W23to23,self.arbor23 = W4.create_matrix_2pop(self.config_dict["W23_params"],Wrec_mode)
            ## not scaled to Nvert, because no Nvert, should be fine

            ## ======================== Afferent conn L4 to L23 ===================================
            ## ====================================================================================
            W4 = connectivity.Connectivity_2pop((self.N4,self.N4),(self.N4,self.N4),\
                                                (self.N23,self.N23),(self.N23,self.N23),\
                                                 random_seed=self.config_dict["random_seed"],\
                                                 Nvert=(self.Nvert,1), verbose=self.verbose)
            Wrec_mode = self.config_dict["W4to23_params"]["Wrec_mode"]
            r_A = None
            if self.config_dict["W4to23_params"]["plasticity_rule"]!="None":
                Wrec_mode = "initialize"
                r_A = self.config_dict["W4to23_params"]["r_A"]
            if (self.config_dict["Wlgn_to4_params"]["W_mode"]=="load_from_external" and\
                self.config_dict["W4to23_params"]["plasticity_rule"]!="None"):
                self.W4to23 = self.load_W4to23(**self.kwargs)
            else:
                self.W4to23,_ = W4.create_matrix_2pop(self.config_dict["W4to23_params"],Wrec_mode)

            # arbor for L4 to L23
            arbor_profile = self.config_dict["W4to23_params"]["arbor_profile"]
            r_A = self.config_dict["W4to23_params"]["r_A"]
            self.arbor4to23 = Wlgn4.create_arbor(radius=r_A,profile=arbor_profile)
            # init normalization
            # syn norm over receiving layer 2/3
            self.init_weights_4to23 = np.sum(self.W4to23,axis=0)
            ## ======================== Feedback conn L23 to L4 ===================================
            ## ====================================================================================
            W4 = connectivity.Connectivity_2pop((self.N23,self.N23),(self.N23,self.N23),\
                                                (self.N4,self.N4),(self.N4,self.N4),\
                                                random_seed=self.config_dict["random_seed"],\
                                                Nvert=(1,self.Nvert), verbose=self.verbose)
            Wrec_mode = self.config_dict["W23to4_params"]["Wrec_mode"]
            self.W23to4,_ = W4.create_matrix_2pop(self.config_dict["W23to4_params"],Wrec_mode)

            self.system = (self.Wret_to_lgn,self.Wlgnto4,self.arbor_on,self.arbor_off,\
                            self.arbor2,self.init_weights,self.W4to4,self.arbor4to4,self.W23to23,\
                            self.arbor23,self.W4to23,self.arbor4to23,self.init_weights_4to23,self.W23to4)



    def generate_inputs(self,**kwargs):
        """
        kwargs : for moving_grating_online:
                    num_freq, num_oris, Nsur, spat_frequencies, orientations
                for white_nosie_online:
                    full_lgn_output, last_timestep,
        """
        Version = self.Version
        Wret_to_lgn = self.Wret_to_lgn

        if self.verbose: print("mode in generate_inputs",self.config_dict["Inp_params"]["input_type"])
        if self.config_dict["Inp_params"]["input_type"]=="moving_grating_online":
            Wret_to_lgn = np.diagflat(np.ones(self.Nlgn**2,dtype=float))

            lgn_input_on,lgn_input_off = [],[]
            num_freq = kwargs["num_freq"]#3
            num_oris = len(kwargs["orientations"])
            Nsur = kwargs["Nsur"]## gives number of input patterns with diff phases
            spat_frequencies = kwargs["spat_frequencies"]#np.array([80,90,120])
            orientations = kwargs["orientations"]
            for spat_frequency in spat_frequencies:
                for orientation in orientations:
                    self.config_dict["Inp_params"]["spat_frequency"] = spat_frequency # vary between 60 and 120 (3 steps?)
                    self.config_dict["Inp_params"]["Nsur"] = Nsur
                    self.config_dict["Inp_params"]["orientation"] = orientation # vary in 8 steps
                    on_inp = inputs.Inputs_lgn((self.Nret,self.Nret),1,2020).create_lgn_input(\
                                                self.config_dict["Inp_params"],\
                                                "moving_grating_online", Wret_to_lgn)
                    off_inp = inputs.Inputs_lgn((self.Nret,self.Nret),1,2020).create_lgn_input(\
                                                 self.config_dict["Inp_params"],\
                                                 "moving_grating_online", -Wret_to_lgn)
                    lgn_input_on.append(on_inp)
                    lgn_input_off.append(off_inp)
            lgn_input_on = np.array(lgn_input_on)
            lgn_input_off = np.array(lgn_input_off)
            lgn = np.stack([lgn_input_on,np.array(lgn_input_off)])
            lgn = lgn.reshape(2,num_freq*num_oris*Nsur,-1)
            lgn = np.swapaxes(lgn,1,2)
            lgn = lgn.reshape(2,-1,num_freq,num_oris,Nsur)
            
        if "moving_grating_periodic_online" in self.config_dict["Inp_params"]["input_type"]:
            Wret_to_lgn = np.diagflat(np.ones(self.Nlgn**2,dtype=float))

            lgn_input_on,lgn_input_off = [],[]
            num_freq = kwargs["num_freq"]#3
            num_oris = len(kwargs["orientations"])
            Nsur = kwargs["Nsur"]## gives number of input patterns with diff phases
            spat_frequencies = kwargs["spat_frequencies"]#np.array([80,90,120])
            orientations = kwargs["orientations"]
            for spat_frequency in spat_frequencies:
                for orientation in orientations:
                    self.config_dict["Inp_params"]["spat_frequency"] = spat_frequency # vary between 60 and 120 (3 steps?)
                    self.config_dict["Inp_params"]["Nsur"] = Nsur
                    self.config_dict["Inp_params"]["orientation"] = orientation # vary in 8 steps
                    on_inp = inputs.Inputs_lgn((self.Nret,self.Nret),1,2020).create_lgn_input(\
                                                self.config_dict["Inp_params"],\
                                                self.config_dict["Inp_params"]["input_type"], Wret_to_lgn)
                    off_inp = inputs.Inputs_lgn((self.Nret,self.Nret),1,2020).create_lgn_input(\
                                                 self.config_dict["Inp_params"],\
                                                 self.config_dict["Inp_params"]["input_type"], -Wret_to_lgn)
                    lgn_input_on.append(on_inp)
                    lgn_input_off.append(off_inp)
            lgn_input_on = np.array(lgn_input_on)
            lgn_input_off = np.array(lgn_input_off)
            lgn = np.stack([lgn_input_on,np.array(lgn_input_off)])
            lgn = lgn.reshape(2,num_freq*num_oris*Nsur,-1)
            lgn = np.swapaxes(lgn,1,2)
            lgn = lgn.reshape(2,-1,num_freq,num_oris,Nsur)
            
        if "moving_grating_photoreceptor" in self.config_dict["Inp_params"]["input_type"]:
            lgn_input_on,lgn_input_off = [],[]
            num_freq = kwargs["num_freq"]#3
            num_oris = len(kwargs["orientations"])
            Nsur = kwargs["Nsur"]## gives number of input patterns with diff phases
            spat_frequencies = kwargs["spat_frequencies"]#np.array([80,90,120])
            orientations = kwargs["orientations"]
            for spat_frequency in spat_frequencies:
                for orientation in orientations:
                    self.config_dict["Inp_params"]["spat_frequency"] = spat_frequency # vary between 60 and 120 (3 steps?)
                    self.config_dict["Inp_params"]["Nsur"] = Nsur
                    self.config_dict["Inp_params"]["orientation"] = orientation # vary in 8 steps
                    on_inp = inputs.Inputs_lgn((self.Nret,self.Nret),1,2020).create_lgn_input(\
                                                self.config_dict["Inp_params"],\
                                                "moving_grating_online", Wret_to_lgn)
                    off_inp = inputs.Inputs_lgn((self.Nret,self.Nret),1,2020).create_lgn_input(\
                                                 self.config_dict["Inp_params"],\
                                                 "moving_grating_online", -Wret_to_lgn)
                    lgn_input_on.append(on_inp)
                    lgn_input_off.append(off_inp)
            lgn_input_on = np.array(lgn_input_on)
            lgn_input_off = np.array(lgn_input_off)
            lgn = np.stack([lgn_input_on,np.array(lgn_input_off)])
            lgn = lgn.reshape(2,num_freq*num_oris*Nsur,-1)
            lgn = np.swapaxes(lgn,1,2)
            lgn = lgn.reshape(2,-1,num_freq,num_oris,Nsur)

        elif self.config_dict["Inp_params"]["input_type"] in ("white_noise_online",\
                "ringlike_online","gaussian_noise_online"):
            lgn,lgnI = [],[]
            if kwargs["full_lgn_output"]:
                # generate only lgn input if not "online" generation of it anyways
                # last_timestep = kwargs["last_timestep"]
                T_pd = self.config_dict["Inp_params"]["pattern_duration"]
                avg_no_inp = self.config_dict["Inp_params"]["avg_no_inp"]
                num_inputs = int(self.config_dict["runtime"]/self.config_dict["dt"]/\
                                self.config_dict["Inp_params"]["pattern_duration"])
                num_plasticity_steps = int(num_inputs/avg_no_inp)
                num_lgn_paths = self.config_dict["num_lgn_paths"]
                rnd_seed_I_diff = 10000 * (1-kwargs["same_EI_input"])
                for istep in range(num_plasticity_steps):
                    for jinput in range(avg_no_inp):

                        rng_seed = self.config_dict["random_seed"]*1000 + jinput + istep*avg_no_inp
                        inp = inputs.Inputs_lgn((self.Nret,self.Nret),Version,rng_seed)
                        ilgn = inp.create_lgn_input(self.config_dict["Inp_params"],\
                                                    self.config_dict["Inp_params"]["input_type"],\
                                                    Wret_to_lgn,\
                                                    expansion_timestep = 0)
                        ilgn = inp.apply_ONOFF_bias(ilgn,self.config_dict["Inp_params"])
                        lgn.append(ilgn)

                lgn = np.swapaxes(np.swapaxes(np.array(lgn),0,1),1,2)
                if num_lgn_paths==4:
                    lgn = np.concatenate([lgn,lgn])

        elif self.config_dict["Inp_params"]["input_type"]=="unstructured":
            pass

        return lgn


    def get_RFs(self,mode,**kwargs):
        """
        generate or load pre-computed feedforward connectivity from LGN to L4
        """
        Wlgn4 = kwargs["Wlgn4"]
        if self.verbose: print("mode in get_RFs",mode)
        if mode in ("initialize","initialize2","initializegauss"):
            W_mode = self.config_dict["Wlgn_to4_params"].get("W_mode","random_delta")
            Won_to_4,_ = Wlgn4.create_matrix(self.config_dict["Wlgn_to4_params"], W_mode,\
                        r_A=self.config_dict["Wlgn_to4_params"]["r_A_on"],profile_A="heaviside")
            Wof_to_4,_ = Wlgn4.create_matrix(self.config_dict["Wlgn_to4_params"], W_mode,\
                        r_A=self.config_dict["Wlgn_to4_params"]["r_A_off"],profile_A="heaviside")
            Wlgnto4 = np.stack([Won_to_4,Wof_to_4])
        elif mode=="gabor":
            conn = connectivity.Connectivity((self.Nlgn,self.Nlgn),(self.N4,self.N4),\
                                            random_seed=12345, verbose=self.verbose)

            ## smooth OPM generation
            grid = np.linspace(0,1,self.N4,endpoint=False)
            xto,yto = np.meshgrid(grid,grid)
            conn_params = {"rng" : np.random.RandomState(20200205)}
            ecp,sigma = conn.gen_ecp(xto, yto, conn_params)
            opm = np.angle(ecp,deg=False)*0.5

            ## smooth phases generation
            grid = np.linspace(0,1,self.N4,endpoint=False)
            xto,yto = np.meshgrid(grid,grid)
            conn_params = {"rng" : np.random.RandomState(20200205), "kc" : 2., "n" : 1}
            ecp,sigma = conn.gen_ecp(xto, yto, conn_params)
            pref_phase = np.angle(ecp,deg=False)



            if "ampl_het" in kwargs.keys():
                ampl_het = kwargs["ampl_het"]
            else:
                ampl_het = None

            if "spatial_freq_het" in kwargs.keys():
                spatial_freq_het = kwargs["spatial_freq_het"]
            else:
                spatial_freq_het = None


            conn_params = {"sigma" : 0.2,
                            "ampl" : 1.,
                            "theta" : opm,#0.3*np.ones((Nlgn,Nlgn)),
                            "psi" : pref_phase,
                            "freq" : 10,
                            "ampl_het" : ampl_het,
                            "spatial_freq_het" : spatial_freq_het,
                            }
            gb,_ = conn.create_matrix(conn_params, "Gabor")
            Wlgnto4_on = np.copy(gb)
            Wlgnto4_off = np.copy(gb)
            Wlgnto4_on[Wlgnto4_on<0] = 0
            Wlgnto4_off[Wlgnto4_off>0] = 0
            Wlgnto4_off *= -1.
            Wlgnto4 = np.stack([Wlgnto4_on,Wlgnto4_off])

        elif mode=="load_from_external":
            Version = self.config_dict["Wlgn_to4_params"]["load_from_prev_run"]
            num_lgn_paths = self.config_dict["num_lgn_paths"]
            if self.verbose:
                print(" ")
                print("Load ff connection from version {}".format(Version))
                print(" ")
            if kwargs["system_mode"]=="two_layer":

                if kwargs["load_location"] in ("","local"):
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "two_layer/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "two_layer/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)

                elif kwargs["load_location"]=="habanero":
                    if os.environ["USER"]=="bh2757":
                        if self.config_dict.get("config_name",False):
                            yfile = np.load(data_dir + "two_layer/{s}/v{v}/y_v{v}.npz".format(
                                s=self.config_dict["config_name"],v=Version))
                        else:
                            yfile = np.load(data_dir + "two_layer/v{v}/y_v{v}.npz".format(v=Version))
                    else:
                        if self.config_dict.get("config_name",False):
                            yfile = np.load("/media/bettina/Seagate Portable Drive/physics/columbia/projects/ori_dev_model/"+\
                                "data/two_layer/habanero/{s}/v{v}/y_v{v}.npz".format(
                                s=self.config_dict["config_name"],v=Version))
                        else:
                            yfile = np.load("/media/bettina/Seagate Portable Drive/physics/columbia/projects/ori_dev_model/"+\
                                "data/two_layer/habanero/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)
                    # with np.load(data_dir + "layer4/v{v}/yt_v{v}.npz".format(v=Version)) as yt:
                    # 	Wlgnto4 = yt["Wt"][-1,:].reshape(2,self.N4**2,self.Nlgn**2)

                elif kwargs["load_location"]=="aws":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load("/media/bettina/Seagate Portable Drive/physics/columbia/projects/"+\
                        "ori_dev_model/data/two_layer/aws/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load("/media/bettina/Seagate Portable Drive/physics/columbia/projects/"+\
                        "ori_dev_model/data/two_layer/aws/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)

            elif kwargs["system_mode"]=="one_layer":

                if os.environ["USER"]=="bh2757":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)
                elif os.environ["USER"]=="tuannguyen":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)
                elif os.environ["USER"]=="thn2112":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)
                elif os.environ["USER"]=="alex":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)
                elif os.environ["USER"]=="ah3913":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)
                elif kwargs["load_location"]=="habanero":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/habanero/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/habanero/y_files/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)

                else:
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(\
                            "/media/bettina/TOSHIBA EXT/physics/columbia/projects/ori_dev_model/"+\
                            "data/ffrec/habanero/{s}/v{v}/y_v{v}.npz".format(
                                s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(\
                            "/media/bettina/TOSHIBA EXT/physics/columbia/projects/ori_dev_model/"+\
                            "data/ffrec/habanero/v{v}/y_v{v}.npz".format(v=Version))
                    Wlgnto4 = yfile["W"].reshape(num_lgn_paths,self.N4**2*self.Nvert,self.Nlgn**2)
                    # with np.load(data_dir + "ffrec/v{v}/yt_v{v}.npz".format(v=Version)) as yt:
                    # 	Wlgnto4 = yt["Wt"][-1,:].reshape(2,self.N4**2,self.Nlgn**2)

        arbor_params = {}
        if self.config_dict["Wlgn_to4_params"].get("ret_scatter",False):
            arbor_params = {"ret_scatter" : self.config_dict["Wlgn_to4_params"]["ret_scatter"]}
        if self.config_dict["Wlgn_to4_params"].get("r_lim",False):
            arbor_params = {"r_lim" : self.config_dict["Wlgn_to4_params"]["r_lim"]}
        arbor_on = Wlgn4.create_arbor(radius=self.config_dict["Wlgn_to4_params"]["r_A_on"],\
                        profile=self.config_dict["Wlgn_to4_params"]["arbor_profile_on"],\
                        arbor_params=arbor_params)
        arbor_on *= self.config_dict["Wlgn_to4_params"]["ampl_on"]
        arbor_off = Wlgn4.create_arbor(radius=self.config_dict["Wlgn_to4_params"]["r_A_off"],\
                            profile=self.config_dict["Wlgn_to4_params"]["arbor_profile_off"],\
                            arbor_params=arbor_params)
        arbor_off *= self.config_dict["Wlgn_to4_params"]["ampl_off"]
        arbor2 = np.stack([arbor_on,arbor_off])

        # if mode=="initializegauss":
        #     new_Wlgnto4 = Wlgnto4 * arbor2
        #     old_norm = np.sum(Wlgnto4,axis=-1)[:,:,None]
        #     new_norm = np.sum(new_Wlgnto4,axis=-1)[:,:,None]
        #     Wlgnto4 = new_Wlgnto4 * old_norm / new_norm * self.config_dict["Wlgn_to4_params"].get("algn",1.0)

        return Wlgnto4,arbor_on,arbor_off,arbor2


    def get_Wrec4(self,mode,conn_type,**kwargs):
        """
        generate or load pre-computed recurrent connectivity in L4
        """
        if "2pop" in mode:
            mode = mode[:-4]
        W4 = kwargs["W4"]
        if self.verbose: print("mode in get_Wrec4",mode)
        aux_dict = self.config_dict["W4to4_params"].copy()
        if mode in ("initialize","initialize2","initializegauss"):
            W_mode = self.config_dict["W4to4_params"].get("Wrec_mode","random_delta")
            if "2pop" in W_mode:
                W_mode = W_mode[:-4]
            aux_dict.update({
                "sigma": self.config_dict["W4to4_params"]["sigma" if conn_type is None else "sigma_"+conn_type],
                "ampl": self.config_dict["W4to4_params"]["a" if conn_type is None else "a"+conn_type],
                "s_noise": self.config_dict["W4to4_params"]["s_noise" if conn_type is None else "s_noise_"+conn_type],
                "u_noise": self.config_dict["W4to4_params"]["u_noise" if conn_type is None else "u_noise_"+conn_type]
            })
            W4to4,_ = W4.create_matrix(aux_dict, W_mode,\
                        r_A=self.config_dict["W4to4_params"]["rA" if conn_type is None else "rA_"+conn_type],
                        profile_A="heaviside")
        elif mode=="gabor":
            conn = connectivity.Connectivity((self.N4,self.N4),(self.N4,self.N4),\
                                            random_seed=12345, verbose=self.verbose)

            # ## smooth OPM generation
            # grid = np.linspace(0,1,self.N4,endpoint=False)
            # xto,yto = np.meshgrid(grid,grid)
            # conn_params = {"rng" : np.random.RandomState(20200205)}
            # ecp,sigma = conn.gen_ecp(xto, yto, conn_params)
            # opm = np.angle(ecp,deg=False)*0.5

            # ## smooth phases generation
            # grid = np.linspace(0,1,self.N4,endpoint=False)
            # xto,yto = np.meshgrid(grid,grid)
            # conn_params = {"rng" : np.random.RandomState(20200205), "kc" : 2., "n" : 1}
            # ecp,sigma = conn.gen_ecp(xto, yto, conn_params)
            # pref_phase = np.angle(ecp,deg=False)



            # if "ampl_het" in kwargs.keys():
            #     ampl_het = kwargs["ampl_het"]
            # else:
            #     ampl_het = None

            # if "spatial_freq_het" in kwargs.keys():
            #     spatial_freq_het = kwargs["spatial_freq_het"]
            # else:
            #     spatial_freq_het = None


            # conn_params = {"sigma" : 0.2,
            #                 "ampl" : 1.,
            #                 "theta" : opm,#0.3*np.ones((N4,N4)),
            #                 "psi" : pref_phase,
            #                 "freq" : 10,
            #                 "ampl_het" : ampl_het,
            #                 "spatial_freq_het" : spatial_freq_het,
            #                 }
            # gb,_ = conn.create_matrix(conn_params, "Gabor")
            # Wlgnto4_on = np.copy(gb)
            # Wlgnto4_off = np.copy(gb)
            # Wlgnto4_on[Wlgnto4_on<0] = 0
            # Wlgnto4_off[Wlgnto4_off>0] = 0
            # Wlgnto4_off *= -1.
            # Wlgnto4 = np.stack([Wlgnto4_on,Wlgnto4_off])
            raise Exception("TODO")

        elif mode=="load_from_external":
            Version = self.config_dict["Wlgn_to4_params"]["load_from_prev_run"]
            if self.config_dict["Wlgn_to4_params"]["connectivity_type"]=="EI":
                num_pops = 2
            else:
                num_pops = 1
            if self.verbose:
                print(" ")
                print("Load rec connection from version {}".format(Version))
                print(" ")
            if kwargs["system_mode"]=="two_layer":

                if kwargs["load_location"] in ("","local"):
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "two_layer/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "two_layer/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)

                elif kwargs["load_location"]=="habanero":
                    if os.environ["USER"]=="bh2757":
                        if self.config_dict.get("config_name",False):
                            yfile = np.load(data_dir + "two_layer/{s}/v{v}/y_v{v}.npz".format(
                                s=self.config_dict["config_name"],v=Version))
                        else:
                            yfile = np.load(data_dir + "two_layer/v{v}/y_v{v}.npz".format(v=Version))
                    else:
                        if self.config_dict.get("config_name",False):
                            yfile = np.load("/media/bettina/Seagate Portable Drive/physics/columbia/projects/ori_dev_model/"+\
                                "data/two_layer/habanero/{s}/v{v}/y_v{v}.npz".format(
                                s=self.config_dict["config_name"],v=Version))
                        else:
                            yfile = np.load("/media/bettina/Seagate Portable Drive/physics/columbia/projects/ori_dev_model/"+\
                                "data/two_layer/habanero/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)
                    # with np.load(data_dir + "layer4/v{v}/yt_v{v}.npz".format(v=Version)) as yt:
                    # 	Wlgnto4 = yt["Wt"][-1,:].reshape(2,self.N4**2,self.Nlgn**2)

                elif kwargs["load_location"]=="aws":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load("/media/bettina/Seagate Portable Drive/physics/columbia/projects/"+\
                        "ori_dev_model/data/two_layer/aws/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load("/media/bettina/Seagate Portable Drive/physics/columbia/projects/"+\
                        "ori_dev_model/data/two_layer/aws/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)

            elif kwargs["system_mode"]=="one_layer":

                if os.environ["USER"]=="bh2757":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)
                elif os.environ["USER"]=="tuannguyen":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)
                elif os.environ["USER"]=="thn2112":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)
                elif os.environ["USER"]=="alex":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)
                elif os.environ["USER"]=="ah3913":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)
                elif kwargs["load_location"]=="habanero":
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(data_dir + "ffrec/habanero/{s}/v{v}/y_v{v}.npz".format(
                            s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(data_dir + "ffrec/habanero/y_files/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)

                else:
                    if self.config_dict.get("config_name",False):
                        yfile = np.load(\
                            "/media/bettina/TOSHIBA EXT/physics/columbia/projects/ori_dev_model/"+\
                            "data/ffrec/habanero/{s}/v{v}/y_v{v}.npz".format(
                                s=self.config_dict["config_name"],v=Version))
                    else:
                        yfile = np.load(\
                            "/media/bettina/TOSHIBA EXT/physics/columbia/projects/ori_dev_model/"+\
                            "data/ffrec/habanero/v{v}/y_v{v}.npz".format(v=Version))
                    W4to4 = yfile["Wrec"]#.reshape(num_pops*self.N4**2*self.Nvert,num_pops*self.N4**2*self.Nvert)
                    # with np.load(data_dir + "ffrec/v{v}/yt_v{v}.npz".format(v=Version)) as yt:
                    # 	Wlgnto4 = yt["Wt"][-1,:].reshape(2,self.N4**2,self.Nlgn**2)

        arbor_params = {}
        if self.config_dict["W4to4_params"].get("ret_scatter",False):
            arbor_params = {"ret_scatter" : self.config_dict["W4to4_params"]["ret_scatter"]}
        if self.config_dict["W4to4_params"].get("r_lim",False):
            arbor_params = {"r_lim" : self.config_dict["W4to4_params"]["r_lim"]}

        if conn_type=="EI_all":
            arbor_EE = W4.create_arbor(radius=self.config_dict["W4to4_params"]["rA_"+"EE"],\
                            profile=self.config_dict["W4to4_params"]["arbor_profile_"+"EE"],\
                            arbor_params=arbor_params)
            arbor_EE *= self.config_dict["W4to4_params"]["ampl_"+"EE"]
            
            arbor_EI = W4.create_arbor(radius=self.config_dict["W4to4_params"]["rA_"+"EI"],\
                            profile=self.config_dict["W4to4_params"]["arbor_profile_"+"EI"],\
                            arbor_params=arbor_params)
            arbor_EI *= self.config_dict["W4to4_params"]["ampl_"+"EI"]

            arbor_IE = W4.create_arbor(radius=self.config_dict["W4to4_params"]["rA_"+"IE"],\
                            profile=self.config_dict["W4to4_params"]["arbor_profile_"+"IE"],\
                            arbor_params=arbor_params)
            arbor_IE *= self.config_dict["W4to4_params"]["ampl_"+"IE"]

            arbor_II = W4.create_arbor(radius=self.config_dict["W4to4_params"]["rA_"+"II"],\
                            profile=self.config_dict["W4to4_params"]["arbor_profile_"+"II"],\
                            arbor_params=arbor_params)
            arbor_II *= self.config_dict["W4to4_params"]["ampl_"+"II"]

            arbor = np.block([[arbor_EE,arbor_EI],[arbor_IE,arbor_II]])
        elif conn_type=="EI1pop_all":
            arbor_EE = W4.create_arbor(radius=self.config_dict["W4to4_params"]["rA_"+"EE"],\
                            profile=self.config_dict["W4to4_params"]["arbor_profile_"+"EE"],\
                            arbor_params=arbor_params)
            arbor_EE *= self.config_dict["W4to4_params"]["ampl_"+"EE"]
            
            arbor_EI = W4.create_arbor(radius=self.config_dict["W4to4_params"]["rA_"+"EI"],\
                            profile=self.config_dict["W4to4_params"]["arbor_profile_"+"EI"],\
                            arbor_params=arbor_params)
            arbor_EI *= self.config_dict["W4to4_params"]["ampl_"+"EI"]

            arbor = np.block([[arbor_EE,arbor_EI]])
        else:
            arbor = W4.create_arbor(radius=self.config_dict["W4to4_params"]["rA" if conn_type is None\
                                                                            else "rA_"+conn_type],\
                            profile=self.config_dict["W4to4_params"]["arbor_profile" if conn_type is None \
                                                                     else "arbor_profile_"+conn_type],\
                            arbor_params=arbor_params)
            arbor *= self.config_dict["W4to4_params"]["ampl" if conn_type is None\
                                                      else "ampl_"+conn_type]

        # if mode=="initializegauss":
        #     new_W4to4 = W4to4 * arbor
        #     old_norm = np.sum(W4to4,axis=-1)[:,None]
        #     new_norm = np.sum(new_W4to4,axis=-1)[:,None]
        #     W4to4 = new_W4to4 * old_norm / new_norm * self.config_dict["W4to4_params"].get("a"+conn_type,1.0)

        return W4to4,arbor


    def load_W4to23(self,**kwargs):
        Version = self.config_dict["Wlgn_to4_params"]["load_from_prev_run"]
        if self.verbose:
            print(" ")
            print("Load W4to23 connection from version {}".format(Version))
            print(" ")
        if kwargs["load_location"] in ("","local"):
            yfile = np.load(data_dir + "two_layer/v{v}/y_v{v}.npz".format(\
                            v=Version))
            W4to23 = yfile["W4to23"].reshape(self.N23**2,self.N4**2*self.Nvert)
        elif kwargs["load_location"]=="habanero":
            yfile = np.load(\
                "/media/bettina/Seagate Portable Drive/physics/columbia/projects/ori_dev_model/"+\
                "data/two_layer/habanero/v{v}/y_v{v}.npz".format(v=Version))
            W4to23 = yfile["W4to23"].reshape(self.N23**2,self.N4**2*self.Nvert)
            # with np.load(data_dir + "layer4/v{v}/yt_v{v}.npz".format(v=Version)) as yt:
            #   Wlgnto4 = yt["Wt"][-1,:].reshape(2,self.N4**2,self.Nlgn**2)
        elif kwargs["load_location"]=="aws":
            yfile = np.load(\
                "/media/bettina/Seagate Portable Drive/physics/columbia/projects/"+\
                "ori_dev_model/data/two_layer/aws/v{v}/y_v{v}.npz".format(v=Version))
            W4to23 = yfile["W4to23"].reshape(self.N23**2,self.N4**2*self.Nvert)

        return W4to23
