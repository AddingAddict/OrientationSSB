import numpy as np
from scipy.integrate import solve_ivp

class Model:
    def __init__(
        self,
        n_e: int=16,
        n_i: int=16,
        n_lgn: int=16,
        init_dict: dict=None,
        seed: int=None,
        rx_wave_start: np.ndarray=None,
        ):
        self.n_e = n_e
        self.n_i = n_i
        self.n_lgn = n_lgn
        
        # postsynaptic weight normalization
        self.wff_sum = 1.0
        self.wee_sum = 0.125
        self.wie_sum = 0.5
        self.wei_sum = 2.25
        self.wii_sum = 0.25

        # presynaptic weight normalization
        self.wlgn_sum = 0.0641025641025641
        self.w4e_sum = 0.25
        self.w4i_sum = 9.25

        # maximum allowed weights
        self.max_wff = 4*self.wff_sum / self.n_lgn
        self.max_wee = 4*self.wee_sum / self.n_e
        self.max_wei = 4*self.wei_sum / self.n_i
        self.max_wie = 4*self.wie_sum / self.n_e
        self.max_wii = 4*self.wii_sum / self.n_i

        # RELU gains
        self.gain_e = 1
        self.gain_i = 2
        self.gain_mat = np.diag(np.concatenate((self.gain_e*np.ones(self.n_e),self.gain_i*np.ones(self.n_i))))
        
        self.dt_dyn = 0.01 # timescale for voltage dynamics
        self.a_avg = 1/60 # smoothing factor for average inputs
        self.targ_dw_rms = 0.0005 # target root mean square weight change
        
        if init_dict is None:
            rng = np.random.default_rng(seed)
            
            # initialize weights
            self.wex = rng.uniform(0.2,0.8,size=(self.n_e,self.n_lgn))
            self.wix = rng.uniform(0.2,0.8,size=(self.n_i,self.n_lgn))
            self.wee = rng.uniform(0.2,0.8,size=(self.n_e,self.n_e))
            self.wei = rng.uniform(0.2,0.8,size=(self.n_e,self.n_i))
            self.wie = rng.uniform(0.2,0.8,size=(self.n_i,self.n_e))
            self.wii = rng.uniform(0.2,0.8,size=(self.n_i,self.n_i))
            
            self.wex *= self.wff_sum / np.sum(self.wex,axis=1,keepdims=True)
            self.wix *= self.wff_sum / np.sum(self.wix,axis=1,keepdims=True)
            self.wee *= self.wee_sum / np.sum(self.wee,axis=1,keepdims=True)
            self.wei *= self.wei_sum / np.sum(self.wei,axis=1,keepdims=True)
            self.wie *= self.wie_sum / np.sum(self.wie,axis=1,keepdims=True)
            self.wii *= self.wii_sum / np.sum(self.wii,axis=1,keepdims=True)
            
            # initialize learning rates
            self.wex_rate = 1e-6
            self.wix_rate = 1e-6
            self.wee_rate = 1e-6
            self.wei_rate = 1e-6
            self.wie_rate = 1e-6
            self.wii_rate = 1e-6
            
            # initialize average inputs and rates
            self.uee = np.zeros(n_e)
            self.uei = np.zeros(n_e)
            self.uie = np.zeros(n_i)
            self.uii = np.zeros(n_i)
            
            # calculate average inputs and rates at the start of a geniculate wave
            self.update_inps(rx_wave_start,100*self.dt_dyn)
            
            self.uee_avg = np.ones(n_e)*np.mean(self.uee)
            self.uei_avg = np.ones(n_e)*np.mean(self.uei)
            self.uie_avg = np.ones(n_i)*np.mean(self.uie)
            self.uii_avg = np.ones(n_i)*np.mean(self.uii)
            self.rx_avg = np.ones(n_lgn)*np.mean(rx_wave_start)
            
            # x_sum = np.sum(self.wex,axis=0,keepdims=True) + np.sum(self.wix,axis=0,keepdims=True)
            # e_sum = np.sum(self.wee,axis=0,keepdims=True) + np.sum(self.wie,axis=0,keepdims=True)
            # i_sum = np.sum(self.wei,axis=0,keepdims=True) + np.sum(self.wii,axis=0,keepdims=True)
            # print(np.mean(x_sum),np.mean(e_sum),np.mean(i_sum))
            
        else:
            self.wex = init_dict['wex']
            self.wix = init_dict['wix']
            self.wee = init_dict['wee']
            self.wei = init_dict['wei']
            self.wie = init_dict['wie']
            self.wii = init_dict['wii']
            self.wex_rate = init_dict['wex_rate']
            self.wix_rate = init_dict['wix_rate']
            self.wee_rate = init_dict['wee_rate']
            self.wei_rate = init_dict['wei_rate']
            self.wie_rate = init_dict['wie_rate']
            self.wii_rate = init_dict['wii_rate']
            self.uee = init_dict['uee']
            self.uei = init_dict['uei']
            self.uie = init_dict['uie']
            self.uii = init_dict['uii']
            self.uee_avg = init_dict['uee_avg']
            self.uei_avg = init_dict['uei_avg']
            self.uie_avg = init_dict['uie_avg']
            self.uii_avg = init_dict['uii_avg']
            self.rx_avg = init_dict['rx_avg']
            
        self.dwex = np.zeros_like(self.wex)
        self.dwix = np.zeros_like(self.wix)
        self.dwee = np.zeros_like(self.wee)
        self.dwei = np.zeros_like(self.wei)
        self.dwie = np.zeros_like(self.wie)
        self.dwii = np.zeros_like(self.wii)

    # voltage dynamics function
    def ode_func(
        self,
        t: float,
        u: np.ndarray,
        w: np.ndarray,
        h: np.ndarray,
        ):
        r = np.fmax(u,0)
        np.matmul(self.gain_mat,r,out=r)
        np.matmul(w,r,out=r)
        r[:] += h - u
        return r / self.dt_dyn

    # integrate rate dynamics and update inputs
    def update_inps(
        self,
        rx: np.ndarray,
        int_time: float,
        ):
        
        # calculate feedforward inputs
        he,hi = self.wex@rx,self.wix@rx
        h = np.concatenate((he,hi))
        
        # create full recurrent weight matrix
        w = np.block([[self.wee,-self.wei],[self.wie,-self.wii]])
        
        # integrate dynamics
        sol = solve_ivp(self.ode_func,[0,int_time],np.concatenate((self.uee-self.uei,self.uie-self.uii)),args=(w,h),t_eval=[int_time],method='RK23')
        
        # compute rates and update inputs
        r = np.fmax(sol.y[:,-1],0)
        np.matmul(self.gain_mat,r,out=r)
        re,ri = r[:self.n_e],r[self.n_e:]
        self.uee[:] = self.wee@re + he
        self.uie[:] = self.wie@re + hi
        self.uei[:] = self.wei@ri
        self.uii[:] = self.wii@ri
        
    # update input and rate averages
    def update_avgs(
        self,
        rx: np.ndarray,
        ):
        self.uee_avg[:] += self.a_avg * (self.uee - self.uee_avg)
        self.uei_avg[:] += self.a_avg * (self.uei - self.uei_avg)
        self.uie_avg[:] += self.a_avg * (self.uie - self.uie_avg)
        self.uii_avg[:] += self.a_avg * (self.uii - self.uii_avg)
        self.rx_avg[:] += self.a_avg * (rx - self.rx_avg)
        
        self.ue = self.uee - self.uei
        self.ui = self.uie - self.uii
        self.ue_avg = self.uee_avg - self.uei_avg
        self.ui_avg = self.uie_avg - self.uii_avg
        
    def reset_dw(
        self,
        ):
        self.dwex[:] = 0
        self.dwix[:] = 0
        self.dwee[:] = 0
        self.dwei[:] = 0
        self.dwie[:] = 0
        self.dwii[:] = 0
        
    # collect weight changes in a batch
    def collect_dw(
        self,
        rx: np.ndarray,
        ):
        
        self.dwex += self.wex_rate * np.outer(self.ue - self.ue_avg,rx - 10*self.rx_avg)
        self.dwix += self.wix_rate * np.outer(self.ui - self.ui_avg,rx - 10*self.rx_avg)
        self.dwee += self.wee_rate * np.outer(self.ue - self.ue_avg,self.ue - 10*self.ue_avg)
        self.dwie += self.wie_rate * np.outer(self.ui - self.ui_avg,self.ue - 10*self.ue_avg)
        self.dwei += self.wei_rate * (np.outer(np.fmax(self.uei - self.uei_avg,0),np.fmax(self.ui - self.ui_avg,0)) -\
                                      np.outer(np.fmax(self.ue - self.ue_avg,0),np.fmax(self.ui - self.ui_avg,0)))
        self.dwii += self.wii_rate * (np.outer(np.fmax(self.uii - self.uii_avg,0),np.fmax(self.ui - self.ui_avg,0)) -\
                                      np.outer(np.fmax(self.ui - self.ui_avg,0),np.fmax(self.ui - self.ui_avg,0)))
        
    def sum_norm_dw(
        self,
        ):
        self.dwex -= np.mean(self.dwex)
        self.dwix -= np.mean(self.dwix)
        self.dwee -= np.mean(self.dwee)
        self.dwei -= np.mean(self.dwei)
        self.dwie -= np.mean(self.dwie)
        self.dwii -= np.mean(self.dwii)
        
    # update learning rates
    def update_learn_rates(
        self,
        ):
        self.wex_rate *= self.targ_dw_rms / np.sqrt(np.mean(self.dwex**2))
        self.wix_rate *= self.targ_dw_rms / np.sqrt(np.mean(self.dwix**2))
        self.wee_rate *= self.targ_dw_rms / np.sqrt(np.mean(self.dwee**2))
        self.wei_rate *= self.targ_dw_rms / np.sqrt(np.mean(self.dwei**2))
        self.wie_rate *= self.targ_dw_rms / np.sqrt(np.mean(self.dwie**2))
        self.wii_rate *= self.targ_dw_rms / np.sqrt(np.mean(self.dwii**2))
        # self.wex_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwex**2))
        # self.wix_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwix**2))
        # self.wee_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwee**2))
        # self.wei_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwei**2))
        # self.wie_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwie**2))
        # self.wii_rate = self.targ_dw_rms / np.sqrt(np.mean(self.dwii**2))
        
        # self.dwex *= self.wex_rate
        # self.dwix *= self.wix_rate
        # self.dwee *= self.wee_rate
        # self.dwei *= self.wei_rate
        # self.dwie *= self.wie_rate
        # self.dwii *= self.wii_rate
        
    # update weights with collected changes, then clip and normalize weights
    def update_weights(
        self,
        ):
        self.wex += self.dwex
        self.wix += self.dwix
        self.wee += self.dwee
        self.wei += self.dwei
        self.wie += self.dwie
        self.wii += self.dwii
        
        # alternate clipping, presynaptic normalization, and postsynaptic normalization
        for _ in range(4):
            # clip weights
            np.clip(self.wex,1e-10,self.max_wff,out=self.wex)
            np.clip(self.wix,1e-10,self.max_wff,out=self.wix)
            np.clip(self.wee,1e-10,self.max_wee,out=self.wee)
            np.clip(self.wei,1e-10,self.max_wei,out=self.wei)
            np.clip(self.wie,1e-10,self.max_wie,out=self.wie)
            np.clip(self.wii,1e-10,self.max_wii,out=self.wii)
            
            # presynaptic normalization
            x_sum = np.sum(self.wex,axis=0,keepdims=True) + np.sum(self.wix,axis=0,keepdims=True)
            self.wex *= self.wlgn_sum / x_sum
            self.wix *= self.wlgn_sum / x_sum
            e_sum = np.sum(self.wee,axis=0,keepdims=True) + np.sum(self.wie,axis=0,keepdims=True)
            self.wee *= self.w4e_sum / e_sum
            self.wie *= self.w4e_sum / e_sum
            i_sum = np.sum(self.wei,axis=0,keepdims=True) + np.sum(self.wii,axis=0,keepdims=True)
            self.wei *= self.w4i_sum / i_sum
            self.wii *= self.w4i_sum / i_sum
            
            # clip weights
            np.clip(self.wex,1e-10,self.max_wff,out=self.wex)
            np.clip(self.wix,1e-10,self.max_wff,out=self.wix)
            np.clip(self.wee,1e-10,self.max_wee,out=self.wee)
            np.clip(self.wei,1e-10,self.max_wei,out=self.wei)
            np.clip(self.wie,1e-10,self.max_wie,out=self.wie)
            np.clip(self.wii,1e-10,self.max_wii,out=self.wii)
            
            # postsynaptic normalization
            self.wex *= self.wff_sum / np.sum(self.wex,axis=1,keepdims=True)
            self.wix *= self.wff_sum / np.sum(self.wix,axis=1,keepdims=True)
            self.wee *= self.wee_sum / np.sum(self.wee,axis=1,keepdims=True)
            self.wei *= self.wei_sum / np.sum(self.wei,axis=1,keepdims=True)
            self.wie *= self.wie_sum / np.sum(self.wie,axis=1,keepdims=True)
            self.wii *= self.wii_sum / np.sum(self.wii,axis=1,keepdims=True)