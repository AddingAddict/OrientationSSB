#import tensorflow as tf
import numpy as np
try:
    import pickle5 as pickle
except:
    import pickle
from dev_ori_sel_RF import connectivity, network_tools
from dev_ori_sel_RF.tools import gen_gaussian_random_field as ggrf

def distance(delta):
    ''' assume periodic boundary conditions'''
    signs = np.sign(delta)
    return np.where(np.abs(delta) > 0.5, -signs*(1 - np.abs(delta)), delta)

def gaussian(x,y,s):
    return np.exp(-(x**2+y**2)/2./s**2)#1./2.*np.pi/s**2*

def stevens_gauss(n_grid,x,y,ori,size=0.088388,aspect_ratio=4.66667,grid_size=1.0):
    pts = np.arange(n_grid)*grid_size/n_grid - grid_size/2
    ysigma = size/2.0
    xsigma = aspect_ratio*ysigma
    
    dx,dy = np.meshgrid(pts,pts)
    
    tmp = np.cos(ori)*dx - np.sin(ori)*dy
    dy = np.sin(ori)*dx + np.cos(ori)*dy
    dx = tmp
    
    dx = dx/xsigma
    dy = dy/ysigma
    
    return np.roll(np.exp(-0.5*(dx**2+dy**2)),(x-n_grid//2,y-n_grid//2),(0,1))

def fio_rect(x):
    return np.fmax(x,0)

def dynamics_system(y,inp_ff,Wrec,gam_o=14.0,gam_s=1.02,k=0.11):
    return fio_rect(gam_o*inp_ff/(k+np.matmul(Wrec,y)))

def integrate(y0,inp,Nt,Wrec,gam_o=14.0,gam_s=1.02,k=0.11):
    y = y0
    for t_idx in range(Nt):
        out = dynamics_system(y,inp,Wrec,gam_o,gam_s,k)
        y = out
    return y

class Inputs:
    '''
    input to retina
    '''
    def __init__(self, size, Version, random_seed):
        self.size = size
        self.Version = Version
        self.random_seed = random_seed

        # rng = np.random.RandomState(self.Version*5101+202)
        self.rng = np.random.RandomState(self.random_seed*5101+2023)

    def create_matrix(self, inp_params, profile, **kwargs):
        self.profile = profile
        ##TODO: choose time-varying or static input
        # if not inp_params:
        # 	from bettina.modeling.ori_dev_model import inp_params

        if self.profile=="GRF":
            N = self.size[0]
            ndim = inp_params["ndim"]
            Nsur = inp_params["Nsur"]
            radius = inp_params["radius"]
            width = inp_params["width"]
            base_vectors = np.empty((N*N,ndim))*np.nan
            for idim in range(ndim):
                rng = np.random.RandomState(idim*10+self.Version*1001)
                cmap = ggrf.generate_topology_map(N, N, radius, width,\
                         rng=rng, return_complex=True, symmetrise=True)
                base_vectors[:,idim] = np.real(cmap).reshape(N*N)

            # householder reflection (qr decomposition of mxn matrix, m>=n
            orth_base_vec,r = np.linalg.qr(base_vectors,mode='reduced')	#q: orthonormal, r: upper triangular
            orth_base_vec = orth_base_vec.T

            # normalization
            #orth_base_vectors = base_vectors
            # orth_base_vec = orth_base_vec/np.sqrt(np.mean(orth_base_vec**2,axis=1))[:,None]
            orth_base_vec = (orth_base_vec - np.nanmean(orth_base_vec,axis=1)[:,None])/np.nanstd(orth_base_vec,axis=1)[:,None]

            # mu = 0.1*rng.randn(Nsur,ndim)
            mu = np.zeros((Nsur,ndim))
            idx_obl = rng.choice(np.arange(ndim),size=Nsur,replace=True)
            idx_car = rng.choice([-1.,1.],size=Nsur,replace=True)
            for isur in range(Nsur):
                mu[isur,idx_obl[isur]] = idx_car[isur]
            input_patterns = 1.*np.dot(mu,orth_base_vec)/ndim
            input_patterns = input_patterns.reshape(Nsur,N,N)

            ## stim indices: -2, -1, 1, 2
            stim_idx = idx_car*(idx_obl+1)
            return input_patterns, orth_base_vec, stim_idx

        if self.profile=="white_noise":
            N = self.size[0]
            # T_pd = inp_params["pattern_duration"]
            T_exp = inp_params["expanse_time"]
            Nsur = (inp_params["Nsur"] + 1) * (T_exp + 1)
            pad_size = 0
            factor = inp_params["onoff_corr_factor"]
            print("factor",factor)

            grid = np.linspace(-0.5,0.5,N+2*pad_size,endpoint=False)
            x,y = np.meshgrid(grid,grid)

            whitenoise = self.rng.randn(Nsur,N,N,3)
            for isur in range(Nsur):
                if (isur+1)%(T_exp+1)==0:
                    whitenoise[isur,...] = 0.
            whitenoise = np.pad(whitenoise,((0,0),(pad_size,pad_size),(pad_size,pad_size),\
                        (0,0)),'constant')
            noise_on = 0 + 1.*whitenoise[...,0] - factor*whitenoise[...,-1]
            noise_of = 0 + 1.*whitenoise[...,1] + factor*whitenoise[...,-1]
            input_patterns = np.stack([noise_on,noise_of])
            print("noise_on",np.nanmin(noise_on),np.nanmax(noise_on))

            input_patterns = np.swapaxes(input_patterns.reshape(2,Nsur,N*N),1,2)
            return np.real(input_patterns)

        if self.profile=="white_noise_online" or self.profile=="white_noise_rect":
            N = self.size[0]
            expansion_timestep = kwargs["expansion_timestep"]
            factor = inp_params["onoff_corr_factor"]
            Nsur = 1
            pad_size = 0

            # grid = np.linspace(-0.5,0.5,N+2*pad_size,endpoint=False)
            # x,y = np.meshgrid(grid,grid)

            whitenoise = self.rng.randn(N,N,3)
            whitenoise = np.pad(whitenoise,((pad_size,pad_size),(pad_size,pad_size),\
                                (0,0)),'constant')
            noise_on = 0 + 1.*whitenoise[...,0] - factor*whitenoise[...,-1]
            noise_of = 0 + 1.*whitenoise[...,1] + factor*whitenoise[...,-1]
            input_patterns = np.stack([noise_on,noise_of])

            # move input pattern to simulate moving waves
            vel = 0.5
            shift = vel * expansion_timestep
            input_patterns = np.roll(np.roll(input_patterns,shift=shift,axis=1),shift=shift,axis=2)

            input_patterns = np.swapaxes(input_patterns.reshape(2,N*N),0,1)
            return np.real(input_patterns)

        if self.profile=="gaussian_noise_online":
            N = int(self.size[0])
            pad_size = 0
            grid = np.linspace(-0.5,0.5,N+2*pad_size,endpoint=False)
            x,y = np.meshgrid(grid,grid)
            sigma = 0.1
            centered_gaussian = np.exp(-(x**2+y**2)/2./sigma**2)
            centered_gaussian = np.fft.fftshift(centered_gaussian)

            random_angles_sym = np.angle(np.fft.fft2(self.rng.randn(2,N,N),axes=(1,2)))

            random_matrix = np.exp(-1j * random_angles_sym)         # Actual complex-number matrix

            # Multiply gaussian=(abs values) with random phases
            ft_gaussian = centered_gaussian[None,:,:] * random_matrix

            # Create topology map by IFFTing the randomized ring
            input_patterns = np.fft.ifft2(ft_gaussian,axes=(1,2))*np.sqrt(2*np.pi*N**2)
            input_patterns = np.swapaxes(input_patterns.reshape(2,N*N),0,1)
            return np.real(input_patterns)

        elif self.profile=="ringlike_online":
            # expansion_timesteps = inp_params["expanse_time"]
            init_radius = inp_params["init_radius"]
            mean_width = inp_params["mean_width"]
            h,w = self.size
            # inputs = np.zeros((h*w,(expansion_timesteps)))
            expansion_timestep = kwargs["expansion_timestep"]

            # random initial position
            ctrx = self.rng.uniform(-init_radius,1+init_radius,1)[0]
            ctry = self.rng.uniform(-init_radius,1+init_radius,1)[0]
            gridx = np.linspace(-ctrx,1-ctrx,self.size[0],endpoint=False)
            gridy = np.linspace(-ctry,1-ctry,self.size[1],endpoint=False)
            x,y = np.meshgrid(gridx,gridy)
            dist = np.sqrt(x**2 + y**2)

            init_r = np.clip(self.rng.randn(1) * 0.1 + init_radius,0.05,0.3)
            vel = np.clip(self.rng.randn(1) * 0.05 + 0.1,0.05,0.1)
            width = np.clip(self.rng.randn(1) * 0.05 + mean_width,0.01,0.3)
            sign = self.rng.choice([-1.,1.],1)
            # print("SETTINGS",np.nanmax(dist),np.nanmin(dist),init_r,vel,width,sign)

            circle = np.logical_and(dist > (init_r + vel*expansion_timestep),\
                                    dist < (init_r + vel*expansion_timestep + width)).astype(float)\
                    * sign * dist
            inputs = circle.flatten()

            return inputs

        elif self.profile=="ringlike_onoff":
            Nsur = inp_params["Nsur"]
            expansion_timesteps = inp_params["expanse_time"]
            init_radius = inp_params["init_radius"]
            h,w = self.size
            inputs = np.zeros((2,h*w,Nsur*(expansion_timesteps+1)))

            ctrs = []
            for input_pattern in range(Nsur):
                ctrx = self.rng.uniform(0,1,1)
                ctry = self.rng.uniform(0,1,1)
                gridx = np.linspace(-ctrx,1-ctrx,w,endpoint=False)
                gridy = np.linspace(-ctry,1-ctry,h,endpoint=False)
                x,y = np.meshgrid(gridx,gridy)
                # x = distance(x)
                # y = distance(y)
                dist_on = np.sqrt(x**2 + y**2)

                init_r = np.clip(self.rng.randn(1) * init_radius*0.9 + init_radius,0.01,1.)
                vel = np.clip(self.rng.randn(1) * 0.02 + 0.1,0.01,0.15)
                width = self.rng.uniform(0.05,1.,1)

                ctrx_of = self.rng.uniform(ctrx-0.3,ctrx+0.3,1)
                ctry_of = self.rng.uniform(ctry-0.3,ctry+0.3,1)
                gridx = np.linspace(-ctrx_of,1-ctrx_of,w,endpoint=False)
                gridy = np.linspace(-ctry_of,1-ctry_of,h,endpoint=False)
                width_of = self.rng.uniform(0.05,1.,1)
                x,y = np.meshgrid(gridx,gridy)
                # x = distance(x)
                # y = distance(y)
                dist_of = np.sqrt(x**2 + y**2)
                print("settings",init_r,vel,width)

                for i in range(expansion_timesteps):
                    circle_on = np.logical_and( dist_on > (init_r + vel*i),\
                                dist_on < (init_r + vel*i + width) ).astype(float) * dist_on
                    inputs[0,:,i+input_pattern*(expansion_timesteps+1)] = circle_on.flatten()

                    circle_of = np.logical_and( dist_of > (init_r + vel*i),\
                                dist_of < (init_r + vel*i + width_of) ).astype(float) * dist_of
                    inputs[1,:,i+input_pattern*(expansion_timesteps+1)] = circle_of.flatten()

                # inputs[:,:,input_pattern*expansion_timesteps:(input_pattern+1)*expansion_timesteps] /=\
                # max(np.nanstd(inputs[:,:,input_pattern*expansion_timesteps:(input_pattern+1)*expansion_timesteps],axis=0))
                ctrs.append(np.array([ctrx,ctry]))

            return inputs, np.array(ctrs)
        
        elif "stevens_bars" in self.profile:
            N = self.size[0]
            bar1 = stevens_gauss(N,self.rng.randint(N),self.rng.randint(N),np.pi*self.rng.random())
            bar2 = stevens_gauss(N,self.rng.randint(N),self.rng.randint(N),np.pi*self.rng.random())
            inputs = 0.25*np.fmax(bar1,bar2).flatten()

            return inputs

        elif self.profile=="moving_grating":
            Nsur = inp_params["Nsur"]
            n = inp_params["spat_frequency"]
            theta = inp_params["orientation"]
            gridx = np.linspace(0,1,self.size[1],endpoint=False)
            gridy = np.linspace(0,1,self.size[0],endpoint=False)
            x0,y0 = np.meshgrid(gridx,gridy)
            x = np.cos(theta) * x0 - np.sin(theta) * y0
            y = np.sin(theta) * x0 + np.cos(theta) * y0
            kx = 2*np.pi/self.size[1] * n
            ky = 2*np.pi/self.size[0] * n
            # Lambda = 2*np.pi/np.sqrt(kx**2+ky**2) #* np.sqrt(2)
            inputs = np.zeros((self.size+(Nsur,)))
            for isur in range(Nsur):
                phi = 1.*isur/Nsur * 2*np.pi
                inputs[:,:,isur] = np.sin(kx * x + 0*ky * y + phi)
            inputs = inputs.reshape(-1,Nsur)
            return inputs

        ## model spatially localised traveling wave like input
        elif self.profile=="localized_waves_online":

            N = self.size[0]
            expansion_timestep = kwargs["expansion_timestep"]
            factor = inp_params["onoff_corr_factor"]
            Nsur = 1
            pad_size = 0

            whitenoise = self.rng.randn(N,N,3)
            whitenoise = np.pad(whitenoise,((pad_size,pad_size),(pad_size,pad_size),\
                                (0,0)),'constant')
            noise_on = 0 + 1.*whitenoise[...,0] - factor*whitenoise[...,-1]
            noise_of = 0 + 1.*whitenoise[...,1] + factor*whitenoise[...,-1]
            input_patterns = np.stack([noise_on,noise_of])
            print("input_patterns",input_patterns.shape)

            vel = 0.5
            shift = vel * expansion_timestep
            input_patterns = np.roll(np.roll(input_patterns,shift=shift,axis=1),shift=shift,axis=2)

            grid = np.linspace(-0.5,0.5,N+2*pad_size,endpoint=False)
            x,y = np.meshgrid(grid,grid)
            sigma = 0.3
            centered_gaussian = np.exp(-(x**2+y**2)/2./sigma**2)
            input_patterns *= centered_gaussian[None,:,:]

            # input_patterns = np.stack([centered_gaussian,centered_gaussian])

            input_patterns = np.swapaxes(input_patterns.reshape(2,N*N),0,1)
            return np.real(input_patterns)


class Inputs_lgn(Inputs):
    '''
    input to lgn which is inputs to retina convolved with connectivity
    from retina to lgn
    '''
    def __init__(self, size, Version, random_seed, gain_control_params=None, cov_mat_params=None):
        super().__init__(size,Version,random_seed)
        self.rng = np.random.RandomState(self.random_seed*5101+2023)
        if gain_control_params is not None:
            self.init_gain_control(gain_control_params)
        if cov_mat_params is not None:
            self.init_cov_mat(cov_mat_params)
            
    def set_seed(self, random_seed):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(self.random_seed*5101+2023)
            
    def init_gain_control(self,params):
        N = self.size[0]
        self.Wgc,_ = connectivity.Connectivity((N,N),(N,N),\
                    random_seed=0).create_matrix(\
                    params,params["profile"])
                    
    def init_cov_mat(self,params):
        N = int(self.size[0])
        
        try:
            self.sqrt_cov = np.load('./../notebooks/cov_{:s}_N={:d}.npy'.format(params.get("name",""),N))
        except:
            conn = connectivity.Connectivity((N,N),(N,N),0)
            
            n_cov,_ = conn.create_matrix(params["n_cov_params"],params["n_cov_params"]['profile'])
            n_cov *= N**2
            n_cov += params["n_cov_base"]

            f_cov,_ = conn.create_matrix(params["f_cov_params"],params["f_cov_params"]['profile'])
            f_cov *= N**2
            f_cov += params["f_cov_base"]

            o_cov,_ = conn.create_matrix(params["o_cov_params"],params["o_cov_params"]['profile'])
            o_cov *= N**2
            o_cov += params["o_cov_base"]
            
            # np.fill_diagonal(n_cov,params["n_var"])
            # np.fill_diagonal(f_cov,params["f_var"])
            
            self.cov = np.block([[n_cov,o_cov],
                                [o_cov,f_cov]])
            evals,evecs = np.linalg.eigh(self.cov)
            self.sqrt_cov = evecs @ np.diag(np.sqrt(np.fmax(0,evals)))
            
            np.save('./../notebooks/cov_{:s}_N={:d}.npy'.format(params.get("name",""),N),self.sqrt_cov)
        
        self.means = np.array([params["n_mean"],params["f_mean"]])
        
    def create_matrix(self, inp_params, profile, **kwargs):
        return super().create_matrix(inp_params, profile, **kwargs)

    def create_lgn_input(self, inp_params, profile, Wret_to_lgn, **kwargs):

        if "white_noise" in profile:
            inp_ret = self.create_matrix(inp_params, profile, **kwargs)
            lgn = np.dot(Wret_to_lgn,inp_ret)
            lgn = np.swapaxes(lgn,0,1)

        elif "GRF" in profile:
            inp_ret, orth_base_vec, stim_idx = self.create_matrix(inp_params, "GRF", **kwargs)
            lgn = np.dot(Wret_to_lgn,inp_ret)
            lgn = np.swapaxes(lgn,0,1)

        elif "ringlike" in profile:
            inp_ret = self.create_matrix(inp_params, "ringlike_online", **kwargs)
            inp_ret = np.stack([inp_ret,-inp_ret])
            lgn = np.matmul(Wret_to_lgn,inp_ret.T)
            lgn = np.swapaxes(lgn,0,1)

        elif "ringlike_onoff" in profile:
            inp_ret, ctrs = self.create_matrix(inp_params, "ringlike_onoff", **kwargs)
            lgn = np.dot(Wret_to_lgn,inp_ret)
            lgn = np.swapaxes(lgn,0,1)

        elif "stevens_bars" in profile:
            inp_ret = self.create_matrix(inp_params, "stevens_bars", **kwargs)
            inp_ret = np.stack([inp_ret,-inp_ret])
            lgn = np.matmul(Wret_to_lgn,inp_ret.T)
            lgn = np.swapaxes(lgn,0,1)

        elif "moving_grating" in profile:
            inp_ret = self.create_matrix(inp_params, "moving_grating", **kwargs)
            lgn = np.dot(Wret_to_lgn,inp_ret)
            lgn = np.swapaxes(lgn,0,1)

        elif "localized_waves" in profile:
            inp_ret = self.create_matrix(inp_params, profile, **kwargs)
            lgn = np.dot(Wret_to_lgn,inp_ret)
            lgn = np.swapaxes(lgn,0,1)

        elif "gaussian_noise_online" in profile:
            inp_ret = self.create_matrix(inp_params, "gaussian_noise_online", **kwargs)
            lgn = np.swapaxes(inp_ret,0,1)
            
        elif "custom_stats" in profile:
            N = self.size[0]
            lgn = self.rng.randn(2*N**2)
            lgn = self.sqrt_cov@lgn
            lgn = lgn.reshape((2,N**2))
            lgn += self.means[:,None]

        ## convolve input with Retina-to-LGN connections
        ## and swap axes such that lgn.shape = 2 x dimensions
        # if Wret_to_lgn.ndim==2:
        # 	lgn = np.dot(Wret_to_lgn,inp_ret)
        # 	lgn = np.swapaxes(lgn,0,1)
        # else:
        # 	lgn = np.stack([np.dot(Wret_to_lgn[0,...],inp_ret[:,0,...]),\
        # 	 				np.dot(Wret_to_lgn[1,...],inp_ret[:,1,...])])


        ## normalise lgn input to SD=1 and shift its mean to positive value
        ## 'online' should only ever produce one input frame per call
        if ("rect" in profile):
            # if ("white_noise" in profile):
            nl = network_tools.nl_rect
            lgn = nl(lgn)

        if ("online" in profile):
            if ("white_noise" in profile or "gaussian_noise" in profile):
                # if True:
                # 	lgn /= np.nanstd(lgn,axis=1)[:,None]
                # 	## lgn += -np.nanmin(lgn,axis=(1))[:,None] + 1

                # 	lgn[0,:] *= 1.
                # 	lgn[1,:] *= inp_params["onoff_rel_weight"]
                # 	# lgn += -np.nanmin(lgn,axis=(1))[:,None] + 1
                # 	# ## old version
                # 	# print("compare (old)",0.1*np.random.randn(1),np.nanmin(lgn),lgn.shape)
                # 	lgn += -np.nanmin(lgn) + 1

                # # # try to keep mean of on/off inputs the same, but weight SD differently
                # else:

                # lgn += -np.nanmean(lgn,axis=(1))[:,None]
                # lgn /= np.nanstd(lgn,axis=1)[:,None]

                # lgn[0,:] *= 1.
                # lgn[1,:] *= inp_params["onoff_rel_weight"]
                # ## make input positive and slight diff in mean between on and off
                lgn += 5 #+ 0.5*self.rng.randn(2)[:,None]
                # lgn = np.clip(lgn,0,np.nanmax(lgn))

                # make input similar in amplitude and sparsity to retinal waves in antolik et al
                # lgn /= 10.
                # lgn += 0.06
                # lgn = np.clip(lgn,0,np.nanmax(lgn))
            elif "sharp" in profile:
                lgn /= np.sqrt(2*np.nanmean(lgn**2))
                lgn = np.clip(lgn,0,1.5)
                lgn = lgn**2
            else:
                # normalise lgn input to between 0 and 1
                # if np.nanmax(lgn)>0:
                # 	lgn /= np.nanmax(lgn)
                # lgn = np.clip(lgn,0,1)
                lgn /= np.sqrt(2*np.nanmean(lgn**2))
                lgn = np.clip(lgn,0,1.5)
        else:
            pass
            # T_exp = inp_params["expanse_time"]
            # for isur in range(inp_params["Nsur"]+1):
            # 	lgn_sd = np.nanstd(lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1],axis=(1,2))
            # 	lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1] /= lgn_sd[:,None,None]
            # 	lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1] +=\
            # 	 - np.nanmin(lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1]) + 1

        # # ====================== add noise to events only ======================
        # for isur in range(inp_params["Nsur"]):
        # 	white_noise = 0.1*np.random.randn(lgn.shape[0],lgn.shape[1],T_exp)
        # 	lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1] += white_noise
        # 	lgn_sd = np.nanstd(lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1],axis=(1,2))
        # 	# print(lgn_sd)
        # lgn = np.clip(lgn,0,np.nanmax(lgn))

        return lgn

    def apply_ONOFF_bias(self,lgn,inp_params):
        ## OFF bias regions
        exp_off_dominant_on = 0.
        exp_off_dominant_of = 0.
        exp = np.array([1.,1.])
        rng = np.random.RandomState(self.random_seed*5101+202)
        if "off_bias_strength" in inp_params.keys():
            if inp_params["off_bias_strength"]>0:
                N = int(self.size[0]) #cast to int, otherwise it's tf int
                rng_bias = np.random.RandomState(20210211)
                radius = N/10.
                width = N/30.
                cmap = ggrf.generate_topology_map(N, N, radius, width,rng=rng_bias,\
                                                  return_complex=True, symmetrise=True)
                off_bias = np.real(cmap).reshape(N*N)
                bias_thr = np.percentile(off_bias,100-inp_params["off_bias_region_size"]*100)
                off_dominant = off_bias - bias_thr
                off_dominant *= inp_params["off_bias_strength"]/np.nanstd(off_dominant)

                ## multiplicative
                exp_off_dominant = 1+off_dominant #np.exp(off_dominant)
                # changes local mean and std:
                lgn[0,:] /= exp_off_dominant
                lgn[1,:] *= exp_off_dominant

                ## change local std
                # exp_off_dominant_on = np.clip(exp_off_dominant,1,np.max(exp_off_dominant))
                # exp_off_dominant_of = np.clip(1./exp_off_dominant,1,np.max(1./exp_off_dominant))

                ## additive constant
                # exp_off_dominant_on = np.zeros_like(off_dominant)
                # exp_off_dominant_on[off_dominant>0] += inp_params["off_bias_strength"]
                # exp_off_dominant_of = np.zeros_like(off_dominant)
                # exp_off_dominant_of[off_dominant<0] += inp_params["off_bias_strength"]

                exp = rng.choice([-1,1],size=2)

        # return np.stack([lgn[0,:]*exp_off_dominant_on**exp[0],\
        # 				 lgn[1,:]*exp_off_dominant_of**exp[1]])

        return np.stack([lgn[0,:]+exp_off_dominant_on*exp[0],\
                         lgn[1,:]+exp_off_dominant_of*exp[1]])
        
    def apply_gain_control(self,lgn,gam_o=1,gam_s=1.02,k=0.11):
        N = self.size[0]
        act_on = integrate(np.zeros(N**2),lgn[0],40,self.Wgc,gam_o,gam_s,k)
        act_off = integrate(np.zeros(N**2),lgn[1],40,self.Wgc,gam_o,gam_s,k)
        return np.stack([act_on,act_off])


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from bettina.modeling.ori_dev_model import Inp_params,connectivity,Wret_to_lgn_params
    from bettina.modeling.ori_dev_model.tools import analysis_tools

    rA = 0.25
    Wret_to_lgn_params["sigma"] = 0.25*rA
    N = 25
    random_seed = 19
    Inp_params.update({"Nsur" : 100})
    Wret_to_lgn_params.update({"mean_eccentricity"	:	0.9,
                                "SD_eccentricity"	:	0.05,
                                "SD_size"			:	0.05,
                                "heterogeneity_type":	None#"independent"#
                                })
    Wret_to_lgn,_ = connectivity.Connectivity((N,N),(N,N),\
                  random_seed=random_seed).create_matrix(\
                  Wret_to_lgn_params,"Gaussian")
    Wret_to_lgn *= Wret_to_lgn_params["gamma_ret"]
    lgn = []

    Inp_params["onoff_rel_weight"] = 1.
    for i in range(Inp_params["Nsur"]):
        # rng_seed = Version*1000 + i
        rng_seed = random_seed*1000 + i
        one_stimulus = Inputs_lgn((N,N),1,rng_seed).create_lgn_input(\
                        Inp_params, "white_noise_online", Wret_to_lgn, expansion_timestep=0)
        # one_stimulus = Inputs_lgn((N,N),1,rng_seed).create_lgn_input(\
        # 				Inp_params, "gaussian_noise_online", Wret_to_lgn, expansion_timestep=0)
        lgn.append( one_stimulus )
    lgn = np.swapaxes(np.swapaxes(np.array(lgn),0,1),1,2)
    print("lgn",lgn.shape)

    lgn_norm = (lgn-np.nanmean(lgn,axis=2)[:,:,None])/np.nanstd(lgn,axis=2)[:,:,None]
    cc = np.nanmean(lgn_norm[:,None,None,:,:] * lgn_norm[None,:,:,None,:],axis=4)
    print("cc.shape",cc.shape)

    print(r"Norm of LGN input: {}$\pm${} (mean pm SD)".format(\
        np.nanmean(lgn),np.nanstd(np.nanmean(lgn,axis=(0,1))),axis=0))

    fig = plt.figure()
    plt.suptitle("Example corr")
    ax = fig.add_subplot(141)
    ax.set_title("onon")
    im=ax.imshow(cc[0,0,:,:],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(142)
    ax.set_title("onoff")
    im=ax.imshow(cc[0,1,:,:],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(143)
    ax.set_title("offoff")
    im=ax.imshow(cc[1,1,:,:],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(144)
    ax.set_title("onon-onoff")
    diff = (0.5*cc[0,0,:,:]+0.5*cc[1,1,:,:]-cc[0,1,:,:])
    im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    plt.colorbar(im,ax=ax)


    cc = cc.reshape(2,2,N,N,N,N)
    fig = plt.figure()
    plt.suptitle("Example corr")
    ax = fig.add_subplot(141)
    ax.set_title("onon")
    im=ax.imshow(cc[0,0,N//2,N//2],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(142)
    ax.set_title("onoff")
    im=ax.imshow(cc[0,1,N//2,N//2],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(143)
    ax.set_title("offoff")
    im=ax.imshow(cc[1,1,N//2,N//2],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(144)
    ax.set_title("onon-onoff")
    diff = (0.5*cc[0,0,N//2,N//2]+0.5*cc[1,1,N//2,N//2]-cc[0,1,N//2,N//2]).reshape(N,N)
    im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    plt.colorbar(im,ax=ax)
    print("Peak CC",cc[:,:,N//2,N//2,N//2,N//2])


    ## average correlation pattern
    cc_avg = 0
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

    # fit_params,fitted_gauss,_ = analysis_tools.determine_LGN_input_correlations_scale(None,diff)
    # fitted_gauss = fitted_gauss.reshape(diff.shape)
    # print("fit_params",fit_params)
    # fig = plt.figure()
    # fig.suptitle("Gaussian fit to LGN input correlation difference")
    # ax = fig.add_subplot(131)
    # im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    # plt.colorbar(im,ax=ax)
    # ax.contour(fitted_gauss,5,colors="b")
    # ax = fig.add_subplot(132)
    # im=ax.imshow(fitted_gauss,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
    # plt.colorbar(im,ax=ax)
    # ax.contour(diff,5,colors="b")
    # ax = fig.add_subplot(133)
    # ax.plot(diff[N//2,:]/np.nanmax(diff),'b',label="Data")
    # ax.plot(fitted_gauss[N//2,:],'r',label="Fit")
    # ax.plot(diff[:,N//2]/np.nanmax(diff),'b--',label="Data")
    # ax.plot(fitted_gauss[:,N//2],'r--',label="Fit")
    # ax.legend(loc="best")


    lgn = lgn.reshape(2,N,N,-1)
    fftabs = np.abs(np.fft.fftshift(np.fft.fft2(lgn[0,:,:,0])))
    fig = plt.figure(figsize=(3*3,2*2))
    plt.suptitle("Example convolved noise pattern")
    ax = fig.add_subplot(231)
    ft = np.abs(np.fft.fftshift(np.fft.fft2(lgn[0,:,:,0]-10)))
    im=ax.imshow(ft,interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(232)
    im=ax.imshow(lgn[0,:,:,1],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(233)
    # ax.plot(lgn[0,N//2,N//2,:],'-')
    ax.plot(np.nanstd(lgn[1,:,:,:],axis=(0,1)),'--')
    ax.plot(np.nanstd(lgn[0,:,:,:],axis=(0,1)),'--')
    # ax.set_ylim(bottom=0)
    ax = fig.add_subplot(234)
    ax.plot(lgn[1,N//2,N//2,:],'-')
    # ax.plot(np.nanmean(lgn[1,:,:,:],axis=(0,1)),'--')
    ax.plot(lgn[0,N//2,N//2,:],'-')
    # ax.plot(np.nanmean(lgn[0,:,:,:],axis=(0,1)),'--')
    # ax.set_ylim(bottom=0)
    ax = fig.add_subplot(235)
    im=ax.imshow(lgn[1,:,:,0],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(236)
    im=ax.imshow(lgn[1,:,:,1],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    plt.show()