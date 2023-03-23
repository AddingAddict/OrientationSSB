import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from bettina.modeling.ori_dev_model import integrator_tf, connectivity,\
dynamics, inputs
from bettina.modeling.ori_dev_model import misc
from bettina.modeling.ori_dev_model import Nlgn, Nret, N4, W4to4_params,\
Wret_to_lgn_params, Wlgn_to4_params, Inp_params, image_dir, data_dir,\
config_dict


config_dict["random_seed"] = 19

N = 25#32
Nlgn, N4, Nret = N,N,N
config_dict["runtime"] = 500
dt = 0.05
t = np.arange(0,config_dict["runtime"]/dt,1).astype("float32")
T_pd = Inp_params["pattern_duration"]
Version = 19


Wret_to_lgn = connectivity.Connectivity((Nret,Nret),(Nlgn,Nlgn),\
		random_seed=config_dict["random_seed"]).create_matrix(\
	Wret_to_lgn_params,"Gaussian") * Wret_to_lgn_params["gamma_ret"]

lgn = []
for i in range(1):
	rng_seed = config_dict["random_seed"]*1000 #+ i*Inp_params["avg_no_inp"]-1
	# ilgn = inputs.Inputs_lgn((Nret,Nret),Version,rng_seed).create_lgn_input(\
	# 	Inp_params, "white_noise_online", Wret_to_lgn)
	ilgn = 1. + np.random.randn(2,Nret*Nret)
	lgn.append( ilgn )
lgn = np.swapaxes(np.swapaxes(np.array(lgn),0,1),1,2)


## LGN to Layer 4
Wlgn4 = connectivity.Connectivity((Nlgn,Nlgn), (N4,N4),\
 random_seed=config_dict["random_seed"])#rng=np.random.RandomState(20200204)
arbor_profile = "heaviside"
arbor = Wlgn4.create_arbor(radius=Wlgn_to4_params["r_A"],profile=arbor_profile)
Won_to_4 = Wlgn4.create_matrix(Wlgn_to4_params, "random_delta",\
	r_A=Wlgn_to4_params["r_A"],profile_A="heaviside")
Wof_to_4 = Wlgn4.create_matrix(Wlgn_to4_params, "random_delta",\
	r_A=Wlgn_to4_params["r_A"],profile_A="heaviside")
Wlgn_to_4 = np.stack([Won_to_4,Wof_to_4])

## Layer 4
gamma_rec = 0.9
W4to4_params["sigma"] = 0.4
W4 = connectivity.Connectivity((N4,N4), (N4,N4),random_seed=config_dict["random_seed"])
# x_I = np.clip(W4to4_params["sigma"]*2,0.1,N4/2.)


deltax = 1./N4
factor = 0.8#2*np.pi#*deltax**2
W4to4_params = {
				"sigma"		:	0.4,
				"ampl"		:	1.0,
				"noise"		:	0.0,

				"sigma_EE"	:	0.2 * factor,
				"sigma_IE"	:	0.25 * factor,
				"sigma_EI"	:	0.15 * factor,
				"sigma_II"	:	0.15 * factor,
				"aEE"		:	11.5,#11.4,
				"aIE"		:	10.8,#11.,
				"aEI"		:	11.,#10.9,
				"aII"		:	10.3,#10.,

				## dina paper parameters
				# "sigma_EE"	:	3.75*deltax,#3*deltax,
				# "sigma_IE"	:	6*deltax,
				# "sigma_EI"	:	2*deltax,
				# "sigma_II"	:	2*deltax,
				# "aEE"		:	0.072 * factor*3**2,#11.4,
				# "aIE"		:	0.06 * factor*6**2,#11.,
				# "aEI"		:	0.0528 * factor*2**2,#10.9,
				# "aII"		:	0.0288 * factor*2**2,#10.,



}
W4to4_params_EE = {"sigma" : W4to4_params["sigma_EE"],\
 "ampl" : W4to4_params["aEE"], "noise" : W4to4_params["noise"]}
conn_EE = W4.create_matrix(W4to4_params_EE, profile="Gaussian")

W4to4_params_IE = {"sigma" : W4to4_params["sigma_IE"],\
 "ampl" : W4to4_params["aIE"], "noise" : W4to4_params["noise"]}
conn_IE = W4.create_matrix(W4to4_params_IE, profile="Gaussian")

W4to4_params_EI = {"sigma" : W4to4_params["sigma_EI"],\
 "ampl" : W4to4_params["aEI"], "noise" : W4to4_params["noise"]}
conn_EI = W4.create_matrix(W4to4_params_EI, profile="Gaussian")

W4to4_params_II = {"sigma" : W4to4_params["sigma_II"],\
 "ampl" : W4to4_params["aII"], "noise" : W4to4_params["noise"]}
conn_II = W4.create_matrix(W4to4_params_II, profile="Gaussian")

W4to4 = np.zeros((N4**2*2, N4**2*2))
W4to4[:N4**2,:N4**2] = conn_EE
W4to4[:N4**2,N4**2:] = -conn_EI
W4to4[N4**2:,:N4**2] = conn_IE
W4to4[N4**2:,N4**2:] = -conn_II
W4to4 = 1.*W4to4/2.#(N4**2*2)
ew,_ = linalg.eig(W4to4,right=True)
W4to4 /= np.nanmax(np.real(ew))/0.8
print("ew",np.nanmax(np.real(ew)))
ew,_ = linalg.eig(conn_EE /np.nanmax(np.real(ew))*0.8,right=True)
print("ew",np.nanmax(np.real(ew)))


I = np.linalg.inv(np.diagflat(np.ones(N4*N4*2)) - W4to4)
l4_I = np.dot(I[:,:N4*N4], np.dot(Wlgn_to_4[0,:,:],lgn[0,:,0]) +\
 np.dot(Wlgn_to_4[1,:,:],lgn[1,:,0]))


fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(231)
ax.set_title("E to E")
connEE = np.fft.fftshift(W4to4[:N4*N4,0].reshape(N4,N4))
# im=ax.imshow(connEE,interpolation="nearest",cmap="binary")
# plt.colorbar(im,ax=ax)
x = np.linspace(-0.5,0.5,N4,endpoint=False)
ax.plot(x,connEE[:,N4//2],'-k')
sigma = 0.15 * 1.1
ax.plot(x,np.nanmax(connEE)*np.exp(-x**2/2./sigma**2),'--r')

ax = fig.add_subplot(232)
ax.set_title("E to I")
im=ax.imshow(W4to4[N4*N4:,:N4*N4],interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(233)
ax.set_title("I to E")
im=ax.imshow(W4to4[:N4*N4,N4*N4:],interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(234)
ax.set_title("I to I")
im=ax.imshow(W4to4[N4*N4:,N4*N4:],interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)

ax = fig.add_subplot(235)
ax.set_title("E filter")
im=ax.imshow(l4_I[:N4*N4].reshape(N4,N4),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(236)
ax.set_title("I filter")
im=ax.imshow(l4_I[N4*N4:].reshape(N4,N4),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)

plt.show()