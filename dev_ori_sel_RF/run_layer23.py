import os
import sys
import numpy as np
import tensorflow as tf
# import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import logging

from bettina.modeling.ori_dev_model import integrator_tf, connectivity,\
dynamics, inputs
from bettina.modeling.ori_dev_model import Ninp, N23, WIto23_params,\
inp_params, image_dir, data_dir

##TODO:
## implement feedback connections


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

## parameters
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
Version = 0

dt = 0.05
t = np.arange(0,500/dt,1).astype("float32")
T_pd = inp_params["pattern_duration"]
inp_params.update({"Nsur" : int(np.ceil(1.*t.size/T_pd))})

## connectivites
W = connectivity.Connectivity((Ninp,Ninp), (N23,N23), "Gaussian")
WIto23 = W.create_matrix(WIto23_params)
# ew = np.linalg.eigvals(WIto23)
# print("EW",ew[np.argsort(np.real(ew))[-10:]])

W23to23_params = {
				"rng" : np.random.RandomState(10200128),
				"sigma1" : 0.06,
				"sigma2" : 0.09,
				"ampl1" : 1.0,
				"ampl2" : 1.0
				}
W = connectivity.Connectivity((N23,N23), (N23,N23), "zmodel")
W23to23 = W.create_matrix(W23to23_params)
ew,ev = np.linalg.eig(W23to23)
id_ew = np.argsort(ew)[::-1]
ev_norm = (ev-np.nanmean(ev,axis=0)[None,:])/np.nanstd(ev,axis=0)[None,:]
print("EW",ew[id_ew[:10]])

if False:
	N = 10
	fig = plt.figure(figsize=(30,30))
	for i in range(N):
		for j in range(N):
			if (i+j*N)<ev.shape[1]:
				ax = fig.add_subplot(N,N,i+1+j*N)
				ax.set_title("Id {}".format(i+j*N),fontsize=8)
				ax.imshow(np.real(ev[:,id_ew[i+j*N]]).reshape(N23,N23),\
					interpolation="nearest",cmap="binary")
	fig.savefig(image_dir + "ev.pdf")
	plt.close(fig)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(np.real(ew[id_ew]),np.imag(ew[id_ew]),"ok")
	# ax.plot(,"-om")
	fig.savefig(image_dir + "ew.pdf")
	plt.close(fig)
	sys.exit()

## inputs
I,I_stim,stim_idx = inputs.Inputs((Ninp,Ninp), "GRF", Version).create_matrix(inp_params)
I = I.reshape(inp_params["Nsur"],(Ninp*Ninp))
Isd = 1./np.nanstd(I,axis=1)
Imean = 1.0 - np.nanmean(I,axis=1)[:,None]
I = Imean + Isd[:,None]*I
I = np.clip(I,0,np.nanmax(I))
# print("I",np.nanmin(I),np.nanmax(I),I.shape)
# print("wrec",np.nanmin(W23to23),np.nanmax(W23to23))
# I = np.random.randn(I.size).reshape(I.shape)
# sys.exit()
inputs = tf.linalg.matvec(WIto23,I)


## initialiyation
tf.random.set_seed(20200128)
u0 = tf.random.uniform([N23*N23], minval=0, maxval=1, dtype=tf.float32)
u0 = tf.convert_to_tensor(u0, name='u0', dtype=tf.float32)

print("check",I.shape,u0.shape,W23to23.shape,WIto23.shape)
# sys.exit()

## run network
params_dict = {"Ninp" : tf.convert_to_tensor(Ninp, name='Ninp', dtype=tf.int32),
				"N23" : tf.convert_to_tensor(N23, name='N23', dtype=tf.int32),
				"inp" : tf.convert_to_tensor(I, name='I', dtype=tf.float32),
				"W_rec_23" : tf.convert_to_tensor(W23to23, name='W23to23', dtype=tf.float32),
				# "WIto23" : tf.convert_to_tensor(WIto23, name='WIto23'),
				"gamma_FF" : tf.constant(1.0, dtype=tf.float32),
				"gamma_23" : tf.constant(.6, dtype=tf.float32),
				"beta_P" : tf.constant(0.01, dtype=tf.float32),
				"beta_post" : tf.constant(0.01,dtype=tf.float32),
				"tau" : tf.constant(1.0, dtype=tf.float32),
				"pattern_duration" : T_pd}

s = N23*N23*Ninp*Ninp
y0 = tf.concat([WIto23.reshape(s), u0], axis=0)
yt = integrator_tf.odeint(dynamics.dynamics, y0, t, dt, params_dict, "Euler")#"RK4")
timepoints = yt.shape[0]
print("yt",yt.shape)

Wff_to_23 = yt[:,:s]
Wff_to_23 = tf.reshape(Wff_to_23, [timepoints,N23*N23,Ninp*Ninp])

l23 = yt[:,s:]
print(l23.shape,Wff_to_23.shape)


# print("check dW", np.sum((Wff_to_23[0,:,:]-Wff_to_23[-1,:,:])>0),900*900)

## save data


## visualize ff W and activity
fig = plt.figure()
ax = fig.add_subplot(231)
ax.set_title("input(t=-2)")
im=ax.imshow(tf.reshape(I[-2,:],(Ninp,Ninp)),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(232)
ax.set_title("input(t=-1)")
im=ax.imshow(tf.reshape(I[-1,:],(Ninp,Ninp)),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(233)
ax.set_title("FF conn (t=-1)")
im=ax.imshow(tf.reshape(Wff_to_23[-1,:,5*11],(N23,N23)),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)

ax = fig.add_subplot(234)
ax.set_title("L23 act (t=-2)")
im=ax.imshow(tf.reshape(l23[-2,:],(N23,N23)),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(235)
ax.set_title("L23 act (t=-1)")
im=ax.imshow(tf.reshape(l23[-1,:],(N23,N23)),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(236)
ax.set_title("Rec conn (t=-1)")
im=ax.imshow(tf.reshape(W23to23[:,5*11],(N23,N23)),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
fig.savefig(image_dir + "IO_test.pdf")


## temporal behaviour
figt = plt.figure(figsize=(12,5))
ax = figt.add_subplot(121)
ax.plot(l23[:,::100],"-")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Act l23")
ax = figt.add_subplot(122)
ax.plot(Wff_to_23[:,5*11,::100],"-")
ax.set_xlabel("Timesteps")
ax.set_ylabel("FF conn")
figt.savefig(image_dir + "dyn_test.pdf")


## how many different patterns are there in l23 activity?
from bettina.tools.dimension_abbott import calc_dimension
# I_norm = (I - np.nanmean(I,axis=1)[:,None])/np.nanstd(I,axis=1)[:,None]
I_norm = (inputs - np.nanmean(inputs,axis=1)[:,None])/np.nanstd(inputs,axis=1)[:,None]
ccI = np.nanmean(I_norm[:,None,:]*I_norm[None,:,:],axis=2)
l23 = l23.numpy()
l23_norm = (l23-np.nanmean(l23,axis=1)[:,None])/np.nanstd(l23,axis=1)[:,None]
cc23 = np.nanmean(l23_norm[:,None,:] * l23_norm[None,:,:],axis=2)
dim23,w23 = calc_dimension(l23_norm, inp='patterns',output=1)

l23_norm = (l23-np.nanmean(l23,axis=0)[None,:])/np.nanstd(l23,axis=0)[None,:]
cc_act23 = np.nanmean(l23_norm[:,None,:] * l23_norm[:,:,None],axis=0)
dim_act23, w_act23 = calc_dimension(l23_norm, inp='patterns',output=1)
print("Dim",dim23,dim_act23)

fig = plt.figure()
ax = fig.add_subplot(231)
ax.set_title("Corr input over space")
ax.imshow(ccI,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
ax = fig.add_subplot(232)
ax.set_title("Corr l23 act over space")
ax.imshow(cc23,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
ax = fig.add_subplot(233)
ax.plot(w23/np.sum(w23),"-k")
ax.set_xlabel("Index")
ax.set_ylabel("Cov l23 act spectrum")
ax.set_xlim(0,10)
ax = fig.add_subplot(234)
ax.set_title("Corr input over time")
ax.imshow(cc_act23,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
ax = fig.add_subplot(235)
ax.plot(w_act23/np.sum(w_act23),"-k")
ax.set_xlabel("Index")
ax.set_ylabel("Cov l23 act spectrum")
ax.set_xlim(0,10)
fig.savefig(image_dir + "cc_test.pdf")

## how well are response and input related
inp_stim = np.dot(Wff_to_23[-1,:,:].numpy(),I_stim.T).T
inp_stim_norm = (inp_stim - np.nanmean(inp_stim,axis=1)[:,None])/np.nanstd(inp_stim,axis=1)[:,None]
print("inputs",np.nanmean(inputs),np.nanstd(inputs),inputs.shape)
l23_norm = (l23-np.nanmean(l23,axis=1)[:,None])/np.nanstd(l23,axis=1)[:,None]
overlap = np.dot(inp_stim_norm,l23_norm.T)/l23_norm.shape[1]
print("overlap",overlap.shape,np.nanmax(overlap),np.nanmin(overlap))

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121)
ax.set_title("Overlap between input and response")
ax.plot(overlap.T,"-")
stim_idx_per_timestep = np.array([0])
print("stim_idx",stim_idx)
for iidx in stim_idx:
	idx_ar = np.array([iidx]*T_pd)
	stim_idx_per_timestep = np.concatenate([stim_idx_per_timestep, idx_ar])
stim_idx_per_timestep = stim_idx_per_timestep[:len(t)]
stim_idx_per_timestep = stim_idx_per_timestep[::T_pd]
ax.plot(stim_idx_per_timestep,"+k")
ax.set_ylim(-1.05,1.05)
ax.set_xlabel("Correlation (Input, Resp)")
ax.set_ylabel("Time points")
print("stim_idx_per_timestep",stim_idx_per_timestep.shape,l23.shape)
ax = fig.add_subplot(122)
ax.plot(stim_idx_per_timestep,overlap[0,:],"o",label="Stim 1")
ax.plot(stim_idx_per_timestep,overlap[1,:],"+",label="Stim 2")
ax.legend(loc="best")
ax.set_xlabel("Stimulus index")
ax.set_ylabel("Overlap")
fig.savefig(image_dir + "overlap_test.pdf")
# plt.close()

## define -2= 135, -1=90, 0=blank, 1=0, 2=45 deg
stim_ori = [np.nan, 0, 45, 135, 90]
stim_ori_dict = {"1" : 0,
				"2" : 45,
				"-1" : 90,
				"-2" : 135}
opm = 0
for istim in [-2,-1,1,2]:
	opm += np.exp(1j*2*stim_ori_dict[str(int(istim))]/180.*np.pi) *\
	 np.nanmean(l23[stim_idx_per_timestep==istim,:],axis=0)
grid = np.linspace(0,1,N23,endpoint=False)
x,y = np.meshgrid(grid,grid)
ecp,_ = connectivity.gen_ecp(x, y, W23to23_params)

fig=plt.figure()
ax = fig.add_subplot(121)
ax.set_title("Output OPM")
ax.imshow(0.5*np.angle(opm).reshape((N23,N23)),interpolation="nearest",cmap="hsv")
ax = fig.add_subplot(122)
ax.set_title("Input OPM")
ax.imshow(0.5*np.angle(ecp),interpolation="nearest",cmap="hsv")
fig.savefig(image_dir + "opm_test.pdf")

fig = plt.figure()
for i,iidx in enumerate([-2,-1,1,2]):
	resps = l23[stim_idx_per_timestep==iidx,:]
	for j in range(10):
		if j<resps.shape[0]:
			ax = fig.add_subplot(10,4,i+1+j*4)
			if j==0:
				ax.set_title("Ori {}, Trial {}".format(stim_ori_dict[str(iidx)],j))
			im=ax.imshow(resps[j,:].reshape((N23,N23)),interpolation="nearest",cmap="binary")
			plt.colorbar(im, ax=ax)
fig.savefig(image_dir + "trial_resp_test.pdf")		

fig = plt.figure()
for i in range(6):
	for j in range(6):
		if (i+j*6+1)<l23.shape[0]:
			ax = fig.add_subplot(6,6,i+1+j*6)
			if j==0:
				ax.set_title("Ori {}, Trial {}".format(stim_ori_dict[str(int(stim_idx_per_timestep[i+j*6+1]))],i+j*6+1))
			resp = l23[i+j*6+1,:]
			im=ax.imshow(resp.reshape((N23,N23)),interpolation="nearest",cmap="binary")
			plt.colorbar(im, ax=ax)
fig.savefig(image_dir + "time_ordered_resp_test.pdf")


if True:
	fig = plt.figure(figsize=(12,6))
	ax = fig.add_subplot(211)
	overlap = np.dot(inp_stim_norm,ev_norm[:,:50])/inp_stim_norm.shape[1]
	ax.plot(np.real(overlap).T,"-")
	ax = fig.add_subplot(212)
	ax.plot(np.imag(overlap).T,"-")
	fig.savefig(image_dir + "ev_inp_overlap.pdf")
	# plt.close(fig)

plt.show()