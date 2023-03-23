import numpy as np
import matplotlib.pyplot as plt

from bettina.modeling.ori_dev_model import image_dir, connectivity, inputs
from bettina.modeling.ori_dev_model.tools import analysis_tools


def energy_symm(Asq, C, I, W):
	"""
	
	"""
	delta = np.diagflat(np.ones((I.shape[0]),dtype=float))
	proj_op = (delta[:,:,None] - Asq[:,None,:]*Asq[None,:,:]/(np.sum(Asq*Asq,axis=0)[None,None,:]))

	proj_op_arb_l = proj_op * Asq[None,:,:]
	proj_op_arb_r = proj_op * Asq[:,None,:]


	## outputl_z',gamma = sum_z P_z,z',gamma * W_z,gamma
	proj_op_arb_l = np.swapaxes(proj_op_arb_l,0,2)
	proj_op_arb_r = np.swapaxes(proj_op_arb_r,0,2)
	W_sw = np.swapaxes(W,0,1)

	outputl = np.squeeze(np.matmul(proj_op_arb_l, W_sw[:,:,None]))
	outputl = np.swapaxes(outputl,0,1)

	## outputr_z2',beta = sum_z2 P_z2',z2,beta * W_z2,beta
	# outputr = np.sum(proj_op_arb* W[None,:,:],axis=1)
	outputr = np.squeeze(np.matmul(W_sw[:,None,:],proj_op_arb_r))
	outputr = np.swapaxes(outputr,0,1)


	# output1_z',beta = sum_gamma output0_z',gamma * C_gamma,beta
	output1 = np.matmul(outputl, C)
	# output2_beta,z2' = sum_z' output1_beta,z' * I_z',z2' 
	output2 = np.matmul( np.swapaxes(output1,0,1), I )
	# output3_z2,z2' = sum_beta outputr_z2',beta * out2_beta,z2'
	energy = -0.5 * np.sum( outputr * output2.T )
	# print("Asq_inv",np.nanmin(output2),np.nanmax(output2),\
	# 	np.nanmin(output1),np.nanmax(output1),\
	# 	np.nanmin(outputr),np.nanmax(outputr),\
	# 	np.nanmin(outputl),np.nanmax(outputl))
	return energy, proj_op_arb_l, proj_op_arb_r, outputl, outputr



Nlgn = 19
N4 = 20

rho = 0.15
eta = 1. * rho
zeta = 1.0 * rho

DA = int(Nlgn*rho)*2

## assume all of these functions are gaussian distributed
Wlgn4 = connectivity.Connectivity((Nlgn,Nlgn), (N4,N4),\
	 random_seed=20200923)
arbor_profile = "gaussian"
arbor = Wlgn4.create_arbor(radius=rho,profile=arbor_profile)# N4 x Nlgn shape = Axalpha
arb_sq = np.sqrt(arbor)
print("arbor",arbor.shape,np.sum(arbor,axis=0)[:20],np.sum(arbor,axis=1)[:20])

Wlgnlgn = connectivity.Connectivity((Nlgn,Nlgn), (Nlgn,Nlgn),\
	 random_seed=20200923)
conn_params = {"sigma" : eta, "ampl" : 1., "noise" : 0.0}
lgn_corr = Wlgnlgn.create_matrix(conn_params,profile="Gaussian")
print("lgn_corr",lgn_corr.shape)

W44 = connectivity.Connectivity((N4,N4), (N4,N4),\
	 random_seed=20200923)
conn_params = {"sigma" : zeta, "ampl" : 1., "noise" : 0.0}
crt_inter = W44.create_matrix(conn_params,profile="Gaussian")
print("W44",crt_inter.shape)

# fig=plt.figure()
# ax = fig.add_subplot(121)
# im=ax.imshow(lgn_corr,interpolation="nearest",cmap="binary")
# plt.colorbar(im,ax=ax)
# ax = fig.add_subplot(122)
# im=ax.imshow(crt_inter,interpolation="nearest",cmap="binary")
# plt.colorbar(im,ax=ax)
# plt.show()
# exit()

## ================ Rec field, shape: N4 x Nlgn =========================================
kappa = 0.2
inp_params = {"ndim" : 1, "Nsur" : 1, "radius" : 0.1, "width" : .5}

kvec_field,_,_ = inputs.Inputs((N4,N4),1,923).create_matrix(inp_params, "GRF")
kvec_fieldx = 2*np.pi/0.8/np.sqrt(2) + 1.5*kvec_field[0,:,:]
kvec_fieldy = 2*np.pi/0.8/np.sqrt(2) - 1.5*kvec_field[0,:,:]
kvec_field_phi = kvec_field[0,:,:]/np.nanmax(kvec_field[0,:,:]) * np.pi * 0.4

# fig=plt.figure()
# ax = fig.add_subplot(121)
# im=ax.imshow(kvec_field_phi,interpolation="nearest",cmap="binary")
# plt.colorbar(im,ax=ax)
# ax = fig.add_subplot(122)
# im=ax.imshow(kvec_fieldy,interpolation="nearest",cmap="binary")
# plt.colorbar(im,ax=ax)
# plt.show()
# exit()

kabs = 2*np.pi/0.8
phi = kvec_field_phi#np.pi*2
rot_matrix = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
kvector = np.array([kabs/np.sqrt(2),kabs/np.sqrt(2)])
# kvector_rot = np.matmul(kvector,rot_matrix)
kvector_rot = np.nansum( kvector[:,None,None,None] *rot_matrix, axis=0)

conn_params = {"sigma" : kappa, "ampl" : 1., "noise" : 0.0,\
				 # "kvector" : [kvec_fieldx,kvec_fieldx]}
				 # "kvector" : kvector}
				 "kvector" : [kvector_rot[0,...],kvector_rot[1,...]],
				 "phase"	:	90./180.*np.pi}
W_gabor = Wlgn4.create_matrix(conn_params,profile="Gabor",r_A=0.15)
print("W_gabor",W_gabor.shape,DA,)

RF1,_,_,_ = analysis_tools.get_RF_form(np.stack([W_gabor,W_gabor]),N4,Nlgn,DA+3,\
			calc_PF=False,Nvert=1)
alpha = 0.5
RF2,_,_,_ = analysis_tools.get_RF_form(np.stack([alpha + (1-alpha) * W_gabor,alpha + (1-alpha) * W_gabor]),\
			N4,Nlgn,int(Nlgn*rho)*2+3,calc_PF=False,Nvert=1)

sd = W_gabor.reshape(N4,N4,Nlgn,Nlgn)
opm,Rn,pref_phase,gabors = analysis_tools.get_response(sd,DA,Nvert=1)
gabors = np.swapaxes(gabors,1,2)
print("gabors",gabors.shape,np.sqrt(gabors.shape[0]),DA,DA*Nlgn*2)
gabors = gabors.reshape(DA*N4*2,DA*2*N4)
pref_ori = 0.5*np.angle(opm,deg=True)
sel = np.abs(opm)
pref_phase /= 2.*np.pi/360

fig=plt.figure(figsize=(3*6,2*5))
ax = fig.add_subplot(231)
im=ax.imshow(W_gabor[108,:].reshape(Nlgn,Nlgn),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(232)
im=ax.imshow(RF1[1,:,:],interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(233)
im=ax.imshow(RF2[1,:,:],interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(234)
im=ax.imshow(pref_ori.reshape(N4,N4),interpolation="nearest",cmap="hsv")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(235)
# im=ax.imshow(pref_phase.reshape(N4,N4),interpolation="nearest",cmap="binary")
# plt.colorbar(im,ax=ax)
ax.hist(pref_phase.flatten())
ax = fig.add_subplot(236)
im=ax.imshow(gabors,interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
plt.show()
exit()
## ====================================================================================

alphas = np.arange(0.,1.1,1.9)
ens = []
for alpha in alphas:
	W = alpha + (1-alpha) * W_gabor
	W[np.logical_not(arbor)] = 0.
	en, proj_op_arb_l,proj_op_arb_r, outputl, outputr = energy_symm(arb_sq, lgn_corr,\
														 crt_inter, W)
	ens.append(en)
print("Energy",ens)


fig=plt.figure()
ax = fig.add_subplot(131)
ax.plot(alphas,ens,'o-')
ax = fig.add_subplot(132)
ax.imshow(RF1[1,:,:],interpolation="nearest",cmap="binary")
ax = fig.add_subplot(133)
ax.imshow(RF2[1,:,:],interpolation="nearest",cmap="binary")

## show that proj_op_arb is symmetric in z,z' (first two dims)
# diff = np.nansum(proj_op_arb_l - np.swapaxes(proj_op_arb_r,0,1),axis=(0,1))
# print("diff",np.sum(diff>10e-8))
# diffc = np.nansum(crt_inter - np.swapaxes(crt_inter,0,1),axis=(0,1))
# print("diffc",np.sum(diffc>10e-8))

print("proj_op_arb_l",proj_op_arb_l.shape,proj_op_arb_r.shape)
a_idx = 20
all_w = []
all_v = []
# for b_idx in range(a_idx,a_idx+1):
for b_idx in range(Nlgn**2):
	A0 = np.matmul(proj_op_arb_l[a_idx,:,:], np.matmul(crt_inter, proj_op_arb_r[b_idx,:,:]))

	w0,v0 = np.linalg.eig(A0)
	idx_sort=np.nanargmax(np.real(w0))
	w0 = w0[idx_sort]
	v0 = v0[:,idx_sort]
	all_w.append(w0)
	all_v.append(v0)
all_v = np.array(all_v)
# A0 = np.roll(A0.reshape(N4,N4,N4,N4),(2,-10,2,-10),(0,1,2,3)).reshape(N4**2,N4**2)
w0,v0 = np.linalg.eig(A0)# * lgn_corr[a_idx,b_idx]
idx_sort=np.argsort(np.real(w0))
w0 = w0[idx_sort]
v0 = v0[:,idx_sort]

fig = plt.figure(figsize=(6*Nlgn,5*Nlgn))
for i in range(Nlgn):
	for j in range(Nlgn):
		ax = fig.add_subplot(Nlgn,Nlgn,j+i*Nlgn+1)
		ax.imshow(np.real(all_v[j+i*Nlgn,:]).reshape(N4,N4),interpolation="nearest",\
					cmap="binary")

fig = plt.figure()
ax= fig.add_subplot(111)
im=ax.imshow(np.real(np.array(all_w)).reshape(Nlgn,Nlgn),interpolation="nearest")
plt.colorbar(im,ax=ax)
# ax.plot(np.arange(Nlgn**2),np.array(all_w2)[:,-1],'-b')

all_v = all_v.reshape(Nlgn,Nlgn,N4,N4)
RFv,PFv,_,_ = analysis_tools.get_RF_form(np.stack([all_v,all_v]),N4,Nlgn,int(Nlgn*rho)*2+3,\
	calc_PF=True,Nvert=Nvert)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(PFv[1,:,:],interpolation="nearest",cmap="binary")


a_idx = 50
b_idx = 50
A1 = np.matmul(proj_op_arb_l[a_idx,:,:], np.matmul(crt_inter, proj_op_arb_r[b_idx,:,:])) #+\
	# np.matmul(proj_op_arb_l[:,:,b_idx], np.matmul(crt_inter, proj_op_arb_r[:,:,a_idx])) ) * 0.5
w1,v1 = np.linalg.eig(A1)
idx_sort=np.argsort(np.real(w1))
w1 = w1[idx_sort]
v1 = v1[:,idx_sort]
print("compare ev",w1[-1],w0[-1],A1.shape,A0.shape)

e0 = np.matmul(np.matmul(v0[:,-1],A0),v0[:,-1])
wg = W_gabor[:,0]/np.sqrt(np.sum(W_gabor[:,0]**2))
erf = np.matmul(np.matmul(wg,A0),wg)
print("compare dot prod",e0,erf)

fig=plt.figure()
ax = fig.add_subplot(241)
im=ax.imshow(A0[:,11].reshape(N4,N4),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
# ax.hist(A0[A0>0],bins=20)
ax = fig.add_subplot(242)
im=ax.imshow(A0[10,:].reshape(N4,N4),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
# ax.hist(A1[A1>0],bins=20)
ax = fig.add_subplot(243)
im=ax.imshow(A1[:,51].reshape(N4,N4),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax = fig.add_subplot(244)
ax.plot(w0,"-o")

ax = fig.add_subplot(245)
im=ax.imshow(np.real(np.roll(v0[:,-1].reshape(N4,N4),(5,5),(0,1))),interpolation="nearest",\
			cmap="binary")
plt.colorbar(im,ax=ax)
ax.contour(np.real(v0[:,-1]).reshape(N4,N4),[0],colors="m")
# ax.plot(w0,'o-')
ax = fig.add_subplot(246)
im=ax.imshow(np.real(v1[:,-1]).reshape(N4,N4),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
ax.contour(v1[:,-1].reshape(N4,N4),[0],colors="m")
ax = fig.add_subplot(247)
im=ax.imshow(np.real(v1[:,-2]).reshape(N4,N4),interpolation="nearest",cmap="binary")
plt.colorbar(im,ax=ax)
# ax.plot(w1,'o-')
ax = fig.add_subplot(248)
ax.plot(w1,"-o")
plt.show()

