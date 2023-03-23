import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors


from bettina.modeling.ori_dev_model.tools import analysis_tools


# plt.rcParams["animation.ffmpeg_path"]="/home/bettina/physics/workenv3/bin/ffmpeg"
# "/home/bettina/physics/workenv3/lib/python3.5/site-packages/ffmpeg/_ffmpeg.py"

def center_of_mass(X):
	# calculate center of mass of a closed polygon
	X = np.concatenate([X,X[:1,:]])
	x = X[:,0]
	y = X[:,1]

	g = (x[:-1]*y[1:] - x[1:]*y[:-1])
	A = 0.5*g.sum()
	cx = ((x[:-1] + x[1:])*g).sum()
	cy = ((y[:-1] + y[1:])*g).sum()
	return 1./(6*A)*np.array([cx,cy])


def plot_RF(RF,PF,add_PF_row=True,**kwargs):
	"""create figure to display RFs and PFs"""
	if add_PF_row:
		nrow = 2
	else:
		nrow = 1
	ncol = 3
	if "avgRF" in kwargs.keys():
		ncol += 1

	N4 = kwargs["N4"]
	Nvert = kwargs["Nvert"]
	DA = kwargs["DA"]
	
	try:
		norm = colors.TwoSlopeNorm(vcenter=0)
	except:
		norm = None
	
	fig = plt.figure(figsize=(6*ncol,5*nrow))
	## receptive field
	ax = fig.add_subplot(nrow,ncol,1)
	ax.set_title("S_D")
	if np.sum(RF[0,:,:]>0)>0:
		vm = np.percentile(RF[0,RF[0,:,:]>0],90)
	else:
		vm = 12
	# vm = np.nanmax(RF)
	im = ax.imshow(RF[0,:,:],interpolation='nearest',cmap='RdBu_r',norm=norm)
	plt.colorbar(im,ax=ax)
	if Nvert>1:
		for iN in range(N4-1):
			ax.axvline(Nvert*DA*(iN+1),ls="--",c="k",lw=1)

	ax = fig.add_subplot(nrow,ncol,2)
	ax.set_title("S_on")
	im = ax.imshow(RF[1,:,:],interpolation='nearest',cmap='binary')
	plt.colorbar(im,ax=ax)
	if Nvert>1:
		for iN in range(N4-1):
			ax.axvline(Nvert*DA*(iN+1),ls="--",c="k",lw=1)

	ax = fig.add_subplot(nrow,ncol,3)
	ax.set_title("S_of")
	im = ax.imshow(RF[2,:,:],interpolation='nearest',cmap='binary')
	plt.colorbar(im,ax=ax)
	if Nvert>1:
		for iN in range(N4-1):
			ax.axvline(Nvert*DA*(iN+1),ls="--",c="k",lw=1)

	if "avgRF" in kwargs.keys():
		ax = fig.add_subplot(nrow,ncol,4)
		ax.set_title("S_avg")
		im = ax.imshow(kwargs["avgRF"][0,:,:],interpolation='nearest',cmap='binary')
		plt.colorbar(im,ax=ax)
		for iN in range(N4-1):
			ax.axvline(Nvert*DA*(iN+1),ls="--",c="k",lw=1)

	## projective field
	if add_PF_row:
		ax = fig.add_subplot(nrow,ncol,ncol+1)
		ax.set_title("S_D (PF)")
		# vm = np.percentile(RF[0,RF[0,:,:]>0],80)
		if np.sum(PF>0)>0:
			vm = np.nanmax(PF)
		else:
			vm = 12
		im = ax.imshow(PF[0,:,:],interpolation='nearest',cmap='RdBu_r',norm=norm)
		plt.colorbar(im,ax=ax)

		ax = fig.add_subplot(nrow,ncol,ncol+2)
		ax.set_title("S_on (PF)")
		im = ax.imshow(PF[1,:,:],interpolation='nearest',cmap='binary')
		plt.colorbar(im,ax=ax)

		ax = fig.add_subplot(nrow,ncol,ncol+3)
		ax.set_title("S_of (PF)")
		im = ax.imshow(PF[2,:,:],interpolation='nearest',cmap='binary')
		plt.colorbar(im,ax=ax)

		if "avgPF" in kwargs.keys():
			ax = fig.add_subplot(nrow,ncol,ncol+4)
			ax.set_title("S_avg")
			im = ax.imshow(kwargs["avgPF"][0,:,:],interpolation='nearest',cmap='binary')
			plt.colorbar(im,ax=ax)

	return fig


def plot_RF_visual_space(Wlgn_to_4,**kwargs):
	"""create figure to display RFs in visual space"""
	Nlgn = kwargs["Nlgn"]
	N4 = kwargs["N4"]
	Nvert = kwargs["Nvert"]
	fig = plt.figure(figsize=(24,10))
	axes = []
	for irow in range(2):
		for jcol in range(4):
			axes.append(fig.add_subplot(2,4,1+jcol+irow*4))
	Wlgn_to_4 = Wlgn_to_4.reshape(2,N4*N4*Nvert,Nlgn*Nlgn)
	deltaS = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
	deltaS = deltaS/np.nanmax(np.abs(deltaS),axis=1)[:,None]
	## normalise max value to 1 such that contours can be drawn at same height for each crt unit
	deltaS = deltaS.reshape(N4,N4*Nvert,Nlgn,Nlgn)
	cntr_lvl = 0.1
	## colormap for on/off subfields
	cmap = plt.get_cmap('viridis')
	cNorm = matplotlib.colors.Normalize(vmin=0,vmax=N4*N4*Nvert)
	cmap_sclw = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
	cmap_sclw.set_array([])
	cNorm = matplotlib.colors.Normalize(vmin=0,vmax=N4*Nvert)
	cmap_vis = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
	cmap_vis.set_array([])
	for i in range(N4):
		for j in range(N4*Nvert):
			col_w = cmap_sclw.to_rgba(i*N4*Nvert+j)
			# ax1.contour(deltaS[i,j,:,:],[-cntr_lvl,cntr_lvl],colors=["b","r"],alpha=0.4,\
			# 			linestyles=["solid","solid"],origin="upper")
			c_on=axes[0].contour(deltaS[i,j,:,:],[cntr_lvl],colors=["r"],alpha=0.4,)
			c_off=axes[0].contour(deltaS[i,j,:,:],[-cntr_lvl],colors=["b"],alpha=0.4,\
									linestyles=["solid"])
			# im=ax3.imshow(deltaS[i,j,:,:],interpolation="nearest",cmap="RdBu_r")
			# # plt.colorbar(im,ax=ax3)
			len_c_on = []
			for c in c_on.allsegs[0]:
				len_c_on.append(len(c))
			len_c_on = np.array(len_c_on)
			on_id = np.argsort(len_c_on)[::-1]
			# longest_on_id = on_id[0]
			len_c_off = []
			for c in c_off.allsegs[0]:
				len_c_off.append(len(c))
			len_c_off = np.array(len_c_off)
			off_id = np.argsort(len_c_off)[::-1]
			# longest_off_id = off_id[0]
			for k,c in enumerate(c_on.allsegs[0]):
				## show center of longest contour only like in Lee et al 2016
				if k not in on_id[:1]:
					continue
				M = center_of_mass(c)
				if (M[0]>=0 and M[0]<=Nlgn and M[1]<=Nlgn and M[1]>=0):
					axes[0].add_patch(plt.Circle((M[0],M[1]),radius=0.5,color="r"))
					axes[1].add_patch(plt.Circle((M[0],M[1]),radius=0.5,color=col_w))
					axes[4].add_patch(plt.Circle((j,i),radius=0.5,color=cmap_vis.to_rgba(M[0])))
					axes[5].add_patch(plt.Circle((j,i),radius=0.5,color=cmap_vis.to_rgba(M[1])))
			for k,c in enumerate(c_off.allsegs[0]):
				if k not in off_id[:1]:
					continue
				M = center_of_mass(c)
				if (M[0]>=0 and M[0]<=(Nlgn) and M[1]<=(Nlgn) and M[1]>=0):
					axes[0].add_patch(plt.Circle((M[0],M[1]),radius=0.5,color="b"))
					axes[2].add_patch(plt.Circle((M[0],M[1]),radius=0.5,color=col_w))
					axes[6].add_patch(plt.Circle((j,i),radius=0.5,color=cmap_vis.to_rgba(M[0])))
					axes[7].add_patch(plt.Circle((j,i),radius=0.5,color=cmap_vis.to_rgba(M[1])))
					# print(i,j,M[0],cmap_sclw.to_rgba(M[0]))
			for ax in axes[4:]:
				ax.set_xlim(0,N4*Nvert)
				ax.set_ylim(0,N4)
			for ax in axes[:4]:
				ax.set_ylim(0,Nlgn)
				ax.set_xlim(0,Nlgn)
			axes[0].set_title("Contour RF")
			axes[1].set_title("Center ON RF")
			axes[2].set_title("Center OFF RF")
			axes[4].set_title("x-location ON")
			axes[5].set_title("y-location ON")
			axes[6].set_title("x-location OFF")
			axes[7].set_title("y-location OFF")

	return fig


def plot_connectivity(W4to4,**kwargs):
	"""creates figures to display rec connectivity of 2pop network
	kwargs must contain: "N4", "Nvert", "Wrec_mode", output_dict
	"""

	## Network params
	N4 = kwargs["N4"]
	Nvert = kwargs["Nvert"]
	Wrec_mode = kwargs["Wrec_mode"]
	fig_list = []

	##  matrix figures of rec connectivity
	nrow,ncol = 2,2
	if "2pop" in Wrec_mode:
		nrow = 5
	fig = plt.figure(figsize=(6*ncol,nrow*5))
	ax = fig.add_subplot(nrow,ncol,1)
	ax.set_title("Total connectivity")
	im=ax.imshow(W4to4,interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(nrow,ncol,2)
	ax.set_title("E to E")
	im=ax.imshow(W4to4[:N4**2*Nvert,:N4**2*Nvert],interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	Wee = W4to4[:N4**2*Nvert,:N4**2*Nvert]
	Wie = W4to4[N4**2*Nvert:,:N4**2*Nvert]
	Wei = W4to4[:N4**2*Nvert,N4**2*Nvert:]
	Wii = W4to4[N4**2*Nvert:,N4**2*Nvert:]
	ax = fig.add_subplot(nrow,ncol,3)
	ax.set_title("Total # of incoming E conn (E)")
	im=ax.imshow(np.sum(Wee>0,axis=1).reshape(N4,N4*Nvert),interpolation="nearest",\
				cmap="binary")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(nrow,ncol,4)
	ax.set_title("Total weight of incoming E conn (E)")
	im=ax.imshow(np.sum(Wee,axis=1).reshape(N4,N4*Nvert),interpolation="nearest",\
				cmap="binary",vmin=np.nanmin(np.sum(Wee,axis=1)),\
				vmax=np.nanmax(np.sum(Wee,axis=1)))
	plt.colorbar(im,ax=ax)
	if "2pop" in Wrec_mode:
		ax = fig.add_subplot(nrow,ncol,5)
		ax.set_title("Total # of incoming E conn (I)")
		im=ax.imshow(np.sum(Wie>0,axis=1).reshape(N4,N4*Nvert),interpolation="nearest",\
					cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,6)
		ax.set_title("Total weight of incoming E conn (I)")
		im=ax.imshow(np.sum(Wie,axis=1).reshape(N4,N4*Nvert),interpolation="nearest",\
					cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,7)
		ax.set_title("Total # of incoming I conn (E)")
		im=ax.imshow(np.sum(Wei<0,axis=1).reshape(N4,N4*Nvert),interpolation="nearest",\
					cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,8)
		ax.set_title("Total weight of incoming I conn (E)")
		im=ax.imshow(np.sum(Wei,axis=1).reshape(N4,N4*Nvert),interpolation="nearest",\
					cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,9)
		ax.set_title("Total # of incoming I conn (I)")
		im=ax.imshow(np.sum(Wii<0,axis=1).reshape(N4,N4*Nvert),interpolation="nearest",\
					cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,10)
		ax.set_title("Total weight of incoming I conn (I)")
		im=ax.imshow(np.sum(Wii,axis=1).reshape(N4,N4*Nvert),interpolation="nearest",\
					cmap="binary")
		plt.colorbar(im,ax=ax)
	fig_list.append(fig)

	## clustering of conections
	if "Gaussian_prob_cluster" in kwargs["Wrec_mode"]:
		if "output_dict" in kwargs.keys():
			cluster_units = kwargs["output_dict"]["EE"]["phase_field_binned"]
			print("cluster_units",cluster_units.shape)
			fig = plt.figure(figsize=(12,5))
			fig.suptitle("Clusters of units used for rec connectivity")
			ax = fig.add_subplot(121)
			im=ax.imshow(cluster_units.reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="viridis")
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(122)
			ax.plot(np.arange(1,len(np.unique(cluster_units))+1),\
					np.bincount(cluster_units)[1:],"sk")
			ax.set_xlabel("Cluster index")
			ax.set_xlim(0.8,len(np.unique(cluster_units))+0.2)
			fig_list.append(fig)


	## examples of individual connectivity matrices
	idx = (N4*N4*Nvert)//2 + ((N4+1)%2) * N4*Nvert//2
	fig = plt.figure()
	fig.suptitle("Examples")
	ax = fig.add_subplot(221)
	ax.set_title("E to E conn")
	im=ax.imshow(W4to4[:N4*N4*Nvert,idx].reshape(N4,N4*Nvert),interpolation="nearest",\
		 		cmap="binary",vmin=0)
	plt.colorbar(im,ax=ax)
	if "2pop" in Wrec_mode:
		ax = fig.add_subplot(222)
		ax.set_title("I to E conn")
		im=ax.imshow(-W4to4[:N4*N4*Nvert,idx-N4**2*Nvert].reshape(N4,N4*Nvert),\
					 interpolation="nearest",cmap="binary",vmin=0)
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(223)
		ax.set_title("E to I conn")
		im=ax.imshow(W4to4[N4*N4*Nvert:,idx].reshape(N4,N4*Nvert),interpolation="nearest",\
					 cmap="binary",vmin=0)
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(224)
		ax.set_title("I to I conn")
		im=ax.imshow(-W4to4[N4*N4*Nvert:,idx+N4**2*Nvert].reshape(N4,N4*Nvert),\
					 interpolation="nearest",cmap="binary",vmin=0)
		plt.colorbar(im,ax=ax)
	fig_list.append(fig)
	plt.close(fig)

	if "2pop" in Wrec_mode:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		print("idx",idx,N4)
		# ax.plot(W4to4[:N4*N4*Nvert,idx].reshape(N4,N4*Nvert)[:,N4//2]/np.max(W4to4[:N4*N4*Nvert,idx].reshape(N4,N4*Nvert)[:,N4//2]),"-r",label="EtoE")
		# ax.plot(W4to4[N4*N4*Nvert:,idx+N4**2*Nvert].reshape(N4,N4*Nvert)[:,N4//2]/np.min(W4to4[N4*N4*Nvert:,idx+N4**2*Nvert].reshape(N4,N4*Nvert)[:,N4//2]),"-c",label="ItoE")
		# ax.plot(W4to4[:N4*N4*Nvert,idx].reshape(N4,N4*Nvert)[:,N4//2]/np.max(W4to4[:N4*N4*Nvert,idx].reshape(N4,N4*Nvert)[:,N4//2]),"-m",label="EtoI")
		# ax.plot(W4to4[N4*N4*Nvert:,idx+N4**2*Nvert].reshape(N4,N4*Nvert)[:,N4//2]/np.min(W4to4[N4*N4*Nvert:,idx+N4**2*Nvert].reshape(N4,N4*Nvert)[:,N4//2]),"-b",label="ItoI")
		ax.plot(W4to4[:N4*N4*Nvert,idx].reshape(N4,N4*Nvert)[:,N4//2]/np.nanmax(W4to4[:N4*N4*Nvert,idx].reshape(N4,N4*Nvert)[:,N4//2]),"-r",label="EtoE")
		ax.plot(W4to4[N4*N4*Nvert:,idx].reshape(N4,N4*Nvert)[:,N4//2]/np.nanmax(W4to4[N4*N4*Nvert:,idx].reshape(N4,N4*Nvert)[:,N4//2]),"-m",label="EtoI")
		ax.plot(W4to4[:N4*N4*Nvert,idx+N4**2*Nvert].reshape(N4,N4*Nvert)[:,N4//2]/np.min(W4to4[:N4*N4*Nvert,idx+N4**2*Nvert].reshape(N4,N4*Nvert)[:,N4//2]),"-c",label="ItoE")
		ax.plot(W4to4[N4*N4*Nvert:,idx+N4**2*Nvert].reshape(N4,N4*Nvert)[:,N4//2]/np.min(W4to4[N4*N4*Nvert:,idx+N4**2*Nvert].reshape(N4,N4*Nvert)[:,N4//2]),"-b",label="ItoI")
		ax.legend(loc="best")
		fig_list.append(fig)
		plt.close(fig)

	## cluster one of the four connectivity matrices and apply clustering to 
	## remaining three
	if "2pop" in Wrec_mode:
		Wee = W4to4[:N4**2*Nvert,:N4**2*Nvert]
		Wei = W4to4[:N4**2*Nvert,N4**2*Nvert:]
		Wie = W4to4[N4**2*Nvert:,:N4**2*Nvert]
		Wii = W4to4[N4**2*Nvert:,N4**2*Nvert:]
		W_cl,leaves,linkage = analysis_tools.hierarchical_clustering_of_correlation(Wei)
		Wee_cl = (Wee[leaves].T)[leaves]
		Wie_cl = (Wie[leaves].T)[leaves]
		Wii_cl = (Wii[leaves].T)[leaves]
		fig = plt.figure(figsize=(18,5))
		ax = fig.add_subplot(141)
		ax.set_title("clustered EE conn")
		im=ax.imshow(Wee_cl,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(142)
		ax.set_title("clustered EI conn")
		im=ax.imshow(-W_cl,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(143)
		ax.set_title("clustered IE conn")
		im=ax.imshow(Wie_cl,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(144)
		ax.set_title("clustered II conn")
		im=ax.imshow(-Wii_cl,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		fig_list.append(fig)

	## EIGENVECTORS and EIGENVALUES of rec conn matrix
	eigvals,eigvec = linalg.eig(W4to4)
	eigvec = eigvec[:,np.argsort(np.real(eigvals))[::-1]]
	eigvals_sorted = np.sort(np.real(eigvals))[::-1]
	print("max,min ev",eigvals_sorted[0],eigvals_sorted[-1])
	fig = plt.figure(figsize=(18,5))
	fig.suptitle("Max ev={:.2f}, min ev={:.5f}".format(eigvals_sorted[0],eigvals_sorted[-1]))
	ax = fig.add_subplot(131)
	ax.plot(eigvals_sorted,'-k')
	ax.set_xlabel("Index")
	ax.set_ylabel("Eigenvalues")
	ax = fig.add_subplot(132)
	ax.plot(eigvals_sorted[:20],'-k')
	ax.set_xlabel("Index")
	ax.set_ylabel("Eigenvalues")
	ax = fig.add_subplot(133)
	ax.plot(np.real(eigvals),np.imag(eigvals),"ok",alpha=0.4)
	ax.set_xlabel("real part eigval")
	ax.set_ylabel("imaginary part eigval")
	fig_list.append(fig)
	nrow,ncol = 2,10
	fig = plt.figure(figsize=(6*ncol,nrow*5))
	for jcol in range(ncol):
		ax = fig.add_subplot(nrow,ncol,1+jcol)
		ax.set_title("Real EV {}".format(jcol+1))
		ax.imshow(np.real(eigvec[:,jcol]).reshape(-1,N4*Nvert),\
					interpolation="nearest",cmap="binary")
		ax = fig.add_subplot(nrow,ncol,1+jcol+ncol)
		ax.set_title("Imag EV {}".format(jcol+1))
		ax.imshow(np.real(eigvec[:N4*N4*Nvert,jcol]).reshape(N4,N4*Nvert),\
					interpolation="nearest",cmap="binary")
	fig_list.append(fig)
	return fig_list


def plot_LGN_input_corr(lgn,**kwargs):
	"""creates figures to display properties of LGN input (amplitude, pattern,correlations)
	kwargs must contain:  "Nlgn"
	"""
	fig_list = []
	# parameters
	N = kwargs["Nlgn"]
	num_lgn_paths = lgn.shape[0]

	labels = ["ON to E", "OFF to E", "ON to I", "OFF to I"]
	figl = plt.figure(figsize=(2*6,2*5))
	figl.suptitle("LGN Input amplitude")
	ax = figl.add_subplot(221)
	for i in range(num_lgn_paths):
		ax.plot(np.nanmean(lgn[i,:,:],axis=0),'-',label=labels[i])
	ax.set_ylabel("LGN mean input")
	ax.set_xlabel("Timesteps")
	ax.legend(loc="best")
	ax = figl.add_subplot(222)
	# ax.plot(np.nanmean(lgn[0,:,:],axis=0),'-')
	# ax.plot(np.nanmean(lgn[1,:,:],axis=0),'-')
	for i in range(num_lgn_paths):
		ax.plot(lgn[i,N-1,:].T,"-",label=labels[i])
	ax.set_ylabel("LGN input (one pxl)")
	ax.set_xlabel("Timesteps")
	ax.legend(loc="best")
	ax = figl.add_subplot(223)
	ax.set_title("On input example to E")
	im=ax.imshow(lgn[0,:,0].reshape([N,N]),interpolation="nearest",\
		cmap="binary")
	plt.colorbar(im,ax=ax)
	ax = figl.add_subplot(224)
	ax.set_title("Off input example to E")
	im=ax.imshow(lgn[1,:,0].reshape([N,N]),interpolation="nearest",\
		cmap="binary")
	plt.colorbar(im,ax=ax)
	fig_list.append(figl)


	##lgn correlations
	lgn = lgn[:,:,:50].reshape(num_lgn_paths,N*N,-1)
	lgn_norm = (lgn - np.nanmean(lgn,axis=2)[:,:,None])/np.nanstd(lgn,axis=2)[:,:,None]
	cc = np.nanmean(lgn_norm[:,None,:,None,:] * lgn_norm[None,:,None,:,:],axis=4)
	# cc = np.nanmean(lgn[:,None,:,None,:] * lgn[None,:,None,:,:],axis=4)
	cc = cc.reshape(num_lgn_paths,num_lgn_paths,N,N,N,N)
	labels = ["ON-ON","ON-OFF","ON-ON - ON-OFF","OFF-OFF"]
	fig = plt.figure(figsize=(4*6,5))
	fig.suptitle("LGN input correlations (to E)")
	for i in range(2):
		for j in range(2):
			ax = fig.add_subplot(1,4,j+1+i*2)
			ax.set_title(labels[j+i*2])
			if not (i==1 and j==0):
				im=ax.imshow(cc[i,j,N//2,N//2].reshape(N,N),interpolation="nearest",\
							 cmap="RdBu_r",vmin=-0.75,vmax=0.75)
			else:
				diff = (0.5*cc[0,0,N//2,N//2]+0.5*cc[1,1,N//2,N//2]-cc[0,1,N//2,N//2]).reshape(N,N)
				im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
			plt.colorbar(im,ax=ax)
	fig_list.append(fig)

	## average (over units) correlation pattern
	cc_avg = 0
	cc[np.logical_not(np.isfinite(cc))] = 0.
	for i in range(N):
		for j in range(N):
			cc_avg += np.roll(np.roll(cc[:,:,i,j,:,:],N//2-i,axis=2),N//2-j,axis=3)
	cc_avg /= N*N
	fig = plt.figure(figsize=(4*6,5))
	fig.suptitle("Avg LGN input correlations (to E)")
	for i in range(2):
		for j in range(2):
			ax = fig.add_subplot(1,4,j+1+i*2)
			ax.set_title(labels[j+i*2])
			if not (i==1 and j==0):
				im=ax.imshow(cc_avg[i,j].reshape(N,N),interpolation="nearest",\
							 cmap="RdBu_r",vmin=-0.75,vmax=0.75)
			else:
				diff = (0.5*cc_avg[0,0]+0.5*cc_avg[1,1]-cc_avg[0,1]).reshape(N,N)
				im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r")#,vmin=0,vmax=1)
			plt.colorbar(im,ax=ax)
	fig_list.append(fig)

	if num_lgn_paths==4:
		##lgn correlations
		fig = plt.figure(figsize=(4*6,5))
		fig.suptitle("LGN input correlations (to I)")
		for i in range(2):
			for j in range(2):
				ax = fig.add_subplot(1,4,j+1+i*2)
				ax.set_title(labels[j+i*2])
				if not (i==1 and j==0):
					im=ax.imshow(cc[i+2,j+2,N//2,N//2].reshape(N,N),interpolation="nearest",\
								 cmap="RdBu_r",vmin=-0.75,vmax=0.75)
				else:
					diff = (0.5*cc[2,2,N//2,N//2]+0.5*cc[3,3,N//2,N//2]-cc[2,3,N//2,N//2]).reshape(N,N)
					im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
				plt.colorbar(im,ax=ax)
		fig_list.append(fig)

		## average (over units) correlation pattern
		fig = plt.figure(figsize=(4*6,5))
		fig.suptitle("Avg LGN input correlations (to I)")
		for i in range(2):
			for j in range(2):
				ax = fig.add_subplot(1,4,j+1+i*2)
				ax.set_title(labels[j+i*2])
				if not (i==1 and j==0):
					im=ax.imshow(cc_avg[i+2,j+2].reshape(N,N),interpolation="nearest",\
								 cmap="RdBu_r",vmin=-0.75,vmax=0.75)
				else:
					diff = (0.5*cc_avg[2,2]+0.5*cc_avg[3,3]-cc_avg[2,3]).reshape(N,N)
					im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
				plt.colorbar(im,ax=ax)
		fig_list.append(fig)


	## average (over units) covariance pattern
	cov_avg = 0
	lgn_norm = (lgn - np.nanmean(lgn,axis=2)[:,:,None])
	cov = np.nanmean(lgn_norm[:,None,:,None,:] * lgn_norm[None,:,None,:,:],axis=4)
	cov = cov.reshape(num_lgn_paths,num_lgn_paths,N,N,N,N)
	cov[np.logical_not(np.isfinite(cov))] = 0.
	for i in range(N):
		for j in range(N):
			cov_avg += np.roll(np.roll(cov[:,:,i,j,:,:],N//2-i,axis=2),N//2-j,axis=3)
	cov_avg /= N*N
	fig = plt.figure(figsize=(4*6,5))
	fig.suptitle("Avg LGN input covariance (to E)")
	for i in range(2):
		for j in range(2):
			ax = fig.add_subplot(1,4,j+1+i*2)
			ax.set_title(labels[j+i*2])
			if not (i==1 and j==0):
				im=ax.imshow(cov_avg[i,j].reshape(N,N),interpolation="nearest",cmap="RdBu_r")
			else:
				diff = (0.5*cov_avg[0,0]+0.5*cov_avg[1,1]-cov_avg[0,1]).reshape(N,N)
				im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r")
			plt.colorbar(im,ax=ax)
	fig_list.append(fig)

	return fig_list


def anim_video(data_init, data, output_file):
	fig = plt.figure()
	big_ax = fig.add_subplot(111)
	big_ax.set_facecolor('none')
	#big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	#fig.subplots_adjust(left=0.01, bottom=0, right=0.99, top=1, wspace=None, hspace=0)
	
	# time_text = big_ax.text(250,300,'-1 s')
	# im = big_ax.imshow(activity[0,:,:],cmap='binary',interpolation='nearest', vmin=0, vmax=255,animated=True)#, aspect=270/320.)
	# big_ax.contour(maps_car,[-0.005],colors='r',linewidths=0.5)
	# big_ax.contour(maps_car,[+0.005],colors='g',linewidths=0.5)
	# big_ax.set_xticks([])
	# big_ax.set_yticks([])

	_,axes,ims = grid_plot_twolayer(data_init,fig)
	ncol = len(data[0])
	nrow = len(data)

	def updatefig(j):
		for icol in range(ncol):
			for jrow in range(nrow):
				# print("shape",data_init[jrow][icol].ndim)
				if data_init[jrow][icol].ndim==2: 
					vmin = np.nanmin(data[jrow][icol][j])
					vmax = np.nanmax(data[jrow][icol][j])
					ims[jrow+icol*nrow].set_array(data[jrow][icol][j])
					ims[jrow+icol*nrow].set_clim(vmin,vmax)
					# axes[jrow+icol*nrow].cla()
					# fig.colorbar(ims[jrow+icol*nrow], cax=axes[jrow+icol*nrow])
					# time_text.set_text(labels[j]+' {:.0f} s'.format(np.floor((j%90)/15.-1)))
				else:
					ims[jrow+icol*nrow].set_data(j, data[jrow][icol][j])
				return ims#,time_text
			

	ani = matplotlib.animation.FuncAnimation(fig, updatefig, frames=data[0][0].shape[0],\
			 blit=True)
	# FFwriter = animation.FFMpegWriter(fps=15, bitrate=2000, extra_args=['-vcodec', 'mpeg4'])
	# ani.save(output_file, writer=FFwriter)
	ani.save(output_file, fps=15, dpi=300, extra_args=['-vcodec', 'mpeg4'])
	# ani.to_html5_video()


def grid_plot_twolayer(data,fig=None,ncol=None,nrow=None,vmin=None,vmax=None):
	"""helper function for anim_video"""
	if ncol is None:
		if isinstance(data,list):
			ncol = len(data[0])
			nrow = len(data)
		else:
			nplots = data.shape[0]
			ncol = int(np.ceil(np.sqrt(np.plots)))
			nrow = int(np.ceil(1.*nplots/ncol))
	if fig is None:
		fig = plt.figure(figsize=(6*ncol,5*nrow))
	axes = []
	ims = []
	for icol in range(ncol):
		for jrow in range(nrow):
			ax = fig.add_subplot(nrow,ncol,jrow+icol*nrow+1)
			if isinstance(data,list):
				if jrow>=len(data):
					continue
				this_data = data[jrow][icol]
			else:
				if (jrow+icol*nrow)>=data.shape[0]:
					continue
				this_data = data[jrow+icol*nrow,...]
			if this_data.ndim==2:
				ax.set_title(str(jrow+icol*nrow))
				im = ax.imshow(this_data,interpolation="nearest",cmap="binary",\
								vmin=vmin,vmax=vmax)
				# plt.colorbar(im,ax=ax)

				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				plt.colorbar(im, cax = cax)

			else:
				im = ax.plot(this_data,"-")
				
			axes.append(ax)
			ims.append(im)

	return fig,axes,ims

def plot_dev_connectivity(conn_t,plastic=False):

	fig_list = []
	fig = plt.figure(figsize=(6,5))
	ax = fig.add_subplot(111)
	ax.set_title("Connectivity Matrix")
	im=ax.imshow(conn_t[-1,:,:],interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	fig_list.append(fig)

	if plastic:
		timesteps,N,N = conn_t.shape
		N4,N23 = int(np.sqrt(N//2)),int(np.sqrt(N//2))
		print("N4",N4,N23,conn_t.shape)
		fig = plt.figure(figsize=(18,5))
		ax = fig.add_subplot(131)
		ax.plot(conn_t[:,N//2,:],'-')
		ax.set_ylabel("Indiv connectivity weights")
		ax = fig.add_subplot(132)
		ax.plot(conn_t[:,N//4,:],'-')
		ax.set_xlabel("Timesteps (saved)")
		ax = fig.add_subplot(133)
		ax.plot(conn_t[:,0,:],'-')
		fig_list.append(fig)

		ncol,nrow = 3,2
		fig = plt.figure(figsize=(6*ncol,5*nrow))
		timepoints = 1,timesteps//2,timesteps-1
		for i,it in enumerate(timepoints):
			conn_EtoE,_,_,_ = analysis_tools.get_RF_form(conn_t[it,:N//2,:N//2],N23,\
												N4,int(N23*0.4+4),calc_PF=False,\
												Nvert=1,mode="other")
			ax = fig.add_subplot(nrow,ncol,i+1)
			ax.set_title("t={}".format(it))
			im=ax.imshow(conn_EtoE,interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax)
			conn_EtoI,_,_,_ = analysis_tools.get_RF_form(conn_t[it,N//2:,:N//2],N23,\
													N4,int(N23*0.4+4),calc_PF=False,\
													Nvert=1,mode="other")
			ax = fig.add_subplot(nrow,ncol,i+ncol+1)
			im=ax.imshow(conn_EtoI,interpolation="nearest",cmap="binary")
			plt.colorbar(im,ax=ax)
		fig_list.append(fig)

	return fig_list


