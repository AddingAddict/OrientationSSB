# import tensorflow as tf
import numpy as np
from dev_ori_sel_RF.tools import calc_tools



def expand_weights_to_orthog_space(weights,c_orth,Nout,Ninp,arbor):
	"""expand ff weight vector w into orthogonal space for divisive normalisation
	(miller and macckay 1992); basically replace s by w
	"""
	arbor_on = arbor[0,:,:]
	arbor_off = arbor[1,:,:]
	arb_bool_on = (arbor_on>0)
	arb_bool_off = (arbor_off>0)
	arb_idx_on = np.arange(1,1+Ninp)[None,:]*arb_bool_on.astype(int)
	arb_idx_off = np.arange(1+Ninp,1+2*Ninp)[None,:]*arb_bool_off.astype(int)

	arbsize_on = np.sum(arb_bool_on)
	arbsize = arbsize_on + np.sum(arb_bool_off)

	arbarea_on = np.sum(arb_bool_on,axis=1)
	arbarea_cs_on = np.concatenate([np.array([0]),np.cumsum(arbarea_on)])
	arbarea_off = np.sum(arb_bool_off,axis=1)
	arbarea_cs_off = np.concatenate([np.array([0]),np.cumsum(arbarea_off)])


	s_vec = np.zeros((Ninp*2+Nout,arbsize))
	c_help = np.zeros((arbsize),dtype=int)
	for i in range(Nout):
		c_help[arbarea_cs_on[i]:arbarea_cs_on[i+1]] = arb_idx_on[i,arb_bool_on[i,:]]
		c_help[arbarea_cs_off[i]+arbsize_on:arbarea_cs_off[i+1]+arbsize_on] =\
														 arb_idx_off[i,arb_bool_off[i,:]]

		s_vec[i+2*Ninp,arbarea_cs_on[i]:arbarea_cs_on[i+1]] = weights[0,i,arb_bool_on[i,:]]
		s_vec[i+2*Ninp,arbarea_cs_off[i]+arbsize_on:arbarea_cs_off[i+1]+arbsize_on] =\
														  weights[1,i,arb_bool_off[i,:]]


	# s_vec = np.zeros((Ninp*2+Nout,2*arbsize))
	# c_help = np.zeros((2*arbsize),dtype=int)
	# for i in range(Nout):
	# 	c_help[arbarea_cs[i]:arbarea_cs[i+1]] = arb_idx[i,arb_bool[i,:]]
	# 	c_help[arbarea_cs[i]+arbsize:arbarea_cs[i+1]+arbsize] = arb_idx[i,arb_bool[i,:]] + Ninp

	# 	s_vec[i+2*Ninp,arbarea_cs[i]:arbarea_cs[i+1]] = weights[0,i,arb_bool[i,:]]
	# 	s_vec[i+2*Ninp,arbarea_cs[i]+arbsize:arbarea_cs[i+1]+arbsize] =\
	# 													 weights[1,i,arb_bool[i,:]]

	s_sum = np.sum(s_vec,axis=0)
	for i in range(1,1+Ninp):
		this_alpha = (c_help==i)
		s_vec[i-1,this_alpha] = s_sum[this_alpha]
	for i in range(1,1+Ninp):
		this_alpha = (c_help==(i+Ninp))
		s_vec[i-1+Ninp,this_alpha] = s_sum[this_alpha]

	s_orth = calc_tools.gs_fast(s_vec, 1, row_vecs=True, norm=True)
	print("s_orth",np.nanmin(s_orth),np.nanmax(s_orth),s_orth[0,:5],c_orth[0,:5])
	# print("c_orth",np.nanmin(c_orth),np.nanmax(c_orth))
	# print("dot product c*w",np.diag(np.dot(s_orth,c_orth.T)))
	# s_orth,r = np.linalg.qr(s_vec.T,mode='complete')	#q: orthonormal, r: upper triangular
	# s_orth = s_orth[:,:s_vec.shape[0]-1].T
	# s_orth = s_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None]
	# print("s_orth2",np.nanmin(s_orth),np.nanmax(s_orth),s_orth[0,:5],c_orth[0,:5])

	return s_orth


def generate_simIO_normalisation(Nlgn,N4,arbor,Nvert=1):
	"""generate s and c vectors for normalisation operator in t-basis [Miller and Mackay 1992]
	s = An^r
	c = n^r
	"""
	Nout = N4**2*Nvert
	Ninp = Nlgn**2
	arb_bool = (arbor>0)
	arb_idx = np.arange(1,1+Ninp)[None,:]*arb_bool.astype(int)
	arbor_sq = np.sqrt(arbor)

	arbsize = np.sum(arb_bool)
	arbarea = np.sum(arb_bool,axis=1)#np.sum(arb_bool[0,:])
	arbarea_cs = np.concatenate([np.array([0]),np.cumsum(arbarea)])

	c_vec = np.zeros((Ninp*2+Nout,2*arbsize))
	s_vec = np.zeros((Ninp*2+Nout,2*arbsize))
	c_help = np.zeros((2*arbsize),dtype=int)
	for i in range(Nout):
		# c_vec[i+2*Ninp,i*arbarea[i]:(i+1)*arbarea[i]] = arbor_sq[i,arb_bool[i,:]]
		# c_vec[i+2*Ninp,i*arbarea[i]+arbsize:(i+1)*arbarea[i]+arbsize] = arbor_sq[i,arb_bool[i,:]]
		# c_help[i*arbarea[i]:(i+1)*arbarea[i]] = arb_idx[i,arb_bool[i,:]]

		c_vec[i+2*Ninp,arbarea_cs[i]:arbarea_cs[i+1]] = arb_bool[i,arb_bool[i,:]]
		c_vec[i+2*Ninp,arbarea_cs[i]+arbsize:arbarea_cs[i+1]+arbsize] =\
														 arb_bool[i,arb_bool[i,:]]
		c_help[arbarea_cs[i]:arbarea_cs[i+1]] = arb_idx[i,arb_bool[i,:]]

		s_vec[i+2*Ninp,arbarea_cs[i]:arbarea_cs[i+1]] = arbor[i,arb_bool[i,:]]
		s_vec[i+2*Ninp,arbarea_cs[i]+arbsize:arbarea_cs[i+1]+arbsize] =\
														 arbor[i,arb_bool[i,:]]

	c_sum = np.sum(c_vec,axis=0)
	for i in range(1,1+Ninp):
		this_alpha = (c_help==i)
		c_vec[i-1,this_alpha] = c_sum[this_alpha]
	c_vec[Ninp:Ninp*2,arbsize:] = c_vec[:Ninp,:arbsize]

	s_sum = np.sum(s_vec,axis=0)
	for i in range(1,1+Ninp):
		this_alpha = (c_help==i)
		s_vec[i-1,this_alpha] = s_sum[this_alpha]
	s_vec[Ninp:Ninp*2,arbsize:] = s_vec[:Ninp,:arbsize]

	c_orth = calc_tools.gs_fast(c_vec, 2*Ninp, row_vecs=True, norm=True)
	s_orth = calc_tools.gs_fast(s_vec, 2*Ninp, row_vecs=True, norm=True)
	## normalise one of the vectors to simplify computation of normalisation
	s_orth = s_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None]

	# print("COMMENT OUT FOLLOWING LINE")
	# p_orth = np.dot(s_orth.T,c_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None])

	# coefficients = np.dot(c_vec,c_orth.T)
	# print("coefficients",coefficients.shape)
	# print("TODO: change output ")
	return c_orth,s_orth#,p_orth,s_vec,c_vec


def generate_simIO_normalisation_onoff(Nlgn,N4,arbor_on,arbor_off,Nvert=1):
	"""generate s and c vectors for normalisation operator in t-basis [Miller and Mackay 1992]
	s = An^r
	c = n^r
	"""
	Nout = N4**2*Nvert
	Ninp = Nlgn**2
	arb_bool_on = (arbor_on>0)
	arb_bool_off = (arbor_off>0)
	arb_idx_on = np.arange(1,1+Ninp)[None,:]*arb_bool_on.astype(int)
	arb_idx_off = np.arange(1+Ninp,1+2*Ninp)[None,:]*arb_bool_off.astype(int)

	arbsize_on = np.sum(arb_bool_on)
	arbsize = arbsize_on + np.sum(arb_bool_off)

	arbarea_on = np.sum(arb_bool_on,axis=1)
	arbarea_cs_on = np.concatenate([np.array([0]),np.cumsum(arbarea_on)])
	arbarea_off = np.sum(arb_bool_off,axis=1)
	arbarea_cs_off = np.concatenate([np.array([0]),np.cumsum(arbarea_off)])

	c_vec = np.zeros((Ninp*2+Nout,arbsize))
	s_vec = np.zeros((Ninp*2+Nout,arbsize))
	c_help = np.zeros((arbsize),dtype=int)
	for i in range(Nout):
		c_vec[i+2*Ninp,arbarea_cs_on[i]:arbarea_cs_on[i+1]] = arb_bool_on[i,arb_bool_on[i,:]]
		c_vec[i+2*Ninp,arbarea_cs_off[i]+arbsize_on:arbarea_cs_off[i+1]+arbsize_on] =\
														 arb_bool_off[i,arb_bool_off[i,:]]
		c_help[arbarea_cs_on[i]:arbarea_cs_on[i+1]] = arb_idx_on[i,arb_bool_on[i,:]]
		c_help[arbarea_cs_off[i]+arbsize_on:arbarea_cs_off[i+1]+arbsize_on] =\
														 arb_idx_off[i,arb_bool_off[i,:]]

		s_vec[i+2*Ninp,arbarea_cs_on[i]:arbarea_cs_on[i+1]] = arbor_on[i,arb_bool_on[i,:]]
		s_vec[i+2*Ninp,arbarea_cs_off[i]+arbsize_on:arbarea_cs_off[i+1]+arbsize_on] =\
														 arbor_off[i,arb_bool_off[i,:]]

	c_sum = np.sum(c_vec,axis=0)
	for i in range(1,1+2*Ninp):
		this_alpha = (c_help==i)
		c_vec[i-1,this_alpha] = c_sum[this_alpha]

	s_sum = np.sum(s_vec,axis=0)
	for i in range(1,1+2*Ninp):
		this_alpha = (c_help==i)
		s_vec[i-1,this_alpha] = s_sum[this_alpha]

	c_orth = calc_tools.gs_fast(c_vec, Ninp, row_vecs=True, norm=True)
	s_orth = calc_tools.gs_fast(s_vec, Ninp, row_vecs=True, norm=True)
	## normalise one of the vectors to simplify computation of normalisation
	s_orth = s_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None]

	# print("COMMENT OUT FOLLOWING LINE")
	# p_orth = np.dot(s_orth.T,c_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None])
	# print("TODO: change output ")
	return c_orth,s_orth#,p_orth,s_vec


def generate_simIO_normalisation_oneUnittype(N4,N23,arbor,Nvert=1):
	"""generate orthogonalised normalisation vectors
	assuming E,I units receive/project with same arbor
	s = An^r
	c = n^r
	input : arbor.shape = N23**2 x N4**2
	"""
	Nout = N23**2
	Ninp = N4**2*Nvert
	arb_bool = (arbor>0)
	arb_idx = np.arange(1,1+Ninp)[None,:]*arb_bool.astype(int)
	arbor_sq = np.sqrt(arbor)

	arbsize = np.sum(arb_bool)
	arbarea = np.sum(arb_bool,axis=1)
	arbarea_cs = np.concatenate([np.array([0]),np.cumsum(arbarea)])

	## order of c_vec is first axis: out E, out I, in E, in I
	## second axis: E to E, I to E, (E to I, I to I) assume only conn from E
	npop = 2
	c_vec = np.zeros((Ninp+Nout*2,npop,arbsize))
	s_vec = np.zeros((Ninp+Nout*2,npop,arbsize))
	c_help = np.zeros((arbsize),dtype=int)
	## fill first vectors for receiving conections
	for i in range(Nout):
		## E receiving/to E
		c_vec[i+Ninp,0,arbarea_cs[i]:arbarea_cs[i+1]] = arb_bool[i,arb_bool[i,:]]
		# c_vec[i+Ninp*2,1,arbarea_cs[i]:arbarea_cs[i+1]] = arb_bool[i,arb_bool[i,:]]
		## I receiving / to I
		c_vec[i+Ninp+Nout,1,arbarea_cs[i]:arbarea_cs[i+1]] = arb_bool[i,arb_bool[i,:]]
		# c_vec[i+Ninp*2+Nout,3,arbarea_cs[i]:arbarea_cs[i+1]] = arb_bool[i,arb_bool[i,:]]
		c_help[arbarea_cs[i]:arbarea_cs[i+1]] = arb_idx[i,arb_bool[i,:]]

		## E receiving/to E
		s_vec[i+Ninp,0,arbarea_cs[i]:arbarea_cs[i+1]] = arbor[i,arb_bool[i,:]]
		# s_vec[i+Ninp*2,1,arbarea_cs[i]:arbarea_cs[i+1]] = arbor[i,arb_bool[i,:]]
		## I receiving / to I
		s_vec[i+Ninp+Nout,1,arbarea_cs[i]:arbarea_cs[i+1]] = arbor[i,arb_bool[i,:]]
		# s_vec[i+Ninp*2+Nout,3,arbarea_cs[i]:arbarea_cs[i+1]] = arbor[i,arb_bool[i,:]]

	c_sum = np.sum(c_vec,axis=0)
	## fill c vectors for outgoing conections
	for i in range(1,1+Ninp):
		this_alpha = (c_help==i)
		c_vec[i-1,0,this_alpha] = c_sum[0,this_alpha]
	c_vec[:Ninp,1,:] = c_vec[:Ninp,0,:]
	# c_vec[Ninp:2*Ninp,1,:] = c_vec[:Ninp,0,:]
	# c_vec[Ninp:2*Ninp,3,:] = c_vec[:Ninp,0,:]
	c_vec = c_vec.reshape(-1,npop*arbsize)

	s_sum = np.sum(s_vec,axis=0)
	## fill s vectors for outgoing conections
	for i in range(1,1+Ninp):
		this_alpha = (c_help==i)
		s_vec[i-1,0,this_alpha] = s_sum[0,this_alpha]
	s_vec[:Ninp,1,:] = s_vec[:Ninp,0,:]
	# s_vec[Ninp:2*Ninp,1,:] = s_vec[:Ninp,0,:]
	# s_vec[Ninp:2*Ninp,3,:] = s_vec[:Ninp,0,:]
	s_vec = s_vec.reshape(-1,npop*arbsize)

	## orthogonalise arbor vectors	
	c_orth = calc_tools.gs_fast(c_vec, Ninp, row_vecs=True, norm=True)
	s_orth = calc_tools.gs_fast(s_vec, Ninp, row_vecs=True, norm=True)
	## normalise one of the vectors to simplify computation of normalisation
	c_orth = c_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None]


	## re-order second axis of c_orth, s_orth to match order of W4to23 for to E and to I
	# arbor_hstack = np.hstack([arbor,arbor])
	# arbor2 = np.concatenate([arbor_hstack,arbor_hstack])
	# arb_idx = (np.arange(1,1+Nout*2,2)[:,None]*arb_bool.astype(int))
	# arb_idx2 = (np.arange(2,2+Nout*2,2)[:,None]*arb_bool.astype(int))
	# arb_idx3 = (np.arange(N23**2*2+1,N23**2*2+1+Nout*2,2)[:,None]*arb_bool.astype(int))
	# arb_idx4 = (np.arange(N23**2*2+2,N23**2*2+2+Nout*2,2)[:,None]*arb_bool.astype(int))
	# arb_sorted = np.concatenate([arb_idx[arb_bool],arb_idx2[arb_bool],arb_idx3[arb_bool],\
	# 							arb_idx4[arb_bool]])

	# idx_cum = 0
	# c_orth_sorted = np.zeros_like(c_orth)
	# s_orth_sorted = np.zeros_like(s_orth)
	# for idx in np.unique(arb_sorted):
	# 	num_idx = np.sum(arb_sorted==idx)
	# 	c_orth_sorted[:,idx_cum:idx_cum+num_idx] = c_orth[:,arb_sorted==idx]
	# 	s_orth_sorted[:,idx_cum:idx_cum+num_idx] = s_orth[:,arb_sorted==idx]
	# 	idx_cum += num_idx
	# c_orth = c_orth_sorted
	# s_orth = s_orth_sorted

	print("COMMENT OUT FOLLOWING LINE")
	p_orth = np.dot(s_orth.T,c_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None])

	print("TODO: change output ")
	return c_orth,s_orth,p_orth


def generate_simIO_normalisation_oneUnittype_rec(N4,arbor,Nvert=1):
	"""generate orthogonalised normalisation vectors
	assuming E,I units receive/project with same arbor
	s = An^r
	c = n^r
	input : arbor.shape = N23**2 x N4**2
	"""
	N = N4**2*Nvert
	arb_bool = (arbor>0)
	arb_idx = np.arange(1,1+Ninp)[None,:]*arb_bool.astype(int)
	arbor_sq = np.sqrt(arbor)

	arbsize = np.sum(arb_bool)
	arbarea = np.sum(arb_bool,axis=1)
	arbarea_cs = np.concatenate([np.array([0]),np.cumsum(arbarea)])

	## order of c_vec is first axis: out E, out I, in E, in I
	## second axis: E to E, I to E, (E to I, I to I) assume only conn from E
	npop = 1
	c_vec = np.zeros((N+N,npop,arbsize))
	s_vec = np.zeros((N+N,npop,arbsize))
	c_help = np.zeros((arbsize),dtype=int)
	## fill first vectors for receiving conections
	for i in range(N):
		## E receiving/to E
		c_vec[i+N,0,arbarea_cs[i]:arbarea_cs[i+1]] = arb_bool[i,arb_bool[i,:]]
		# c_vec[i+N*2,1,arbarea_cs[i]:arbarea_cs[i+1]] = arb_bool[i,arb_bool[i,:]]
		# c_vec[i+N*2+N,3,arbarea_cs[i]:arbarea_cs[i+1]] = arb_bool[i,arb_bool[i,:]]
		c_help[arbarea_cs[i]:arbarea_cs[i+1]] = arb_idx[i,arb_bool[i,:]]

	c_sum = np.sum(c_vec,axis=0)
	## fill c vectors for outgoing conections
	for i in range(1,1+Ninp):
		this_alpha = (c_help==i)
		c_vec[i-1,0,this_alpha] = c_sum[0,this_alpha]
	c_vec[:Ninp,1,:] = c_vec[:Ninp,0,:]
	# c_vec[Ninp:2*Ninp,1,:] = c_vec[:Ninp,0,:]
	# c_vec[Ninp:2*Ninp,3,:] = c_vec[:Ninp,0,:]
	c_vec = c_vec.reshape(-1,npop*arbsize)

	s_sum = np.sum(s_vec,axis=0)
	## fill s vectors for outgoing conections
	for i in range(1,1+Ninp):
		this_alpha = (c_help==i)
		s_vec[i-1,0,this_alpha] = s_sum[0,this_alpha]
	s_vec[:Ninp,1,:] = s_vec[:Ninp,0,:]
	# s_vec[Ninp:2*Ninp,1,:] = s_vec[:Ninp,0,:]
	# s_vec[Ninp:2*Ninp,3,:] = s_vec[:Ninp,0,:]
	s_vec = s_vec.reshape(-1,npop*arbsize)

	## orthogonalise arbor vectors	
	c_orth = calc_tools.gs_fast(c_vec, Ninp, row_vecs=True, norm=True)
	s_orth = calc_tools.gs_fast(s_vec, Ninp, row_vecs=True, norm=True)
	## normalise one of the vectors to simplify computation of normalisation
	c_orth = c_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None]


	## re-order second axis of c_orth, s_orth to match order of W4to23 for to E and to I
	# arbor_hstack = np.hstack([arbor,arbor])
	# arbor2 = np.concatenate([arbor_hstack,arbor_hstack])
	# arb_idx = (np.arange(1,1+Nout*2,2)[:,None]*arb_bool.astype(int))
	# arb_idx2 = (np.arange(2,2+Nout*2,2)[:,None]*arb_bool.astype(int))
	# arb_idx3 = (np.arange(N23**2*2+1,N23**2*2+1+Nout*2,2)[:,None]*arb_bool.astype(int))
	# arb_idx4 = (np.arange(N23**2*2+2,N23**2*2+2+Nout*2,2)[:,None]*arb_bool.astype(int))
	# arb_sorted = np.concatenate([arb_idx[arb_bool],arb_idx2[arb_bool],arb_idx3[arb_bool],\
	# 							arb_idx4[arb_bool]])

	# idx_cum = 0
	# c_orth_sorted = np.zeros_like(c_orth)
	# s_orth_sorted = np.zeros_like(s_orth)
	# for idx in np.unique(arb_sorted):
	# 	num_idx = np.sum(arb_sorted==idx)
	# 	c_orth_sorted[:,idx_cum:idx_cum+num_idx] = c_orth[:,arb_sorted==idx]
	# 	s_orth_sorted[:,idx_cum:idx_cum+num_idx] = s_orth[:,arb_sorted==idx]
	# 	idx_cum += num_idx
	# c_orth = c_orth_sorted
	# s_orth = s_orth_sorted

	print("COMMENT OUT FOLLOWING LINE")
	p_orth = np.dot(s_orth.T,c_orth / np.diag(np.dot(s_orth,c_orth.T))[:,None])

	print("TODO: change output ")
	return c_orth,s_orth,p_orth


def generate_simIO_normalisation_old(Nlgn,N4,arbor):
	Nout = N4**2*Nvert
	Ninp = Nlgn**2
	arb_bool = (arbor>0)
	print("CHECK",Nout,Ninp,arb_bool.shape)
	arb_idx = np.arange(1,1+Nout)[None,:]*arb_bool.astype(int)
	arbor_sq = np.sqrt(arbor)

	arbsize = np.sum(arb_bool)
	arbarea = np.sum(arb_bool[0,:])
	# print("arbarea",np.sum(arb_bool,axis=1))
	print("arbsize",arbsize,arbarea,Nlgn,N4)

	c_vec = np.zeros((2*Ninp+Nout,2*arbsize))
	c_help = np.zeros((2*arbsize),dtype=int)
	for i in range(Nout):
		c_vec[i+2*Ninp,i*arbarea:(i+1)*arbarea] = arbor_sq[i,arb_bool[i,:]]
		c_vec[i+2*Ninp,i*arbarea+arbsize:(i+1)*arbarea+arbsize] = arbor_sq[i,arb_bool[i,:]]

		c_help[i*arbarea:(i+1)*arbarea] = arb_idx[i,arb_bool[i,:]]

	c_sum = np.sum(c_vec,axis=0)
	for i in range(1,1+Ninp):
		this_alpha = (c_help==i)
		c_vec[i-1,this_alpha] = c_sum[this_alpha]
	c_vec[Ninp:Ninp+Nout,arbsize:] = c_vec[:Ninp,:arbsize]
	
	# householder reflection (qr decomposition of mxn matrix, m>=n
	# c_orth,_ = np.linalg.qr(c_vec.T,mode='reduced')	#q: orthonormal, r: upper triangular
	# c_orth = c_orth.T
	# c_orth = calc_tools.gs(c_vec, row_vecs=True, norm = True)
	c_vec = calc_tools.gs_fast(c_vec, 2*Ninp, row_vecs=True, norm = True)
	
	
	s_orth = c_vec[:,:arbsize] * arbor_sq[None,arb_bool]
	s_orth = np.hstack([s_orth,c_vec[:,arbsize:] * arbor_sq[None,arb_bool]])
	
	c_vec[:,:arbsize] = c_vec[:,:arbsize]/arbor_sq[None,arb_bool]
	c_vec[:,arbsize:] = c_vec[:,arbsize:]/arbor_sq[None,arb_bool]
	
	c_vec = c_vec / np.diag(np.dot(s_orth,c_vec.T))[:,None]

	print("COMMENT OUT FOLLOWING LINE")
	p_orth = np.dot(s_orth.T,c_vec / np.diag(np.dot(s_orth,c_vec.T))[:,None])

	print("TODO: change output ")
	return c_vec,s_orth,p_orth

def generate_simIO_normalisation_coarse(Ninp,Nout):
	## generate coarse-grained orthogonalized c_x vectors
	##(calpha are already orthogonal to each other)
	c_x = np.zeros((Nout,2,Nout))
	for i in range(Nout):
		c_x[i,:,i] = 1

	u_x = np.zeros((Nout,2,Nout))
	nvec = np.ones((Nout))
	for i in range(Nout):
		u_x[i,:,:] = c_x[i,:,:] + 1./(Nout-i)*( -nvec + np.sum(c_x[:i,:,:],axis=0) )

	c_x = c_x.reshape(Nout,-1)
	u_x = u_x.reshape(Nout,-1)

	p_orth = 0
	for i in range(Nout-1):
		p_orth += np.dot(u_x[i:i+1,:].T,u_x[i:i+1,:])/np.dot(u_x[i,:],u_x[i,:])/Nout
	p_orth_comp = np.ones((Nout*2*Ninp,Nout*2*Ninp))*(-1./2/Nout**2)
	for i in range(Nout):
		p_orth_comp[Ninp*i:Ninp*(i+1),Ninp*i:Ninp*(i+1)] = (Nout-1.)/2./Nout**2
	p_orth_comp[Ninp*Nout:,:Ninp*Nout] = p_orth_comp[:Ninp*Nout,:Ninp*Nout]
	p_orth_comp[:Ninp*Nout,Ninp*Nout:] = p_orth_comp[:Ninp*Nout,:Ninp*Nout]
	p_orth_comp[:Ninp*Nout,:Ninp*Nout] +=\
		 1./Nout*np.diagflat(np.ones(Ninp*Nout),k=0)

	for i in range(1,Nout):
		p_orth_comp[:Ninp*Nout,:Ninp*Nout] +=\
		 1./Nout*np.diagflat(np.ones(Ninp*Nout-i*Ninp),k=i*Ninp)\
		 + 1./Nout*np.diagflat(np.ones(Ninp*Nout-i*Ninp),k=-i*Ninp)
	p_orth_comp[Ninp*Nout:,Ninp*Nout:] = p_orth_comp[:Ninp*Nout,:Ninp*Nout]

	return p_orth_comp


def generate_elements_simIO_normalisation(Nlgn,N4,Narb):
	Ninp,Nout = Narb,Narb
	row_x = np.ones((2*Nlgn*N4)) * (-1./2/Nout**2)
	row_x[:N4] = (Nout-1.)/2./Nout**2
	row_x[Nlgn*N4:N4*(Nlgn+1)] = (Nout-1.)/2./Nout**2

	## following lines should be ok
	row_alpha = np.zeros((Nlgn*N4))
	row_alpha[::N4] = 1./Nout
	return row_x,row_alpha


# def apply_simIO_normalisation(delta,row_x,row_alpha,Ninp,Nout):
# 	delta = tf.reshape(delta, [-1])
# 	delta2 = np.zeros_like(delta)

# 	for i in range(Nout): 
# 		irow_x = np.roll(row_x,i*Nout,axis=0)
# 		vecs = np.sum(delta * irow_x,axis=0)
# 		delta2[i*Nout:(i+1)*Nout] += vecs
# 		delta2[i*Nout+Ninp*Nout:(i+1)*Nout+Ninp*Nout] += vecs

# 	for i in range(Ninp):
# 		irow_alpha = np.roll(row_alpha,i,axis=0)
# 		delta2[i:Ninp*Nout:Nout] += np.sum(delta[:Ninp*Nout] * irow_alpha,axis=0)
# 		delta2[(i+Ninp*Nout)::Nout] += np.sum(delta[Ninp*Nout:] * irow_alpha,axis=0)

# 	return delta2


if __name__=="__main__":
	import matplotlib.pyplot as plt
	from bettina.modeling.ori_dev_model import connectivity,Wlgn_to4_params,W4to23_params,\
	W4to4_params

	#### TEST expand_weights_to_orthog_space() #####
	# Nlgn,N4 = 5,5
	# Nvert = 1
	# Nout,Ninp = N4**2*Nvert,Nlgn**2
	# Wlgn4 = connectivity.Connectivity((Nlgn,Nlgn), (N4,N4),\
	# 								   random_seed=19,Nvert=(1,Nvert))
	# arbor = Wlgn4.create_arbor(radius=0.4,profile="gaussian")#heaviside
	# arbor2 = np.stack([arbor,arbor])
	# Wlgn_to4_params["sigma"] = 0.2
	# Won_to_4 = Wlgn4.create_matrix(Wlgn_to4_params,"Gaussian",r_A=0.4)
	# Wof_to_4 = Wlgn4.create_matrix(Wlgn_to4_params,"Gaussian",r_A=0.4)
	# Wlgn_to_4_init = np.stack([Won_to_4,Wof_to_4])
	# frozen_init = np.logical_or(Wlgn_to_4_init<=0,Wlgn_to_4_init>=12)
	# print("Wlgn_to_4_init",np.nanmin(Wlgn_to_4_init),np.nanmax(Wlgn_to_4_init))
	# print("frozen,init",np.sum(frozen_init[arbor2>0]))
	# c_orth,s_orth = generate_simIO_normalisation(Nlgn,N4,arbor.reshape(Nout,Ninp))

	# ## update step with additive normalisation
	# np.random.seed(23243)
	# dW = 0.1*np.random.randn(Wlgn_to_4_init.size).reshape(arbor2.shape)*arbor2
	# print("dW",np.nanmax(dW),np.nanmin(dW))
	# delta_mask = dW[arbor2>0] ## complete update incl multiplied by arbor
	# mask_fl = np.logical_not(frozen_init)[arbor2>0]	## boolean mask as type float

	# norm = np.dot(s_orth,c_orth.T)
	# delta_mask -= np.sum(s_orth*np.dot(c_orth,delta_mask)[:,None],axis=0)
	# p_orth = 0
	# for i in range(s_orth.shape[0]):
	# 	p_orth += s_orth[i:i+1,:].T*c_orth[i:i+1,:]/norm[i,i]


	# delta_mask *= mask_fl
	# dWnorm = np.zeros_like(dW)
	# dWnorm[arbor2>0] = delta_mask
	# Wlgn_to_4 = Wlgn_to_4_init + 0.5 * dWnorm
	# print("dWnorm",np.sum(dWnorm,axis=1),np.sum(dWnorm,axis=(0,2)),dWnorm.shape)


	# ## clipping
	# Wlgn_to_4 = np.clip(Wlgn_to_4,0,12)
	# # Wlgn_to_4[arbor2<=0] = 0.
	# print("Wlgn_to_4",np.nanmin(Wlgn_to_4),np.nanmax(Wlgn_to_4))
	# frozen = np.logical_or(Wlgn_to_4<=0,Wlgn_to_4>=12)
	# print("frozen,after update",np.sum(frozen[arbor2>0]))

	# ## multiplicative normalisation
	# w_orth,w_vec = expand_weights_to_orthog_space(Wlgn_to_4,c_orth,Ninp,Nout,arbor2)
	# dW_eff = Wlgn_to_4 - Wlgn_to_4_init
	# print("dW_eff",np.sum(dW_eff,axis=1),np.sum(dW_eff,axis=(0,2)),np.nanmax(dW_eff))
	# dW_eff = dW_eff[arbor2>0]
	# update = dW_eff - np.sum(w_orth*np.dot(c_orth,dW_eff)[:,None],axis=0)
	# update2d = np.zeros_like(Wlgn_to_4)
	# update2d[arbor2>0] = update + 0*Wlgn_to_4_init[arbor2>0]
	# # print("Wlgn_to_4_init",np.sum(Wlgn_to_4_init,axis=1),np.sum(Wlgn_to_4_init,axis=(0,2)))
	# print("update",np.sum(update2d,axis=1),np.sum(update2d,axis=(0,2)),np.nanmax(update2d),\
	# 	np.nanmin(update2d))


	# idx = 24
	# Wlgn_to_4_x = Wlgn_to_4[:,idx,:]
	# arbor_x = arbor2[:,idx,:]
	# print("Wlgn_to_4",Wlgn_to_4_x[arbor_x>0])
	# print("sorth",s_vec[2*Ninp+idx,s_vec[2*Ninp+idx,:]!=0])
	# idx = 23
	# Wlgn_to_4_x = Wlgn_to_4[:,:,idx]
	# arbor_x = arbor2[:,:,idx]
	# print("Wlgn_to_4",Wlgn_to_4_x[arbor_x>0])
	# print("sorth",s_vec[idx,s_vec[idx,:]!=0],s_vec[Ninp+idx,s_vec[Ninp+idx,:]!=0])

	# fig = plt.figure()
	# ax = fig.add_subplot(131)
	# im=ax.imshow(np.dot(s_orth,c_orth.T),interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(132)
	# im=ax.imshow(np.dot(w_orth,c_orth.T),interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(133)
	# im=ax.imshow(s_orth,interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# plt.show()

	# exit()

	#### TEST generate_simIO_normalisation_oneUnittype() #####
	# N23,N4 = 5,5
	# Nvert = 1
	# Nout,Ninp = N23**2,N4**2*Nvert
	# W4 = connectivity.Connectivity_2pop((N4,N4),(N4,N4),(N23,N23),(N23,N23),\
	# 									 random_seed=19,Nvert=(Nvert,1))
	# Wrec_mode = W4to23_params["Wrec_mode"]
	# W4to23 = W4.create_matrix_2pop(W4to23_params,Wrec_mode)
	# print("W4to23",W4to23.shape)

	# W = connectivity.Connectivity((N4,N4),(N23,N23),random_seed=19,Nvert=(1,Nvert))
	# arbor = W.create_arbor(radius=0.2,profile="heaviside")
	# print("arbor",arbor.shape)
	# arbor_hstack = np.hstack([arbor,arbor])
	# arbor_4to23 = arbor#np.concatenate([arbor_hstack,arbor_hstack])
	# Narb = np.sum(arbor_4to23[0,:]>0)
	# print("arbor",arbor_4to23.shape,Ninp,Nout,Narb,np.sum(arbor_4to23>0)*4)

	# c_orth,s_orth,p_orth = generate_simIO_normalisation_oneUnittype(N4,N23,\
	# 							arbor_4to23.reshape(Nout,Ninp))
	# print("c_orth,s_orth,p_orth",c_orth.shape,s_orth.shape)
	# WEE = W4to23[:N4**2,:N4**2]

	# # fig = plt.figure()
	# # ax = fig.add_subplot(121)
	# # im=ax.imshow(c_orth,interpolation="nearest",cmap="binary")
	# # plt.colorbar(im,ax=ax)
	# # ax = fig.add_subplot(122)
	# # im=ax.imshow(c_vec,interpolation="nearest",cmap="binary")
	# # plt.colorbar(im,ax=ax)
	# # plt.show()
	# # # exit()

	# arbor_hstack = np.hstack([arbor,0*arbor])
	# print("arbor_hstack",arbor_hstack.shape)
	# arbor2 = np.concatenate([arbor_hstack,arbor_hstack])
	# W4to23[arbor2<=0] = 0.
	# # arbor2 = np.concatenate([arbor,arbor])

	# # Wtest = np.copy(W4to23)
	# # Wtest[:N4**2,:N4**2] = 1
	# # Wtest[N4**2:,:N4**2] = 2
	# # Wtest[:N4**2,N4**2:] = 3
	# # Wtest[N4**2:,N4**2:] = 4
	# # Wtest[arbor2<=0] = 0.
	# # Wtest = Wtest[:,:N4**2].reshape(2,N23**2,N4**2)

	# # fig = plt.figure()
	# # ax = fig.add_subplot(131)
	# # im=ax.imshow(Wtest.reshape(2*N23**2,N4**2),interpolation="nearest",cmap="binary")
	# # plt.colorbar(im,ax=ax)
	# # ax = fig.add_subplot(132)
	# # Wtest = Wtest.reshape(N23**2,2*N4**2)
	# # ax.plot(Wtest.flatten(),'-')
	# # ax = fig.add_subplot(133)
	# # arb_bool = (arbor>0)
	# # arb_idx = (np.arange(1,1+Nout*2,2)[:,None]*arb_bool.astype(int))
	# # arb_idx2 = (np.arange(2,2+Nout*2,2)[:,None]*arb_bool.astype(int))
	# # arb_idx3 = (np.arange(N23**2*2+1,N23**2*2+1+Nout*2,2)[:,None]*arb_bool.astype(int))
	# # arb_idx4 = (np.arange(N23**2*2+2,N23**2*2+2+Nout*2,2)[:,None]*arb_bool.astype(int))
	# # arb_sorted = np.concatenate([arb_idx[arb_bool],arb_idx2[arb_bool],arb_idx3[arb_bool],\
	# # 							arb_idx4[arb_bool]])
	# # arb_sorted = np.concatenate([arb_idx[arb_bool],arb_idx2[arb_bool]])
	# # # im=ax.imshow(arb_sorted,interpolation="nearest",cmap="binary")
	# # # plt.colorbar(im,ax=ax)
	# # ax.plot(arb_sorted.flatten(),'-')
	# # plt.show()
	# # exit()


	# W = (W4to23)[arbor2>0] + np.random.randn(np.sum(arbor2>0))*0.001+1
	# print("W ",W.shape,arbor2.shape,np.sum(np.sum(arbor2>0,axis=1)),c_orth.shape)
	# # iteration = (W - np.dot(p_orth,W) )
	# iteration = (W - np.sum(s_orth*np.dot(c_orth,W)[:,None],axis=0))

	# iteration_fullshape = np.zeros_like(arbor2)
	# iteration_fullshape[arbor2>0] = iteration
	# iteration_fullshape = iteration_fullshape.reshape(2*Nout,2*Ninp)
	# print("check",np.sum(iteration_fullshape,axis=1),np.sum(iteration_fullshape,axis=(0)))
	# print("check W",np.sum(W4to23,axis=1),np.sum(W4to23,axis=(0)))

	# exit()

	# ############### TEST generate_simIO_normalisation() ###############
	Nlgn,N4 = 7,7
	Nvert = 1
	Ninp,Nout = Nlgn**2,N4**2*Nvert
	arbor_params = {"mean_eccentricity"		:	0.2,
					"SD_eccentricity"	:	0.0,
					"SD_size"			:	0.0,
					"heterogeneity_type":	"independent"}

	Wlgn4 = connectivity.Connectivity((Nlgn,Nlgn), (N4,N4),\
									   random_seed=19,Nvert=(1,Nvert))
	arbor = Wlgn4.create_arbor(radius=0.4,profile="gaussian")
	arbor_g = Wlgn4.create_arbor(radius=0.4,profile="gaussian",arbor_params=arbor_params)
	arbor_g2 = arbor_g
	Won_to_4 = Wlgn4.create_matrix(Wlgn_to4_params,"Gaussian",r_A=0.4)
	Wof_to_4 = Wlgn4.create_matrix(Wlgn_to4_params,"Gaussian",r_A=0.4)
	Wlgn_to_4 = np.stack([Won_to_4,Wof_to_4])
	Narb = np.sum(arbor[0,:]>0)
	print("arbor",arbor.shape,arbor_g.shape,Ninp,Nout,np.sum(arbor[5,:]>0),np.sum(arbor_g[5,:]>0))

	c_orth,s_orth,p_orth,s_vec,c_vec = generate_simIO_normalisation(Nlgn,N4,arbor.reshape(Nout,Ninp))
	c_orthg,s_orthg,p_orthg,s_vecg,c_vecg = generate_simIO_normalisation(Nlgn,N4,arbor_g.reshape(Nout,Ninp))
	# p_orth_comp = generate_simIO_normalisation_coarse(Ninp,Nout)
	np.random.seed(45454)
	delta = 1+np.random.randn(2*Ninp*Nout)

	# arbor_g2 = Wlgn4.create_arbor(radius=0.4,profile="heaviside",arbor_params=arbor_params)
	# c_orthg,s_orthg,p_orthg,c_vecg = generate_simIO_normalisation_onoff(Nlgn,\
	# 								N4,arbor_g.reshape(Nout,Ninp),arbor.reshape(Nout,Ninp))

	arbor2 = np.concatenate([arbor_g.flatten(),arbor_g2.flatten()])
	arbor3 = np.concatenate([arbor_g.flatten(),arbor_g.flatten()])
	Wlgn_to_4 = Wlgn_to_4.flatten()
	Wlgn_to_4 = Wlgn_to_4.reshape(2,Nout,Ninp)


	# q,r = np.linalg.qr(c_vec.T,mode='complete')	#q: orthonormal, r: upper triangular
	# c_orth = -q[:,:c_vec.shape[0]-1].T
	# q,r = np.linalg.qr(s_vec.T,mode='complete')	#q: orthonormal, r: upper triangular
	# s_orth = -q[:,:c_vec.shape[0]-1].T

	delta = np.random.randn(np.sum(arbor3>0))
	sum_delta = np.dot(np.dot(c_orth,delta),s_orth)
	sum_delta3d = np.zeros_like(Wlgn_to_4)
	sum_delta3d[arbor3.reshape(2,Nout,Ninp)>0] = sum_delta
	delta3d = np.zeros_like(Wlgn_to_4)
	delta3d[arbor3.reshape(2,Nout,Ninp)>0] = delta
	print("sum_delta",np.sum(sum_delta3d,axis=1),np.sum(sum_delta3d,axis=(0,2)))
	print("TEST sum_x NORM",np.sum(delta3d,axis=1),np.sum(delta3d,axis=(0,2)))
	# print("s_orth",np.sum(s_orth,axis=0),np.sum(s_vec,axis=1))
	fig = plt.figure()
	ax = fig.add_subplot(121)
	# ax.plot(s_orth[1,:],s_orthg[1,:],"o")
	# ax.plot([-0.15,0.15],[-0.15,0.15],'--k')
	im=ax.imshow(np.dot(s_orth,s_orth.T),interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(122)
	im=ax.imshow(np.dot(s_orth,c_orth.T),interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	plt.show()
	exit()
	# ## TEST sum_x NORM
	# W = Wlgn_to_4[arbor3>0]
	# iteration = (delta - np.dot(np.dot(c_vec[:2*Ninp,:],delta),s_vec[:2*Ninp,:]/np.diag(np.dot(s_vec[:2*Ninp,:],c_vec[:2*Ninp,:].T))[:,None]))
	# iteration_fullshape = np.zeros_like(arbor3)
	# iteration_fullshape[arbor3>0] = iteration
	# iteration_fullshape = iteration_fullshape.reshape(2,Nout,Ninp)
	# print("TEST sum_x NORM",np.sum(iteration_fullshape,axis=1),np.sum(iteration_fullshape,axis=(0,2)))
	# print("")
	# ## TEST sum_alpha,i NORM
	# iteration = (delta - np.dot(np.dot(c_vec[2*Ninp:,:],delta),s_vec[2*Ninp:,:]/np.diag(np.dot(s_vec[2*Ninp:,:],c_vec[2*Ninp:,:].T))[:,None]))
	# iteration_fullshape = np.zeros_like(arbor3)
	# iteration_fullshape[arbor3>0] = iteration
	# iteration_fullshape = iteration_fullshape.reshape(2,Nout,Ninp)
	# print("TEST sum_alpha,i NORM",np.sum(iteration_fullshape,axis=1),np.sum(iteration_fullshape,axis=(0,2)))
	# print("")


	W = (Wlgn_to_4.flatten())[arbor2>0] #* arbor2[arbor2>0]
	iteration = (delta - np.dot(p_orthg,delta) )
	Wfull_shape = np.zeros_like(arbor2)
	Wfull_shape[arbor2>0] = W
	Wfull_shape = Wfull_shape.reshape(2,Nout,Ninp)
	iteration_fullshape = np.zeros_like(arbor2)
	iteration_fullshape[arbor2>0] = iteration
	iteration_fullshape = iteration_fullshape.reshape(2,Nout,Ninp)
	iteration_full = np.zeros_like(Wlgn_to_4)
	iteration_full[(arbor2>0).reshape(2,Nout,Ninp)] = iteration.flatten()
	print("check",np.sum(iteration_fullshape,axis=1),np.sum(iteration_fullshape,axis=(0,2)))
	print("")

	W = W.flatten()
	step1 = (W * arbor2[arbor2>0] - s_orthg[10,:] * np.dot(c_orthg[10,:],W * arbor2[arbor2>0]))
	step1full_shape = np.zeros_like(arbor2)
	step1full_shape[arbor2>0] = step1
	step1full_shape = step1full_shape.reshape(2,Nout,Ninp)
	step1_full = np.zeros_like(Wlgn_to_4)
	step1_full[(arbor2>0).reshape(2,Nout,Ninp)] = step1.flatten()
	print("check step1",np.sum(step1full_shape,axis=1),np.sum(step1full_shape,axis=(0,2)))
	print("")


	# iteration2 = (W - np.sum(s_orthg*np.dot(c_orthg,W)[:,None],axis=0)) * arbor2[arbor2>0]
	W = (Wlgn_to_4.flatten())[arbor2>0]
	iteration2 = (delta -
				  np.sum(s_orthg*np.dot(c_orthg,delta)[:,None],axis=0))
	# iteration2 = (W - np.dot(np.dot(c_orthg,W),s_orthg))
	iteration2_full = np.zeros_like(arbor2)
	iteration2_full[(arbor2>0)] = iteration2
	iteration2_full = iteration2_full.reshape(2,Nout,Ninp)
	print("checkfull2",np.sum(iteration2_full,axis=1),np.sum(iteration2_full,axis=(0,2)))
	print("")

	## check c_vec
	# fig = plt.figure()
	# ax = fig.add_subplot(131)
	# ax.plot(np.arange(Nout*Ninp),Wlgn_to_4[1,:,:].flatten(),"x")
	# ax.plot(np.where(arbor.flatten())[0],W[1,:].flatten(),"+")
	# ax = fig.add_subplot(132)
	# ax.plot(np.sum(Wlgn_to_4,axis=(0,2)).flatten(),"o")
	# # ax.plot(np.dot(c_vec[2*Ninp:,:],W.flatten()),"+")
	# ax.plot(np.dot(c_orth[2*Ninp:,:],W.flatten()),"x")
	# ax = fig.add_subplot(133)
	# ax.plot(np.sum(Wlgn_to_4,axis=(1)).flatten(),"o")
	# # ax.plot(np.dot(c_vec[:2*Ninp,:],W.flatten()),"+")
	# ax.plot(np.dot(c_orth,W.flatten()),"x")


	fig = plt.figure(figsize=(12,5))
	ax = fig.add_subplot(221)
	ax.set_title("Wlgn_to_4[0,:,:]")
	im=ax.imshow(Wlgn_to_4[0,:,:],interpolation="nearest",cmap="binary",vmin=0.,vmax=0.17)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(222)
	ax.set_title("W[0,:,:]")
	im=ax.imshow(iteration_full[0,:,:],interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	# ax.plot(Wlgn_to_4[0,arbor>0].flatten(),'-')
	ax = fig.add_subplot(223)
	# ax.set_title("Wlgn_to_4[0,:,:]")
	# im = ax.imshow(iteration2[0,:,:],interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(224)
	ax.set_title("iteration2_full[0,:,:]")
	im = ax.imshow(iteration2_full[0,:,:],interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	
	# fig = plt.figure()
	# ax = fig.add_subplot(131)
	# ax.plot(row_x[Ninp*Nout:]+row_alpha,'-k')
	# # ax.plot(row_alpha,'-b')
	# ax.plot(p_orth[Ninp*Nout:,Ninp*Nout],'--m')
	# ax = fig.add_subplot(132)
	# ax.plot(out.flatten(),'-k')
	# # ax.plot(norm_arb,'-g')
	# # ax.plot(delta-np.dot(p_orth,delta)*arbor,'--r')
	# ax = fig.add_subplot(133)
	# # ax.plot(delta,'-b')
	# # ax.plot(delta+delta2,'-g')
	# # ax.plot((delta-delta2*arbor) - (delta-np.dot(p_orth,delta)*arbor),'--r')

	plt.show()
