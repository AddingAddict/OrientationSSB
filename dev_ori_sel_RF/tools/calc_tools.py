import numpy as np

def gs(X, row_vecs=True, norm=True):
	"""
	gram-schmidt orthogonalization of input vectors X
	"""
	if not row_vecs:
	    X = X.T
	Y = X[0:1,:].copy()
	# print(Y)
	for i in range(1, X.shape[0]):
	    proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
	    Y = np.vstack((Y, X[i,:] - proj.sum(0)))
	    # print(i,Y)
	# if norm:
	#     Y = np.diag(1./np.linalg.norm(Y,axis=1)).dot(Y)
	if row_vecs:
	    return Y
	else:
	    return Y.T

def gs_fast(X, Ninp, row_vecs=True, norm=True):
	"""
	gram-schmidt orthogonalization of input vectors X
	assuming first Ninp vectors are already orthogonal to each other
	"""
	if not row_vecs:
	    X = X.T
	norm = np.sqrt( np.sum(X[0,:]*X[0,:]) )
	Y = (X[:Ninp,:]/norm).copy()
	for i in range(Ninp, X.shape[0]-1):#
		Ynew = X[i,:] - np.sum(np.dot( Y, X[i,:] )[:,None] * Y, axis=0)
		Y = np.vstack((Y, Ynew/np.linalg.norm(Ynew) ))
	# if norm:
	# 	Y = np.diag(1./np.linalg.norm(Y,axis=1)).dot(Y)
	if row_vecs:
	    return Y
	else:
	    return Y.T