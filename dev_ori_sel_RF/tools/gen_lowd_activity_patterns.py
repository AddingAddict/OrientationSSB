import numpy as np
from dev_ori_sel_RF.tools import gen_gaussian_random_field as ggrf


def create_activity_patterns(frequency,size,ndim,width,random_seed=19*114):
    """ generate ndim set of orthogonal activity patterns
    """
    radius = frequency/2./np.pi
    # width = 0.2
    # ndim = 2
    lowD_subset = np.empty((size*size,ndim))*np.nan
    for idim in range(ndim):
        rng = np.random.RandomState(random_seed*10000+idim)
        cmap = ggrf.generate_topology_map(size,size,radius,width,rng=rng,\
                                          return_complex=True, symmetrise=True)
        lowD_subset[:,idim] = np.real(cmap).flatten()
    lowD_subset,_ = np.linalg.qr(lowD_subset,mode='reduced')
    return lowD_subset




if __name__=="__main__":
    import matplotlib.pyplot as plt

    freq = 19
    size = 32
    ndim = 4
    width = 0.2
    act = create_activity_patterns(freq,size,ndim,width)
    print("act",act.shape,np.nanmean(act,axis=0),np.nanstd(act,axis=0),np.nanmean(act**2,axis=0))

    # act = np.clip(act,0,100000)
    cc = np.corrcoef(act)

    nrow,ncol = 1,3
    fig = plt.figure(figsize=(ncol*6,nrow*5))
    ax = fig.add_subplot(nrow,ncol,1)
    im=ax.imshow(act[:,0].reshape(size,size),interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(nrow,ncol,2)
    im=ax.imshow(cc[:,0].reshape(size,size),interpolation="nearest",cmap="RdBu_r",vmin=-1,vmax=1)
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(nrow,ncol,3)
    im=ax.imshow(cc,interpolation="nearest",cmap="RdBu_r",vmin=-1,vmax=1)
    plt.colorbar(im,ax=ax)
    plt.show()