import numpy as np
from scipy.stats import norm,bernoulli,poisson
    
unif_vec = np.linspace(0,1,1001)[1:-1]

def bin_corr_bnds(
    prob_vec: np.ndarray,
    ):
    lo_bnd = np.fmax(-np.sqrt(((1-prob_vec)/prob_vec)[:,None]*((1-prob_vec)/prob_vec)[None,:]),
                     -1/np.sqrt(((1-prob_vec)/prob_vec)[None,:]*((1-prob_vec)/prob_vec)[:,None]))
    up_bnd = np.fmin(np.sqrt(((1-prob_vec)/prob_vec)[:,None]/((1-prob_vec)/prob_vec)[None,:]),
                     np.sqrt(((1-prob_vec)/prob_vec)[None,:]/((1-prob_vec)/prob_vec)[:,None]))
    return lo_bnd,up_bnd

def pois_corr_bnds(
    expec_vec: np.ndarray,
    ):
    unifs = poisson.ppf(unif_vec[:,None],expec_vec[None,:])
    means = np.mean(unifs,axis=0)
    stds = np.std(unifs,axis=0)
    unifs[:] = (unifs - means[None,:])/stds[None,:]

    lo_bnd = unifs.T @ unifs[::-1,:] / len(unif_vec)
    up_bnd = unifs.T @ unifs / len(unif_vec)
    return lo_bnd,up_bnd

def gen_corr_bin_vars(
    prob_vec: np.ndarray,
    corr_mat: np.ndarray,
    rng,
    nsamp: int=1,
    max_l: int=np.inf,
    tol: float=1e-12,
    return_prms: bool=False,
    debug: bool=False,
    ) -> np.ndarray:
    ndim = len(prob_vec)
    
    # calculate the matrix of alpha values defined in Park et al 1996
    np.fill_diagonal(corr_mat,1)
    a = np.log(1 + corr_mat*np.sqrt(((1-prob_vec)/prob_vec)[:,None]*((1-prob_vec)/prob_vec)[None,:]))
    
    # iteratively find the smallest nonzero alpha value, build beta values and S index sets, and update alpha matrix
    bs = []
    Ss = []
    l = 0
    while l < max_l:
        l += 1
        # try to find the smallest nonzero alpha value
        try:
            this_b = np.min(a[a>tol])
        except:
            break
        # if found, append to list of beta values and find its location
        b_idx = np.argwhere(a==this_b)[0]
        # if the associated diagonal elements are positive, add the indices to the S index set
        if a[b_idx[0],b_idx[0]]>tol and a[b_idx[1],b_idx[1]]>tol:
            bs.append(this_b)
            this_S = []
            not_S = np.setdiff1d(np.arange(ndim),b_idx)
            if b_idx[0]!=b_idx[1]:
                this_S.extend([b_idx[0],b_idx[1]])
            else:
                this_S.append(b_idx[0])
            # add any other indices to S such that a[i,j] > 0 for all i,j in S
            for repeat in range(ndim//4):
                for i in range(ndim):
                    if i not in this_S:
                        if np.all(a[i,this_S]>tol):# and\
                            # a[i,np.setdiff1d(not_S,[i])].max() < a[i,i] - this_b:
                            this_S.append(i)
                            not_S = np.setdiff1d(not_S,i)
            Ss.append(this_S)
        else:
            raise ValueError('Infeasible correlation matrix')
        # subtract all beta values from the alpha matrix elements with indices in S
        for i in Ss[-1]:
            for j in Ss[-1]:
                a[i,j] -= this_b
        a[:] = np.where(a<=tol,0,a)
        
    pois_vec = np.zeros((ndim,nsamp))
    # compute correlated Poisson random variables with beta values and S index sets
    for b,S in zip(bs,Ss):
        pois_vec[S,:] += rng.poisson(lam=b,size=nsamp)
        
    if return_prms:
        if debug:
            return pois_vec==0,bs,Ss,l,a
        else:
            return pois_vec==0,bs,Ss
    else:
        if debug:
            return pois_vec==0,l,a
        else:
            return pois_vec==0
    
def gen_corr_bin_vars_from_prms(
    ndim: int,
    bs: list,
    Ss: list,
    rng,
    nsamp: int=1
    ) -> np.ndarray:
    pois_vec = np.zeros((ndim,nsamp))
    # compute correlated Poisson random variables with beta values and S index sets
    for b,S in zip(bs,Ss):
        pois_vec[S,:] += rng.poisson(lam=b,size=nsamp)
        
    return pois_vec==0

def gen_corr_pois_vars(
    expec_vec: np.ndarray,
    corr_mat: np.ndarray,
    rng,
    nsamp: int=1,
    return_prms: bool=False,
    debug: bool=False,
    ) -> np.ndarray:
    ndim = len(expec_vec)
    
    # calculate lower and upper bounds on Poisson correlations
    lo_corr,up_corr = pois_corr_bnds(expec_vec)
    
    # compute exponential fit coeffs for relationship between desired and actual correlations under NORTA
    # assuming rho_{pois} = a*[exp(b * rho_{norm}) - 1]
    a = - up_corr*lo_corr / (up_corr + lo_corr)
    b = np.log((up_corr+a) / a)
    
    # compute the matrix of normal correlations vis rho_{norm} = log[(rho_{pois} + a) / a] / b
    np.fill_diagonal(corr_mat,1)
    norm_corr_mat = np.log((corr_mat + a) / a) / b
    if debug:
        print((corr_mat + a) / a)
        print(norm_corr_mat)
    try:
        norm_vec = rng.multivariate_normal(mean=np.zeros(ndim),cov=norm_corr_mat,size=nsamp,method='cholesky').T
    except:
        vals,vecs = np.linalg.eigh(norm_corr_mat)
        print(np.min(vals))
        norm_corr_mat = vecs @ np.diag(np.fmax(vals,1e-12)) @ vecs.T
        norm_vec = rng.multivariate_normal(mean=np.zeros(ndim),cov=norm_corr_mat,size=nsamp,method='cholesky').T
        
    if return_prms:
        return poisson.ppf(norm.cdf(norm_vec),expec_vec[:,None]),norm_corr_mat
    else:
        return poisson.ppf(norm.cdf(norm_vec),expec_vec[:,None])

def gen_mov_bar(
    bar_cent: np.ndarray,
    bar_dir: float,
    bar_len: float,
    bar_dist: float,
    ):
    def bar_to_box(
        s: np.ndarray,
        t: np.ndarray,
        periodic: bool=True,
        ):
        bar_pos = bar_cent + bar_dist*np.array([np.cos(bar_dir),np.sin(bar_dir)])*np.array(t)[...,None] +\
            bar_len*np.array([np.sin(bar_dir),-np.cos(bar_dir)])*(np.array(s)[...,None]-0.5)
        if periodic:
            bar_pos = np.mod(bar_pos,1)
        return bar_pos
        
    return bar_to_box

def bar_pass_time(
    bar_poss: np.ndarray,
    box_poss: np.ndarray,
    box_ts: np.ndarray,
    res: float=1e-3,
    ):
    dists = np.sqrt(np.sum((bar_poss[...,None,:,:]-box_poss[...,:,None,:])**2,axis=-1))
    pass_times = np.zeros(box_poss.shape[:-1])
    in_bar_pass = np.any(dists <= res,axis=-1)
    pass_times[np.logical_not(in_bar_pass)] = np.nan
    pass_times[in_bar_pass] = box_ts[np.argmin(dists[in_bar_pass],axis=-1)]
    return pass_times