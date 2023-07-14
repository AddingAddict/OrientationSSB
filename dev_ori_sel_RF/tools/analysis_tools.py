import numpy as np
import skimage
from skimage import measure
from scipy import interpolate
from scipy import optimize
import scipy.spatial.distance as sd
import scipy.cluster.hierarchy as ch

from dev_ori_sel_RF import connectivity,dynamics_np


def distance_in_space_pbc(x1,x2=None,border_value=1):
    ''' compute difference of x1 and x2 with periodic boundary conditions
    if x2 is None, assumed that x1 is already the difference
    '''
    if x2 is None:
        difference = x1
    else:
        difference = x1 - x2
    max_diff = np.nanmax(difference)/2.

    signs = np.sign(difference)
    return np.where(np.abs(difference) > border_value/2.,\
             -signs*(border_value - np.abs(difference)), difference)


def difference_in_pref_ori(array1,array2,in_deg=True,all_to_all=True):
    if in_deg:
        array1 = array1/360.*2*np.pi
        array2 = array2/360.*2*np.pi
    if all_to_all:
        return 0.5*np.abs(np.angle( np.exp(1j*2*(array1[:,None] - array2[None,:])), deg=in_deg ))
    else:
        return 0.5*np.abs(np.angle( np.exp(1j*2*(array1 - array2)), deg=in_deg ))

def difference_in_phase(array1,array2,in_deg=True,all_to_all=True):
    if in_deg:
        array1 = array1/360.*2*np.pi
        array2 = array2/360.*2*np.pi
    if all_to_all:
        return np.abs(np.angle( np.exp(1j*(array1[:,None] - array2[None,:])), deg=in_deg ))
    else:
        return np.abs(np.angle( np.exp(1j*(array1 - array2)), deg=in_deg ))

def correlate(x,y):
    """computes cross-correlation coefficient between normalised x and y"""
    x_norm = x.reshape(np.prod(x.shape))
    y_norm = y.reshape(np.prod(y.shape))
    assert x_norm.size==y_norm.size, "given arrays in analysis_tools.correlate have different\
                                      sizes{} and {}".format(x.size,y.size)
    x_norm = (x_norm-np.nanmean(x_norm))/np.nanstd(x_norm)
    y_norm = (y_norm-np.nanmean(y_norm))/np.nanstd(y_norm)
    return np.nanmean(x_norm*y_norm)


def compute_MR(activity,axis,f1_idx):
    spectrum = np.abs(np.fft.fft(activity,axis=axis))
    f0 = spectrum[0,...]
    f1 = spectrum[f1_idx,...]
    MR = f1/f0
    return MR

def compute_orientation_tuning_from_activity(activity,orientations,norm=False):
    original_shape = activity.shape
    if activity.ndim>2:
        activity = activity.reshape(activity.shape[0],-1)
    opm = np.sum(activity * np.exp(1j*orientations)[:,None], axis=0)
    if norm:
        opm /= np.sum(activity,axis=0)
    activity = activity.reshape(original_shape)
    return opm.reshape(original_shape[1:])


def hierarchical_clustering_of_correlation(correlation_matrix):
    ### Choose METRIC ###
    distSquareMatrix = sd.squareform(sd.pdist(correlation_matrix, metric='euclidean'))
    # distSquareMatrix  = 1. -abs(correlation_matrix)
    # distSquareMatrix  = (1. - correlation_matrix )

    CM_linkageMatrix = ch.linkage(distSquareMatrix, method='average')
    #CM_linkageMatrix = ch.linkage(distSquareMatrix, method='centroid')#signle,median,centroid,average,weighted,ward

    #CM_dendro = ch.dendrogram(CM_linkageMatrix, p=6, truncate_mode='lastp',no_plot=True)
    #CM_dendro = ch.dendrogram(CM_linkageMatrix, labels=CM_l_list,no_plot=True)
    #CM_dendro = ch.dendrogram(CM_linkageMatrix,no_plot=True)
    #CM_leaves = np.array(CM_dendro['leaves'])
    CM_leaves = ch.leaves_list(CM_linkageMatrix)	#sorted pixel indices
    CM_transformed_array = (correlation_matrix[CM_leaves].T)[CM_leaves]
    return CM_transformed_array,CM_leaves,CM_linkageMatrix


def generate_grating_stimuli(input_size,num_phases,spat_freq,orientations):
    if isinstance(input_size,int):
        input_size = (input_size,input_size)
    h,w = input_size

    wavelength = h/spat_freq[0]
    phases = np.linspace(0,wavelength,num_phases,endpoint=False)

    stimuli = np.empty((len(phases),len(spat_freq),len(orientations),h,w))
    y,x = np.meshgrid(np.arange(h),np.arange(w))
    for i,iphase in enumerate(phases):
        for j,jfreq in enumerate(spat_freq):
            for k,kori in enumerate(orientations):
                x_ori = x * np.cos(kori) - y * np.sin(kori)
                y_ori = x * np.sin(kori) + y * np.cos(kori)
                stimuli[i,j,k,:,:] = \
                 np.sin(2*np.pi/h*jfreq*x_ori + 2*np.pi/w*jfreq*y_ori + 2*np.pi/w*jfreq*iphase)
    return stimuli

def stimulate_system(stimuli,config_dict,Wff,W4to4=None,W23to4=None,W23to23=None,W4to23=None):
    num_phases,num_spat_freq,num_oris,dim_y,dim_x = stimuli.shape
    stimuli = stimuli.reshape(num_phases,num_spat_freq,num_oris,-1)
    timesteps = 10#config_dict["Inp_params"]["pattern_duration"]
    dt = 0.05
    N4 = config_dict["N4"]
    Nvert = config_dict["Nvert"]
    N23 = config_dict["N23"]

    ## ============================ Lateral conn Layer 4 ==================================
    ## ====================================================================================
    if W4to4 is None:
        W4 = connectivity.Connectivity_2pop((N4,N4), (N4,N4), (N4,N4), (N4,N4),\
            random_seed=config_dict["random_seed"],Nvert=Nvert)
        Wrec_mode = config_dict["W4to4_params"]["Wrec_mode"]
        W4to4 = W4.create_matrix_2pop(config_dict["W4to4_params"],Wrec_mode)
    ## ============================ Lateral conn Layer 2/3 ================================
    ## ====================================================================================
    if W23to23 is None:
        W4 = connectivity.Connectivity_2pop((N23,N23), (N23,N23), (N23,N23), (N23,N23),\
            random_seed=config_dict["random_seed"])
        Wrec_mode = config_dict["W23_params"]["Wrec_mode"]
        W23to23 = W4.create_matrix_2pop(config_dict["W23_params"],Wrec_mode)
    ## ======================== Afferent conn L4 to L23 ===================================
    ## ====================================================================================
    if W4to23 is None:
        W4 = connectivity.Connectivity_2pop((N4,N4), (N4,N4), (N23,N23), (N23,N23),\
            random_seed=config_dict["random_seed"],Nvert=(Nvert,1))
        Wrec_mode = config_dict["W4to23_params"]["Wrec_mode"]
        W4to23 = W4.create_matrix_2pop(config_dict["W4to23_params"],Wrec_mode)
    ## ======================== Feedback conn L4 to L23 ===================================
    ## ====================================================================================
    if W23to4 is None:
        W4 = connectivity.Connectivity_2pop((N23,N23), (N23,N23), (N4,N4), (N4,N4),\
            random_seed=config_dict["random_seed"],Nvert=(1,Nvert))
        Wrec_mode = config_dict["W23to4_params"]["Wrec_mode"]
        W23to4 = W4.create_matrix_2pop(config_dict["W23to4_params"],Wrec_mode)
    ## ====================================================================================

    ## ======================== Seed Initial condition ====================================
    rng = np.random.RandomState(config_dict["random_seed"]*113)


    resp_l4 = np.empty((num_phases,num_spat_freq,num_oris,N4**2*Nvert*2))*np.nan
    resp_l23 = np.empty((num_phases,num_spat_freq,num_oris,N23**2*2))*np.nan
    num_params = num_phases*num_spat_freq*num_oris
    for i in range(num_phases):
        for j in range(num_spat_freq):
            for k in range(num_oris):
                if (k+j*num_oris+i*num_spat_freq*num_oris)&10:
                    print("Doing run {} of {} (phase={}, spat freq={}, ori={}).".format(\
                        k+j*num_oris+i*num_spat_freq*num_oris,num_params,i,j,k))
                l40 = rng.uniform(0, 1, N4**2*2*Nvert)
                l230 = rng.uniform(0, 1, N23**2*2)
                l4 = l40
                l23 = l230
                # l4_t = [l4]
                # l23_t = [l23]
                stimulus = np.stack([stimuli[i,j,k,:],stimuli[i,j,k,:]])
                for tstep in range(timesteps):
                    print("CHECK NAN",np.sum(np.isfinite(l4)),np.sum(np.isfinite(l23)),\
                        np.nanmax(l4),np.nanmax(l23),config_dict["gamma_4"],config_dict["gamma_lgn"])
                    l4 += dt*dynamics_np.dynamics_l4(l4,W4to4,stimulus,Wff,l23,W23to4,\
                     config_dict["gamma_4"],config_dict["gamma_lgn"],N4**2*Nvert)
                    l23 += dt*dynamics_np.dynamics_l23(l23,W23to23,l4,W4to23,N23**2)
                    # l4_t.append(l4)
                    # l23_t.append(l23)

                resp_l4[i,j,k,:] = l4
                resp_l23[i,j,k,:] = l23

    return resp_l4, resp_l23

def find_phase_sensitivity(responses):
    resp_l4, resp_l23 = responses
    size_l4 = resp_l4.shape[-1]
    size_l23 = resp_l23.shape[-1]
    resp_l4_phases = np.empty((resp_l4.shape[0],size_l4))*np.nan
    resp_l23_phases = np.empty((resp_l23.shape[0],size_l23))*np.nan

    pref_ori = np.nanargmax(np.nanmax(resp_l4,axis=0),axis=0)
    for i in range(size_l4):
        resp_l4_phases[:,i] = resp_l4[:,pref_ori[i],i]
    pref_ori = np.nanargmax(np.nanmax(resp_l23,axis=0),axis=0)
    for i in range(size_l23):
        resp_l23_phases[:,i] = resp_l23[:,pref_ori[i],i]

    fft_l4 = np.abs(np.fft.fft(resp_l4_phases,axis=0))
    fft_l23 = np.abs(np.fft.fft(resp_l23_phases,axis=0))
    ## modulation ratio = F1/F0
    modulation_ratio_l4 = fft_l4[1]/fft_l4[0]
    modulation_ratio_l23 = fft_l23[1]/fft_l23[0]

    sgl_cell_props_dict = {}
    sgl_cell_props_dict["modulation_ratio_l4"] = modulation_ratio_l4
    sgl_cell_props_dict["modulation_ratio_l23"] = modulation_ratio_l23
    return resp_l4_phases,resp_l23_phases,sgl_cell_props_dict

def find_preferred_abs_phase(responses,temp_freq=None):
    original_shape = responses.shape
    timesteps = responses.shape[0]
    if responses.ndim>2:
        responses = responses.reshape(timesteps,-1)
    if temp_freq is None:
        spectrum = np.abs(np.fft.fft(responses-np.nanmean(responses),axis=0))
        peak_freq = np.argmax(spectrum)
        oscillation_duration = timesteps//peak_freq
        num_oscillations = peak_freq
    else:
        oscillation_duration = timesteps//temp_freq
        num_oscillations = temp_freq
    response_per_oscillation = responses[:num_oscillations*oscillation_duration,...]
    response_per_oscillation = response_per_oscillation.reshape(num_oscillations,\
                                oscillation_duration,-1)

    avg_response_per_oscillation = np.nanmean(response_per_oscillation,axis=0)
    idx_preferred_phase = np.nanargmax(avg_response_per_oscillation,axis=0)
    preferred_phase = 2*np.pi*idx_preferred_phase/oscillation_duration
    return preferred_phase.reshape(original_shape[1:])


def gabor(sigma, theta, Lambda, psi, gamma, size):
    """Gabor feature extraction.
    sigma : width of envelope in x direction
    theta : orientation of gabor in rad
    Lambda : wavelength of gabor
    psi : phase of gabor in rad
    gamma : decay of envelope in y relative to x direction
    size : size of gabor array"""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    if isinstance(size,int):
        size = (size,size)

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.linspace(ymin, ymax + 1, size[0]),\
                          np.linspace(xmin, xmax + 1, size[1]))
    # Rotation
    x_theta = x * np.cos(theta) - y * np.sin(theta)
    y_theta = x * np.sin(theta) + y * np.cos(theta)

    if isinstance(psi,np.ndarray):
        gb = np.exp(-.5 * (x_theta[:,:,None] ** 2 / sigma_x ** 2 + y_theta[:,:,None] ** 2 / sigma_y ** 2)) *\
         np.cos(2 * np.pi / Lambda * x_theta[:,:,None] + psi[None,None,:])
    else:
        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) *\
         np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def gabor_fit(sigma, theta, psi, Lambda, gamma, x, y):
    """Gabor feature extraction.
    sigma : width of envelope in x direction
    theta : orientation of gabor in rad
    Lambda : wavelength of gabor
    psi : phase of gabor in rad
    gamma : decay of envelope in y relative to x direction
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Rotation
    x_theta = x * np.cos(theta) - y * np.sin(theta)
    y_theta = x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) *\
         np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def get_response(sd,DA,Nvert=1):
    """
    sd : s_on - s_off, sd.shape = N4 x N4*Nvert x Nlgn x Nlgn
    DA : arbor diameter
    """
    assert sd.ndim==4, "reshape sd such that it is four dimensional"
    N4 = sd.shape[0]
    Nlgn = sd.shape[2]

    ## Fourier transform SD
    fct = 1
    delta_bins = 20
    sdfft = np.abs(np.fft.fftshift(np.fft.fftn(sd,s=(fct*Nlgn,fct*Nlgn),axes=(2,3)),axes=(2,3)))
    h,w = sdfft.shape[2:]

    ## bin orientation in fourier space
    kx,ky = np.meshgrid(np.arange(-w/2.,w/2.),np.arange(-h/2.+1,h/2.+1)[::-1])
    angles = np.arctan2(ky,kx)*180/np.pi + (np.arctan2(ky,kx)<0)*360
    frequency = np.sqrt((kx/w)**2 + (ky/h)**2).flatten()
    angle_disc = np.arange(0,180,delta_bins)
    ori_bins = np.searchsorted(angle_disc,angles,side='right')

    half = h//2 + h%2
    sdfft_long = sdfft[:,:,:half,:].reshape(N4,N4*Nvert,-1)
    ori_bins = ori_bins[:half,:].flatten()

    ## best response for each binned orientation across spatial frequency
    Rn = np.empty((180//delta_bins,N4,N4*Nvert))*np.nan
    maxk = np.zeros((180//delta_bins,N4,N4*Nvert),dtype=int)
    for ibin in range(1,1+180//delta_bins):
        sd_k = sdfft_long[:,:,ori_bins==ibin]
        sd_maxk = np.argmax(sd_k,axis=2)
        Rn[ibin-1,:,:] = np.max(sd_k,axis=2)
        maxk[ibin-1,:,:] = sd_maxk

    ## vector sum of best responses
    phi = np.linspace(0,2*np.pi,num=180//delta_bins,endpoint=False) + delta_bins/2./180*np.pi
    opm = np.sum(np.exp(1j*phi)[:,None,None]*Rn, axis=0)
    rms = np.sum(Rn,axis=0)
    opm = opm/rms

    ## find preferred phase
    # pref_ori = np.angle(opm,deg=False)*0.5
    # RFsd,_,_,_ = get_RF_form(sd,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,mode="diff_only")
    # num_phases = 12
    # psi = np.linspace(0,2*np.pi,num_phases,endpoint=False)
    # pref_phase = np.empty_like(pref_ori)*np.nan
    # gabors = []
    # xold,yold = np.meshgrid(np.arange(DA),np.arange(DA))
    # xnew,ynew = np.arange(0,DA,0.5),np.arange(0,DA,0.5)
    # for y in range(N4):
    # 	for x in range(N4*Nvert):
    # 		# print(x,y,np.argmax(Rn[:,y,x]),maxk[np.argmax(Rn[:,y,x]),y,x])
    # 		pref_freq = frequency[maxk[np.argmax(Rn[:,y,x]),y,x]]
    # 		pref_ori = angle_disc[np.argmax(Rn[:,y,x])] + delta_bins/2.
    # 		# if (y<4 and x<4):
    # 		# print("pref_freq",y,x,2*np.pi/pref_freq,pref_ori[y,x]/2/np.pi*360)
    # 		g = -gabor(sigma=2.,
    # 				 theta=(90-pref_ori)/180*np.pi,
    # 				 Lambda=2*np.pi/pref_freq,
    # 				 psi=psi,
    # 				 gamma=1.,
    # 				 size=(ynew.shape[0],xnew.shape[0]))

    # 		tck = interpolate.bisplrep(xold,yold,RFsd[y*DA:(y+1)*DA,x*DA:(x+1)*DA])
    # 		RFsd_intp = interpolate.bisplev(xnew,ynew,tck)
    # 		#conv = np.nanmean(RFsd_intp[:,:,None] * g,axis=(0,1))
    # 		RFsd_intp = (RFsd_intp-np.nanmean(RFsd_intp,axis=(0,1)))/np.nanstd(RFsd_intp,axis=(0,1))
    # 		gnorm = (g-np.nanmean(g,axis=(0,1))[None,None,:])/np.nanstd(g,axis=(0,1))[None,None,:]
    # 		conv = np.nanmean(RFsd_intp[:,:,None] * gnorm, axis=(0,1))
    # 		pref_phase[y,x] = psi[np.argmax(conv)]
    # 		gabors.append(g[:,:,0])

    # gabors = np.array(gabors).reshape(N4,N4*Nvert,DA*2,DA*2)

    return opm,Rn#,pref_phase,gabors


def get_RF_form(Wff,N4,Nlgn,DA,calc_PF=True,Nvert=1,mode="full_onoff"):
    if mode=="full_onoff":
        W = Wff.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
        RF = np.zeros((3,DA*N4,DA*N4*Nvert))
        PF = np.zeros((3,DA*Nlgn,((DA-3)*Nvert)*Nlgn+DA%2))
        avg_RF = np.zeros((3,N4,N4*Nvert))
        avg_PF = np.zeros((3,Nlgn,Nlgn))
        for i in range(N4*Nvert):
            for j in range(N4):
                # son_ij = np.roll(np.roll(W[0,j,i,:,:],shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
                # sof_ij = np.roll(np.roll(W[1,j,i,:,:],shift=Nlgn//2-j,axis=0),shift=Nlgn//2-i,axis=1)
                son_ij = np.roll(np.roll(W[0,j,i,:,:],shift=Nlgn//2-int(1.*j/N4*Nlgn),axis=0),\
                    shift=Nlgn//2-int(1.*i/N4/Nvert * Nlgn),axis=1)
                sof_ij = np.roll(np.roll(W[1,j,i,:,:],shift=Nlgn//2-int(1.*j/N4*Nlgn),axis=0),\
                    shift=Nlgn//2-int(1.*i/N4/Nvert * Nlgn),axis=1)
                RF[0,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
                 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2] -\
                 sof_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
                RF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
                 son_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
                RF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
                 sof_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
                avg_RF[:,j,i] = np.mean(RF[:,j*DA:(j+1)*DA,i*DA:(i+1)*DA],axis=(1,2))
        if calc_PF:
            for i in range(Nlgn):
                for j in range(Nlgn):
                    son_ij = np.roll(np.roll(W[0,:,:,j,i],shift=N4//2-j,axis=0),\
                            shift=N4*Nvert//2-i*Nvert,axis=1)
                    sof_ij = np.roll(np.roll(W[1,:,:,j,i],shift=N4//2-j,axis=0),\
                            shift=N4*Nvert//2-i*Nvert,axis=1)
                    PF[0,j*DA:(j+1)*DA,i*(DA-3)*Nvert:(i+1)*(DA-3)*Nvert] = \
                     son_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4*Nvert//2-(DA-3)*Nvert//2:N4*Nvert//2+(DA-3)*Nvert//2+(DA-3)%2] -\
                     sof_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4*Nvert//2-(DA-3)*Nvert//2:N4*Nvert//2+(DA-3)*Nvert//2+(DA-3)%2]
                    PF[1,j*DA:(j+1)*DA,i*(DA-3)*Nvert:(i+1)*(DA-3)*Nvert] = \
                     son_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4*Nvert//2-(DA-3)*Nvert//2:N4*Nvert//2+(DA-3)*Nvert//2+(DA-3)%2]
                    PF[2,j*DA:(j+1)*DA,i*(DA-3)*Nvert:(i+1)*(DA-3)*Nvert] = \
                     sof_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4*Nvert//2-(DA-3)*Nvert//2:N4*Nvert//2+(DA-3)*Nvert//2+(DA-3)%2]
                    avg_PF[:,j,i] = np.mean(PF[:,j*DA:(j+1)*DA,i*(DA-3)*Nvert:(i+1)*(DA-3)*Nvert+DA%2],axis=(1,2))
    else:
        W = Wff.reshape(N4,N4*Nvert,Nlgn,Nlgn)
        RF = np.zeros((DA*N4,DA*N4*Nvert))
        PF = np.zeros((DA*Nlgn,(DA-3)*Nvert*Nlgn+DA%2))
        avg_RF = np.zeros((N4,N4*Nvert))
        avg_PF = np.zeros((Nlgn,Nlgn))
        for i in range(N4*Nvert):
            for j in range(N4):
                s_ij = np.roll(np.roll(W[j,i,:,:],shift=Nlgn//2-int(1.*j/N4 * Nlgn),axis=0),\
                                        shift=Nlgn//2-int(1.*i/N4/Nvert * Nlgn),axis=1)
                # s_ij = W[j,i,:,:]
                RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
                 s_ij[Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2, Nlgn//2-DA//2:Nlgn//2+DA//2+DA%2]
                avg_RF[j,i] = np.mean(RF[j*DA:(j+1)*DA,i*DA:(i+1)*DA])
        if calc_PF:
            for i in range(Nlgn):
                for j in range(Nlgn):
                    s_ij = np.roll(np.roll(W[:,:,j,i],shift=N4//2-j,axis=0),\
                                            shift=N4*Nvert//2-i*Nvert,axis=1)
                    PF[j*DA:(j+1)*DA,i*(DA-3)*Nvert:(i+1)*(DA-3)*Nvert] = \
                     s_ij[N4//2-DA//2:N4//2+DA//2+DA%2, N4*Nvert//2-(DA-3)*Nvert//2:N4*Nvert//2+(DA-3)*Nvert//2+(DA-3)%2]
                    avg_PF[j,i] = np.mean(PF[j*DA:(j+1)*DA,i*(DA-3)*Nvert:(i+1)*(DA-3)*Nvert+(DA-3)%2])
    return RF, PF, avg_RF, avg_PF


def get_center_of_mass_subfields(RF_array,**kwargs):
    """find coordinates of center of mass sub receptive fields in coordinate space
    given by last two dimensions of RF_array
    """
    h,w = RF_array.shape[-2:]
    if RF_array.ndim>3:
        RF_array = RF_array.reshape(-1,h,w)
    ## normalise max value to 1 such that contours can be drawn at same height for each crt unit
    RF_array = RF_array/np.nanmax(np.abs(RF_array),axis=(1,2))[:,None,None]
    RF_array_ON = np.copy(RF_array)
    RF_array_ON[RF_array_ON<0] = 0.0
    RF_array_OFF = np.copy(RF_array)
    RF_array_OFF[RF_array_OFF>0] = 0.0

    nlocs = RF_array.shape[0]
    cntr_lvl = 0.1
    centroids = np.empty((nlocs,2,2))*np.nan
    for i in range(nlocs):
        c_on = measure.find_contours(RF_array[i,:,:],cntr_lvl)
        c_off = measure.find_contours(RF_array[i,:,:],-cntr_lvl)
        if len(c_on)>0:
            len_c_on = [len(item) for item in c_on]
            on_id = np.argsort(len_c_on)[::-1]
            max_val_id = on_id[0]
            max_val = 0
            for iid in on_id:
                mask_on_field = measure.grid_points_in_poly((h,w),c_on[iid])
                if np.sum(mask_on_field)>0:
                    mv = np.nanmax(RF_array_ON[i,mask_on_field])>max_val
                    if mv>max_val:
                        max_val = mv
                        max_val_id = iid
            mask_on_field = measure.grid_points_in_poly((h,w),c_on[max_val_id])
            idx_on_field_y,idx_on_field_x = np.where(mask_on_field)
            M00 = np.sum(RF_array_ON[i,mask_on_field])*1.
            M10 = np.sum(RF_array_ON[i,mask_on_field]*idx_on_field_x)
            M01 = np.sum(RF_array_ON[i,mask_on_field]*idx_on_field_y)
            centroids[i,:,0] = np.array([M10/M00, M01/M00])
        if len(c_off)>0:
            len_c_off = [len(item) for item in c_off]
            off_id = np.argsort(len_c_off)[::-1]
            longest_off_id = off_id[0]
            min_val_id = longest_off_id
            min_val = 0
            for iid in off_id:
                mask_off_field = measure.grid_points_in_poly((h,w),c_off[iid])
                if np.sum(mask_off_field)>0:
                    mv = np.nanmin(RF_array_OFF[i,mask_off_field])
                    if mv<min_val:
                        min_val = mv
                        min_val_id = iid
            mask_off_field = measure.grid_points_in_poly((h,w),c_off[min_val_id])
            idx_off_field_y,idx_off_field_x = np.where(mask_off_field)
            M00 = np.sum(RF_array_OFF[i,mask_off_field])*1.
            M10 = np.sum(RF_array_OFF[i,mask_off_field]*idx_off_field_x)
            M01 = np.sum(RF_array_OFF[i,mask_off_field]*idx_off_field_y)
            centroids[i,:,1] = np.array([M10/M00, M01/M00])
    return centroids


def least_square_diff(args,x,y,data,func):
    # sigma,theta,psi,Lambda,gamma = args[:]
    return np.sqrt((func(*args,x=x,y=y)-data)**2)

def fit_gabor_to_RF(RFsd,**kwargs):
    DA = kwargs["DA"]
    N4 = kwargs["N4"]
    Nvert = kwargs["Nvert"]
    #DA*N4,DA*N4*Nvert = RFsd.shape
    xold,yold = np.meshgrid(np.arange(-DA/2,DA/2,1),np.arange(-DA/2,DA/2,1))
    xnew,ynew = np.arange(-DA/2,DA/2,0.5),np.arange(-DA/2,DA/2,0.5)
    xg,yg = np.meshgrid(xnew,ynew)
    outside_rf = np.sqrt(xg**2 + yg**2)>(DA//2-1)
    inside_rf = np.logical_not(outside_rf)
    DAnew = xnew.size

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    sigma_x = 2.
    sigma_y = 2.
    theta = 0
    size = (ynew.size,xnew.size)
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.linspace(ymin, ymax + 1, size[0]),\
                          np.linspace(xmin, xmax + 1, size[1]))
    y = y.flatten()
    x = x.flatten()
    scaling_factor = xmax*2./xold[0,:].size
    print("scaling_factor",scaling_factor,xmax,DA,DA/2)
    delta_bins = 20
    angle_disc = np.arange(0,180,delta_bins) + delta_bins/2.
    x0 = np.array([2,0,0,2*np.pi/1.,1.])	##sigma, theta, psi, Lambda, gamma,

    fit_params = np.empty((N4,N4*Nvert,len(x0)))*np.nan
    fitted_gabor = np.empty((2,N4*DAnew,N4*Nvert*DAnew))*np.nan
    fit_cost = np.empty((N4,N4*Nvert))*np.nan
    num_half_cycles = np.empty((N4,N4*Nvert))*np.nan
    for i in range(N4):
        for j in range(N4*Nvert):
            tck = interpolate.bisplrep(xold,yold,RFsd[i*DA:(i+1)*DA,j*DA:(j+1)*DA],s=0.1)
            RFsd_intp = interpolate.bisplev(xnew,ynew,tck).T#[::-1,:]
            RFsd_intp[outside_rf] = 0.0
            RFsd_intp = RFsd_intp.flatten()
            # RFsd_intp = RFsd[i*DA:(i+1)*DA,j*DA:(j+1)*DA].flatten()
            RFsd_intp /= np.nanmax(np.abs(RFsd_intp))
            if np.sum(np.logical_not(np.isfinite(RFsd_intp)))>0:
                continue

            if "Rn" in kwargs.keys():
                pref_ori = angle_disc[np.argmax(kwargs["Rn"][:,i,j])]
                # print("pref_ori",pref_ori,x0[3])
                x0[1] = (90-pref_ori)/180*np.pi
            # res = optimize.minimize(least_square_diff,x0,gabor,RFsd_intp,sigma,theta,Lambda,psi)
            # fitp,_ = optimize.leastsq(least_square_diff, x0[:], args=(x,y,RFsd_intp))
            ## Solve a nonlinear least-squares problem with bounds on the variables.
            # bounds = ([0,10],np.inf,np.inf,[0.5,12],[0.1,10])
            ## fit parameters: sigma_x, theta, psi, Lambda, gamma
            bounds = ([1,-np.inf,-np.inf,5,0.1],[12,np.inf,np.inf,20,5])
            opt_res = optimize.least_squares(least_square_diff,x0,args=(x,y,RFsd_intp,\
                                             gabor_fit),method="trf",bounds=bounds)
            fitp = opt_res.x
            fitted_gabor[0,i*DAnew:(i+1)*DAnew,j*DAnew:(j+1)*DAnew] =\
                                  gabor_fit(*fitp[:],x=x,y=y).reshape(ynew.size,-1)
            fitted_gabor[1,i*DAnew:(i+1)*DAnew,j*DAnew:(j+1)*DAnew] =\
                                 RFsd_intp.reshape(ynew.size,-1)
            fitp[0] = np.abs(fitp[0])/scaling_factor
            fitp[1] = (fitp[1]%np.pi)
            fitp[2] = (fitp[2]*(1+np.sign(fitp[2]))*0.5 +\
                         0.5*(1-np.sign(fitp[2]))*(2*np.pi+fitp[2]))%(2*np.pi)
            fitp[3] = np.abs(fitp[3])/scaling_factor
            if opt_res.success:
                fit_params[i,j,:] = fitp
            fit_cost[i,j] = opt_res.cost
            num_half_cycles[i,j] = 1./fitp[3]*8*fitp[0]
    return fit_params,fitted_gabor,fit_cost,xmax,ymax,num_half_cycles


def gaussian2d(sigma,x,y):
    return np.exp(-(x**2+y**2)/2./sigma**2)

def determine_LGN_input_correlations_scale(lgn_inputs,avg_cc_diff=None):
    """
    lgn_inputs 		:	shape: num_lgn_paths x dimension_x x dim_y x # patterns
    avg_cc_diff 	:	average correlation matrix of LGN input, if None it is computed here
    """
    if avg_cc_diff is None:
        lgn_norm = (lgn_inputs-np.nanmean(lgn_inputs,axis=3)[:,:,:,None])/\
                    np.nanstd(lgn_inputs,axis=3)[:,:,:,None]
        cc = np.nanmean(lgn_norm[:,None,None,:,None,:,:] * lgn_norm[None,:,:,None,:,None,:],\
                        axis=6)
        N = cc.shape[2]
        avg_cc_diff = 0
        for i in range(N):
            for j in range(N):
                avg_cc_diff += np.roll(np.roll(cc[:,:,i,:,j,:],N//2-i,axis=2),N//2-j,axis=3)
        avg_cc_diff /= N*N
        avg_cc_diff = 0.5*avg_cc_diff[0,0,...]+0.5*avg_cc_diff[1,1,...]-avg_cc_diff[0,1,...]

    h,w = avg_cc_diff.shape
    x,y = np.meshgrid(np.linspace(-0.5,0.5,h),np.linspace(-0.5,0.5,w))
    x = x.flatten()
    y = y.flatten()
    avg_cc_diff = avg_cc_diff.flatten()
    avg_cc_diff /= np.nanmax(avg_cc_diff)
    sigma0 = 0.3
    bounds = ([0],[1])
    opt_res = optimize.least_squares(least_square_diff,sigma0,args=(x,y,avg_cc_diff,\
                                    gaussian2d),method="trf",bounds=bounds)
    fitparams = opt_res.x
    fitted_gauss = gaussian2d(*fitparams,x=x,y=y)
    return fitparams,fitted_gauss.reshape(h,w),avg_cc_diff.reshape(h,w)


def calc_dimension(array, inp='covariance',output=0):
    if inp=='covariance':
        w = np.linalg.eigvalsh(array)		# eigenvalues in ascending order
    elif inp=='patterns':
        ## array shape: num patterns x x,y-dimension
        array = array.reshape(array.shape[0], np.prod(array.shape[1:]))
        array[np.logical_not(np.isfinite(array))] = 0

        #cov = np.cov(array,rowvar=False)
        #w = np.linalg.eigvalsh(cov)

        #array_norm = (array - np.nanmean(array,axis=0)[None,:])
        #cov = np.nanmean(array_norm[:,None,:]*array_norm[None,:,:],axis=2)
        #w = np.linalg.eigvalsh(cov)

        ## interested in pixel covariance matrix <r_i(t)*r_j(t)>_t
        array_norm = (array - np.nanmean(array,axis=0)[None,:])
        s = np.linalg.svd(array_norm,compute_uv=False)
        w = s**2
        #print('w',np.sum(w))
    else:
        print('Input should either be covariance matrix or activity patterns!')
        w = None
    eff_dim = np.sum(w)**2/np.sum(w**2)
    if output==0:
        return eff_dim
    elif output==1:
        return eff_dim,w



if __name__=="__main__":
    import matplotlib.pyplot as plt

    input_size = (50,50)
    phases = np.linspace(0,1,10)
    spat_freq = np.array([np.sqrt(0.5* 1./0.2)])
    orientations = np.linspace(0,np.pi,8)


    ## test gabor fit
    # DA = 5
    # N4 = 4
    # Nvert = 1
    # RFsd = np.random.randn(N4**2*Nvert*DA**2).reshape(DA*N4,DA*N4*Nvert)
    # fit=fit_gabor_to_RF(RFsd,DA=DA,Nvert=Nvert,N4=N4)
    # print("fit",fit)
    # exit()

    # test gabor
    pref_ori = 0
    pref_freq = 4.
    x,y = np.meshgrid(np.linspace(-0.5,0.5,50),np.linspace(-0.5,0.5,50))
    gabor = gabor_fit(sigma=.11,
                 theta=(90-pref_ori)/180*np.pi,
                 psi=0/180*np.pi,
                 Lambda=0.3,
                 gamma=.5,
                 x=x,y=y)
    print("gabor",gabor.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im=ax.imshow(gabor,interpolation="nearest",cmap="RdBu_r",vmin=-1.,vmax=1.)
    plt.colorbar(im,ax=ax)
    plt.savefig("/home/bettina/physics/columbia/conference/cosyne2022/image/tmp/logaspectratio03.pdf",\
                bbox_inches="tight")
    plt.show()
    exit()

    ## test generate_grating_stimuli
    stimuli = generate_grating_stimuli(input_size,10,spat_freq,orientations)
    print(stimuli.shape)

    fig = plt.figure()
    ax = fig.add_subplot(141)
    im=ax.imshow(stimuli[0,0,2,:,:],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(142)
    im=ax.imshow(stimuli[1,0,2,:,:],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(143)
    im=ax.imshow(stimuli[5,0,2,:,:],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    ax = fig.add_subplot(144)
    im=ax.imshow(stimuli[-1,0,2,:,:],interpolation="nearest",cmap="binary")
    plt.colorbar(im,ax=ax)
    plt.show()
