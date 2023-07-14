"""
estimate spatial wavelength of activity patterns (do they change during plasticity?)
"""

import numpy as np
from collections import defaultdict
from scipy.special import jn,jv,erf,jn_zeros
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.ndimage import map_coordinates


## Fit functions
def gauss(x,a,sd):
    return a*np.exp(-x**2/(2*sd**2))

def gabor(x,sd,alpha):
    return np.exp(-x**2/(2*sd**2))*np.cos(alpha*x)

def expcos(x,var,alpha):
    return np.exp(-abs(x)/(2*var))*np.cos(alpha*x)

def cauchy(x,gamma,alpha):
    return gamma**2/(x**2+gamma**2)*np.cos(alpha*x)

def damped_bessel(x,var,alpha):
    return np.exp(-abs(x)/2./var)*jv(0,alpha*x)

def linfct(x,a,b):
    return a*x+b


## Connectivity
def H(k,sigma):
    """ FT of gaussian connectivity """
    return np.exp(-k**2/2.*sigma*sigma)

## Trace, Determinant and Eigenvalue
def tracek(k,aee,aii,see,sii,tau=1,alpha=0):
    aii_s = aii*(1-alpha)
    aii_a = aii*alpha
    return -1 - (1 + aii_s*H(k,sii) + aii_a)/tau + aee*H(k,see)

def detk(k,aee,aeiaie,aii,see,sei,sii,tau=1,alpha=0):
    aii_s = aii*(1-alpha)
    aii_a = aii*alpha
    return ((1 + aii_s*H(k,sii) + aii_a)*(1 - aee*H(k,see)) + aeiaie*H(k,see)*H(k,sei))/tau

def eigval_max(k,aee,aeiaie,aii,see,sei,sii,tau=1,alpha=0):
    tr = tracek(k,aee,aii,see,sii,tau,alpha)
    arg = tr**2 - 4*detk(k,aee,aeiaie,aii,see,sei,sii,tau,alpha)

    sign = np.ones_like(arg,dtype=float)
    sign[arg<0] = -1
    factor = np.ones_like(arg,dtype=complex)
    factor[arg<0] = 1j
    return tr/2. + factor*1./2*np.sqrt(arg*sign)



## helper functions
def get_autocorrelation(pattern,max_spatial_lag=None):
    """calculates autocorrelation of array pattern"""
    hs,ws = pattern.shape
    max_lag = max_spatial_lag
    norm = hs*ws
    if max_lag is None:
        max_lag = hs//2#min([hs//2,ws//2])

    ## wiener-khinchin
    ## first normalise pattern to 0 mean and SD 1
    pattern_norm = pattern - np.nanmean(pattern)
    pattern_norm /= np.nanstd(pattern_norm)
    pattern_norm[np.logical_not(np.isfinite(pattern_norm))] = 0.0
    ## then do fourier transforms
    fft_spectrum = abs(np.fft.fft2(pattern_norm, s=(2*hs,2*ws)))**2
    autocorr = np.fft.ifft2( fft_spectrum )/norm
    autocorr = np.fft.fftshift(autocorr)[max([0,hs-max_lag]):hs+max_lag+1,\
                max([0,ws-max_lag]):ws+max_lag+1]
    return np.real(autocorr)

def radial_average(array,distance_conversion_factor,nangles):
    """calculates radial average of interpolated copy of array
    nangles : number of angles on which to interpolate"""
    if array.ndim>2:
        ys,xs = [],[]
        for frame in array:
            y,x = radial_average(frame,distance_conversion_factor,nangles)
            ys.append(y)
            xs.append(x)
        return np.array(ys),np.array(xs)
    else:
        h,w = array.shape
        # print("ARRAY SHAPE",h,w)
        if h!=w:
            hn = np.min([h,w])
            array = array[h//2-hn//2:h//2+hn//2,w//2-hn//2:w//2+hn//2]
        h,w = array.shape
        ## generate coordinates of line cut through array at y=0
        ## each rotated cut will be interpolated and added to average
        xnew = np.linspace(-(w//2),w//2,w,endpoint=True)
        # print("xnew",xnew)
        ynew = 0*xnew
        coords_new = np.vstack([xnew,ynew])
        angles = np.linspace(0,2*np.pi,nangles)
        spectrum = []
        for phi in angles:
            rotm = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
            this_coord = np.dot(rotm,coords_new)
            ## coordinate origin is at upper left corner for this function!
            spectrum.append( map_coordinates(array,np.vstack([this_coord[1,:],\
                            this_coord[0,:]])+np.array([w/2.,h/2.])[:,None],order=1) )
        return np.nanmean(np.array(spectrum),axis=0), xnew*distance_conversion_factor

def estimate_wavelength_by_fits(spectrum,distance):
    ## FIT
    fit_vals = np.isfinite(spectrum)
    init_var = np.array([0.1,0.25,0.3])
    init_k = np.array([10,15])
    pcov = np.array([[1,1],[1,1]])
    popt = np.array([np.nan,np.nan])
    try:
        for ivar in init_var:
            for ik in init_k:
                ipopt,ipcov = curve_fit(gabor,distance[fit_vals],spectrum[fit_vals],\
                    p0=[ivar,ik])
                if np.mean(np.sqrt(np.diag(ipcov)))<np.mean(np.sqrt(np.diag(pcov))):
                    pcov = ipcov
                    popt = ipopt
    except:
        pass
    perr = np.sqrt(np.diag(pcov))
    w_gabor = np.array([2*np.pi/abs(popt[1]), 2*np.pi/abs(popt[1])**2*perr[1]])
    popt_gabor = popt

    bessel_roots = jn_zeros(0,3)[1:]
    initvar = 0.15
    initk = 20.
    pcov = np.array([[1,1],[1,1]])
    popt = np.array([np.nan,np.nan])
    try:
        popt,pcov = curve_fit(damped_bessel,distance[fit_vals],spectrum[fit_vals],\
                                p0=[initvar,initk])
    except Exception as e:
        # print(e)
        pass
    perr = np.sqrt(np.diag(pcov))
    bessel_roots_mm = bessel_roots/abs(popt[1])
    w_bessel_approx = bessel_roots_mm[0] + (bessel_roots_mm[1]-bessel_roots_mm[0])/2.
    w_bessel = np.array([w_bessel_approx,perr[1]])
    popt_bessel = popt

    return {"bessel_fit" : popt_bessel, "gabor_fit" : popt_gabor,\
             "bessel_wavelength" : w_bessel, "gabor_wavelength" : w_gabor}


## method functions
def spatial_wavelength_by_autocorrelation(patterns):
    if patterns.ndim>2:
        num_frames,h,w = patterns.shape
        result_dict = defaultdict(list)
        for i in range(num_frames):
            result = spatial_wavelength_by_autocorrelation(patterns[i,:,:])
            for key in result:
                result_dict[key].append(result[key])
        return result_dict

    pattern = patterns
    autocorr = get_autocorrelation(pattern)
    resolution = 1./patterns.shape[0]
    spectrum,distance = radial_average(autocorr,resolution,360)
    fit_params_dict = estimate_wavelength_by_fits(spectrum,distance)
    fit_params_dict["spectrum"] = spectrum
    fit_params_dict["distance"] = distance
    return fit_params_dict


def spatial_wavelength_by_avg_correlation(patterns):
    nframes,h,w = patterns.shape
    if patterns.ndim>2:
        patterns = patterns.reshape(nframes,-1)
    cc = np.corrcoef(patterns,rowvar=0).reshape(h*w,h,w)
    avg_cc = 0
    for y in range(h):
        for x in range(w):
            ## not sure about order of array here
            cc_centered = np.roll(np.roll(cc[x+y*w,:,:],shift=h//2-y,axis=0),shift=w//2-x,axis=1)
            # print("x",x,y,cc_centered[h//2,w//2])
            avg_cc += cc_centered
    # print("cc_centered",cc_centered)
    avg_cc /= h*w
    # print("center",avg_cc,avg_cc[h//2,w//2])
    resolution = 1./h
    spectrum,distance = radial_average(avg_cc,resolution,360)
    fit_params_dict = estimate_wavelength_by_fits(spectrum,distance)
    fit_params_dict["spectrum"] = spectrum
    fit_params_dict["distance"] = distance
    fit_params_dict["avg_cc"] = avg_cc
    return fit_params_dict


def spatial_wavelength_by_FFT(patterns):
    if patterns.ndim>2:
        num_frames,h,w = patterns.shape
        result_dict = defaultdict(list)
        for i in range(num_frames):
            result = spatial_wavelength_by_FFT(patterns[i,:,:])
            for key in result:
                result_dict[key].append(result[key])
        return result_dict

    pattern = patterns - np.nanmean(patterns)
    spectrum2d = np.abs(np.fft.fftshift(np.fft.fft2(pattern)))
    spectrum,distance = radial_average(spectrum2d,1,360)
    position_right_peak = distance[np.nanargmax(spectrum[len(spectrum)//2:])+len(spectrum)//2]
    position_left_peak = np.abs(distance[np.nanargmax(spectrum[:len(spectrum)//2])])
    freq_peak_position = (position_left_peak + position_right_peak)*0.5
    wavelength = 1./freq_peak_position
    result = {"spectrum2d" : spectrum2d, "spectrum" :  spectrum, "distance" :\
                distance, "wavelength" : wavelength}
    return result

def create_overview_plot(input_patterns,resultdict1,result_dict2,result_dict3,show=True):
    import matplotlib.pyplot as plt

    Nsur = input_patterns.shape[0]
    ncol,nrow = 3,4
    fig = plt.figure(figsize=(6*ncol,5*nrow))
    for i in range(3):
        ax = fig.add_subplot(nrow,ncol,i+1)
        ax.set_title("Input pattern {}".format(i+1))
        im=ax.imshow(input_patterns[i,:,:],interpolation="nearest",cmap="binary")
        plt.colorbar(im,ax=ax)

    for i in range(3):
        ax = fig.add_subplot(nrow,ncol,i+1+ncol)
        ax.set_title("Autocorrelation Fits {}".format(i+1))
        ax.plot(result_dict1["distance"][i],result_dict1["spectrum"][i],"-k")
        ax.plot(result_dict1["distance"][i],gabor(result_dict1["distance"][i],\
                *result_dict1["gabor_fit"][i]),"--r")
        ax.plot(result_dict1["distance"][i],damped_bessel(result_dict1["distance"][i],\
                *result_dict1["bessel_fit"][i]),"--m")
        ax.plot([-0.5,0.5],[1.,1.],"--",c="gray")

    for i in range(3):
        ax = fig.add_subplot(nrow,ncol,i+1+2*ncol)
        ax.set_title("FFT spectrum {}".format(i+1))
        im=ax.imshow(result_dict3["spectrum2d"][i],interpolation="nearest",cmap="binary")
        plt.colorbar(im,ax=ax)

    ax = fig.add_subplot(nrow,ncol,1+3*ncol)
    ax.set_title("Autocorrelation Fits")
    ax.plot(result_dict2["distance"],result_dict2["spectrum"],"-k")
    ax.plot([-0.5,0.5],[1.,1.],"--",c="gray")
    ax = fig.add_subplot(nrow,ncol,2+3*ncol)
    im=ax.imshow(result_dict2["avg_cc"],interpolation="nearest",cmap="RdBu_r")
    plt.colorbar(im,ax=ax)

    ax = fig.add_subplot(nrow,ncol,3+3*ncol)
    ax.plot(result_dict3["distance"][0],result_dict3["spectrum"][0],"-k")

    ## Distribution of wavelength values
    distribution_dict = defaultdict(list)
    for i in range(Nsur):
        distribution_dict["gabor"].append(result_dict1["gabor_wavelength"][i][0])
        distribution_dict["bessel"].append(result_dict1["bessel_wavelength"][i][0])
        distribution_dict["fft"].append(result_dict3["wavelength"][i])
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.plot(np.sort(distribution_dict["gabor"]),np.linspace(0,1,Nsur),"-b",\
            label="Gabor autocorr")
    ax.plot(np.sort(distribution_dict["bessel"]),np.linspace(0,1,Nsur),"-r",\
            label="Bessel autocorr")
    ax.plot(np.sort(distribution_dict["fft"]),np.linspace(0,1,Nsur),"-g",label="FFT")
    ax.plot(result_dict2["gabor_wavelength"][0],[0.5],"oc",label="Gabor crosscorr")
    ax.plot(result_dict2["bessel_wavelength"][0],[0.5],"om",label="Bessel crosscorr")
    ax.legend(loc="best")

    if show:
        plt.show()
    return fig,fig2,distribution_dict


def create_test_patterns(Nsur):
    from bettina.modeling.ori_dev_model import inputs,Inp_params
    ## create test input pattern of given
    N = 50
    random_seed = 19
    Inp_params.update({"Nsur" : Nsur})

    rng_seed = random_seed*1000
    input_patterns,_,_ = inputs.Inputs((N,N),1,rng_seed).create_matrix(Inp_params, "GRF")
    return input_patterns

if __name__=="__main__":
    import re
    import pickle
    from os import listdir
    from copy import copy
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from collections import defaultdict

    from bettina.modeling.ori_dev_model import config_dict,data_dir,scan_simulations,image_dir
    from bettina.modeling.ori_dev_model.tools import parse_args,update_params_dict,\
    get_spatial_wavelength,misc

    ## TEST PATTERNS
    # Nsur = 100
    # input_patterns = create_test_patterns(Nsur)
    ## test various methods to estimate spatial scale in patterns
    # result_dict1 = spatial_wavelength_by_autocorrelation(input_patterns)
    # result_dict2 = spatial_wavelength_by_avg_correlation(input_patterns)
    # result_dict3 = spatial_wavelength_by_FFT(input_patterns)
    # fig,fig2 = create_overview_plot(input_patterns,result_dict1,result_dict2,result_dict3,\
    # 								show=True)

    ## LOAD ACTUAL L4 ACTIVITY PATTERNS
    args_dict = vars(parse_args.args)
    print("args_dict",args_dict)
    print("")

    lookup_dir = parse_args.get_data_dir()
    if isinstance(args_dict["idx"],int):
        Versions = [args_dict["idx"]]
    elif args_dict["idx"] is None:
        Versions = []
        listfiles = listdir(lookup_dir)
        for item in listfiles:
            name_match = re.match("v(\d+)",item)
            if name_match:
                Versions.append(int(name_match.group(1)))
    else:
        for item in args_dict["idx"]:
            if ":" in item:
                version_list = item.split(":")
                assert version_list[0]!="",\
                 "Error: start and end value expected for list of indices"
                assert version_list[1]!="",\
                 "Error: start and end value expected for list of indices"
                Versions = np.arange(int(version_list[0]),int(version_list[1])+1,1)
            elif "-" in item:
                version_list = item.split("-")
                assert version_list[0]!="",\
                 "Error: start and end value expected for list of indices"
                assert version_list[1]!="",\
                 "Error: start and end value expected for list of indices"
                Versions = np.arange(int(version_list[0]),int(version_list[1])+1,1)
            else:
                assert isinstance(int(item),int), "Error: int value expected for index"
                Versions = [int(item)]

    search_variables = copy(args_dict)
    for key in args_dict.keys():
        if search_variables[key] is None:
            search_variables.pop(key, None)
    search_variables.pop("idx", None)
    search_variables.pop("load_external_from", None)
    search_variables.pop("not_saving_temp", None)
    print("search_variables",search_variables)
    print("Versions",Versions)
    matching_versions = scan_simulations.scan_simulation_for_params(\
                         args_dict["load_external_from"][0],Versions,**search_variables)
    print("")
    print("matching_versions",matching_versions)

    load_external_from = args_dict["load_external_from"][0]
    cluster_name = "local" if load_external_from=="" else load_external_from
    print("cluster_name",cluster_name)
    for version in matching_versions:
        try:
            file_dir = lookup_dir + "v{}/".format(version)
            params = pickle.load(open(file_dir + "config_v{v}.p".format(v=version),"rb"))
            _,_,l4_t,l4I_t,_,_,keys = misc.load_data(version,file_dir,params)
            l4_t = l4_t[1::2,:]
            l4I_t = l4I_t[1::2,:]
            print("check shape Wlgn_to_4_t",l4_t.shape,l4I_t.shape,keys)
            ## take only every 10 pattern
            l4_t = l4_t[::10,:]
            l4I_t = l4I_t[::10,:]
            # analysis_results = pickle.load(open(data_dir +\
            # 					 "layer4/habanero/results/v{v}_results.p".format(v=VERSION),"rb"))
            num_patterns = l4_t.shape[0]
            h,w = params["N4"],params["N4"]*params["Nvert"]
            l4_t = l4_t.reshape(num_patterns,h,w)
            l4I_t = l4I_t.reshape(num_patterns,h,w)

            result_dict1 = get_spatial_wavelength.spatial_wavelength_by_autocorrelation(l4_t)
            result_dict2 = get_spatial_wavelength.spatial_wavelength_by_avg_correlation(l4_t)
            result_dict3 = get_spatial_wavelength.spatial_wavelength_by_FFT(l4_t)

            fig,fig2,distribution_dict = create_overview_plot(l4_t,result_dict1,result_dict2,\
                                            result_dict3,show=False)
            pp = PdfPages(image_dir + "layer4/wavelength/{}_v{}.pdf".format(cluster_name,version))
            pp.savefig(fig,dpi=300,bbox_inches="tight")
            pp.savefig(fig2,dpi=300,bbox_inches="tight")
            plt.close()
            pp.close()

            results_dict = {"cortical_wavelength" : np.nanmedian(distribution_dict["bessel"])}
            misc.write_to_hdf5(results_dict,cluster_name,version)
        except:
            misc.PrintException()


