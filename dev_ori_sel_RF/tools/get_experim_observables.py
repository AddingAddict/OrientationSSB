import numpy as np

from bettina.modeling.ori_dev_model.tools import analysis_tools



def on_off_ratio(RF,**kwargs):
    """
    """

    N4 = kwargs["N4"]
    Nvert = kwargs["Nvert"]
    DA = kwargs["DA"]

    RF_ratio = np.empty((N4,N4*Nvert))*np.nan
    for k in range(N4):
        for l in range(N4*Nvert):
            R_on = np.sum(RF[1,k*DA:(k+1)*DA,l*DA:(l+1)*DA]>0)
            R_off = np.sum(RF[2,k*DA:(k+1)*DA,l*DA:(l+1)*DA]>0)
            # R_on_max = np.nanmax(RF[1,k*DA:(k+1)*DA,l*DA:(l+1)*DA])
            # R_off_max = np.nanmax(RF[2,k*DA:(k+1)*DA,l*DA:(l+1)*DA])
            R_on_mean = np.nanmean(RF[1,k*DA:(k+1)*DA,l*DA:(l+1)*DA])
            R_off_mean = np.nanmean(RF[2,k*DA:(k+1)*DA,l*DA:(l+1)*DA])
            ## consider only units that respond to both dark and light stimuli
            if (R_on>0 and R_off>0):
                # RF_ratio[k,l] = R_on_max/(R_on_max+R_off_max)
                RF_ratio[k,l] = R_on_mean/(R_on_mean+R_off_mean)

    return RF_ratio


def on_off_segregation(RF,**kwargs):
    """
    """

    N4 = kwargs["N4"]
    Nvert = kwargs["Nvert"]
    DA = kwargs["DA"]

    RF_seg = np.empty((N4,N4*Nvert))*np.nan
    for k in range(N4):
        for l in range(N4*Nvert):
            R_on = np.sum(RF[1,k*DA:(k+1)*DA,l*DA:(l+1)*DA]>0)
            R_off = np.sum(RF[2,k*DA:(k+1)*DA,l*DA:(l+1)*DA]>0)
            ## consider only units that respond to both dark and light stimuli
            counter =\
                np.abs(RF[1,k*DA:(k+1)*DA,l*DA:(l+1)*DA] - RF[2,k*DA:(k+1)*DA,l*DA:(l+1)*DA])
            num = RF[1,k*DA:(k+1)*DA,l*DA:(l+1)*DA] + RF[2,k*DA:(k+1)*DA,l*DA:(l+1)*DA]
            ## consider only units that respond to both dark and light stimuli
            if (R_on>0 and R_off>0):
                RF_seg[k,l] = np.nanmean(counter)/np.nanmean(num)

    return RF_seg




def off_anchoring(RF,**kwargs):
    """
    """

    N4 = kwargs["N4"]
    Nvert = kwargs["Nvert"]
    DA = kwargs["DA"]

    RF_array = np.empty((N4,N4*Nvert,DA,DA))*np.nan
    for k in range(N4):
        for l in range(N4*Nvert):
            RF_array[k,l,:,:] = RF[0,k*DA:(k+1)*DA,l*DA:(l+1)*DA]
    RF_array = RF_array.reshape(N4**2*Nvert,DA,DA)
    centroids_on_off_RF = analysis_tools.get_center_of_mass_subfields(RF_array)
    centroids_on_off_RF = centroids_on_off_RF.reshape(N4**2*Nvert,2,2)
    dist_on_center = np.sqrt((DA//2 - centroids_on_off_RF[:,0,0])**2 +\
                         (DA//2 - centroids_on_off_RF[:,1,0])**2)
    dist_off_center = np.sqrt((DA//2 - centroids_on_off_RF[:,0,1])**2 +\
                         (DA//2 - centroids_on_off_RF[:,1,1])**2)
    dist_on_minus_off = dist_on_center - dist_off_center
    ## - RF value at center
    rf_center_value = RF_array[:,DA//2,DA//2]
    return dist_on_center,dist_off_center,dist_on_minus_off,rf_center_value



def average_RF(RF,**kwargs):
    """
    """

    N4 = kwargs["N4"]
    Nvert = kwargs["Nvert"]
    DA = kwargs["DA"]

    ARF = 0
    for k in range(N4):
        for l in range(N4*Nvert):
                ARF += RF[0,k*DA:(k+1)*DA,l*DA:(l+1)*DA]
    ARF /= N4**2*Nvert

    return ARF



def RF_fitparams(sd,RF,**kwargs):
    """
    """

    N4 = kwargs["N4"]
    Nvert = kwargs["Nvert"]
    DA = kwargs["DA"]
    Nlgn = kwargs["Nlgn"]

    ## fit variables in fit_params:
    ## sigma : width of envelope in x direction
    ## theta : orientation of gabor in rad
    ## Lambda : wavelength of gabor
    ## psi : phase of gabor in rad
    ## gamma : decay of envelope in y relative to x direction


    sd = sd.reshape(N4,N4*Nvert,Nlgn,Nlgn)
    opm,Rn = analysis_tools.get_response(sd,DA,Nvert=Nvert)
    sel = np.abs(opm)

    print("RF[0,...]",RF[0,...].size,np.sum(np.isfinite(RF[0,...])))
    fit_params,fitted_gabor,fit_cost,xmax,ymax,num_halfcycles =\
         analysis_tools.fit_gabor_to_RF(RF[0,...],DA=DA,Nvert=Nvert,N4=N4,Rn=Rn)
    fit_params = fit_params.reshape(N4**2*Nvert,-1)
    ncols = fit_params.shape[-1]
    reasonable_fits = fit_params[:,0]>1.
    labels = ["Envelope width","Orientation","Relative phase","# half-cycles",\
              "Log Aspect ratio"]

    gabor_params = np.empty_like(fit_params)*np.nan
    # num_halfcycles = np.nanmean([xmax,ymax])*4/fit_params[:,3]
    gabor_params[:,:3] = fit_params[:,:3]
    gabor_params[:,3] = num_halfcycles.flatten()
    gabor_params[:,4] = -np.log10(fit_params[:,4])

    return gabor_params,fitted_gabor,fit_cost,sel



def compute_experimental_observables(Wlgn_to_4, RF, **kwargs):
    """
    execute all functions that compute experimentally measured observables
    """

    if RF is None:
        N4 = kwargs["N4"]
        Nvert = kwargs["Nvert"]
        DA = kwargs["DA"]
        Nlgn = kwargs["Nlgn"]
        sf = Wlgn_to_4.reshape(2,N4,N4*Nvert,Nlgn,Nlgn)
        RF,_,_,_ = analysis_tools.get_RF_form(sf,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert)

    sd = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
    gabor_params,fitted_gabor,fit_cost,sel = RF_fitparams(sd,RF,**kwargs)

    ARF = average_RF(RF,**kwargs)

    dist_on_center,dist_off_center,dist_on_minus_off,rf_center_value = off_anchoring(RF,**kwargs)

    RF_seg = on_off_segregation(RF,**kwargs)

    RF_ratio = on_off_ratio(RF,**kwargs)

    DA = kwargs["DA"]
    Nvert = kwargs["Nvert"]
    Nlgn = kwargs["Nlgn"]
    N4 = kwargs["N4"]
    sd = sd.reshape(N4,N4*Nvert,Nlgn,Nlgn)
    opm,Rn = analysis_tools.get_response(sd,DA,Nvert=Nvert)

    observables = {
                    "Envelope width" : gabor_params[:,0],
                    "Orientation_fit" : gabor_params[:,1],
                    "Orientation_FT" : opm,
                    "Relative phase" : gabor_params[:,2],
                    "# half cycles" : gabor_params[:,3],
                    "Log aspect ratio" : gabor_params[:,4],
                    "Fitted Gabor" : fitted_gabor,
                    "Fit cost" : fit_cost,
                    "Distance ON to center" : dist_on_center,
                    "Distance OFF to center" : dist_off_center,
                    "ON-OFF Distance to center" : dist_on_minus_off,
                    "Center value RF" : rf_center_value,
                    "ONOFF segregation" : RF_seg,
                    "ONOFF ratio" : RF_ratio,
                    "Average RF" : ARF,
                    "Selectivity" : sel,
    }
    return observables
