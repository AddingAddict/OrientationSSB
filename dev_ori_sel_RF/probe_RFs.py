"""
analyse RF of specific form (either generated via simulation or theoretically assuming eg.
gabor shape) to test phase sensitivity, orientation tuning
one or two layer system with moving grating input
"""

import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dev_ori_sel_RF import image_dir,data_dir,config_dict,dynamics_np,dynamics,\
system_generation,network,network_ffrec
from dev_ori_sel_RF.tools import analysis_tools,misc,update_params_dict


def fio_lin(x):
    return x
def fio_rect(x):
    x[x<0] = 0
    return x
def fio_powerlaw(x):
    x[x<0] = 0
    return x**2

def probe_responses(probe_config_dict,inp_dict,t,lgn,y0,dynamics_system,Wlgn_to_4,W4to4,W4to23,W23to23,W23to4,tau):
    Nret,Nlgn,N4,N23,Nvert = probe_config_dict["Nret"],probe_config_dict["Nlgn"],probe_config_dict["N4"],\
                             probe_config_dict["N23"],probe_config_dict["Nvert"]
    
    print("Wlgn_to_4",Wlgn_to_4.shape,lgn.shape)
    gamma_rec = probe_config_dict["gamma_4"]
    gamma_ff = probe_config_dict["gamma_lgn"]
    dt = config_dict["dt"]
    temporal_duration = inp_dict["temporal_duration"]
    act = np.zeros((len(inp_dict["spat_frequencies"]),len(inp_dict["orientations"]),inp_dict["Nsur"],len(y0)))
    inp = np.zeros((len(inp_dict["spat_frequencies"]),len(inp_dict["orientations"]),inp_dict["Nsur"],len(y0)))
    yt = np.zeros((len(inp_dict["spat_frequencies"]),len(inp_dict["orientations"]),len(t),len(y0)))
    It = np.zeros((len(inp_dict["spat_frequencies"]),len(inp_dict["orientations"]),len(t),len(y0)))
    for i,spat_frequency in enumerate(inp_dict["spat_frequencies"]):
        for j,orientation in enumerate(inp_dict["orientations"]):
            y = y0
            for kt in t:
                # calculate dynamics of activity
                lgn_t = int((kt//temporal_duration)%inp_dict["Nsur"])#
                # lgn_inp = lgn[:2,:,0]
                lgn_inp = lgn[:,:,i,j,lgn_t]
                out = dynamics_system(y,lgn_inp,Wlgn_to_4,W4to4,W4to23,W23to23,\
                                     W23to4,gamma_rec,gamma_ff,N4*N4*Nvert,N23**2,tau)
                try:
                    y = y + dt*out
                except:
                    y = y + dt*out[0]
                yt[i,j,kt] = y

                # only calculate ff input at beginning of new phase
                if kt%temporal_duration == 0:
                    ff_inp_E = gamma_ff*(np.dot(Wlgn_to_4[0,:,:],lgn_inp[0,:])+\
                                        np.dot(Wlgn_to_4[1,:,:],lgn_inp[1,:]))
                    ff_inp_I = gamma_ff*(np.dot(Wlgn_to_4[2,:,:],lgn_inp[0,:])+\
                                        np.dot(Wlgn_to_4[3,:,:],lgn_inp[1,:]))
                It[i,j,kt] = np.concatenate([ff_inp_E,ff_inp_I])
                
                # save last activity and ff input for given phase
                if (kt+1)%temporal_duration == 0:
                    act[i,j,lgn_t,:] = y
                    inp[i,j,lgn_t,:] = np.concatenate([ff_inp_E,ff_inp_I])
    
    return act,inp,yt,It

def plot_probe_RFs(pdf_path,probe_config_dict,inp_dict,lgn,Wlgn_to_4,all_yt,all_It,
                   system_mode='one_layer',RF_mode='load_from_external'):
    Nret,Nlgn,N4,N23,Nvert = probe_config_dict["Nret"],probe_config_dict["Nlgn"],probe_config_dict["N4"],\
                             probe_config_dict["N23"],probe_config_dict["Nvert"]
    suffix = "_ampl{}".format(probe_config_dict["gamma_lgn"])
    last_timestep = all_yt.shape[2]-1 #t[-1]
    gamma_rec = probe_config_dict["gamma_4"]
    gamma_ff = probe_config_dict["gamma_lgn"]
    dt = config_dict["dt"]
    temporal_duration = inp_dict["temporal_duration"]
    if system_mode=="one_layer":
        npop = 2
        L4_size = N4*N4*Nvert
        all_phase = np.zeros((len(inp_dict["spat_frequencies"]),len(inp_dict["orientations"]),npop,N4,N4*Nvert))
        all_modrat = np.zeros((len(inp_dict["spat_frequencies"]),len(inp_dict["orientations"]),npop,N4,N4*Nvert))
    act_last_timestep = []
    for i,spat_frequency in enumerate(inp_dict["spat_frequencies"]):
        for j,orientation in enumerate(inp_dict["orientations"]):
            yt = all_yt[i,j]
            It = all_It[i,j]
            print("It",It.shape,yt.shape)

            # ## collect last responses for all orientations
            # ## take max response over last moving grating
            # last_response = np.nanmax(yt[-temporal_duration*inp_dict["Nsur"]:,...],axis=0)
            ## take mean response over last moving grating
            # last_response = np.nanmean(yt[-temporal_duration*inp_dict["Nsur"]:,...],axis=0)
            ## take response over last moving grating
            last_response = yt[-1,...]
            ## take last frame of ff input to show pattern
            last_input = It[-1,...]
            act_last_timestep.append(last_response)

            # t1 = 500
            t1 = 0
            t2 = int(probe_config_dict["last_timestep"]+1)
            dT = t2-t1
            temp_freq = int((last_timestep+1)//temporal_duration//inp_dict["Nsur"])
            print("temp_freq",temp_freq)
            if system_mode=="one_layer":
                labels = ["L4,E","L4,I"]
                yt_list = [yt[t1:t2,:L4_size].reshape(dT,N4,N4*Nvert),\
                            yt[t1:t2,L4_size:].reshape(dT,N4,N4*Nvert)]
                yfinal_list = [last_response[:L4_size].reshape(N4,N4*Nvert),\
                                last_response[L4_size:].reshape(N4,N4*Nvert)]
                It_list = [It[t1:t2,:L4_size].reshape(dT,N4,N4*Nvert),\
                            It[t1:t2,L4_size:L4_size*2].reshape(dT,N4,N4*Nvert)]
                Ifinal_list = [last_input[:L4_size].reshape(N4,N4*Nvert),\
                                last_input[L4_size:L4_size*2].reshape(N4,N4*Nvert)]


            trace_list = []
            spectrum_list = []
            for k,yt_pop in enumerate(yt_list):
                ## compute modulation ratio per pixel
                # trace = np.nanmean(yt_pop,axis=(1,2))
                # spectrum_avg = np.abs(np.fft.fftshift(np.fft.fft(trace-np.nanmean(trace))))
                # spectrum = np.abs(np.fft.fft(yt_pop-np.nanmean(yt_pop,axis=0),axis=0))
                # trace_list.append(trace)
                # spectrum_list.append([spectrum/np.nanmax(spectrum[:40,:],axis=0)[None,:],\
                #                       spectrum_avg/np.nanmax(spectrum_avg[:40])])
                # mod_ratio.append(analysis_tools.compute_MR(yt_pop,0,temp_freq))
                # pref_phase = analysis_tools.find_preferred_abs_phase(yt_pop,\
                #                                                         temp_freq=temp_freq)
                this_yt = yt_pop
                trace = np.nanmean(this_yt,axis=(1,2))
                spectrum_avg = np.abs(np.fft.fftshift(np.fft.fft(trace-np.nanmean(trace))))
                spectrum = np.abs(np.fft.fft(this_yt-np.nanmean(this_yt,axis=0),axis=0))
                trace_list.append(trace)
                spectrum_list.append([spectrum/np.nanmax(spectrum[:40,:],axis=0)[None,:],\
                                      spectrum_avg/np.nanmax(spectrum_avg[:40])])
                all_modrat[i,j,k] = analysis_tools.compute_MR(this_yt[temporal_duration-1::temporal_duration],
                                                              0,temp_freq)
                pref_phase = analysis_tools.find_preferred_abs_phase(this_yt[temporal_duration-1::temporal_duration],
                                                                     temp_freq=temp_freq)
                print("pref_phase",np.nanmax(pref_phase),np.nanmin(pref_phase))
                all_phase[i,j,k] = pref_phase
            print("Spat frequency={:.0f}, orientation={:.0f} deg".format(spat_frequency,\
                    orientation*180/np.pi),npop)

            ## figure names
            if system_mode=="one_layer":
                if RF_mode in ("gabor","initialize"):
                    filename = "MR_freq{:.0f}_ori{:.0f}_Gabor{}.pdf".format(\
                                spat_frequency,orientation/np.pi*180,suffix)
                elif RF_mode=="load_from_external":
                    filename = "MR_freq{:.0f}_ori{:.0f}_V{}{}.pdf".format(\
                                spat_frequency,orientation/np.pi*180,probe_config_dict["Version"],\
                                suffix)

            pp = PdfPages(pdf_path + filename)
            ## figures for comparing stats of L4 and L23
            ncols,nrows = 4,1
            figc = plt.figure(figsize=(6*ncols,5*nrows))
            figc.suptitle("Comparing L4 and L23")
            axes = []
            for k in range(nrows):
                for l in range(ncols):
                    axes.append(figc.add_subplot(nrows,ncols,l+k*ncols+1))
            for k,(yt_pop,label) in enumerate(zip(yt_list,labels)):
                ## figs for L4 E,I and L23 E,I
                ncols,nrows = 8,1
                fig = plt.figure(figsize=(6*ncols,5*nrows))
                fig.suptitle(label)

                idmax = np.argmax(all_modrat[i,j,k])
                ymax,xmax = idmax//N4,idmax%N4
                idmin = np.argmin(all_modrat[i,j,k])
                ymin,xmin = idmin//N4,idmin%N4

                if (k<2 and "moving_grating_periodic_online" in probe_config_dict["Inp_params"]["input_type"]):
                    ## raw input trace
                    ax = fig.add_subplot(nrows,ncols,1)
                    ax.set_ylabel("Raw input")
                    ax.set_xlabel("Timesteps")
                    ax.plot(np.nanmean(lgn[0,:,i,j,:],axis=1),"-k",label="avg")
                    ax.plot(np.nanmean(lgn[1,:,i,j,:],axis=1),"--k",label="avg")
                    ax.plot(lgn[0,:,i,j,:].reshape(-1,N4,N4*Nvert)[:,ymax,xmax],"-",c="gray",label="max mod")
                    ax.plot(lgn[1,:,i,j,:].reshape(-1,N4,N4*Nvert)[:,ymax,xmax],"--",c="gray",label="max mod")
                    ax.legend(loc="best")

                if (k<2):
                    ## input trace to L4 (raw input convolved with RF)
                    ax = fig.add_subplot(nrows,ncols,2)
                    ax.set_ylabel("Input to LGN")
                    ax.set_xlabel("Timesteps")
                    ax.plot(np.arange(t1,t2),np.nanmean(It_list[k],axis=(1,2)),"-k",label="avg")
                    ax.plot(np.arange(t1,t2),It_list[k][:,ymax,xmax],"-",c="gray",label="max mod")
                    ax.plot(np.arange(t1,t2),It_list[k][:,ymin,xmin],"--",c="gray",label="min mod")
                    ax.legend(loc="best")


                ## activity trace
                ax = fig.add_subplot(nrows,ncols,3)
                ax.set_ylabel("Activity")
                ax.set_xlabel("Timesteps")
                ax.plot(np.arange(t1,t2),trace_list[k],"-k",label="avg")
                ax.plot(np.arange(t1,t2),yt_pop[:,ymax,xmax],"-",c="gray",label="max mod")
                ax.plot(np.arange(t1,t2),yt_pop[:,ymin,xmin],"--",c="gray",label="min mod")
                ax.legend(loc="best")

                ## frequency spectrum
                ax = fig.add_subplot(nrows,ncols,4)
                ax.set_ylabel("Spectrum")
                ax.set_xlabel("Frequency")
                ax.plot(spectrum_list[k][0][:,ymax,xmax],"-",c="gray",label="max mod")
                ax.plot(spectrum_list[k][0][:,ymin,xmin],"--",c="gray",label="min mod")
                ax.plot(spectrum_list[k][1],"-k",label="avg")
                ax.set_xlim(0,inp_dict["Nsur"]*temp_freq)
                ax.set_ylim(0,1)

                if k<2:
                    ## input from LGN to L4
                    ax = fig.add_subplot(nrows,ncols,5)
                    ax.set_title("FF input last timestep")
                    im=ax.imshow(Ifinal_list[k],interpolation="nearest",cmap="binary")
                    plt.colorbar(im,ax=ax)

                ## max/mean resposne over last moving grating
                ax = fig.add_subplot(nrows,ncols,6)
                ax.set_title("Activity last timestep")
                # im=ax.imshow(yfinal_list[k],interpolation="nearest",cmap="binary")
                im=ax.imshow(np.nanmean(yt_pop[-temporal_duration*inp_dict["Nsur"]:,:,:],axis=0),interpolation="nearest",cmap="binary")
                plt.colorbar(im,ax=ax)

                ax = fig.add_subplot(nrows,ncols,7)
                ax.set_title("Modulation ratio")
                im=ax.imshow(all_modrat[i,j,k],interpolation="nearest",cmap="binary",vmin=0,vmax=1)
                plt.colorbar(im,ax=ax)

                ax = fig.add_subplot(nrows,ncols,8)
                ax.set_title("Phase preference")
                im=ax.imshow(all_phase[i,j,k],interpolation="nearest",cmap="hsv",vmin=0,vmax=2*np.pi)
                plt.colorbar(im,ax=ax)
                pp.savefig(fig,dpi=300,bbox_inches="tight")
                plt.close(fig)


                axes[0].plot(np.sort(yt_pop[-temporal_duration*inp_dict["Nsur"]:,:].mean(0).flatten()),\
                            np.linspace(0,1,yfinal_list[k].size),'-',label=label)
                axes[1].plot(np.sort(all_modrat[i,j,k].flatten()),\
                            np.linspace(0,1,all_modrat[i,j,k].size),'-',label=label)

            axes[0].set_xlabel("Activity")
            axes[0].set_ylabel("Cumulative distribution")
            axes[1].set_xlabel("Modulation ratio")
            axes[1].set_ylabel("Cumulative distribution")
            axes[1].set_xlim(0,1)
            for ax in axes:
                ax.legend(loc="best")
            pp.savefig(figc,dpi=300,bbox_inches="tight")
            plt.close(figc)


            ## LGN input
            lgn_inp = lgn[:,:,i,j,-1]
            # exc_ff_inp = gamma_ff * (np.dot(Wlgn_to_4[0,:,:],lgn[0,i,j,0,:]) +\
            # 						 np.dot(Wlgn_to_4[1,:,:],lgn[1,i,j,0,:]))
            exc_ff_inp = gamma_ff * (np.dot(Wlgn_to_4[0,:,:],lgn_inp[0,:]) +\
                                     np.dot(Wlgn_to_4[1,:,:],lgn_inp[1,:]))
            ## visual stimuli
            fig = plt.figure(figsize=(18,5))
            ax = fig.add_subplot(131)
            ax.set_title("Input ON")
            im=ax.imshow(gamma_ff*lgn_inp[0,:].reshape(Nlgn,Nlgn),interpolation="nearest",\
                        cmap="binary")
            plt.colorbar(im,ax=ax)
            ax = fig.add_subplot(132)
            ax.set_title("Input OFF")
            im=ax.imshow(gamma_ff*lgn_inp[1,:].reshape(Nlgn,Nlgn),interpolation="nearest",\
                        cmap="binary")
            plt.colorbar(im,ax=ax)
            ax = fig.add_subplot(133)
            ax.set_title("Total LGN input")
            im=ax.imshow(exc_ff_inp.reshape(Nlgn,Nlgn),interpolation="nearest",\
                        cmap="binary")
            plt.colorbar(im,ax=ax)
            pp.savefig(fig,dpi=300,bbox_inches="tight")
            plt.close(fig)

            ## VARIABILITY IN RESPONSE PATTERN DURING ONE GRATING OSCILLATION
            oscillation_duration = (last_timestep+1)/temp_freq
            timesteps = np.array([last_timestep+1-oscillation_duration,\
                                 last_timestep+1-3/4.*oscillation_duration,\
                                 last_timestep+1-1./2*oscillation_duration,\
                                 last_timestep+1-1./4*oscillation_duration]).astype(int)
            ncol,nrow = len(timesteps),1
            fig = plt.figure(figsize=(6*ncol,5*nrow))
            for k in range(ncol):
                ax = fig.add_subplot(nrow,ncol,k+1)
                ax.set_title("t={}".format(timesteps[k]))
                im=ax.imshow(yt[timesteps[k],:L4_size].reshape(N4,N4),interpolation="nearest",\
                            cmap="binary")
                plt.colorbar(im,ax=ax)
            pp.savefig(fig,dpi=300,bbox_inches="tight")
            plt.close(fig)

            pp.close()
            print("plot done",orientation,spat_frequency)

    act_last_timestep = np.array(act_last_timestep)
    num_oris = len(inp_dict["orientations"])
    if system_mode=="one_layer":
        labels = ["L4,E","L4,I"]
        L4_size = N4*N4*Nvert
        final_act_list = [act_last_timestep[:,:L4_size].reshape(num_oris,N4,N4*Nvert),\
                          act_last_timestep[:,L4_size:].reshape(num_oris,N4,N4*Nvert)]
        all_phase_list = [all_phase[-1,:,0,:,:],all_phase[-1,:,1,:,:]]

    ## file names
    if system_mode=="one_layer":
        if RF_mode=="gabor":
            filename = "ori_phase_freq{:.0f}_Gabor{}.pdf".format(spat_frequency,suffix)
        elif RF_mode=="initialize":
            filename = "ori_phase_freq{:.0f}_init{}.pdf".format(spat_frequency,suffix)
        elif RF_mode=="load_from_external":
            filename = "ori_phase_freq{:.0f}_V{}{}.pdf".format(spat_frequency,\
                                                             probe_config_dict["Version"],suffix)
    pp = PdfPages(pdf_path + filename)

    sf = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
    rA = int(probe_config_dict["Wlgn_to4_params"]["r_A_on"] * probe_config_dict["Wlgn_to4_params"].get("r_lim",1.) * N4)
    DA = 2*rA + 5
    RF,_,_,_ = analysis_tools.get_RF_form(sf,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,mode="o")
    ext = np.nanmax(np.abs(RF))
    fig = plt.figure(figsize=(6,5))
    fig.suptitle("RFs used")
    ax = fig.add_subplot(111)
    im=ax.imshow(RF,interpolation="nearest",cmap="RdBu_r",vmin=-ext,vmax=ext)
    plt.colorbar(im,ax=ax)
    pp.savefig(dpi=300,bbox_inches="tight")
    plt.close(fig)

    ncols,nrows = 4,1
    orientations_binning = np.concatenate([inp_dict["orientations"],np.array([np.pi])])
    # orientations_binning -= np.pi/2./num_oris
    for j,(final_act,jall_phase) in enumerate(zip(final_act_list,all_phase_list)):
        opm = analysis_tools.compute_orientation_tuning_from_activity(final_act,\
                                                        inp_dict["orientations"]*2,norm=True)
        sel = np.abs(opm)
        ori = np.angle(opm)*0.5
        ori = ori - (np.sign(ori)-1)*0.5*np.pi

        ori_bin = np.searchsorted(orientations_binning,ori,side='left')
        ori_bin[ori_bin==num_oris] = 0
        ori_bin[ori_bin==num_oris+1] = 0
        pref_phase_full = np.zeros_like(ori,dtype=float)
        for y in range(ori.shape[0]):
            for x in range(ori.shape[1]):
                pref_phase_full[y,x] = jall_phase[ori_bin[y,x],y,x]

        fig = plt.figure(figsize=(6*ncols,nrows*5))
        ax = fig.add_subplot(nrows,ncols,1)
        ax.set_title("Orientation preference map {}".format(labels[j]))
        # im=ax.imshow(misc.plot_complex_map(opm),interpolation="nearest")
        im=ax.imshow(ori,cmap="twilight",interpolation="nearest",vmin=0,vmax=np.pi)
        plt.colorbar(im,ax=ax,orientation="horizontal")

        ax = fig.add_subplot(nrows,ncols,2)
        opm_fft = np.abs(np.fft.fftshift(np.fft.fft2(opm - np.nanmean(opm))))
        opm_fps = np.zeros(int(np.ceil(N4//2*np.sqrt(2))))
        grid = np.arange(-N4//2,N4//2)
        x,y = np.meshgrid(grid,grid)
        bin_idxs = np.digitize(np.sqrt(x**2+y**2),np.arange(0,np.ceil(N4//2*np.sqrt(2)))+0.5)
        for idx in range(0,int(np.ceil(N4//2*np.sqrt(2)))):
            opm_fps[idx] = np.mean(opm_fft[bin_idxs == idx])
        ax.set_title("Spectrum Orientation preference map {}".format(labels[j]))
        im=ax.imshow(opm_fft,cmap="binary",interpolation="nearest")
        plt.colorbar(im,ax=ax,orientation="horizontal")
        ax.plot(np.arange(N4//2,N4),(N4//4)*opm_fps[:N4//2]/np.nanmax(opm_fps[:N4//2-1]))

        ax = fig.add_subplot(nrows,ncols,3)
        ax.set_title("Preferred absolute phase {}".format(labels[j]))
        im=ax.imshow(pref_phase_full,interpolation="nearest",cmap="twilight",vmin=0,vmax=2*np.pi)
        plt.colorbar(im,ax=ax,orientation="horizontal")

        ax = fig.add_subplot(nrows,ncols,4)
        opm_fft = np.abs(np.fft.fftshift(np.fft.fft2(pref_phase_full - np.nanmean(pref_phase_full))))
        opm_fps = np.zeros(int(np.ceil(N4//2*np.sqrt(2))))
        grid = np.arange(-N4//2,N4//2)
        x,y = np.meshgrid(grid,grid)
        bin_idxs = np.digitize(np.sqrt(x**2+y**2),np.arange(0,np.ceil(N4//2*np.sqrt(2)))+0.5)
        for idx in range(0,int(np.ceil(N4//2*np.sqrt(2)))):
            opm_fps[idx] = np.mean(opm_fft[bin_idxs == idx])
        ax.set_title("Spectrum Preferred absolute phase {}".format(labels[j]))
        im=ax.imshow(opm_fft,cmap="binary",interpolation="nearest")
        plt.colorbar(im,ax=ax,orientation="horizontal")
        ax.plot(np.arange(N4//2,N4),(N4//4)*opm_fps[:N4//2]/np.nanmax(opm_fps[:N4//2-1]))

        pp.savefig(dpi=300,bbox_inches="tight")
        plt.close(fig)

    pp.close()

    return all_phase,all_modrat

def probe_RFs_one_layer(Version,config_name,freqs=np.array([60,80,100]),oris=np.linspace(0,np.pi,4,endpoint=False),Nsur=8,temporal_duration=500,calc_pref_freq=False,outdir=None):
    RF_mode = "load_from_external"
    system_mode = "one_layer"
    connectivity_type = "EI"
    load_location = 'local'

    load_path = data_dir + "layer4/{s}/v{v}/".format(s=config_name,v=Version)
    probe_config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
    Nret,Nlgn,N4,N23,Nvert = probe_config_dict["Nret"],probe_config_dict["Nlgn"],probe_config_dict["N4"],\
                             probe_config_dict["N23"],probe_config_dict["Nvert"]
    
    if outdir is None:
        pdf_path = image_dir + "grating_responses/{}/v{}_{}/".format(config_name,Version,load_location)
    else:
        pdf_path = os.path.join(outdir,"grating_responses/{}/v{}_{}/".format(config_name,Version,load_location))
    misc.ensure_path(pdf_path)

    probe_config_dict["Wlgn_to4_params"].update({
        "W_mode": RF_mode,
        "load_from_prev_run" : Version})

    dt = probe_config_dict["dt"]
    # T_pd = 1000
    T_pd = temporal_duration*dt*Nsur
    t = np.arange(0,T_pd/dt,1).astype(int)
    probe_config_dict["Inp_params"].update({"input_type" : "moving_grating_periodic_online"})
    # probe_config_dict["Inp_params"].update({"input_type" : "moving_grating_periodic_online_sharp"})
    # probe_config_dict["Inp_params"].update({"input_type" : "white_noise_online"})
    last_timestep = t[-1]
    probe_config_dict.update({
                        "config_name" : config_name,
                        "last_timestep" : last_timestep,
                        "RF_mode" : RF_mode,
                        "system" : system_mode,
                        "Version" : Version
                        })
    n = network.Network(Version,probe_config_dict,load_location=load_location)#,ampl_het=0.,spatial_freq_het=0.
    kwargs = {
                ## parameters for moving gratings
                "num_freq" : 1,
                "spat_frequencies" : freqs,#40,60,90
                "orientations" : oris,
                "Nsur" : Nsur,
                "temporal_duration": temporal_duration,
                "dt": dt
    }

    _,Wlgn_to_4,arbor_on,arbor_off,arbor2,_,W4to4 = n.system

    ################################# calc pref freq ###############################
    if calc_pref_freq:
        sd = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
        sd = sd.reshape((N4,N4,Nlgn,Nlgn))
        
        xs,ys = np.meshgrid(np.arange(Nlgn),np.arange(Nlgn))
        xs[xs > Nlgn//2] -= Nlgn
        ys[ys > Nlgn//2] -= Nlgn
        ls = np.sqrt(xs**2+ys**2)
        ls_bins = np.digitize(ls,np.arange(0,Nlgn//2)+0.5)
        
        sd_fft = np.fft.fft2(sd,axes=(2,3))
        sd_fft_mag = np.zeros((N4,N4,Nlgn//2))
        for i in range(N4):
            for j in range(N4):
                for k in range(Nlgn//2):
                    sd_fft_mag[i,j,k] = np.max(np.abs(sd_fft[i,j,ls_bins == k]))
        sd_fft_mag = np.mean(sd_fft_mag,axis=(0,1))
        pref_lam = np.argmax(sd_fft_mag)
        kwargs.update({
            "spat_frequencies" : np.array([pref_lam*Nlgn,])
        })
        
    lgn = n.generate_inputs(full_lgn_output=True,last_timestep=last_timestep,\
                            same_EI_input=True,**kwargs)
    # lgn += 0.1
    lgn -= np.nanmin(lgn) #- 0.5
    print("lgn",lgn.shape)

    lgn_rshp = lgn.reshape(2,-1,Nlgn**2,len(oris),Nsur)
    print(lgn_rshp.shape)
    lgn_rshp = lgn_rshp.transpose((0,2,3,4,1))
    print(lgn_rshp.shape)
    lgn_rshp = lgn_rshp.reshape(2,Nlgn**2,len(oris),len(freqs),Nsur)
    print(lgn_rshp.shape)
    lgn_rshp = lgn_rshp.reshape(2,Nlgn**2,len(freqs),len(oris),Nsur)
    print(lgn_rshp.shape)
    # lgn_rshp /= 4
    
    ################################# initialization ###############################
    np.random.seed(probe_config_dict["random_seed"]*113)
    l40 = np.random.uniform(0,1,2*N4*N4*Nvert)*0.1
    y0 = l40
    def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                        W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,tau):
        return dynamics_np.dynamics_onelayer(y,inp_ff,Wff,W4to4,gamma_rec,gamma_ff,N4,\
                                            tau,fio=fio_rect)
    if connectivity_type=="EI":
        def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                            W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,tau):
            return dynamics_np.dynamics_onelayer_fullinput(y,inp_ff,Wff,W4to4,gamma_rec,\
                                                            gamma_ff,N4,tau,fio=fio_rect)
    W4to23 = 0
    W23to4 = 0
    W23to23= 0

    if probe_config_dict["tau"]!=1:
        tau = np.ones((N4**2*2*Nvert),dtype=float)
        tau[N4**2*Nvert:] *= probe_config_dict["tau"]
    else:
        tau = 1.
        
    act,inp,yt,It = probe_responses(probe_config_dict,kwargs,t,lgn_rshp,y0,dynamics_system,
                                    Wlgn_to_4,W4to4,W4to23,W23to23,W23to4,tau)
    phase,modrat = plot_probe_RFs(pdf_path,probe_config_dict,kwargs,lgn,Wlgn_to_4,yt,It,
                                  system_mode=system_mode,RF_mode=RF_mode)
    return act,inp,phase,modrat

def probe_RFs_ffrec(Version,config_name,freqs=np.array([60,80,100]),oris=np.linspace(0,np.pi,4,endpoint=False),Nsur=10,calc_pref_freq=False,outdir=None):
    RF_mode = "load_from_external"
    system_mode = "one_layer"
    connectivity_type = "EI"
    load_location = 'local'

    load_path = data_dir + "ffrec/{s}/v{v}/".format(s=config_name,v=Version)
    probe_config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
    Nret,Nlgn,N4,N23,Nvert = probe_config_dict["Nret"],probe_config_dict["Nlgn"],probe_config_dict["N4"],\
                             probe_config_dict["N23"],probe_config_dict["Nvert"]
    
    if outdir is None:
        pdf_path = image_dir + "grating_responses/{}/v{}_{}/".format(config_name,Version,load_location)
    else:
        pdf_path = os.path.join(outdir,"grating_responses/{}/v{}_{}/".format(config_name,Version,load_location))
    misc.ensure_path(pdf_path)

    probe_config_dict["Wlgn_to4_params"].update({
        "W_mode": RF_mode,
        "load_from_prev_run" : Version})
    if "2pop" in probe_config_dict["W4to4_params"]["Wrec_mode"]:
        probe_config_dict["W4to4_params"].update({
            "Wrec_mode": "load_from_external2pop"})
    else:
        probe_config_dict["W4to4_params"].update({
            "Wrec_mode": "load_from_external"})

    temporal_duration = 500
    dt = probe_config_dict["dt"]
    # T_pd = 1000
    T_pd = temporal_duration*dt*Nsur
    t = np.arange(0,T_pd/dt,1).astype(int)
    probe_config_dict["Inp_params"].update({"input_type" : "moving_grating_periodic_online"})
    # probe_config_dict["Inp_params"].update({"input_type" : "moving_grating_periodic_online_sharp"})
    # probe_config_dict["Inp_params"].update({"input_type" : "white_noise_online"})
    last_timestep = t[-1]
    probe_config_dict.update({
                        "config_name" : config_name,
                        "last_timestep" : last_timestep,
                        "RF_mode" : RF_mode,
                        "system" : system_mode,
                        "Version" : Version
                        })
    n = network_ffrec.Network(Version,probe_config_dict,load_location=load_location)#,ampl_het=0.,spatial_freq_het=0.
    kwargs = {
                ## parameters for moving gratings
                "num_freq" : 1,
                "spat_frequencies" : freqs,#40,60,90
                "orientations" : oris,
                "Nsur" : Nsur,
                "temporal_duration": temporal_duration,
                "dt": dt
    }

    _,Wlgn_to_4,arbor_on,arbor_off,arbor2,_,W4to4,arbor4to4,_ = n.system

    ################################# calc pref freq ###############################
    if calc_pref_freq:
        sd = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
        sd = sd.reshape((N4,N4,Nlgn,Nlgn))
        
        xs,ys = np.meshgrid(np.arange(Nlgn),np.arange(Nlgn))
        xs[xs > Nlgn//2] -= Nlgn
        ys[ys > Nlgn//2] -= Nlgn
        ls = np.sqrt(xs**2+ys**2)
        ls_bins = np.digitize(ls,np.arange(0,Nlgn//2)+0.5)
        
        sd_fft = np.fft.fft2(sd,axes=(2,3))
        sd_fft_mag = np.zeros((N4,N4,Nlgn//2))
        for i in range(N4):
            for j in range(N4):
                for k in range(Nlgn//2):
                    sd_fft_mag[i,j,k] = np.max(np.abs(sd_fft[i,j,ls_bins == k]))
        sd_fft_mag = np.mean(sd_fft_mag,axis=(0,1))
        pref_lam = np.argmax(sd_fft_mag)
        kwargs.update({
            "spat_frequencies" : np.array([pref_lam*Nlgn,])
        })
        
    lgn = n.generate_inputs(full_lgn_output=True,last_timestep=last_timestep,\
                            same_EI_input=True,**kwargs)
    # lgn += 0.1
    lgn -= np.nanmin(lgn) #- 0.5
    print("lgn",lgn.shape)

    lgn_rshp = lgn.reshape(2,-1,Nlgn**2,len(oris),Nsur)
    print(lgn_rshp.shape)
    lgn_rshp = lgn_rshp.transpose((0,2,3,4,1))
    print(lgn_rshp.shape)
    lgn_rshp = lgn_rshp.reshape(2,Nlgn**2,len(oris),len(freqs),Nsur)
    print(lgn_rshp.shape)
    lgn_rshp = lgn_rshp.reshape(2,Nlgn**2,len(freqs),len(oris),Nsur)
    print(lgn_rshp.shape)
    # lgn_rshp /= 4
    
    ################################# initialization ###############################
    np.random.seed(probe_config_dict["random_seed"]*113)
    l40 = np.random.uniform(0,1,2*N4*N4*Nvert)*0.1
    y0 = l40
    def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                        W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,tau):
        return dynamics_np.dynamics_onelayer(y,inp_ff,Wff,W4to4,gamma_rec,gamma_ff,N4,\
                                            tau,fio=fio_rect)
    if connectivity_type=="EI":
        def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                            W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,tau):
            return dynamics_np.dynamics_onelayer_fullinput(y,inp_ff,Wff,W4to4,gamma_rec,\
                                                            gamma_ff,N4,tau,fio=fio_rect)
    W4to23 = 0
    W23to4 = 0
    W23to23= 0

    if probe_config_dict["tau"]!=1:
        tau = np.ones((N4**2*2*Nvert),dtype=float)
        tau[N4**2*Nvert:] *= probe_config_dict["tau"]
    else:
        tau = 1.
        
    act,inp,yt,It = probe_responses(probe_config_dict,kwargs,t,lgn_rshp,y0,dynamics_system,
                                    Wlgn_to_4,W4to4,W4to23,W23to23,W23to4,tau)
    phase,modrat = plot_probe_RFs(pdf_path,probe_config_dict,kwargs,lgn,Wlgn_to_4,yt,It,
                                  system_mode=system_mode,RF_mode=RF_mode)
    return act,inp,phase,modrat

if __name__=="__main__":

    ## set up system
    RF_mode = "load_from_external"#"load_from_external"#"gabor"#"initialize"
    system_mode = "two_layer" #"two_layer" "one_layer"
    connectivity_type = "EI"
    N = 25
    Version = 13
    load_location = "local"

    T_pd = 1000
    gamma_ff = 0.01#config_dict["gamma_lgn"]


    if RF_mode=="load_from_external":
        ## load config_dict
        ## load parameters
        if load_location=="habanero":
            if os.environ["USER"]=="bh2757":
                if system_mode=="two_layer":
                    load_path = data_dir + "two_layer/v{v}/".format(v=Version)
                else:
                    load_path = data_dir + "layer4/v{v}/".format(v=Version)
            else:
                # load_path = data_dir + "two_layer/habanero/v{v}/".format(v=Version)
                if system_mode=="two_layer":
                    load_path =\
                     "/media/bettina/Seagate Portable Drive/physics/columbia/projects/" +\
                     "ori_dev_model/data/two_layer/habanero/v{v}/".format(v=Version)
                else:
                    load_path =\
                     "/media/bettina/Seagate Portable Drive/physics/columbia/projects/" +\
                     "ori_dev_model/data/layer4/habanero/v{v}/".format(v=Version)

        elif load_location=="aws":
            # load_path = data_dir + "two_layer/aws/v{v}/".format(v=Version)
            load_path =\
                 "/media/bettina/Seagate Portable Drive/physics/columbia/projects/" +\
                 "ori_dev_model/data/two_layer/aws/v{v}/".format(v=Version)
        else:
            if system_mode=="two_layer":
                load_path = data_dir + "two_layer/v{v}/".format(v=Version)
            else:
                load_path = data_dir + "layer4/v{v}/".format(v=Version)
        config_dict = pickle.load(open(load_path + "config_v{v}.p".format(v=Version),"rb"))
        update_params_dict.update_params(config_dict)
        if "plastic" not in config_dict["W4to23_params"].keys():
            config_dict["W4to23_params"]["plastic"] = False
            config_dict["W4to23_params"]["arbor_profile"] = "gaussian"
            config_dict["W4to23_params"]["s_noise"] = 0.2
        config_dict["Wlgn_to4_params"].update({"W_mode": RF_mode,
                                               "load_from_prev_run" : Version,\
                                               "connectivity_type" : connectivity_type})
        pdf_path = image_dir + "grating_responses/v{}_{}/".format(Version,load_location)
        Nret,Nlgn,N4,N23,Nvert = config_dict["Nret"],config_dict["Nlgn"],config_dict["N4"],\
                                 config_dict["N23"],config_dict["Nvert"]
        suffix = "_ampl{}".format(gamma_ff)

    else:
        config_dict["Wlgn_to4_params"].update({"load_from_prev_run" : Version,\
                                                 "connectivity_type" : connectivity_type})
        Nret,Nlgn,N4,N23 = N,N,N,N
        Nvert = 1
        config_dict.update({"Nlgn" : Nlgn,
                            "Nvert" : Nvert,
                            "N23" : N23,
                            "N4" : N4,
                            "Nret" : Nret})
        config_dict["tau"] = 0.2
        config_dict["W4to4_params"]["sigma_factor"] = 0.5
        config_dict["W4to4_params"]["max_ew"] = 0.95
        # config_dict["W23_params"]["max_ew"] = 1.02
        # config_dict["W4to23_params"]["aEE"] *= 2
        # config_dict["W4to23_params"]["sigma_EE"] = 0.15
        # config_dict["W4to23_params"]["sigma_IE"] = 0.15
        # config_dict["W4to23_params"]["aEE"] *= 2
        # config_dict["W4to23_params"]["aIE"] *= 2
        # config_dict["W4to23_params"]["max_ew"] = "orig"
        # config_dict["W23to4_params"]["aEE"] *= 5
        # config_dict["W23to4_params"]["aIE"] *= 5
        # config_dict["W23to4_params"]["max_ew"] = "orig"
        # config_dict["W4to23_params"]["r_A"] = 0.3
        suffix = ""#"_ff*2_fb*5_init"
        pdf_path = image_dir + "grating_responses/gabor/"
    misc.ensure_path(pdf_path)

    ## SIMULATION PARAMETERS
    dt = config_dict["dt"]
    t = np.arange(0,T_pd/dt,1).astype(int)
    config_dict["Inp_params"].update({"input_type" : "moving_grating_online"})
    # config_dict["Inp_params"].update({"input_type" : "white_noise_online"})
    last_timestep = t[-1]
    config_dict.update({
                        "last_timestep" : last_timestep,
                        "RF_mode" : RF_mode,
                        "system" : system_mode,
                        "Version" : Version
                        })
    n = network.Network(Version,config_dict,load_location=load_location)#,ampl_het=0.,spatial_freq_het=0.
    kwargs = {
                ## parameters for moving gratings
                "num_freq" : 1,
                "spat_frequencies" : np.array([80,]),#40,60,90
                "orientations" : np.linspace(0,np.pi,8,endpoint=False),
                "Nsur" : 10,
    }
    lgn = n.generate_inputs(full_lgn_output=True,last_timestep=last_timestep,\
                            same_EI_input=True,**kwargs)
    # lgn += 0.1
    lgn -= np.nanmin(lgn) #- 0.5
    print("lgn",lgn.shape)
    # print("lgn",lgn.shape,np.nanmax(lgn,axis=(0,1,2,4)),np.nanstd(lgn,axis=(0,1,2,4)))

    # _,Wlgn_to_4,_,_,_,_,_,_,_,_,_,_ = n.system
    # sf = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
    # rA = int(config_dict["Wlgn_to4_params"]["r_A_on"] * N4)
    # DA = 2*rA + 5
    # RF,_,_,_ = analysis_tools.get_RF_form(sf,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,mode="o")
    # ext = np.nanmax(np.abs(RF))
    # print("RF",RF.shape)
    # fig = plt.figure(figsize=(6,5))
    # ax = fig.add_subplot(111)
    # im=ax.imshow(RF,interpolation="nearest",cmap="RdBu_r",vmin=-ext,vmax=ext)
    # plt.colorbar(im,ax=ax)
    # plt.savefig(pdf_path+"RF{}.pdf".format(suffix))
    # plt.close()
    # exit()


    if system_mode=="one_layer":
        _,Wlgn_to_4,arbor_on,arbor_off,arbor2,_,W4to4 = n.system
        ################################# initialization ###############################
        np.random.seed(config_dict["random_seed"]*113)
        l40 = np.random.uniform(0,1,2*N4*N4*Nvert)*0.1
        y0 = l40
        def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                            W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,tau):
            return dynamics_np.dynamics_onelayer(y,inp_ff,Wff,W4to4,gamma_rec,gamma_ff,N4,\
                                                tau,fio=fio_rect)
        if connectivity_type=="EI":
            def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                                W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,tau):
                return dynamics_np.dynamics_onelayer_fullinput(y,inp_ff,Wff,W4to4,gamma_rec,\
                                                                gamma_ff,N4,tau,fio=fio_rect)
        W4to23 = 0
        W23to4 = 0
        W23to23= 0
    elif system_mode=="two_layer":
        _,Wlgn_to_4,arbor_on,arbor_off,arbor2,_,W4to4,arbor4to4,W23to23,arbor23,W4to23,arbor4to23,_,W23to4 = n.system
        ################################# initialization ###############################
        np.random.seed(config_dict["random_seed"]*113)
        l40 = np.random.uniform(0,1,2*N4*N4*Nvert)*0.1
        l230 = np.random.uniform(0,1,2*N23**2)*0.1
        y0 = np.concatenate([l40,l230])

        def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                            W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,tau):
            return dynamics_np.dynamics_twolayer(y,inp_ff,Wff,W4to4,W4to23,W23to23,\
                                                 W23to4,gamma_rec,gamma_ff,N4,N23,tau,\
                                                 fio=fio_rect)

        if connectivity_type=="EI":
            def dynamics_system(y,inp_ff,Wff,W4to4,W4to23,\
                                W23to23,W23to4,gamma_rec,gamma_ff,N4,N23,tau):
                return dynamics_np.dynamics_twolayer_fullinput(y,inp_ff,Wff,W4to4,W4to23,\
                                                                W23to23,W23to4,gamma_rec,\
                                                                 gamma_ff,N4,N23,tau,\
                                                                 fio=fio_rect)

        # W23to4 *= 0

    ################################# DYNAMICS ###############################
    if config_dict["tau"]!=1:
        tau = np.ones((N4**2*2*Nvert),dtype=float)
        tau[N4**2*Nvert:] *= config_dict["tau"]
    else:
        tau = 1.


    I = np.linalg.inv(np.diagflat(np.ones(N4*N4*2*Nvert)) - W4to4)

    print("Wlgn_to_4",Wlgn_to_4.shape,lgn.shape)
    gamma_rec = config_dict["gamma_4"]
    temporal_duration = 500
    num_reps = t[-1]/temporal_duration
    for i,spat_frequency in enumerate(kwargs["spat_frequencies"]):
        all_phase = []
        act_last_timestep = []
        for j,orientation in enumerate(kwargs["orientations"]):
            mod_ratio = []
            phases = []
            yt = [y0]
            y = y0
            It = [l40*0]
            for kt in t:
                lgn_t = int((kt//temporal_duration)%kwargs["Nsur"])#
                # inp = lgn[:2,:,0]
                inp = lgn[:,:,i,j,lgn_t]
                dy,_ = dynamics_system(y,inp,Wlgn_to_4,W4to4,W4to23,W23to23,\
                                     W23to4,gamma_rec,gamma_ff,N4*N4*Nvert,N23**2,tau)
                y = y + dt*dy
                yt.append( y )

                # ff input
                ff_inp_E = gamma_ff*(np.dot(Wlgn_to_4[0,:,:],inp[0,:])+\
                                     np.dot(Wlgn_to_4[1,:,:],inp[1,:]))
                ff_inp_I = gamma_ff*(np.dot(Wlgn_to_4[2,:,:],inp[0,:])+\
                                     np.dot(Wlgn_to_4[3,:,:],inp[1,:]))
                It.append(np.concatenate([ff_inp_E,ff_inp_I]))

            # l4_I_avg = 0
            # for k in range(lgn.shape[-1]):
            # 	l4_I = np.dot(I[:,:N4*N4*Nvert], np.dot(Wlgn_to_4[0,:,:],lgn[0,:,i,j,k]) +\
            # 	 		np.dot(Wlgn_to_4[1,:,:],lgn[1,:,i,j,k])) * gamma_ff
            # 	if connectivity_type=="EI":
            # 		l4_I_toI = np.dot(I[:,N4*N4*Nvert:], np.dot(Wlgn_to_4[2,:,:],lgn[0,:,i,j,k]) +\
            # 		 			np.dot(Wlgn_to_4[3,:,:],lgn[1,:,i,j,k])) * gamma_ff
            # 		l4_I += l4_I_toI
            # 	l4_I = np.clip(l4_I,0,np.nanmax(l4_I))
            # 	l4_I_avg += l4_I
            # yt = np.ones_like(t)

            # ff_inp_E = gamma_ff*(np.dot(Wlgn_to_4[0,:,:],lgn[0,:,i,j,:])+\
            # 					 np.dot(Wlgn_to_4[1,:,:],lgn[1,:,i,j,:]))
            # ff_inp_I = gamma_ff*(np.dot(Wlgn_to_4[2,:,:],lgn[0,:,i,j,:])+\
            # 					 np.dot(Wlgn_to_4[3,:,:],lgn[1,:,i,j,:]))
            # It = np.stack([np.repeat(ff_inp_E,num_reps,axis=1),np.repeat(ff_inp_E,num_reps,axis=1)])

            yt = np.array(yt)
            It = np.array(It)
            print("It",It.shape,yt.shape)

            ## collect last responses for all orientations
            ## take max response over last moving grating
            last_response = np.nanmax(yt[-temporal_duration*kwargs["Nsur"]:,...],axis=0)
            ## take mean response over last moving grating
            # last_response = np.nanmean(yt[-temporal_duration*kwargs["Nsur"]:,...],axis=0)
            ## take last frame of ff input to show pattern
            last_input = It[-1,...]
            act_last_timestep.append(last_response)

            t1 = 500
            t2 = int(config_dict["last_timestep"]+1)
            dT = t2-t1
            temp_freq = int((last_timestep+1)//temporal_duration//kwargs["Nsur"])
            print("temp_freq",temp_freq)
            if system_mode=="one_layer":
                labels = ["L4,E","L4,I"]
                L4_size = N4*N4*Nvert
                yt_list = [yt[t1:t2,:L4_size].reshape(dT,N4,N4*Nvert),\
                            yt[t1:t2,L4_size:].reshape(dT,N4,N4*Nvert)]
                yfinal_list = [last_response[:L4_size].reshape(N4,N4*Nvert),\
                                last_response[L4_size:].reshape(N4,N4*Nvert)]
                It_list = [It[t1:t2,:L4_size].reshape(dT,N4,N4*Nvert),\
                            It[t1:t2,L4_size:L4_size*2].reshape(dT,N4,N4*Nvert)]
                Ifinal_list = [last_input[:L4_size].reshape(N4,N4*Nvert),\
                                last_input[L4_size:L4_size*2].reshape(N4,N4*Nvert)]
            elif system_mode=="two_layer":
                labels = ["L4,E","L4,I","L23,E","L23,I"]
                L4_size = N4*N4*Nvert
                L23_size = N23**2
                yt_list = [yt[t1:t2,:L4_size].reshape(dT,N4,N4*Nvert),\
                            yt[t1:t2,L4_size:L4_size*2].reshape(dT,N4,N4*Nvert),\
                            yt[t1:t2,L4_size*2:L4_size*2+L23_size].reshape(dT,N23,N23),\
                            yt[t1:t2,L4_size*2+L23_size:].reshape(dT,N23,N23)]
                yfinal_list = [last_response[:L4_size].reshape(N4,N4*Nvert),\
                                last_response[L4_size:L4_size*2].reshape(N4,N4*Nvert),\
                                last_response[L4_size*2:L4_size*2+L23_size].reshape(N23,N23),\
                                last_response[L4_size*2+L23_size:].reshape(N23,N23)]
                It_list = [It[t1:t2,:L4_size].reshape(dT,N4,N4*Nvert),\
                            It[t1:t2,L4_size:L4_size*2].reshape(dT,N4,N4*Nvert)]
                Ifinal_list = [last_input[:L4_size].reshape(N4,N4*Nvert),\
                                last_input[L4_size:L4_size*2].reshape(N4,N4*Nvert)]


            trace_list = []
            spectrum_list = []
            for yt_pop in yt_list:
                ## compute modulation ratio per pixel
                trace = np.nanmean(yt_pop,axis=(1,2))
                spectrum_avg = np.abs(np.fft.fftshift(np.fft.fft(trace-np.nanmean(trace))))
                spectrum = np.abs(np.fft.fft(yt_pop-np.nanmean(yt_pop,axis=0),axis=0))
                trace_list.append(trace)
                spectrum_list.append([spectrum/np.nanmax(spectrum[:40,:],axis=0)[None,:],\
                                      spectrum_avg/np.nanmax(spectrum_avg[:40])])
                mod_ratio.append(analysis_tools.compute_MR(yt_pop,0,temp_freq))
                pref_phase = analysis_tools.find_preferred_abs_phase(yt_pop,\
                                                                        temp_freq=temp_freq)
                print("pref_phase",np.nanmax(pref_phase),np.nanmin(pref_phase))
                phases.append(pref_phase)
            all_phase.append(phases)
            print("Spat frequency={:.0f}, orientation={:.0f} deg".format(spat_frequency,\
                    orientation*180/np.pi),len(phases))

            ## figure names
            if system_mode=="one_layer":
                if RF_mode in ("gabor","initialize"):
                    filename = "MR_freq{:.0f}_ori{:.0f}_Gabor{}.pdf".format(\
                                spat_frequency,orientation/np.pi*180,suffix)
                elif RF_mode=="load_from_external":
                    filename = "MR_freq{:.0f}_ori{:.0f}_V{}{}.pdf".format(\
                                spat_frequency,orientation/np.pi*180,config_dict["Version"],\
                                suffix)
            elif system_mode=="two_layer":
                if RF_mode in ("gabor","initialize"):
                    filename = "MR_freq{:.0f}_ori{:.0f}_Gabor_2layer{}.pdf".format(\
                                spat_frequency,orientation/np.pi*180,suffix)
                elif RF_mode=="load_from_external":
                    filename = "MR_freq{:.0f}_ori{:.0f}_V{}_2layer{}.pdf".format(\
                                spat_frequency,orientation/np.pi*180,config_dict["Version"],\
                                suffix)

            pp = PdfPages(pdf_path + filename)
            ## figures for comparing stats of L4 and L23
            ncols,nrows = 4,1
            figc = plt.figure(figsize=(6*ncols,5*nrows))
            figc.suptitle("Comparing L4 and L23")
            axes = []
            for k in range(nrows):
                for l in range(ncols):
                    axes.append(figc.add_subplot(nrows,ncols,l+k*ncols+1))
            for k,(yt_pop,label) in enumerate(zip(yt_list,labels)):
                ## figs for L4 E,I and L23 E,I
                ncols,nrows = 8,1
                fig = plt.figure(figsize=(6*ncols,5*nrows))
                fig.suptitle(label)

                idmax = np.argmax(mod_ratio[k])
                ymax,xmax = idmax//N4,idmax%N4
                idmin = np.argmin(mod_ratio[k])
                ymin,xmin = idmin//N4,idmin%N4

                if (k<2 and config_dict["Inp_params"]["input_type"]=="moving_grating_online"):
                    ## raw input trace
                    ax = fig.add_subplot(nrows,ncols,1)
                    ax.set_ylabel("Raw input")
                    ax.set_xlabel("Timesteps")
                    ax.plot(np.nanmean(lgn[0,:,i,j,:],axis=1),"-k",label="avg")
                    ax.plot(np.nanmean(lgn[1,:,i,j,:],axis=1),"--k",label="avg")
                    ax.plot(lgn[0,:,i,j,:].reshape(-1,N4,N4*Nvert)[:,ymax,xmax],"-",c="gray",label="max mod")
                    ax.plot(lgn[1,:,i,j,:].reshape(-1,N4,N4*Nvert)[:,ymax,xmax],"--",c="gray",label="max mod")
                    ax.legend(loc="best")

                if (k<2):
                    ## input trace to L4 (raw input convolved with RF)
                    ax = fig.add_subplot(nrows,ncols,2)
                    ax.set_ylabel("Input to LGN")
                    ax.set_xlabel("Timesteps")
                    ax.plot(np.arange(t1,t2),np.nanmean(It_list[k],axis=(1,2)),"-k",label="avg")
                    ax.plot(np.arange(t1,t2),It_list[k][:,ymax,xmax],"-",c="gray",label="max mod")
                    # ax.plot(np.arange(t1,t2),yt_pop[:,ymin,xmin],"--",c="gray",label="min mod")
                    ax.legend(loc="best")


                ## activity trace
                ax = fig.add_subplot(nrows,ncols,3)
                ax.set_ylabel("Activity")
                ax.set_xlabel("Timesteps")
                ax.plot(np.arange(t1,t2),trace_list[k],"-k",label="avg")
                ax.plot(np.arange(t1,t2),yt_pop[:,ymax,xmax],"-",c="gray",label="max mod")
                # ax.plot(np.arange(t1,t2),yt_pop[:,ymin,xmin],"--",c="gray",label="min mod")
                ax.legend(loc="best")

                ## frequency spectrum
                ax = fig.add_subplot(nrows,ncols,4)
                ax.set_ylabel("Spectrum")
                ax.set_xlabel("Frequency")
                ax.plot(spectrum_list[k][0][:,ymax,xmax],"-",c="gray",label="max mod")
                ax.plot(spectrum_list[k][0][:,ymin,xmin],"--",c="gray",label="min mod")
                ax.plot(spectrum_list[k][1],"-k",label="avg")
                ax.set_xlim(0,10*temp_freq)
                ax.set_ylim(0,1)

                if k<2:
                    ## input from LGN to L4
                    ax = fig.add_subplot(nrows,ncols,5)
                    ax.set_title("FF input last timestep")
                    im=ax.imshow(Ifinal_list[k],interpolation="nearest",cmap="binary")
                    plt.colorbar(im,ax=ax)

                ## max/mean resposne over last moving grating
                ax = fig.add_subplot(nrows,ncols,6)
                ax.set_title("Activity last timestep")
                # im=ax.imshow(yfinal_list[k],interpolation="nearest",cmap="binary")
                im=ax.imshow(np.nanmean(yt_pop[-temporal_duration*kwargs["Nsur"]:,:,:],axis=0),interpolation="nearest",cmap="binary")
                plt.colorbar(im,ax=ax)

                ax = fig.add_subplot(nrows,ncols,7)
                ax.set_title("Modulation ratio")
                im=ax.imshow(mod_ratio[k],interpolation="nearest",cmap="binary",vmin=0,vmax=1)
                plt.colorbar(im,ax=ax)

                ax = fig.add_subplot(nrows,ncols,8)
                ax.set_title("Phase preference")
                im=ax.imshow(all_phase[-1][k],interpolation="nearest",cmap="hsv",vmin=0,vmax=2*np.pi)
                plt.colorbar(im,ax=ax)
                pp.savefig(fig,dpi=300,bbox_inches="tight")
                plt.close(fig)


                axes[0].plot(np.sort(yt_pop[-temporal_duration*kwargs["Nsur"]:,:].mean(0).flatten()),\
                            np.linspace(0,1,yfinal_list[k].size),'-',label=label)
                axes[1].plot(np.sort(mod_ratio[k].flatten()),\
                            np.linspace(0,1,mod_ratio[k].size),'-',label=label)

            axes[0].set_xlabel("Activity")
            axes[0].set_ylabel("Cumulative distribution")
            axes[1].set_xlabel("Modulation ratio")
            axes[1].set_ylabel("Cumulative distribution")
            axes[1].set_xlim(0,1)
            if system_mode=="two_layer":
                axes[2].plot(yt_list[0][-temporal_duration*kwargs["Nsur"]:,:,:].mean(0).flatten(),\
                            yt_list[2][-temporal_duration*kwargs["Nsur"]:,:,:].mean(0).flatten(),'or',alpha=0.4,\
                            rasterized=True,label="E")
                axes[2].plot(yt_list[1][-temporal_duration*kwargs["Nsur"]:,:,:].mean(0).flatten(),\
                            yt_list[3][-temporal_duration*kwargs["Nsur"]:,:,:].mean(0).flatten(),'ob',alpha=0.4,\
                            rasterized=True,label="I")
                axes[2].plot([0,np.max(yt_list[0][-temporal_duration*kwargs["Nsur"]:,:,:].mean(0))],\
                             [0,np.max(yt_list[0][-temporal_duration*kwargs["Nsur"]:,:,:].mean(0))],"--",c="gray")
                axes[3].plot(mod_ratio[0].flatten(),mod_ratio[2].flatten(),'or',alpha=0.4,\
                            rasterized=True,label="E")
                axes[3].plot(mod_ratio[1].flatten(),mod_ratio[3].flatten(),'ob',alpha=0.4,\
                            rasterized=True,label="I")
                axes[3].plot([0,1],[0,1],"--",c="gray")
                axes[2].set_xlabel("Activity L4")
                axes[2].set_ylabel("Activity L23")
                axes[3].set_xlabel("Mod ratio L4")
                axes[3].set_ylabel("Mod ratio L23")
            for ax in axes:
                ax.legend(loc="best")
            pp.savefig(figc,dpi=300,bbox_inches="tight")
            plt.close(figc)


            ## LGN input
            # exc_ff_inp = gamma_ff * (np.dot(Wlgn_to_4[0,:,:],lgn[0,i,j,0,:]) +\
            # 						 np.dot(Wlgn_to_4[1,:,:],lgn[1,i,j,0,:]))
            exc_ff_inp = gamma_ff * (np.dot(Wlgn_to_4[0,:,:],inp[0,:]) +\
                                     np.dot(Wlgn_to_4[1,:,:],inp[1,:]))
            ## visual stimuli
            fig = plt.figure(figsize=(18,5))
            ax = fig.add_subplot(131)
            ax.set_title("Input ON")
            im=ax.imshow(gamma_ff*inp[0,:].reshape(Nlgn,Nlgn),interpolation="nearest",\
                        cmap="binary")
            plt.colorbar(im,ax=ax)
            ax = fig.add_subplot(132)
            ax.set_title("Input OFF")
            im=ax.imshow(gamma_ff*inp[1,:].reshape(Nlgn,Nlgn),interpolation="nearest",\
                        cmap="binary")
            plt.colorbar(im,ax=ax)
            ax = fig.add_subplot(133)
            ax.set_title("Total LGN input")
            im=ax.imshow(exc_ff_inp.reshape(Nlgn,Nlgn),interpolation="nearest",\
                        cmap="binary")
            plt.colorbar(im,ax=ax)
            pp.savefig(fig,dpi=300,bbox_inches="tight")
            plt.close(fig)

            ## VARIABILITY IN RESPONSE PATTERN DURING ONE GRATING OSCILLATION
            oscillation_duration = (last_timestep+1)/temp_freq
            timesteps = np.array([last_timestep-oscillation_duration,\
                                 last_timestep-3/4.*oscillation_duration,\
                                 last_timestep-1./2*oscillation_duration,\
                                 last_timestep-1./4*oscillation_duration]).astype(int)
            ncol,nrow = len(timesteps),1
            fig = plt.figure(figsize=(6*ncol,5*nrow))
            for k in range(ncol):
                ax = fig.add_subplot(nrow,ncol,k+1)
                ax.set_title("t={}".format(timesteps[k]))
                im=ax.imshow(yt[timesteps[k],:L4_size].reshape(N4,N4),interpolation="nearest",\
                            cmap="binary")
                plt.colorbar(im,ax=ax)
            pp.savefig(fig,dpi=300,bbox_inches="tight")
            plt.close(fig)

            pp.close()
            print("plot done",orientation,spat_frequency)



        ## compute orientation tuning and selectivity
        all_phase = np.array(all_phase)
        act_last_timestep = np.array(act_last_timestep)
        num_oris = len(kwargs["orientations"])
        if system_mode=="one_layer":
            labels = ["L4,E","L4,I"]
            L4_size = N4*N4*Nvert
            final_act_list = [act_last_timestep[:,:L4_size].reshape(num_oris,N4,N4*Nvert),\
                                act_last_timestep[:,L4_size:].reshape(num_oris,N4,N4*Nvert)]
            all_phase_list = [all_phase[:,0,:,:],all_phase[:,1,:,:]]
        elif system_mode=="two_layer":
            labels = ["L4,E","L4,I","L23,E","L23,I"]
            L4_size = N4*N4*Nvert
            L23_size = N23**2
            final_act_list = [\
                            act_last_timestep[:,:L4_size].reshape(num_oris,N4,N4*Nvert),\
                            act_last_timestep[:,L4_size:L4_size*2].reshape(num_oris,N4,N4*Nvert),\
                            act_last_timestep[:,L4_size*2:L4_size*2+L23_size].reshape(num_oris,N23,N23),\
                            act_last_timestep[:,L4_size*2+L23_size:].reshape(num_oris,N23,N23)
                            ]
            all_phase_list = [all_phase[:,0,:,:],all_phase[:,1,:,:],\
                              all_phase[:,2,:,:],all_phase[:,3,:,:]]

        ## file names
        if system_mode=="one_layer":
            if RF_mode=="gabor":
                filename = "ori_phase_freq{:.0f}_Gabor{}.pdf".format(spat_frequency,suffix)
            elif RF_mode=="initialize":
                filename = "ori_phase_freq{:.0f}_init{}.pdf".format(spat_frequency,suffix)
            elif RF_mode=="load_from_external":
                filename = "ori_phase_freq{:.0f}_V{}{}.pdf".format(spat_frequency,\
                                                                 config_dict["Version"],suffix)
        elif system_mode=="two_layer":
            if RF_mode=="gabor":
                filename = "ori_phase_freq{:.0f}_Gabor_2layer{}.pdf".format(spat_frequency,suffix)
            elif RF_mode=="initialize":
                filename = "ori_phase_freq{:.0f}_init_2layer{}.pdf".format(spat_frequency,suffix)
            elif RF_mode=="load_from_external":
                filename = "ori_phase_freq{:.0f}_V{}_2layer{}.pdf".format(\
                            spat_frequency,config_dict["Version"],suffix)
        pp = PdfPages(pdf_path + filename)

        sf = Wlgn_to_4[0,...] - Wlgn_to_4[1,...]
        rA = int(config_dict["Wlgn_to4_params"]["r_A_on"] * N4)
        DA = 2*rA + 5
        RF,_,_,_ = analysis_tools.get_RF_form(sf,N4,Nlgn,DA,calc_PF=False,Nvert=Nvert,mode="o")
        ext = np.nanmax(np.abs(RF))
        fig = plt.figure(figsize=(6,5))
        fig.suptitle("RFs used")
        ax = fig.add_subplot(111)
        im=ax.imshow(RF,interpolation="nearest",cmap="RdBu_r",vmin=-ext,vmax=ext)
        plt.colorbar(im,ax=ax)
        pp.savefig(dpi=300,bbox_inches="tight")
        plt.close(fig)

        ncols,nrows = 4,1
        orientations_binning = np.concatenate([kwargs["orientations"],np.array([np.pi])])
        # orientations_binning -= np.pi/2./num_oris
        for j,(final_act,jall_phase) in enumerate(zip(final_act_list,all_phase_list)):
            opm = analysis_tools.compute_orientation_tuning_from_activity(final_act,\
                                                            kwargs["orientations"]*2,norm=True)
            sel = np.abs(opm)
            ori = np.angle(opm)*0.5
            ori = ori - (np.sign(ori)-1)*0.5*np.pi

            ori_bin = np.searchsorted(orientations_binning,ori,side='left')
            ori_bin[ori_bin==num_oris] = 0
            ori_bin[ori_bin==num_oris+1] = 0
            pref_phase_full = np.zeros_like(ori,dtype=float)
            for y in range(ori.shape[0]):
                for x in range(ori.shape[1]):
                    pref_phase_full[y,x] = jall_phase[ori_bin[y,x],y,x]

            fig = plt.figure(figsize=(6*ncols,nrows*5))
            ax = fig.add_subplot(nrows,ncols,1)
            ax.set_title("Orientation preference map {}".format(labels[j]))
            # im=ax.imshow(misc.plot_complex_map(opm),interpolation="nearest")
            im=ax.imshow(ori,cmap="hsv",interpolation="nearest",vmin=0,vmax=np.pi)
            plt.colorbar(im,ax=ax,orientation="horizontal")

            ax = fig.add_subplot(nrows,ncols,2)
            opm_fft = np.abs(np.fft.fftshift(np.fft.fft2(opm - np.nanmean(opm))))
            ax.set_title("Spectrum Orientation preference map {}".format(labels[j]))
            im=ax.imshow(opm_fft,cmap="binary",interpolation="nearest")
            plt.colorbar(im,ax=ax,orientation="horizontal")

            ax = fig.add_subplot(nrows,ncols,3)
            ax.set_title("Preferred absolute phase {}".format(labels[j]))
            im=ax.imshow(pref_phase_full,interpolation="nearest",cmap="hsv",vmin=0,vmax=2*np.pi)
            plt.colorbar(im,ax=ax,orientation="horizontal")

            ax = fig.add_subplot(nrows,ncols,4)
            opm_fft = np.abs(np.fft.fftshift(np.fft.fft2(pref_phase_full - np.nanmean(pref_phase_full))))
            ax.set_title("Spectrum Preferred absolute phase {}".format(labels[j]))
            im=ax.imshow(opm_fft,cmap="binary",interpolation="nearest")
            plt.colorbar(im,ax=ax,orientation="horizontal")

            pp.savefig(dpi=300,bbox_inches="tight")
            plt.close(fig)

        pp.close()
    # plt.show()

    print("All plots saved!")
    print("Done.")