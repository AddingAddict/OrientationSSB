import numpy as np

def generate_topology_map(patchsize_y,patchsize_x, ring_size, ring_thickness,\
							Nsur=1, rng=None, return_complex=False, symmetrise=False):

	if rng is None:
		rng = np.random.RandomState()

	# Create a centered ring
	ring_image = generate_ring(patchsize_y,patchsize_x, ring_size, ring_thickness)

	# Move the ring's center to (0, 0), s.t. the ring is split into four parts.
	ring_image = np.fft.fftshift(ring_image)

	if Nsur>1:
		ring_iffts = []
		for isur in range(Nsur):
			random_angles = rng.rand(patchsize_y,patchsize_x) * np.pi  # Array of uniformly random angles
			if symmetrise:
				random_angles_sym = np.angle(np.fft.fft2(rng.randn(patchsize_y,patchsize_x)))
				random_angles = random_angles_sym
			
			random_matrix = np.exp(-1j * random_angles)         # Actual complex-number matrix

			# Multiply ring=(abs values) with random phases
			randomized_ring = ring_image * random_matrix

			# Create topology map by IFFTing the randomized ring
			ring_ifft = np.fft.ifft2(randomized_ring)*np.sqrt(2*np.pi*patchsize_y*patchsize_x)

			if return_complex:
				ring_iffts.append(ring_ifft)
		
		ring_iffts = np.array(ring_iffts)
		return ring_iffts, ring_image
	else:
		# Create a matrix of random complex variable
		random_angles = rng.rand(patchsize_y,patchsize_x) * np.pi  # Array of uniformly random angles
		if symmetrise:
			random_angles_sym = np.angle(np.fft.fft2(rng.randn(patchsize_y,patchsize_x)))
			random_angles = random_angles_sym
		random_matrix = np.exp(-1j * random_angles)         # Actual complex-number matrix

		# Multiply ring=(abs values) with random phases
		randomized_ring = ring_image * random_matrix

		# Create topology map by IFFTing the randomized ring
		ring_ifft = np.fft.ifft2(randomized_ring)*np.sqrt(2*np.pi*patchsize_y*patchsize_x)

		if return_complex:
			return ring_ifft

		#topology_map = (np.angle(ring_ifft) + np.pi) / 2
		topology_map = ( (np.angle(ring_ifft)) % (2 * np.pi) ) / 2

		return topology_map,np.abs(ring_ifft)


def generate_ring(window_size_y, window_size_x, mean, stds):
	mean = np.float(mean)
	stds = np.float(stds)
	coords_x = np.tile((np.arange(window_size_x) - ((window_size_x) / 2))[None, :], (window_size_y, 1)).astype(float)
	coords_y = np.tile((np.arange(window_size_y) - ((window_size_y) / 2))[:, None], (1, window_size_x)).astype(float)

	radii = ((coords_x ** 2) + (coords_y ** 2)) ** 0.5
	# Check that np.fft.fftshift(radii) has a 0 as its first (leftmost and uppermost) element.
	height = 1.#/2./np.pi/stds**2#1
	ring_image = height * np.exp(-(((radii - mean)**2 / (2 * stds**2))))
	return ring_image
    
def gauss(x,A,sigma,mu):
	return A/sigma*np.exp(-(x-mu)**2/2./sigma**2)



if __name__=="__main__":
	import matplotlib
	import matplotlib.pyplot as plt
	from bettina.tools import dimension_abbott
	
	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'
	
	window_size = 20
	mean = 2.5#10
	stds = .4#*2.8
	patchsize = window_size
	ring_size = mean
	ring_thickness = stds
	VERSION = 2
	

	#cmap = generate_topology_map(window_size,window_size, ring_size, ring_thickness, rng=np.random.RandomState(4), return_complex=True)
	cmap = generate_topology_map(window_size,window_size, ring_size, ring_thickness,\
			 rng=np.random.RandomState(5565), return_complex=True,symmetrise=True)
	print(np.mean(np.real(cmap)),np.std(np.real(cmap)),np.std(np.abs(cmap)),mean*stds/np.sqrt(2*np.pi))
	#print(np.nanmax(sel),np.nanmin(sel))
	#exit()
	
	from bettina.tools import average_over_360degrees
	from scipy.optimize import curve_fit
	
	spont = np.stack([np.real(cmap),np.imag(cmap)])
	nframes,h,w = spont.shape
	print('spont',spont.shape)
	spont[np.logical_not(np.isfinite(spont))] = 0
	spectrum = np.fft.fftshift(np.abs(np.fft.fft2(spont-np.nanmean(spont,axis=(1,2))[:,None,None],axes=(1,2))),axes=(1,2))
	spectrum1d,distance = average_over_360degrees.average_angles(spectrum,weights=None,interpolate=False)
	spectrum1d = np.nanmean(spectrum1d,axis=0)
	fdistance = distance/h/(mean/window_size)#1./(1.*distance/h/0.1/1000.)#

	mean = np.nansum(spectrum1d[distance<20]*fdistance[distance<20])/np.nansum(spectrum1d[distance<20])
	sd = np.sqrt(np.nansum(spectrum1d[distance<20]*fdistance[distance<20]**2)/np.nansum(spectrum1d[distance<20]) - mean**2)
	
	fit_spect = spectrum1d[distance<20]
	fit_dist = fdistance[distance<20]
	ampl = np.max(spectrum1d[distance<20])
	#popt,pcov = curve_fit(gauss,fit_dist,fit_spect,p0=[ampl,sd,mean])
	
	# Plot spectrum
	print('fdistance',fdistance.shape,distance.shape)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(fdistance,spectrum1d,'-',label='sd={:.5f}'.format(sd))
	#ax.plot(fdistance,gauss(fdistance,ampl,sd,popt[2])*sd,'-r')
	#ax.plot(fdistance,gauss(fdistance,*popt),'-m',label='sd_f={:.5f}'.format(popt[1]))
	plt.legend()				
	#fig.savefig('/home/hein/data_mpif2/image/tmp/spectral_width/spectrum1d_{}_{}.pdf'.format(ferret,dataset.date),dpi=100)
	#plt.close(fig)
	
	#spectrum = generate_ring(window_size,window_size, mean, stds)
	print('DIM',np.sum(spectrum)**2/np.sum(spectrum**2))
	plt.figure()
	plt.subplot(121)
	plt.imshow(np.real(cmap)*500+1000,interpolation='nearest',cmap='binary')
	plt.colorbar()
	plt.subplot(122)
	plt.imshow(np.angle(np.fft.fftshift(np.fft.fft2(cmap))),interpolation='nearest',cmap='hsv')
	#plt.imshow(np.fft.fftshift(spect),interpolation='nearest',cmap='hsv')
	plt.colorbar()
	plt.show()
	
	
	
