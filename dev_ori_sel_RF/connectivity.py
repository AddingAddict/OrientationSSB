import numpy as np
from scipy import linalg
from scipy.signal import convolve2d
from copy import copy

from dev_ori_sel_RF.tools import gen_gaussian_random_field as ggrf


def distance(delta):
	''' assume periodic boundary conditions'''
	signs = np.sign(delta)
	return np.where(np.abs(delta) > 0.5, -signs*(1 - np.abs(delta)), delta)


def linear_arbor_falloff(d,radius,inner_radius):
	outer_radius = np.nanmin(d[d>radius])
	arbor = np.zeros_like(d,dtype=float)
	arbor[d<=inner_radius] = 1.
	tapering_region = np.logical_and(d>inner_radius,d<outer_radius)
	arbor[tapering_region] = d[tapering_region]/(inner_radius-outer_radius) +\
							 outer_radius/(outer_radius-inner_radius)
	return arbor


def gaussian(x,y,s):
	return np.exp(-(x**2+y**2)/2./s**2)#1./2.*np.pi/s**2*

def exponential(x,y,s):
	return np.exp(-np.sqrt(x**2+y**2)/s)


def gaussian_filter(ar,sigma):
	"""
	filters array ar with gaussian of width sigma

	"""
	N = ar.shape[-1]
	grid = np.linspace(-0.5,0.5,N,endpoint=False)
	x,y = np.meshgrid(grid,grid)
	g = gaussian(x,y,sigma)

	if ar.ndim==3:
		filtererd_ar = []
		for iar in ar:
			filtererd_ar.append(convolve2d(g,iar,mode="same",boundary="wrap"))
		return np.array(filtererd_ar)
	else:
		filtered_ar = convolve2d(g,ar,mode="same",boundary="wrap")
		return filtered_ar


class Connectivity:
	def __init__(self, from_size, to_size, random_seed, Nvert=1, full_output=False):
		self.from_size = from_size
		self.to_size = to_size
		if isinstance(Nvert,int):
			Nvert = (Nvert,Nvert)
		self.Nvert = Nvert
		self.rng = np.random.RandomState(random_seed*90)
		# self.rng = np.random.default_rng(random_seed*90)
		self.full_output = full_output
		self.full_output_dict = {}
		
	
	def gen_ecp(self, x, y, conn_params):
		"""
		generate ecp - solution of wolf model
		kc 		:	critical wave number
		n 		:	number of modes
		"""
		if "kc" in conn_params.keys():
			kc = conn_params["kc"]
		else:
			kc = 3.## number of wavelengths per layer
		if "n" in conn_params.keys():
			n = conn_params["n"]
		else:
			n = 30
		A = (self.rng.randn(n)*0.2+0.8)[:,None,None]#1.0

		## Long-range interactions
		j = np.arange(n)
		kj = kc*2*np.pi*np.stack([np.cos(j*np.pi/n), np.sin(j*np.pi/n)])
		lj = self.rng.choice([-1.,1.],size=n)
		lk = (lj[None,:]*kj)[:,:,None,None]
		phi = self.rng.uniform(low=0,high=2*np.pi,size=n)[:,None,None]
		ecp = np.sum(A*np.exp(1j*lk[0,...]*x[None,...] + 1j*lk[1,:,...]*y[None,...] + phi),axis=0)
		
		sigma = 0.3/kc/2/np.pi*n

		return ecp, sigma

	def create_matrix(self, conn_params, profile, **kwargs):
		grid = np.linspace(0,1,self.from_size[0],endpoint=False)
		# grid = np.repeat(grid,self.Nvert[0])
		xfrom,yfrom = np.meshgrid(grid,grid)
		xfrom = np.repeat(xfrom,self.Nvert[0]).reshape(self.from_size[0],self.from_size[1]*self.Nvert[0])
		yfrom = np.repeat(yfrom,self.Nvert[0]).reshape(self.from_size[0],self.from_size[1]*self.Nvert[0])
		grid = np.linspace(0,1,self.to_size[0],endpoint=False)
		# grid = np.repeat(grid,self.Nvert[1])
		xto,yto = np.meshgrid(grid,grid)
		xto = np.repeat(xto,self.Nvert[1]).reshape(self.to_size[0],self.to_size[1]*self.Nvert[1])
		yto = np.repeat(yto,self.Nvert[1]).reshape(self.to_size[0],self.to_size[1]*self.Nvert[1])
		xdelta = distance(xto[:,:,None,None]-xfrom[None,None,:,:])
		ydelta = distance(yto[:,:,None,None]-yfrom[None,None,:,:])

		arbor_params = {}
		if conn_params.get("ret_scatter",False):
			arbor_params = {"ret_scatter" : conn_params["ret_scatter"]}

		if "r_A" in kwargs:
			if isinstance(kwargs["r_A"],float):
				profile_A = kwargs.get("profile_A", "heaviside")
				arbor = self.create_arbor(radius=kwargs["r_A"],profile=profile_A,\
						arbor_params=arbor_params)
				arbor = arbor.reshape(xdelta.shape)
				print("CHECK radius",kwargs["r_A"])
			else:
				arbor = None
		else:
			arbor = None


		## Heterogeneously varying anisotropic Connectivity (incoming conn are elliptical)
		if "heterogeneity_type" in conn_params.keys():
			if conn_params["heterogeneity_type"] is not None:
				## draw spatially uncorrelated random fields
				ecc_field = self.rng.randn(*xto.shape)
				size_field_x = self.rng.randn(*xto.shape)
				z_field = self.rng.randn(*xto.shape) + 1j*self.rng.randn(*xto.shape)

				## spatially filter rnd fields with gaussian
				if conn_params["heterogeneity_type"]=="smooth":
					sigma = conn_params["smoothness_width"]
					ecc_field = gaussian_filter(ecc_field,sigma)
					size_field_x = gaussian_filter(size_field_x,sigma)
					z_field = gaussian_filter(z_field,sigma)

				## eccentricity distribution
				ecc_field = conn_params["mean_eccentricity"] +\
							ecc_field/np.std(ecc_field)*conn_params["SD_eccentricity"]
				ecc_field = np.clip(ecc_field,0.0,0.95)
				
				## size distribution
				size_field_x = (1 + size_field_x/np.std(size_field_x)*conn_params["SD_size"])
				size_field_y = size_field_x*np.sqrt(1 - ecc_field**2)
				
				## orientations of elongated connections				
				orientation_field = np.angle(z_field)*0.5
				cos_theta = np.cos(orientation_field)[None,None,:,:]
				sin_theta = np.sin(orientation_field)[None,None,:,:]

				xdelta2 = (xdelta*cos_theta - ydelta*sin_theta) * size_field_x[None,None,:,:]
				ydelta = (xdelta*sin_theta + ydelta*cos_theta) * size_field_y[None,None,:,:]
				xdelta = xdelta2

		if arbor_params.get("ret_scatter",False):
			scatter_ampl = 0.25
			if isinstance(arbor_params["ret_scatter"],float):
				scatter_ampl = arbor_params["ret_scatter"]
			rng = np.random.default_rng(2001)
			x_scat = scatter_ampl*rng.standard_normal(self.to_size)
			y_scat = scatter_ampl*rng.standard_normal(self.to_size)
			xdelta = distance(xto[:,:,None,None]-xfrom[None,None,:,:]-x_scat[:,:,None,None])
			ydelta = distance(yto[:,:,None,None]-yfrom[None,None,:,:]-y_scat[:,:,None,None])


		if profile=="linear":
			pairwise_distance = np.sqrt(xdelta**2 + ydelta**2)
			conn_matrix = pairwise_distance
		
		elif profile in ("Gaussian","Gaussian_broadOFF"):
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			noise_str = conn_params["noise"]

			# noise_str = 0
			# print("")
			# print("UNCOMMENT L 176 in connectivity.py")
			# print("")
			
			if noise_str>0:
				noise_field = self.rng.uniform(1-noise_str,1,xdelta.size).reshape(xdelta.shape)
			else:
				noise_field = 1.
			disc_gaussian = gaussian(xdelta,ydelta,sigma) #/ np.prod(self.from_size)
			disc_gaussian *= noise_field
			## norm_factor is uniform
			# norm_factor = np.sum(disc_gaussian,axis=(0,1))[None,None,:,:]
			
			if arbor is not None:
				disc_gaussian[np.logical_not(arbor)] = 0.
				print("arbor",arbor.shape)
			norm_factor = np.sum(disc_gaussian,axis=(2,3))[:,:,None,None]
			conn_matrix = ampl * disc_gaussian / norm_factor
			
		elif profile in ("Exponential",):
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			noise_str = conn_params["noise"]
			if noise_str>0:
				noise_field = self.rng.uniform(1-noise_str,1,xdelta.size).reshape(xdelta.shape)
			else:
				noise_field = 1.
			disc_exponential = exponential(xdelta,ydelta,sigma)
			disc_exponential *= noise_field
			
			if arbor is not None:
				disc_exponential[np.logical_not(arbor)] = 0.
			norm_factor = np.sum(disc_exponential,axis=(2,3))[:,:,None,None]
			conn_matrix = ampl * disc_exponential / norm_factor

		elif profile=="Gaussian_inv":
			disc_gaussian = self.create_matrix(conn_params, profile="Gaussian")
			disc_gaussian /= np.nanmax(disc_gaussian)
			conn_matrix = disc_gaussian -\
			 np.diag(np.ones(np.prod(self.from_size)*self.Nvert[0])).reshape(disc_gaussian)

			if arbor is not None:
				conn_matrix[np.logical_not(arbor)] = 0.
			conn_matrix =  conn_matrix / np.sum(conn_matrix,axis=0)[None,:]
			# print("conn_matrix",conn_matrix.size,np.sum(np.isfinite(conn_matrix)),\
			# 	np.nanmin(conn_matrix),np.nanmax(conn_matrix))

		elif profile=="Gaussian_prob":
			"""construct recurrent connectivity such that connections have given
			density. weights are determined by spatial distance, prob of existing
			connection prop to distance"""
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			noise_str = conn_params["noise"]
			if noise_str>0:
				noise_field = self.rng.uniform(1-noise_str,1,xdelta.size).reshape(xdelta.shape)
			else:
				noise_field = 1.
			no_conn_number = int(np.prod(self.to_size) * (1-conn_params["density"]))
			disc_gaussian = gaussian(xdelta,ydelta,sigma)
			rnd_field = self.rng.uniform(0,1,xdelta.size).reshape(xdelta.shape)
			prob_existing_conn = (disc_gaussian - rnd_field).reshape(\
									np.prod(self.to_size)*self.Nvert[1],\
									np.prod(self.from_size)*self.Nvert[0])
			thresholds = np.sort(prob_existing_conn,axis=(0))[no_conn_number,:]
			conn_matrix = disc_gaussian.reshape(np.prod(self.to_size)*self.Nvert[1],\
												np.prod(self.from_size)*self.Nvert[0])
			conn_matrix[prob_existing_conn<thresholds[None,:]] = 0.0

			conn_matrix = conn_matrix.reshape(xdelta.shape)
			conn_matrix *= noise_field
			if arbor is not None:
				conn_matrix[np.logical_not(arbor)] = 0.
			norm_factor = np.sum(conn_matrix,axis=(2,3))[None,None,:,:]
			conn_matrix = ampl * conn_matrix / norm_factor

		elif profile=="Gaussian_sparse":
			## connectivity not really clustered
			"""construct recurrent connectivity such that connections have given
			density. weight is determined by spatial distance, randomly chosen 
			nonzero weights"""
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			noise_str = conn_params["noise"]
			if noise_str>0:
				noise_field = self.rng.uniform(1-noise_str,1,xdelta.size).reshape(xdelta.shape)
			else:
				noise_field = 1.
			total_conn_number = int(np.prod(self.to_size) * np.prod(self.from_size) *\
								 np.prod(self.Nvert) * conn_params["density"])
			disc_gaussian = gaussian(xdelta,ydelta,sigma)
			conn_matrix = disc_gaussian.reshape(np.prod(self.to_size)*self.Nvert[1],\
							np.prod(self.from_size)*self.Nvert[0])

			## set random seed for different types:
			if conn_params["type"]=="EE":
				rng = np.random.RandomState(12345*90+7)
			elif conn_params["type"]=="IE":
				rng = np.random.RandomState(12345*90+77)
			elif conn_params["type"]=="EI":
				rng = np.random.RandomState(12345*90+177)
			elif conn_params["type"]=="II":
				rng = np.random.RandomState(12345*90+1)

			## set average density of connectiosn
			rnd_field = rng.uniform(0,1,np.prod(self.from_size)*np.prod(self.to_size)\
									*np.prod(self.Nvert))
			rnd_field = rnd_field.reshape(np.prod(self.to_size)*self.Nvert[1],\
											np.prod(self.from_size)*self.Nvert[0])
			conn_matrix[rnd_field>conn_params["density"]] = 0.0

			conn_matrix = conn_matrix.reshape(xdelta.shape)
			conn_matrix *= noise_field
			# norm_factor = np.sum(conn_matrix,axis=(0,1))[None,None,:,:]
			if arbor is not None:
				conn_matrix[np.logical_not(arbor)] = 0.
			norm_factor = np.sum(conn_matrix,axis=(2,3))[:,:,None,None]
			conn_matrix = ampl * conn_matrix / norm_factor
			# print("norm_factor",np.sum(norm_factor==0),conn_matrix.shape)

		elif profile=="Gaussian_prob_density":
			## connectivity not really clustered
			"""construct recurrent connectivity such that connections have given
			density. realization is determined by differnece in their phase (rnd field),
			weight is determined by spatial distance"""
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			noise_str = conn_params["noise"]
			if noise_str>0:
				noise_field = self.rng.uniform(1-noise_str,1,xdelta.size).reshape(xdelta.shape)
			else:
				noise_field = 1.
			total_conn_number = int(np.prod(self.to_size) * np.prod(self.from_size)\
								 * conn_params["density"] * np.prod(self.Nvert)) 
			disc_gaussian = gaussian(xdelta,ydelta,sigma)
			conn_matrix = disc_gaussian.reshape(np.prod(self.to_size)*self.Nvert[1],\
												np.prod(self.from_size)*self.Nvert[0])

			phase_field = self.rng.uniform(0,np.pi,np.prod(self.from_size)*self.Nvert[0])
			diff_phase = np.abs(phase_field[:,None] - phase_field[None,:]) # from 0 to pi

			## overal threhsold to obtain over density of conns
			threshold = np.sort(diff_phase.flatten())[total_conn_number-1]

			if conn_params["type"] in ("EE","EI"):
				above_thr = diff_phase>threshold
				conn_matrix[above_thr] = 0.0
			else:
				below_thr = diff_phase<threshold
				conn_matrix[below_thr] = 0.0

			conn_matrix = conn_matrix.reshape(xdelta.shape)
			conn_matrix *= noise_field
			# norm_factor = np.sum(conn_matrix,axis=(0,1))[None,None,:,:]
			if arbor is not None:
				conn_matrix[np.logical_not(arbor)] = 0.
			norm_factor = np.sum(conn_matrix,axis=(2,3))[:,:,None,None]
			conn_matrix = ampl * conn_matrix / norm_factor
			# print("norm_factor",np.sum(norm_factor==0),conn_matrix.shape)

		elif profile=="Gaussian_prob_cluster":
			## connectivity clustered
			"""construct recurrent connectivity such that connections has given number of
			clusters. realization is determined by differnece in their phase, weight
			is determined by spatial distance"""
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			noise_str = conn_params["noise"]
			if noise_str>0:
				noise_field = self.rng.uniform(1-noise_str,1,xdelta.size).reshape(xdelta.shape)
			else:
				noise_field = 1.
			disc_gaussian = gaussian(xdelta,ydelta,sigma)
			conn_matrix = disc_gaussian.reshape(np.prod(self.to_size)*self.Nvert[1],\
												np.prod(self.from_size)*self.Nvert[0])

			ncluster = conn_params["ncluster"]
			phase_field = self.rng.uniform(0,np.pi,np.prod(self.from_size)*self.Nvert[0])
			phase_field_binned = np.searchsorted(np.linspace(0,np.pi,ncluster,endpoint=False),\
												 phase_field,side="right")
			diff_phase = np.abs(phase_field_binned[:,None] - phase_field_binned[None,:]) # from 0 to pi
			## overal threhsold to obtain over density of conns
			threshold = 0.5

			if conn_params["type"] in ("EE","EI"):
				above_thr = diff_phase>threshold
				# print("above_thr",np.sum(above_thr),above_thr.size,np.sum(above_thr,axis=1)[:5])
				conn_matrix[above_thr] = 0.0
			else:
				below_thr = diff_phase<threshold
				# print("below_thr",np.sum(below_thr),below_thr.size,np.sum(below_thr,axis=1)[:5])
				conn_matrix[below_thr] = 0.0

			conn_matrix = conn_matrix.reshape(xdelta.shape)
			conn_matrix *= noise_field
			# norm_factor = np.sum(conn_matrix,axis=(0,1))[None,None,:,:]
			if arbor is not None:
				conn_matrix[np.logical_not(arbor)] = 0.
			norm_factor = np.sum(conn_matrix,axis=(2,3))[:,:,None,None]
			norm_factor[norm_factor==0] = 1.
			conn_matrix = ampl * conn_matrix / norm_factor
			# print("norm_factor",np.sum(norm_factor==0),conn_matrix.shape)
			self.full_output_dict.update({"phase_field_binned" : phase_field_binned})

		elif profile=="Mexican-hat":
			sigma1 = conn_params["sigma1"]
			sigma2 = conn_params["sigma2"]
			ampl1 = conn_params["ampl1"]
			noise_str = conn_params["noise"]
			ampl2 = conn_params["ampl2"]
			if noise_str>0:
				noise_field1 = self.rng.uniform(1-noise_str,1,xdelta.size).reshape(xdelta.shape)
				noise_field2 = self.rng.uniform(1-noise_str,1,xdelta.size).reshape(xdelta.shape)
			else:
				noise_field1 = 1.
				noise_field2 = 1.
			disc_gaussian1 = gaussian(xdelta,ydelta,sigma1) / np.prod(self.from_size) / self.Nvert[0]
			if arbor is not None:
				disc_gaussian1[np.logical_not(arbor)] = 0.
			disc_gaussian1 /= np.sum(disc_gaussian1,axis=(0,1))[None,None,:,:]
			disc_gaussian2 = gaussian(xdelta,ydelta,sigma2) / np.prod(self.from_size) / self.Nvert[0]
			if arbor is not None:
				disc_gaussian2[np.logical_not(arbor)] = 0.
			disc_gaussian2 /= np.sum(disc_gaussian2,axis=(0,1))[None,None,:,:]
			conn_matrix = ampl1 * noise_field1 * disc_gaussian1 -\
						  ampl2 * noise_field2 * disc_gaussian2
			
		elif profile=="zmodel":
			assert self.from_size==self.to_size, "choose same fromsize and tosize"
			assert self.Nvert==(1,1), "currently implemented for Nvert=1"

			## Short-range Mexican-hat like interactions
			# mh = self.create_matrix(conn_params, profile="Mexican-hat")
			# mh = mh.reshape(self.to_size[0],self.to_size[1]*self.Nvert[1],\
			# 				self.from_size[0],self.from_size[1]*self.Nvert[0])
			## short-range Gaussian interaction
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			disc_gaussian = gaussian(xdelta,ydelta,sigma)
			norm_factor = np.sum(disc_gaussian,axis=(2,3))[:,:,None,None]
			disc_gaussian = ampl * disc_gaussian / norm_factor

			## Long-range interactions
			dist_dep_gauss = gaussian(xdelta,ydelta,0.5) / np.prod(self.from_size) / self.Nvert[0]
			ecp, sigma_ecp = self.gen_ecp(xfrom, yfrom, conn_params)
			sigmaz = 0.5*np.nanmean(np.abs(ecp))
			deltaz = ecp[:,:,None,None] - ecp[None,None,:,:]
			ori_dep_gauss = gaussian(np.abs(deltaz),0,sigmaz) / np.prod(self.from_size) / self.Nvert[0]
			zconn = dist_dep_gauss * ori_dep_gauss
			zconn[np.sqrt(xdelta**2+ydelta**2)<(sigma*2.)] = 0.
			zconn /= np.sum(zconn,axis=(2,3))[:,:,None,None]

			conn_matrix = (disc_gaussian + 5*zconn)
			if arbor is not None:
				conn_matrix[np.logical_not(arbor)] = 0.
			# conn_matrix /= np.sum((conn_matrix),axis=(2,3))[:,:,None,None]
			# if arbor is not None:
			# 	conn_matrix[np.logical_not(arbor)] = 0.

		elif profile in ("random_delta","initialize","initialize2"):
			s_noise = conn_params["s_noise"]
			if profile=="initialize2":
				disc_gaussian = gaussian(xdelta,ydelta,0.2)
				noise_field = self.rng.uniform(1-s_noise,1+s_noise,xdelta.size)
				conn_matrix = disc_gaussian * noise_field.reshape(disc_gaussian.shape)
			else:
				conn_matrix = self.rng.uniform(1-s_noise,1+s_noise,xdelta.size)
				conn_matrix = conn_matrix.reshape(xdelta.shape)
			
			if arbor is not None:
				conn_matrix[np.logical_not(arbor)] = 0.
			if profile in ("initialize2","initialize"):
				norm_factor = np.sum(conn_matrix,axis=(2,3))[:,:,None,None]
				conn_matrix /= norm_factor
			
			conn_matrix *= conn_params["ampl"]

		elif profile=="Gabor":
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			theta = conn_params["theta"]
			psi = conn_params["psi"]
			disc_gaussian = ampl * gaussian(xdelta,ydelta,sigma)
			Lambda = 2*np.pi/conn_params["freq"]
			if "spatial_freq_het" in conn_params.keys():
				if conn_params["spatial_freq_het"] is not None:
					N = self.to_size[0]
					radius = 2.5
					width = 0.4
					cmap = ggrf.generate_topology_map(N, N, radius, width,rng=self.rng, \
			 										return_complex=True, symmetrise=True)
					spatial_freq_het = np.real(cmap)
					spatial_freq_het *= conn_params["spatial_freq_het"]/np.nanstd(spatial_freq_het)
					Lambda = Lambda + spatial_freq_het
				else:
					N = self.to_size[0]
					Lambda = np.ones((N,N))*Lambda
			else:
				N = self.to_size[0]
				Lambda = np.ones((N,N))*Lambda

			gamma = 1.
			sigma_x = sigma
			sigma_y = float(sigma) / gamma

			# Bounding box
			nstds = 3  # Number of standard deviation sigma
			xmax = np.max(np.concatenate([np.abs(nstds * sigma_x * np.cos(theta)),\
									   np.abs(nstds * sigma_y * np.sin(theta))]))
			xmax = np.ceil(max(1, xmax))
			ymax = np.max(np.concatenate([np.abs(nstds * sigma_x * np.sin(theta)),\
					 				   np.abs(nstds * sigma_y * np.cos(theta))]))
			ymax = np.ceil(max(1, ymax))
			x = xdelta * 2 * xmax
			y = ydelta * 2 * ymax

			# Rotation
			x_theta = x * np.cos(theta[:,:,None,None]) - y * np.sin(theta[:,:,None,None])
			y_theta = x * np.sin(theta[:,:,None,None]) + y * np.cos(theta[:,:,None,None])

			if isinstance(psi,np.ndarray):
				conn_matrix =\
					 np.exp(-.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) *\
				 	 np.cos(2 * np.pi / Lambda[:,:,None,None] * x_theta + psi[:,:,None,None])
			else:
				conn_matrix =\
					 np.exp(-.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) *\
				 	 np.cos(2 * np.pi / Lambda[:,:,None,None] * x_theta + psi)

			## apply heterogeneous amplitudes to RFs, some cortical unit receive stronger input
			if "ampl_het" in conn_params.keys():
				if conn_params["ampl_het"] is not None:
					amplitude_het = conn_params["ampl_het"]
					
					N = self.to_size[0]
					radius = 2.5
					width = 0.4
					cmap = ggrf.generate_topology_map(N, N, radius, width,rng=self.rng, \
			 										return_complex=True, symmetrise=True)
					het_amplitudes = np.real(cmap)
					het_amplitudes *= amplitude_het/np.nanstd(het_amplitudes)
					conn_matrix *= het_amplitudes[:,:,None,None]

			if arbor is not None:
				conn_matrix[np.logical_not(arbor)] = 0.
			# kx,ky = conn_params["kvector"]
			# if isinstance(kx,np.ndarray):
			# 	kx = kx[:,:,None,None]
			# if isinstance(ky,np.ndarray):
			# 	ky = ky[:,:,None,None]
			# if "phase" in conn_params.keys():
			# 	phases = conn_params["phase"]
			# else:
			# 	phases = 0
			# conn_matrix = disc_gaussian * np.sin(xdelta * kx + ydelta * ky + phases)

		else:
			print("Specified connectivity profile ({}) not found.".format(profile))

		self.conn_matrix = conn_matrix.reshape(np.prod(self.to_size)*self.Nvert[1],\
												np.prod(self.from_size)*self.Nvert[0])
		
		if arbor is not None:
			self.arbor = arbor.reshape(np.prod(self.to_size)*self.Nvert[1],\
										np.prod(self.from_size)*self.Nvert[0])
		else:
			self.arbor = None
		

		if self.full_output:
			return self.conn_matrix, (self.full_output_dict, self.arbor)
		else:
			return self.conn_matrix, self.arbor
	
	def create_arbor(self, radius, profile="heaviside", arbor_params={}):
		"""
		arbor function: crude model of overlap of geniculocortical terminal
		arbors with cortical dendritic arbors
		"""
		grid = np.linspace(0,1,self.from_size[0],endpoint=False)
		xfrom,yfrom = np.meshgrid(grid,grid)
		xfrom = np.repeat(xfrom,self.Nvert[0]).reshape(self.from_size[0],self.from_size[1]*self.Nvert[0])
		yfrom = np.repeat(yfrom,self.Nvert[0]).reshape(self.from_size[0],self.from_size[1]*self.Nvert[0])
		grid = np.linspace(0,1,self.to_size[0],endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		xto = np.repeat(xto,self.Nvert[1]).reshape(self.to_size[0],self.to_size[1]*self.Nvert[1])
		yto = np.repeat(yto,self.Nvert[1]).reshape(self.to_size[0],self.to_size[1]*self.Nvert[1])
		xdelta = distance(xto[:,:,None,None]-xfrom[None,None,:,:])
		ydelta = distance(yto[:,:,None,None]-yfrom[None,None,:,:])
	

		## Heterogeneously varying anisotropic Connectivity
		if "heterogeneity_type" in arbor_params.keys():
			if arbor_params["heterogeneity_type"] is not None:
				## eccentricity distribution
				np.random.seed(2000)
				ecc_field = np.random.randn(*xto.shape)[:,:,None,None]
				ecc_field = arbor_params["mean_eccentricity"] +\
							ecc_field/np.std(ecc_field)*arbor_params["SD_eccentricity"]
				ecc_field = np.clip(ecc_field,0.0,0.95)
				## size distribution
				size_field_x = np.random.randn(*xto.shape)[:,:,None,None]
				size_field_x = (1 + size_field_x/np.std(size_field_x)*arbor_params["SD_size"])
				size_field_y = size_field_x*np.sqrt(1 - ecc_field**2)
				## orientations of elongated connections
				z_field = np.random.randn(*xto.shape) + 1j*np.random.randn(*xto.shape)
				orientation_field = np.angle(z_field)*0.5
				cos_theta = np.cos(orientation_field)[:,:,None,None]
				sin_theta = np.sin(orientation_field)[:,:,None,None]
				if arbor_params["heterogeneity_type"]=="smooth":
					print('TODO: implement smooth heterogeneity_type')
					pass
				xdelta2 = (xdelta*cos_theta - ydelta*sin_theta) * size_field_x
				ydelta = (xdelta*sin_theta + ydelta*cos_theta) * size_field_y
				xdelta = xdelta2
		
		if arbor_params.get("ret_scatter",False):
			rng = np.random.default_rng(2001)
			scatter_ampl = 0.25
			if isinstance(arbor_params["ret_scatter"],float):
				scatter_ampl = arbor_params["ret_scatter"]
			x_scat = scatter_ampl*rng.standard_normal(self.to_size)
			y_scat = scatter_ampl*rng.standard_normal(self.to_size)
			xdelta = distance(xto[:,:,None,None]-xfrom[None,None,:,:]-x_scat[:,:,None,None])
			ydelta = distance(yto[:,:,None,None]-yfrom[None,None,:,:]-y_scat[:,:,None,None])


		d = np.sqrt(xdelta**2 + ydelta**2)		## absolute distance |vec(x)-vec(y)|
		d = d.reshape(np.prod(self.to_size)*self.Nvert[1],\
					  np.prod(self.from_size)*self.Nvert[0])

		if profile=="heaviside":
			arbor = (np.around(d,2)<=radius).astype(float)

		elif profile=="gaussian":
			arbor = gaussian(d,0,radius)
			arbor[np.around(d,2)>radius] = 0.0
		
		elif profile=="overlap":
			d_unique = np.unique(d[np.around(d,2)<=radius])
			cA = 0.5
			N = radius*(cA+2)

			delta_bin = 0.05
			i,j = np.meshgrid(np.arange(-(self.to_size[0]//2),self.to_size[0]//2+delta_bin,\
					delta_bin),\
					np.arange(-(self.from_size[0]//2),self.from_size[0]//2+delta_bin,delta_bin))
			delta1 = np.sqrt(i**2+j**2)
			delta2 = np.sqrt((d_unique[:,None,None]-i[None,:,:])**2+(j[None,:,:])**2)
			circles = (delta1 < radius)[None,:,:]*(delta2 < (radius*cA))
			norm = np.sum(delta2 < (radius*cA),axis=(1,2))
			overlap = 1.*np.sum(circles,axis=(1,2))/norm

			arbor = np.zeros_like(d,dtype=float)
			for iduni in range(d_unique.size):
				arbor[d==d_unique[iduni]] = overlap[iduni]

		elif profile=="linear_falloff":
			arbor = linear_arbor_falloff(np.around(d,2),radius=radius,\
										 inner_radius=radius*0.5)
			arbor[np.around(d,2)>radius] = 0.0

		else:
			print("Specified arbor profile ({}) not found. Use one of the following: \
				  'heaviside', 'gaussian', 'overlap, 'linear_falloff'.".format(profile))

		return arbor



class Connectivity_2pop():
	'''
	input to lgn which is inputs to retina convolved with connectivity
	from retina to lgn
	'''
	def __init__(self, from_size_E, from_size_I, to_size_E, to_size_I, random_seed,\
				 Nvert=1, full_output=False):
		self.random_seed = random_seed
		self.from_Ne = from_size_E
		self.from_Ni = from_size_I
		self.to_Ne = to_size_E
		self.to_Ni = to_size_I
		if isinstance(Nvert,int):
			Nvert = (Nvert,Nvert)
		self.Nvert = Nvert
		self.full_output = full_output
		self.full_output_dict = {}

	def create_matrix_2pop(self, conn_params, profile, **kwargs):
		# profile : "Lognormal_2pop" "Gaussian_2pop" "Gaussian_2pop_dense"

		sigma_rec = conn_params["sigma_factor"]
		max_ew = conn_params["max_ew"]
		##TODO: different random seed for E, I conn

		profile_conn = profile.replace("2pop","")
		print("profile_conn",profile_conn)
		print("Warning: Currently E and I units are matched in their abs phases/same realization of existing conns")
		if not "density" in conn_params.keys():
			conn_params["density"] = 1.
		if not "ncluster" in conn_params.keys():
			conn_params["ncluster"] = 1
		profile_conn_EE,profile_conn_EI,profile_conn_IE,profile_conn_II = \
			profile_conn,profile_conn,profile_conn,profile_conn
		if profile_conn=="zmodel":
			profile_conn_EE = "zmodel"
			profile_conn_IE = "zmodel"
			profile_conn_EI,profile_conn_II = ["Gaussian"]*2

		rA_E = conn_params["rA_E"]
		rA_I = conn_params["rA_I"]
		W4to4_params_EE = copy(conn_params)
		W4to4_params_EE.update({"sigma" : conn_params["sigma_EE"] * sigma_rec,\
						 		"ampl" : conn_params["aEE"], "type" : "EE"})
		conn_EE,arb_EE = Connectivity(self.from_Ne, self.to_Ne,\
				 self.random_seed, self.Nvert,full_output=self.full_output\
				 ).create_matrix(W4to4_params_EE, profile_conn_EE, r_A=rA_E, **kwargs)

		W4to4_params_IE = copy(conn_params)
		W4to4_params_IE.update({"sigma" : conn_params["sigma_IE"] * sigma_rec,\
						 		"ampl" : conn_params["aIE"], "type" : "IE"})
		conn_IE,arb_IE = Connectivity(self.from_Ne, self.to_Ni,\
		 			self.random_seed, self.Nvert,full_output=self.full_output\
		 			).create_matrix(W4to4_params_IE, profile_conn_IE, r_A=rA_E, **kwargs)

		W4to4_params_EI = copy(conn_params)
		W4to4_params_EI.update({"sigma" : conn_params["sigma_EI"] * sigma_rec,\
								 "ampl" : conn_params["aEI"], "type" : "EI"})
		conn_EI,arb_EI = Connectivity(self.from_Ni, self.to_Ne,\
				 self.random_seed, self.Nvert,full_output=self.full_output\
				 ).create_matrix(W4to4_params_EI, profile_conn_EI, r_A=rA_I, **kwargs)

		W4to4_params_II = copy(conn_params)
		W4to4_params_II.update({"sigma" : conn_params["sigma_II"] * sigma_rec,\
						 		"ampl" : conn_params["aII"], "type" : "II"})
		conn_II,arb_II = Connectivity(self.from_Ni, self.to_Ni,\
		 			self.random_seed, self.Nvert,full_output=self.full_output\
		 			).create_matrix(W4to4_params_II, profile_conn_II, r_A=rA_I, **kwargs)

		if self.full_output:
			output_dict_EE,arb_EE = arb_EE
			output_dict_IE,arb_IE = arb_IE
			output_dict_EI,arb_EI = arb_EI
			output_dict_II,arb_II = arb_II

			self.full_output_dict.update({"EE" : output_dict_EE,
										  "IE" : output_dict_IE,
										  "EI" : output_dict_EI,
										  "II" : output_dict_II,})
		print("")
		print("CHECK CONN VALS",profile_conn,np.nanmax(conn_EE),np.nanmax(conn_IE),\
				np.nanmax(conn_EI),np.nanmax(conn_II))
		
		from_Ne_total = np.prod(self.from_Ne) * self.Nvert[0]
		from_Ni_total = np.prod(self.from_Ni) * self.Nvert[0]
		to_Ne_total = np.prod(self.to_Ne) * self.Nvert[1]
		to_Ni_total = np.prod(self.to_Ni) * self.Nvert[1]
		Wrec = np.zeros((to_Ne_total+to_Ni_total,from_Ne_total+from_Ni_total))
		Wrec[:to_Ne_total,:from_Ne_total] = conn_EE
		Wrec[:to_Ne_total,from_Ne_total:] = -conn_EI
		Wrec[to_Ne_total:,:from_Ne_total] = conn_IE
		Wrec[to_Ne_total:,from_Ne_total:] = -conn_II

		# normalise conn matrix if the maximal real eigenvalue is given
		if Wrec.shape[0]==Wrec.shape[1]:
			ew,_ = linalg.eig(Wrec,right=True)
			print("orig max ew",np.nanmax(np.real(ew)),max_ew)
			if isinstance(max_ew,float):
				Wrec /= np.nanmax(np.real(ew)) / max_ew
				print("max_ew",max_ew)

			Wee = np.sum(Wrec[:to_Ne_total,:from_Ne_total],axis=1)[0]
			Wei = np.abs(np.sum(Wrec[:to_Ne_total,from_Ne_total:],axis=1)[0])
			Wie = np.sum(Wrec[to_Ne_total:,:from_Ne_total],axis=1)[0]
			Wii = np.abs(np.sum(Wrec[to_Ne_total:,from_Ne_total:],axis=1)[0])
			print("WEE={},WEI={},WIE={},WII={}".format(Wee,Wei,Wie,Wii))
			print("omega_i=Wie-Wee={}".format(Wie-Wee))
			print("omega_e=Wii-Wei={}".format(Wii-Wei))
		else:
			svds = linalg.svdvals(Wrec)
			Wrec /= np.nanmax(np.real(svds)) / max_ew
			print('svd,Wrec',np.sqrt(svds[:5]))
		print("")
		
		if arb_EE is None:
			arbor = None
		else:
			arbor = np.zeros((to_Ne_total+to_Ni_total,from_Ne_total+from_Ni_total))
			arbor[:to_Ne_total,:from_Ne_total] = arb_EE
			arbor[:to_Ne_total,from_Ne_total:] = arb_EI
			arbor[to_Ne_total:,:from_Ne_total] = arb_IE
			arbor[to_Ne_total:,from_Ne_total:] = arb_II

		if self.full_output:
			return Wrec, self.full_output_dict, arbor
		else:
			return Wrec, arbor


		
if __name__=="__main__":
	import matplotlib.pyplot as plt
	from bettina.modeling.ori_dev_model import W4to4_params
	from bettina.ferretV1 import hierarchical_clustering_correlation as hcc


	N = 25
	Nvert = 1
	fromsize = (N,N)
	tosize = (N,N)
	conn = Connectivity(fromsize,tosize,random_seed=12345)

	## test "gabor"
	if False:
		grid = np.linspace(0,1,N,endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		conn_params = {"rng" : np.random.RandomState(20200205)}
		ecp,sigma = conn.gen_ecp(xto, yto, conn_params)
		thetas = np.angle(ecp,deg=False)*0.5
		
		## smooth phases generation
		grid = np.linspace(0,1,N,endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		conn_params = {"rng" : np.random.RandomState(20200205), "kc" : 2., "n" : 1}
		ecp,sigma = conn.gen_ecp(xto, yto, conn_params)
		pref_phase = np.angle(ecp,deg=False)

		np.random.seed(12)
		# thetas = (np.random.random(N*N)*2*np.pi).reshape(N,N)
		conn_params = {"sigma" : 0.2,
						"ampl" : 1.,
						"theta" : thetas,
						"psi" : 0,
						"freq" : 10}
		gb = conn.create_matrix(conn_params, "Gabor")
		gb = gb.reshape(N,N,N,N)
		print("gb",gb.shape)

		idx = 10
		fig = plt.figure()
		ax = fig.add_subplot(141)
		im=ax.imshow(thetas.reshape(N,N),interpolation="nearest",cmap="hsv")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(142)
		im=ax.imshow(pref_phase.reshape(N,N),interpolation="nearest",cmap="hsv")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(143)
		im=ax.imshow(gb[:,:,idx,idx],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(144)
		im=ax.imshow(gb[:,:,15,15],interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		plt.show()
		exit()

	## test "2pop"
	if False:
		N4 = 25
		W4 = Connectivity_2pop((N4,N4), (N4,N4), (N4,N4), (N4,N4),\
								random_seed=12345, Nvert=Nvert)
		W4to4_params["sigma_factor"] = 0.5
		W4to4 = W4.create_matrix_2pop(W4to4_params,"Gaussian2pop")

		grid = np.linspace(0,1,N4,endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		ecp, sigma_ecp = Connectivity((N4,N4),(N4,N4),19).gen_ecp(xto, yto, W4to4_params)

		fig = plt.figure()
		ax = fig.add_subplot(131)
		im=ax.imshow(W4to4,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(132)
		im=ax.imshow(W4to4[N4//2+N4**2//2-2,:N4**2*Nvert].reshape(N4,N4*Nvert),\
						interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(133)
		im=ax.imshow(np.imag(ecp),\
						interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax)
		plt.show()
		exit()

	## test z-model connectivity
	if True:
		N4 = N
		W4to4_params.update({"kc"	:	3.,
							 "n"	:	30,
							 "rng"	:	np.random.RandomState(13*90),
							 "sigma":	0.3,
							 "ampl"	:	1.,
							})
		W4 = Connectivity_2pop((N4,N4), (N4,N4), (N4,N4), (N4,N4),random_seed=12345, Nvert=Nvert)
		W4to4_params["sigma_factor"] = 0.5
		W4to4_params["rA_E"] = None
		W4to4_params["rA_I"] = None
		conn,_ = W4.create_matrix_2pop(W4to4_params,"zmodel2pop")

		fig = plt.figure()
		ax = fig.add_subplot(131)
		im=ax.imshow(conn[:N4**2*Nvert,N4//2+N4**2//2+4].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(132)
		im=ax.imshow(conn[:N4**2*Nvert,N4//2+N4**2//2+4+N4**2].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(133)
		im=ax.imshow(conn[N4//2+N4**2//2-2,:N4**2*Nvert].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)

		eigvals,eigvec = linalg.eig(conn)
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


		plt.show()
		exit()

	## test heterogeneity
	if False:
		N4 = N
		sigma_factor = 0.4
		W4to4_params.update({"sigma"	:	0.2*sigma_factor,
							"mean_eccentricity"	:	0.8,
							"SD_eccentricity"	:	0.05,
							"SD_size"			:	0.05,
							"heterogeneity_type":	"independent",})
		W4 = Connectivity((N4,N4), (N4,N4),random_seed=12345, Nvert=Nvert)
		conn = W4.create_matrix(W4to4_params,profile="linear")

		fig = plt.figure()
		ax = fig.add_subplot(131)
		im=ax.imshow(conn[:N4**2*Nvert,N4//2+N4**2//2+4].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(132)
		im=ax.imshow(conn[N4//2+N4**2//2-2,:N4**2*Nvert].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)

		W4to4_params.update({"sigma"	:	0.2*sigma_factor,
							"mean_eccentricity"	:	0.8,
							"SD_eccentricity"	:	0.05,
							"SD_size"			:	0.05,
							"heterogeneity_type":	None,})
		W4 = Connectivity((N4,N4), (N4,N4),random_seed=12345, Nvert=Nvert)
		conn = W4.create_matrix(W4to4_params,profile="linear")

		fig = plt.figure()
		ax = fig.add_subplot(131)
		im=ax.imshow(conn[:N4**2*Nvert,N4//2+N4**2//2+4].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(132)
		im=ax.imshow(conn[N4//2+N4**2//2-2,:N4**2*Nvert].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)

		plt.show()
		exit()

	## test arbor
	if True:
		N4 = N
		arbor_params = {"mean_eccentricity"		:	0.5,
						"SD_eccentricity"	:	0.0,
						"SD_size"			:	0.0,
						"heterogeneity_type":	None,#"independent"
						"ret_scatter" : False
						}
		W4 = Connectivity((N4,N4), (N4,N4),random_seed=12345, Nvert=Nvert)
		arbor = W4.create_arbor(0.3, profile="gaussian",arbor_params=arbor_params)
		# print("arbor",arbor.shape,np.sum(arbor>0,axis=0),np.unique(np.sum(arbor,axis=1)))
		fig = plt.figure()
		ax = fig.add_subplot(131)
		ax.set_title("sum={}".format(np.sum(arbor[:N4**2*Nvert,1])))
		im=ax.imshow(arbor[:N4**2*Nvert,N4//2+N4**2//2+4].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(132)
		ax.set_title("sum={}".format(np.sum(arbor[:N4**2*Nvert,0])))
		im=ax.imshow(arbor[:N4**2*Nvert,N4//2+N4**2//2+5].reshape(N4,N4*Nvert),interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(133)
		im=ax.imshow(arbor,interpolation="nearest",\
						cmap="binary")
		plt.colorbar(im,ax=ax)
		plt.show()
		exit()

	Wee = W4to4[:N4**2,:N4**2]
	print("to one unit",np.sum(W4to4[N4//2+N4**2//2,:N4**2]),np.sum(Wee,axis=1)[:5])
	print("from one unit",np.sum(W4to4[:N4**2,N4//2+N4**2//2]),np.sum(Wee,axis=0)[:5])
	Wee_cl,leaves,linkage = hcc.hierarchical_clustering_of_correlation(Wee, visualise=False)

	fig = plt.figure(figsize=(18,5))
	plt.suptitle("Example corr")
	ax = fig.add_subplot(131)
	ax.set_title("EE conn")
	im=ax.imshow(Wee,interpolation="nearest",cmap="RdBu_r")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(132)
	ax.set_title("clustered EE conn")
	im=ax.imshow(Wee_cl,interpolation="nearest",cmap="RdBu_r")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(133)
	ax.set_title("EI conn")
	im=ax.imshow(W4to4[:N4**2*Nvert,N4**2*Nvert:],interpolation="nearest",cmap="RdBu_r")
	plt.colorbar(im,ax=ax)
	plt.show()
	exit()

	arbor = conn.create_arbor(0.2,profile="gaussian")
	# print("arbareas",np.sum(arbor,axis=0))

	# fig = plt.figure()
	# ax = fig.add_subplot(131)
	# im=ax.imshow(arbor[0,:].reshape(N,N),interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# ax.contour(arbor[0,:].reshape(N,N),[0.2],colors="m")
	# ax = fig.add_subplot(132)
	# ax.plot((arbor[0,:].reshape(N,N))[0,:],"-o")
	# print((arbor[0,:].reshape(N,N))[0,:])
	# ax = fig.add_subplot(133)
	# im=ax.imshow(arbor[12,:].reshape(N,N),interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# plt.show()
	# exit()

	
	W = conn.create_matrix({"sigma1" : 0.06,
							 "sigma2" : 0.09,
							 "ampl1" : 1,
							 "ampl2" : 1.0,
							 "rng" : np.random.RandomState(1555),
							 "sigma" : 0.1,
							 "ampl"  : 1.,
							 "noise"	: 0.0})
	W = W.reshape(N,N,N,N)
	# ffta = np.abs(np.fft.fftshift(np.fft.fft2(W[:,:,N//2,N//2])))
	# print(np.argmax(ffta)//N,np.argmax(ffta)%N)
	W2 = 1.-np.linalg.inv(W[:,N//2,:,N//2-1])
	im=plt.imshow(W2,interpolation="nearest",cmap="binary")
	plt.colorbar(im)
	plt.show()
	exit()

	conn = Connectivity(fromsize,fromsize,"zmodel")
	W,mh, zc = conn.create_matrix({"sigma1" : 0.08,
								 "sigma2" : 0.12,
								 "ampl1" : 1,
								 "ampl2" : 1.0,
								 "rng" : np.random.RandomState(1555)})
	print(mh.shape,W.shape)
	
	#W = conn.create_arbor(0.2)
	#print(W.shape,np.sum(W))
	
	fig = plt.figure(figsize=(18,6))
	ax = fig.add_subplot(131)
	im=ax.imshow(mh[:,:,N//2,N//2],interpolation="nearest",cmap="binary")
	plt.colorbar(im)
	ax = fig.add_subplot(132)
	im=ax.imshow(zc[:,:,N//2,N//2],interpolation="nearest",cmap="binary")
	plt.colorbar(im)
	ax.contour(mh[:,:,N//2,N//2],[0],colors="m")
	ax = fig.add_subplot(133)
	# im=im=ax.imshow(W[:,N//2*N+N//2].reshape(tosize),interpolation="nearest",cmap="binary")
	im=im=ax.imshow(W[:,:,N//2,N//2],interpolation="nearest",cmap="binary")
	plt.colorbar(im)
	plt.show()
	
		
