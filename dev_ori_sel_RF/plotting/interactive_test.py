#!/usr/bin/python

''' 
interactive widget to test stuffs
'''

__author__ = 'Bettina Hein'
__email__ = 'hein@fias.uni-frankfurt.de'

import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
import matplotlib.gridspec as gridspec
import numpy as np
# from scipy.interpolate import RectBivariateSpline
# import h5py
import pickle
from copy import copy

from bettina.modeling.ori_dev_model import data_dir,image_dir,scan_simulations
from bettina.modeling.ori_dev_model.tools import misc,analysis_tools
from bettina.modeling.ori_dev_model.plotting import helper_definitions as hd
from bettina.modeling.ori_dev_model.data import data_lee

#TODO:
	#- add filter to show only simulations with certain parameter settings:
	#- wrec_mode, ncl, density, Nvert, LGN correlation (MH), noise_rec, aEE, sI, rC, rA, beta
	# (find available parameters online in .p files)

# os.environ["QT_QPA_PLATFORM"] = "offscreen" ##otherwise following error is thrown and
# application  ## is aborted:  qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
## in "" even though it was found.
## This application failed to start because no Qt platform plugin could be initialized.
## Reinstalling the application may fix this problem.
## Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc,
##wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.

class Parameter_selection_window(QMainWindow):
	def __init__(self, parameter_settings, labels, parent=None):
		super(Parameter_selection_window, self).__init__(parent)

		self.setMinimumSize(1200,360)
		self._main = QWidget()
		self.setCentralWidget(self._main)
		self.mainlayout = QGridLayout()
		self._main.setLayout(self.mainlayout)
		## add min width/height

		self.setWindowTitle("Select simulation parameters")
		self.label1 = QLabel('Available parameter settings')
		self.mainlayout.addWidget(self.label1,0,0,1,1)

		self.default_parameter_settings = parameter_settings
		self.current_parameter_settings = copy(parameter_settings)
		self.selected_filters = []
		self.parameter_keys = sorted(self.default_parameter_settings.keys())
		

		self.create_parameter_menu()

		
	def create_parameter_menu(self):
		# QCheckbox | Parameter List | set_default Button
		## add uncheck all/check all buttons

		num_row = len(self.default_parameter_settings.keys())
		num_col = 4

		self.buttongroups = []
		self.checkboxes = []
		self.lists = []
		self.pushbuttons = []
		for i,key in enumerate(self.parameter_keys):
			# self.buttongroups.append( QButtonGroup() )
			self.checkboxes.append(QCheckBox(key))
			self.checkboxes[-1].setChecked(False)
			self.checkboxes[-1].setCheckable(True)
			self.checkboxes[-1].toggled.connect(self.select_parameter)

			self.lists.append( QLineEdit() )
			self.lists[-1].setText( " ".join( [str(item) for item in\
							 self.default_parameter_settings[key]] )  )
			self.lists[-1].returnPressed.connect(self.select_parameter_range)

			self.pushbuttons.append(QPushButton("Reset default"))
			self.pushbuttons[-1].setCheckable(True)
			self.pushbuttons[-1].clicked.connect(self.set_default_parameter_settings)

			self.mainlayout.addWidget(self.checkboxes[-1],i+1,0,1,1)
			self.mainlayout.addWidget(self.lists[-1],i+1,1,1,1)
			self.mainlayout.addWidget(self.pushbuttons[-1],i+1,2,1,1)


	def select_parameter(self):
		self.selected_filters = []
		for i,checkbox in enumerate(self.checkboxes):
			if checkbox.isChecked():
				self.selected_filters.append(self.parameter_keys[i])

	def set_default_parameter_settings(self):
		for i,pushbutton in enumerate(self.pushbuttons):
			if pushbutton.isChecked():
				default = " ".join([str(item) for item in\
						 self.default_parameter_settings[self.parameter_keys[i]]])
				self.lists[i].setText(default)
				pushbutton.setChecked(False)

	def select_parameter_range(self):
		for i,line in enumerate(self.lists):
			if line.isModified():
				items = str(line.text()).split()
				self.current_parameter_settings[self.parameter_keys[i]] = items
				line.setModified(False)



class Main_window(QMainWindow):
	def __init__(self, load_external_from=False, parent=None):
		# super().__init__()
		# QWidget.__init__(self)
		QMainWindow.__init__(self, parent)
		self.setMinimumSize(800,800)
		self._main = QWidget()
		self.setCentralWidget(self._main)
		self.mainlayout = QVBoxLayout()
		self.setWindowTitle("OSRF - interactive")
		self.init_width,self.init_height = 4*4,9

		## DATA PARAMETERS
		self.load_external_from = load_external_from
		# self.initVERSIONS = [537,538,539,540,541,542,543,544,545,546,547]
		self.initVERSIONS = np.arange(901,907)
		self.initVERSIONS = np.arange(884,896)
		self.initVERSIONS = np.arange(798,807)
		self.initVERSIONS = np.arange(1007,1022)
		self.initVERSIONS = np.arange(1022,1027)
		# self.initVERSIONS = np.concatenate([np.arange(695,700),np.arange(798,807)])
		self.varied_params = np.array(['Wrec',"Nvert"])
		self.text_to_key = {'L4 rec. conn. scheme' 		: 	'Wrec',
							"# vertical units"			:	'Nvert',
							}
		self.find_all_parameter_ranges(self.load_external_from)
		self.init_data_parameters()

		self.create_menu()
		self.create_main_frame()
		self.create_status_bar()
		
		
		# self.init_labels()
		self.load_data()
		# self.get_data_comb()
		# self.sort_data()
		self.on_draw()

		

	def init_data_parameters(self):
		self.data_parameters = {
								"Wrec_mode"		:	["Gaussian2pop", "Gaussian_prob2pop",\
													 "Gaussian_prob_cluster2pop",\
													 "Gaussian_sparse2pop",\
													 "Gaussian_prob_density2pop"],
		}
		self.default_params = {
								"file_path" 	: 	image_dir
		}

	def find_all_parameter_ranges(self,load_external_from):
		# last_version = misc.get_version(data_dir+"layer4/",version=None,readonly=True)
		last_version = 599
		versions = np.arange(0,last_version)
		self.args_to_scan = [["W4to4_params","ncluster"],\
							 ["Wret_to_lgn_params","ampl2"],\
							 ["Wlgn_to4_params","r_A"],\
							  "Nvert",\
							 ["Wret_to_lgn_params","sigma"],\
							 ["Wret_to_lgn_params","sigma2"],\
							 ["W4to4_params","density"],\
							 ["W4to4_params","Wrec_mode"],\
							 ["W4to4_params","aEE"],\
							 ["W4to4_params","noise"],\
							 ["W4to4_params","sigma_factor"]]
		self.arg_labels = ["# cluster rec conn",
							"MH strength",\
							"Arbor radius",\
							"# vertical units",\
							"LGN corr range (Gauss)",\
							"LGN corr range (MH)",\
							"Density rec conn",\
							"Rec conn mode","E to E conn strength",\
							"Noise rec conn",\
							"Intracortical range"]
		print_out = False
		self.param_range = scan_simulations.range_of_params(load_external_from, versions,\
															self.arg_labels,print_out,\
															*self.args_to_scan)

	def scrolling(self, event):
		val = self.scroll.verticalScrollBar().value()
		if event.button =="down":
			self.scroll.verticalScrollBar().setValue(val+100)
		else:
			self.scroll.verticalScrollBar().setValue(val-100)

	def create_main_frame(self):
		self._main.setLayout(self.mainlayout)
		self._main.layout().setContentsMargins(0,0,0,0)
		self._main.layout().setSpacing(0)

		self.dpi = 100
		self.fig = Figure((self.init_width,self.init_height), dpi=self.dpi)#5*4,3*4
		# self.fig.set_size_inches(6,5)
		# self.fig.canvas.layout.width = "500px"

		self.canvas = FigureCanvas(self.fig)
		self.canvas.draw()
		# self.canvas.setParent(self)
		self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		self.canvas.updateGeometry()
		mpl_toolbar = NavigationToolbar(self.canvas, self._main)
		
		self.mainlayout.addWidget(mpl_toolbar)
		self.mainlayout.addWidget(self.canvas)

		## scroll bar
		self.scroll = QScrollArea(self._main)
		self.scroll.setWidget(self.canvas)
		self._main.layout().addWidget(self.scroll)
		self.canvas.mpl_connect("scroll_event", self.scrolling)

		grid = QGridLayout()

		## Draw Button
		self.slider_label = QLabel('Actions')
		self.sliders = []
		self.sliders.append(QPushButton("&Draw"))
		self.sliders[-1].clicked.connect(self.on_draw)
		grid.addWidget(self.slider_label,0,0,1,1)
		grid.addWidget(self.sliders[-1],0,1,1,1)

		## radiobutton which kind of plots to be shown
		## individual: shows one plot per version
		## Summary: shows one plot including all versions
		self.radiobutton_label = QLabel('Plotstyle')
		self.rbtn_ind = QRadioButton("Individual")
		self.rbtn_sum = QRadioButton("Summary")
		self.rbtn_ind.setChecked(True)
		self.rbtn_sum.setChecked(False)
		self.rbtn_ind.toggled.connect(self.change_plotstyle)
		grid.addWidget(self.radiobutton_label,1,0,1,1)
		grid.addWidget(self.rbtn_ind,1,2,1,1)
		grid.addWidget(self.rbtn_sum,1,3,1,1)

		
		## combobox of metrics to be plotted
		self.combobox_label = QLabel('Metric')
		self.combobox = QComboBox()
		self.combobox.addItem("RF")
		self.combobox.addItem("Orientation")
		self.combobox.addItem("Relative phase")
		self.combobox.addItem("L4")
		self.combobox.addItem("Selectivity")
		self.combobox.addItem("ONOFF ratio")
		self.combobox.addItem("ONOFF segregation")
		self.combobox.addItem("ON-OFF Distance to center")
		self.combobox.addItem("Center value RF")
		self.combobox.addItem("Envelope width")
		self.combobox.addItem("Log aspect ratio")
		self.combobox.addItem("# half cycles")
		self.combobox.currentTextChanged.connect(self.combobox_changed)
		grid.addWidget(self.combobox_label,1,4,1,1)
		grid.addWidget(self.combobox,1,4,1,1)


		## Line for simulation ID/ version number
		self.versionbox = QLineEdit()
		self.versionlabel = QLabel('Simulation ID')
		s = ' '
		self.versionbox.setText( s.join( [str(item) for item in self.initVERSIONS] ) )
		grid.addWidget(self.versionlabel,3,0,1,1)
		grid.addWidget(self.versionbox,3,1,1,1)
		self.versionbox.returnPressed.connect(self.load_data)

		## Button to make parameter selection
		self.pushButton = QPushButton("Simulation settings")
		self.pushButton.clicked.connect(self.show_simulation_setting_setter)
		self.dialog = Parameter_selection_window( self.param_range, self.arg_labels)
		grid.addWidget(self.pushButton,0,2,1,1)

		self.mainlayout.addLayout(grid)

	def show_simulation_setting_setter(self):
		self.dialog.show()

	def change_plotstyle(self):
		# print("change_plotstyle",self.rbtn_ind.isChecked(),self.rbtn_sum.isChecked())
		self.on_draw()

	def combobox_changed(self):
		self.on_draw()

	def create_status_bar(self):
		self.status_text = QLabel("Version 1.0")
		self.statusBar().addWidget(self.status_text, 1)

	def create_menu(self):
		self.file_menu = self.menuBar().addMenu("&File")

		load_file_action = self.create_action("&Save plot",
							shortcut="Ctrl+S", slot=self.save_plot, 
							tip="Save the plot")
		quit_action = self.create_action("&Quit", slot=self.window_close, 
						shortcut="Ctrl+Q", tip="Close the application")
		self.add_actions(self.file_menu,(load_file_action, None, quit_action))
		
		self.help_menu = self.menuBar().addMenu("&Help")
		about_action = self.create_action("&About", 
						shortcut='F1', slot=self.on_about, 
						tip='About the gui')	
		self.add_actions(self.help_menu, (about_action,))

	def save_plot(self):
		# file_choices = "PDF (*.pdf)|*.pdf"
		# path = unicode(QFileDialog.getSaveFileName(self, 
		# 				'Save file', self.default_params["file_path"], 
		# 				file_choices))
		# if path:
		# 	self.canvas.print_figure(path, dpi=self.dpi)
		# 	self.statusBar().showMessage('Saved to %s' % path, 2000)
		versions = '-'.join( [str(item) for item in self.VERSIONS] )
		filename = self.default_params["file_path"] + 'layer4/overview/{}_v{}.pdf'.format(\
					self.combobox.currentText()[4:],versions)
		self.fig.savefig(filename,bbox_inches='tight',dpi=300,format='pdf')
		print('Saved to {}'.format(filename))

	def window_close(self):
		print("window_close")

	def on_about(self):
		print("on about")

	def add_actions(self, target, actions):
		for action in actions:
			if action is None:
				target.addSeparator()
			else:
				target.addAction(action)

	def create_action(  self, text, slot=None, shortcut=None, 
						icon=None, tip=None, checkable=False):
		action = QAction(text, self)
		if icon is not None:
			action.setIcon(QIcon(":/%s.png" % icon))
		if shortcut is not None:
			action.setShortcut(shortcut)
		if tip is not None:
			action.setToolTip(tip)
			action.setStatusTip(tip)
		if slot is not None:
			action.triggered.connect(slot)
			pass
		if checkable:
			action.setCheckable(True)
		return action

	def get_list_of_simulation_id(self):
		self.VERSIONS = np.array([],dtype=int)
		items = str(self.versionbox.text()).split()
		try:
			for item in items:
				if ":" in item:
					version_list = item.split(":")
					assert version_list[0]!="",\
					 "Error: start and end value expected for list of indices"
					assert version_list[1]!="",\
					 "Error: start and end value expected for list of indices"
					self.VERSIONS = np.concatenate([self.VERSIONS,np.arange(int(version_list[0]),\
						int(version_list[1])+1,1)])
				elif "-" in item:
					version_list = item.split("-")
					assert version_list[0]!="",\
					 "Error: start and end value expected for list of indices"
					assert version_list[1]!="",\
					 "Error: start and end value expected for list of indices"
					self.VERSIONS = np.concatenate([self.VERSIONS,np.arange(int(version_list[0]),\
						int(version_list[1])+1,1)])
				else:
					assert isinstance(int(item),int), "Error: int value expected for index"
					self.VERSIONS = np.concatenate([self.VERSIONS,np.array([int(item)])])
		except:
			misc.PrintException()
			self.VERSIONS = self.initVERSIONS


	def load_results(self,Version,load_external_from):
		if os.path.expanduser('~') == '/home/bettina':
			path_to_files = data_dir + "layer4/"
		elif os.path.expanduser('~') == '/rigel/theory/users/bh2757':
			path_to_files = data_dir + "layer4/"
		'''add more paths here:'''
		#elif os.path.expanduser('~') == '/home/USERNAME':
			#path_to_files = PATH_TO_FILES

		if load_external_from=="":
			filepath = data_dir + "layer4/"
			cluster_name = "local"
		else:
			# Seagate Portable Drive
			filepath = "/media/bettina/TOSHIBA EXT/physics/columbia/projects" + \
					   "/ori_dev_model/data/layer4/{}/".format(load_external_from)
			cluster_name = load_external_from

		filename = filepath + "observables/observables_v{}.hdf5".format(Version)
		observables = misc.load_from_hdf5(cluster_name,[Version,],filename)
		observables = observables[Version]
		params = pickle.load(open(filepath +"pickle_files/config_v{v}.p".format(\
						 	v=Version),"rb"))

		y = np.load(filepath + "y_files/y_v{}.npz".format(Version))
		if "Wlgn_to_4" in y.files:
			Wlgn_to_4 = y["Wlgn_to_4"].reshape(params["num_lgn_paths"],\
											   params["N4"]**2*params["Nvert"],\
											   params["Nlgn"]**2)
		else:
			Wlgn_to_4 = y["W"].reshape(params["num_lgn_paths"],\
									   params["N4"]**2*params["Nvert"],\
									   params["Nlgn"]**2)
		## by default look at excitatory activity
		if "l4" in y.files:
			l4 = y["l4"][:params["N4"]**2*params["Nvert"]]

		# dataset = pickle.load(open(data_dir +\
		# 		 			"layer4/habanero/results/v{v}_results.p".format(v=VERSION),"rb"))
		# params = pickle.load(open(data_dir +\
		# 					"layer4/habanero/pickle_files/config_v{v}.p".format(\
		# 					v=VERSION),"rb"))

		observables.update({"L4" : l4, "Wlgn_to_4" : Wlgn_to_4})
		print("observables",observables.keys())
		return observables,params

	def load_data(self):
		self.get_list_of_simulation_id()
		self.loaded_data = {}
		self.loaded_params = {}
		delete_id = []
		for i,VERSION in enumerate(self.VERSIONS):
			try:
				observables,params = self.load_results(VERSION,load_external_from)
				self.loaded_data[VERSION] = observables
				self.loaded_params[VERSION] = params
			except:
				misc.PrintException()
				delete_id.append(i)
				print("File for {} not found.".format(VERSION))

		self.VERSIONS = np.delete(self.VERSIONS,delete_id)


	def filter_simulation_id(self):
		filtered_id = []
		for i,VERSION in enumerate(self.VERSIONS):
			for j,arg in enumerate(self.args_to_scan):
				if self.arg_labels[j] not in self.dialog.selected_filters:
					continue
				if arg[0] in self.loaded_params[VERSION].keys():
					if arg[1] in self.loaded_params[VERSION][arg[0]].keys():
						value_Version = self.loaded_params[VERSION][arg[0]][arg[1]]
						values_filter =\
						 self.dialog.current_parameter_settings[self.arg_labels[j]]
						if str(value_Version) not in values_filter:
							filtered_id.append(i)
							# self.VERSIONS.pop(i)

		self.VERSIONS = np.delete(self.VERSIONS,filtered_id)

	def define_colormap(self,num_colors):
		"""define colormap for num_colors different colors
		use as cmap.to_rgba(i) where i in [0,num_colors)"""
		cmap = plt.get_cmap('rainbow')
		cNorm = matplotlib.colors.Normalize(vmin=0,vmax=num_colors)
		cmap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
		cmap.set_array([])
		return cmap


	def on_draw(self):
		self.filter_simulation_id()
		self.fig.clear()
		# self.fig.set_size_inches(6,5)
		# self.fig.canvas.layout.width = "500px"

		self.fig.suptitle(self.combobox.currentText())
		if self.rbtn_ind.isChecked():
			num_plots = len(self.VERSIONS)
			## maximally four plots in one row, add enough rows to accomodate all plots
			self.ncol = 4#np.min([num_plots,4])
			self.nrow = int(np.ceil(1.*num_plots/self.ncol))
			# self.fig.set_figwidth(6*self.ncol)
			# self.fig.set_figheight(5*self.nrow)
			# self.fig.set_dpi(self.dpi)
		elif self.rbtn_sum.isChecked():
			x_value_list = ["lgn_corr_length","intracortical_length","onoff_input_factor"]
			self.ncol = len(x_value_list)
			self.nrow = 2
			num_plots = self.nrow * self.ncol
		gs_ovdh = gridspec.GridSpec(nrows=self.nrow,ncols=self.ncol,\
									height_ratios=[1.0]*self.nrow,\
									width_ratios=[1.0]*self.ncol,hspace=0.2,wspace=0.2)
		self.axes = []
		for i in range(self.nrow):
			for j in range(self.ncol):
				if (i*self.ncol+j)<num_plots:
					self.axes.append(self.fig.add_subplot(gs_ovdh[i,j]))

		chosen_key = self.combobox.currentText()

		if self.rbtn_ind.isChecked():

			if self.combobox.currentText() in hd.individual_plot_measures:
				cmap = hd.cmap_dict[chosen_key]
				for ax_id,VERSION in enumerate(self.VERSIONS):
					N4 = self.loaded_params[VERSION]["N4"]
					Nvert = self.loaded_params[VERSION]["Nvert"]
					array = self.loaded_data[VERSION][chosen_key]#
					if array.ndim==3:
						array = array[0,:,:]
					elif array.ndim==1:
						array = array.reshape(N4,N4*Nvert)
					im = self.axes[ax_id].imshow(array,interpolation='nearest',cmap=cmap)
					plt.colorbar(im,ax=self.axes[ax_id])
					Wrec_mode = self.loaded_params[VERSION]["W4to4_params"]["Wrec_mode"]
					self.axes[ax_id].set_title("v{}".format(VERSION)+" "+Wrec_mode)


		elif self.rbtn_sum.isChecked():

			x_list,x_uni,cmaps = [],[],[]
			for i,xvalue in enumerate(x_value_list):
				xvalues = scan_simulations.get_all_parameters(self.loaded_params,\
																self.VERSIONS,xvalue)
				x_uni.append(np.unique(xvalues))
				x_list.append(np.array(xvalues))
				cmaps.append(self.define_colormap(len(np.unique(xvalues))))
				self.axes[i+self.ncol].set_ylabel("Cumulative distribution")
				self.axes[i].set_xlabel(xvalue)
			
			## Histogram
			if self.combobox.currentText() in hd.histogram_list:
				for i,xvalue in enumerate(x_value_list):
					self.axes[i].set_ylabel("Average")
				if self.combobox.currentText() in data_lee.exp_data.keys():
					x = data_lee.exp_data[self.combobox.currentText()][:,0]
					bin_diff = x[1] - x[0]
					y = data_lee.exp_data[self.combobox.currentText()][:,1]
					for j,xvalue in enumerate(x_value_list):
						self.axes[j+self.ncol].plot(x,y/np.sum(y)/bin_diff,'--',c="k",\
												label="Exp Data")
				for loc_id,VERSION in enumerate(self.VERSIONS):
					measure = self.loaded_data[VERSION][chosen_key]
					measure = measure[np.isfinite(measure)]
					for j,xvalue in enumerate(x_value_list):
						self.axes[j].plot(x_list[j][loc_id],np.nanmean(measure),'o')
						self.axes[j].set_ylim(bottom=0)
						color_id = np.searchsorted(x_uni[j],x_list[j][loc_id],side="left")
						color = cmaps[j].to_rgba(color_id)
						n,_,_ = plt.hist(measure,bins=hd.bins_dict[chosen_key],density=True)
						self.axes[j+self.ncol].plot(hd.bins_dict[chosen_key][1:],n,'-o',c=color,\
													label="{}={}".format(xvalue,x_uni[j][color_id]))
						self.axes[j+self.ncol].legend(loc="best")
						

			## plot cumulative distribution
			elif self.combobox.currentText() in hd.cumulative_dist_list:
				for i,xvalue in enumerate(x_value_list):
					self.axes[i].set_ylabel("Average")
				for loc_id,VERSION in enumerate(self.VERSIONS):
					measure = self.loaded_data[VERSION][chosen_key]
					measure = measure[np.isfinite(measure)]
					for j,xvalue in enumerate(x_value_list):
						self.axes[j].plot(x_list[j][loc_id],np.nanmean(measure),'o')
						color_id = np.searchsorted(x_uni[j],x_list[j][loc_id],side="left")
						color = cmaps[j].to_rgba(color_id)
						label = str(Version)
						self.axes[j+self.ncol].plot(np.sort(measure),np.linspace(0,1,len(measure)),\
													"-",label=label,c=color)

				for ax in self.axes:
					ax.legend(loc="best")

		## resize(width, height)
		# self.canvas.resize(300*self.ncol,300*self.nrow)
		self.canvas.draw()



if __name__=="__main__":
	import argparse

	## ==================Optional Input Arguments ===========================
	parser = argparse.ArgumentParser(description="Plot results from simulation.")
	# parser.add_argument("--index", dest='idx', help="specifying run number (either\
	#  individual runs as x1 x2 x3 or endpoints of successive numbers of runs as x1:xn)",\
	#  required=False,  nargs="*", default=None)
	parser.add_argument("--cl", dest='load_external_from', help="specify where data lies\
						(e.g. None, aws, habanero)", required=False,  nargs="*", default=[""])
	args = parser.parse_args()
	args_dict = vars(args)

	load_external_from = args_dict["load_external_from"][0]
	print("load_external_from",load_external_from)

	app = QApplication(sys.argv)
	screen = Main_window(load_external_from)
	screen.show()
	# sys.exit(app.exec_())
	app.exec_()
