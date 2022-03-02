import os
import sys
import datetime
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import bresenhamline as bresenhamline
import scipy.interpolate as interpolate
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from cml import *
import gcopula_sparaest as sparest

# create a function to calculate probability of zero P from input data
# not used for now - just use a single p) value for now (the median of the 5 stations = 
def calcP0(x):
    result_p0 = (x < 0.1).sum() / len(x) # wet/dry threshold is < 0.1 mm
    return result_p0

# start time
start = datetime.datetime.now()

# use random seed if you want to ensure reproducibility
np.random.seed(121)

# paperstyle_plot
# if True -> same plot style as in paper which requires the installation of some non-standard modules and the download
# of an additional nc file
paperstyle_plot = False 

# PREPROCESSING
#--------------
# define start and end time step
start_time_idx = 0
end_time_idx = 1

# get path to input_data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # sets path for importing input data
datafolder = os.path.join(project_root, r'input_data')

# load input data
# rain gauge data for generating P fields (stattion coords are in eastings and northings)
p_data     = np.genfromtxt(os.path.join(datafolder,'MCRB_Pinputs_20km.csv'), delimiter='\t') # read in P data for stations
# dataframe of rain gauge data for calculating probability of zero P (P0)
p0_data_df = pd.read_csv(os.path.join(datafolder,'P0inputs_20km_stns.csv'))
# gauge Px values in 100 increments for generating probability distributions: read in values from one csv file for all gauges
# Px_data = np.genfromtxt(os.path.join(datafolder,'Pxvals_for_Pdistns.csv'), delimiter='\t')

# rain gauge coordinates
p_lat_lon = p_data[:,(0,1)] # extracts values from all rows (gauge sites) for column indexes 0 and 1 only 
# print("p_lat_lon = ", p_lat_lon)

# CML data
mwl_data   = np.genfromtxt(os.path.join(datafolder,'syn_obs_CML_2015_08_MCRBcoords.csv'), delimiter='\t') # read in CML data with MCRB coords
# CML coordinates
mwl_lat_lon = mwl_data[:,0:4]	 # extracts values from all 100 rows (CML links) for column indexes 0 to 4 only

# define grid for MCRB catchment (eastings and northings)
xs = 623445 # xllcorner
xsize = 400 # ncols increased from 164 to incorporate other stations
xinc = 50 # cellsize
ys = 5642972 # yllcorner
ysize = 400 # nrows increased from 164 to incorporate other stations
yinc = 50 # cellsize
xg = np.arange(xs,(xs+xinc*xsize),xinc) # set up a numpy array for grid x axis: start=xs, stop=xs+(cellsize*columns), step=cellsize
yg = np.arange(ys,(ys+yinc*ysize),yinc) # set up a numpy array for grid y axis: start=ys, stop=ys+(cellsize*rows), step=cellsize
grid = np.array(np.meshgrid(yg,xg)).reshape(2,-1).T # use -1 as we don't know this array dimension and we want numpy to figure it out for us

# transform to standard xy grid
grid_ = np.copy(grid)
grid_[:,0] = (grid_[:,0]-ys)/yinc
grid_[:,1] = (grid_[:,1]-xs)/xinc

# transform p_lat_lon to same grid
p_xy = np.copy(p_lat_lon) # returns an array copy of p_lat_lon
p_xy[:,0] = (p_xy[:,0] - ys)/yinc
p_xy[:,1] = (p_xy[:,1] - xs)/xinc
p_xy = p_xy.astype(int) # gauge locations on standard xy grid

print("p_xy =", p_xy)
# sys.exit("That'll do - stop there!")

# transform mwl_lat_lon to same grid
mwl_xy = np.copy(mwl_lat_lon)
mwl_xy[:,0] = (mwl_xy[:,0] - ys)/yinc
mwl_xy[:,2] = (mwl_xy[:,2] - ys)/yinc
mwl_xy[:,1] = (mwl_xy[:,1] - xs)/xinc
mwl_xy[:,3] = (mwl_xy[:,3] - xs)/xinc
mwl_xy = mwl_xy.astype(int)

# define line integrals between the two coordinates of mwl using Bresenham's Line Algorithm
mwl_integrals = []
for integ in range(mwl_xy.shape[0]):
	mwl_integrals.append(np.array(
	bresenhamline.get_line(mwl_xy[integ,:2], mwl_xy[integ,2:])))

# loop over time steps
for tstep in range(start_time_idx, end_time_idx):

	# rain gauge values
	prec = p_data[:, (2+tstep)] # 2+tstep because the first 2 columns are coord data
	print("prec = ", prec)

	# CML integral values
	mwl_prec = mwl_data[:, (5+tstep)] # 5+tstep because the first columns are coord and other data


# fit the marginal distribution
# convert prec numpy array to pandas df so I can use df.apply on the calcP0 function
# calculate p0 using the function calcP0 created above
	p0arr = p0_data_df.apply(calcP0, axis=0).values
	p0 = np.median(p0arr) # use just a single value for p0 (the median of the values at each station)
# 	print("prob of zero P = ", p0)
    
# 	build cdf and invcdf from pdf
	x = prec[prec >= 0.1] # apply wet/dry threshold

	# transform observations to standard normal using the fitted cdf;
	# zero (dry) observations
	mp0 = prec == 0.0   
	lecp = p_xy[mp0] # lecp are the coords of gauges with obs P = 0 
	lecv = np.ones(lecp.shape[0]) * st.norm.ppf(p0) # lecv are the less or equal constraint values, i.e. obs that recorded zero P
    
	print("lecp (coords of zero P obs = ", lecp)
	print("lecp.shape[0] = ", lecp.shape[0])
	print("lecv = ", lecv)

	# non-zero (wet) observations (st.gamma.cdf generates the gamma cdf, st.norm.ppf generates the inverse cdf)
	cp = p_xy[~mp0] # cp are the equality constraint coords, i.e. gauges with non-zero P 
	cv = st.norm.ppf((1.-p0) * st.gamma.cdf(x, 0.669, scale = 6.85) + p0) # cv are the equality constraint non-zero values

# 	# fit a Gaussian copula -> spatial model
# 	outputfile = None               			# if you want to specify an outputfile -> os.path.join(savefolder, 'MLM.sparaest')   
	u = (st.rankdata(prec) - 0.5) / prec.shape[0]   # observations in copula (rank) space
# 	covmods = ['Mat', 'Exp', 'Sph',]     # covariance function that will be tried for the fitting
# 	ntries = 6                                  # number of tries per covariance function with random subsets
    
	print("p_xy = ", np.copy(p_xy)) # standard grid gauge coords
	print("st.rankdata(prec) = ", st.rankdata(prec))
	print("prec.shape[0] = ", prec.shape[0])
	print("u = ", u)
# 	print('stop')
# 	sys.exit()

# 	cmods = sparest.paraest_multiple_tries(np.copy(p_xy),
# 										   u,
# 										   ntries = [ntries, ntries],
# 										   n_in_subset = 3,               # number of values in subsets
# 										   neighbourhood = 'nearest',     # subset search algorithm
# 										   covmods = covmods,             # covariance functions
# 										   outputfile = outputfile)       # store all fitted models in an output file

# 	# take the copula model with the highest likelihood
# 	# reconstruct from parameter array
# 	likelihood = -666
# 	for model in range(len(cmods)):
# 		for tries in range(ntries):
# 			if cmods[model][tries][1]*-1. > likelihood:
# 				likelihood = cmods[model][tries][1]*-1.
# 				cmod = '0.01 Nug(0.0) + 0.99 %s(%1.3f)'%(covmods[model], cmods[model][tries][0][0]) # last part (cmods) ~ range parameter
# 				if covmods[model] == 'Mat':
# 					cmod += '^%1.3f'%(cmods[model][tries][0][1])

	cmod = '0.01 Nug(0.58) + 0.99 Exp(40)%s(%1.3f)' # parameters from correlation Exp curve fit (Steps5n6_corr_gauges_daily_ts_unijit)

	print("cmod = ", cmod)
# 	print("cmods = ", cmods)
# 	print("len(cmods) = ", len(cmods))
# 	print('stop')
# 	sys.exit()

	# SIMULATION USING RMWSPy
	#------------------------
	# number of conditional fields to be simulated
	nfields = 20

	# marginal distribution variables
	marginal = {}
	marginal['p0'] = p0

	# initialize CMLModel
	my_CMLModel = CMLModel(mwl_prec, marginal, mwl_integrals)

	# initialize Random Mixing Whittaker-Shannon
	CS = RMWS(my_CMLModel,
			 domainsize = (ysize, xsize),
			 covmod = cmod,
			 nFields = nfields,
			 cp = cp,
			 cv = cv,
			 le_cp = lecp,
			 le_cv = lecv,
			 optmethod = 'no_nl_constraints',
			 minObj = 0.4,
			 maxbadcount= 20,    
			 maxiter = 300,
			 )

	print("cp ~ equality constraint coords (those with zon-zero P) = ", cp)
	print("cv ~ equality constraint (non-zero) values = ", cv)
# 	print('stop')
# 	sys.exit()

    
	# run RMWS
	CS()

	# POST-PROCESSING
	#----------------
	# backtransform simulated fields to original data space
	f_prec_fields = st.norm.cdf(CS.finalFields)

	mp0f = f_prec_fields <= p0
	f_prec_fields[mp0f] = 0.0

	f_prec_fields[~mp0f] = (f_prec_fields[~mp0f]-p0)/(1.-p0)
    
	f_prec_fields[~mp0f] = st.gamma.ppf(f_prec_fields[~mp0f], 0.669, scale = 6.85)
    
	# save simulated precipitation fields
	np.save('sim_precfields_tstp=%i.npy'%tstep, f_prec_fields)

	# create box plot for simulated MWL values
	mwldict = {}
	for i in range(mwl_prec.shape[0]):
		mwldict[i] = []
		coords = mwl_integrals[i]

		for j in range(nfields):
			mwldict[i].append(f_prec_fields[j, coords[:,0], coords[:,1]].mean())

	boxlist = []
	for i in range(mwl_prec.shape[0]):
		boxlist.append(mwldict[i])
	x = np.arange(1, mwl_prec.shape[0]+1)


	# random index for plotting single realization
	rix = np.random.randint(0, f_prec_fields.shape[0], 1)

	if paperstyle_plot:
		import plot_paper
		plot_paper.plot_pp(datafolder, p_data, mwl_data, f_prec_fields, mwl_prec, boxlist, x, f_prec_fields.shape[0], tstep, rix)
	else:
		# basic plots
		# box plot
		plt.figure(figsize=(12,5))
		plt.boxplot(boxlist)
		plt.plot(x, mwl_prec, 'x', c='red')
# 		plt.savefig(r'boxplot_mwl_tstp=%i.png'%tstep, dpi=150)
		plt.savefig(r'boxplot_mwl_tstp=%i.png'%tstep, dpi=300)
		plt.clf()
		plt.close()

		# plot single realization
		plt.figure()
		plt.imshow(f_prec_fields[rix[0]], origin='lower', interpolation='nearest', cmap='jet')
		plt.plot(p_xy[:,1],p_xy[:,0],'x',c='black')
		for i in range(len(mwl_integrals)):
			plt.plot(mwl_integrals[i][:,1], mwl_integrals[i][:,0], '.', c='green')
		plt.plot(mwl_xy[:,1], mwl_xy[:,0], '.', c='red')
		plt.plot(mwl_xy[:,3], mwl_xy[:,2], '.', c='blue')
		plt.colorbar()
		plt.savefig(r'prec_field_tstp=%i.png'%tstep)
		plt.clf()
		plt.close()

		# plot mean field and standard deviation field
		meanfield = np.mean(f_prec_fields, axis=0)
		stdfield = np.std(f_prec_fields, axis=0)
		fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
		img1 = axs[0].imshow(meanfield,
							 origin='lower',
							 interpolation='nearest',
							 cmap='Blues')
		axs[0].plot(p_xy[:,1],p_xy[:,0],'x',c='black')
		for i in range(len(mwl_integrals)):
			axs[0].plot(mwl_integrals[i][:,1], mwl_integrals[i][:,0], '.', c='green')
		axs[0].plot(mwl_xy[:,1], mwl_xy[:,0], '.', c='red')
		axs[0].plot(mwl_xy[:,3], mwl_xy[:,2], '.', c='blue')
		axs[0].set_title("mean field")
		divider1 = make_axes_locatable(axs[0])
		cax1 = divider1.append_axes("right", size="10%", pad=0.1)
		cbar1 = plt.colorbar(img1, cax=cax1) 

		img2 = axs[1].imshow(stdfield,
							 origin='lower',
							 interpolation='nearest',
							 cmap='Reds'
							 )
		axs[1].plot(p_xy[:,1], p_xy[:,0], 'x', c='black')
		for i in range(len(mwl_integrals)):
			axs[1].plot(mwl_integrals[i][:,1], mwl_integrals[i][:,0], '.', c='green')
		axs[1].plot(mwl_xy[:,1], mwl_xy[:,0], '.', c='red')
		axs[1].plot(mwl_xy[:,3], mwl_xy[:,2], '.', c='blue')
		axs[1].set_title("standard deviation field")
		divider2 = make_axes_locatable(axs[1])
		cax2 = divider2.append_axes("right", size="10%", pad=0.1)
		cbar2 = plt.colorbar(img2, cax=cax2)
		plt.savefig(r'meanf_stdf_tstp=%i.png'%tstep, dpi=150)
		plt.clf()
		plt.close()

	end = datetime.datetime.now()

	print('time needed:', end - start)