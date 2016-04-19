#!/usr/bin/env python
"""Plot output from 3D pylith greens functions problem

Modified from examples/2d/greensfncs/strikeslip

Usage: plot_inv_results3d.py output_directory

Author: Scott Henderson (sth54@cornell.edu)
"""
from __future__ import print_function
import numpy as np
import h5py
import os
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from matplotlib.mlab import griddata



def load_h5(dataFile, field='displacement', components=(0,1,2)):
	"""get data values and coordinates from h5 file
	"""
	print('Reading {0} (components {1}) from {2}...'.format(field, components, dataFile))
	
	data = h5py.File(dataFile, "r", driver="sec2")
	dataC = data['geometry/vertices'][:]
	dataV = data['vertex_fields/{0}'.format(field)][:,:,components]
	data.close()
	
	return (dataC, dataV)



class MidpointNormalize(Normalize):
    ''' Center divergent colorbar on zero '''
    #http://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # ignoring masked values and all kinds of edge cases
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



def plot_fault_slip_magnitude(solutionFile, outputFile, lcurveFile, dampVal=0.001, components=0):
	"""Compare estimated versus true fault slip
	
	Inputs
	-------
	solutionFile	hdf5 file with forward (true) fault slip
	outputFile 		text file with fault slip (estimated) from greens function inversion
	lcurveFilew 	text file with damping params, model norm, residual norm
	
	Add quiver vectors?
	"""
	# Load files
	(solCoords, solVals) = load_h5(solutionFile, field='slip', components=component)
	penalties, modelNorms, residualNorms = np.loadtxt(lcurveFile, unpack=True)
	
	# Get predicted fault slip from inversion
	predicted = np.loadtxt(outputFile)
	predCoords = predicted[:,0:3]
	predSlip = predicted[:, 3:] #each column for different damping paramter
	
	# Get damping parameters from header of penalty file
	damp_vals = np.loadtxt('penalty_params.txt')
	

	ind, = np.nonzero(damp_vals == dampVal)[0]
	#ind = -1 #too much damping forces model length to zero (no slip='smoothest' solution, but large residual)
	title = 'Solution for dump_vals={}'.format(damp_vals[ind])
	
	# Negate fault slip if right-lateral, normal, or opening
	true = -1*solVals
	xt,yt,zt = solCoords.T * 1e-3
	
	model = predSlip[:,ind]
	xm,ym,zm = predCoords.T * 1e-3
	
	ms = 100
	fig, (ax,ax1,ax2) = plt.subplots(1,3, 
									subplot_kw=dict(aspect=1.0, adjustable='box-forced'),  #some irrelevant for inline
									sharex=True, sharey=True, figsize=(11,8.5))
	
	sc = ax.scatter(yt, zt, c=true, s=ms, cmap=plt.cm.hot) #norm=MidpointNormalize(midpoint=0)
	ax.set_title('True Fault Slip')
	ax.set_xlabel('Y [km]')
	ax.set_ylabel('Z [km]')
	cb = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, ticks=MaxNLocator(nbins=5)) #5 ticks only)
	cb.set_label('[m]')
	
	sc1 = ax1.scatter(ym, zm, c=model, s=ms, norm=sc.norm, cmap=plt.cm.hot) #norm=MidpointNormalize(midpoint=0),
	ax1.set_title('Model (beta={0})'.format(damp_vals[ind]))
	cb1 = plt.colorbar(sc1, ax=ax1, orientation='horizontal', pad=0.1,ticks=MaxNLocator(nbins=5))
	cb1.set_label('[m]')
	
	# Match coordinates (equivalent locations to within numerical accuracy)
	big = solCoords #'true'
	small = predCoords #'model'
	indMatchSmall, indMatchBig = np.array(np.all((small[:,None,:]==big[None,:,:]),axis=-1).nonzero())
	tol = 0.0001
	indMatchSmall, indMatchBig = np.array(np.all( np.abs(small[:,None,:]-big[None,:,:])<=tol ,axis=-1).nonzero())
	x,y,z = small.T
	residual = model[indMatchSmall] - true[indMatchBig]
	
	sc2 = ax2.scatter(ym, zm, c=residual, s=ms, cmap=plt.cm.bwr) #norm=MidpointNormalize(midpoint=0), 
	cb2 = plt.colorbar(sc2, ax=ax2, orientation='horizontal', pad=0.1,ticks=MaxNLocator(nbins=5))
	cb2.set_label('[m]')
	ax2.set_title('Residual')
	
	plt.suptitle(title, fontsize=16, fontweight='bold')



def plot_Lcurve(lcurveFile):
	"""model norm versus residual norm for tested penalty parameters
	"""
	# Load L-curve file
	penalties, modelNorms, residualNorms = np.loadtxt(lcurveFile, unpack=True)
	
	plt.figure(figsize=(11,8.5))
	plt.loglog(residualNorms,modelNorms,'ko-')
	plt.xlabel('|residual|')
	plt.ylabel('|model|')
	plt.title('L-Curve')
	
	# Annotate damping parameters
	for i,p in enumerate(penalties):
		plt.annotate(str(p), (residualNorms[i], modelNorms[i]) )
		
	plt.show()
	

def run_forward_problem(G,m):
	"""re-run forward problem with augmented G-matrix and estimated model slip
	"""
	(impCoords, impVals, respCoords, respVals) = getImpResp(impulseFile, responseFile)

	# Get observed displacements and observation locations.
	(dataCoords, dataVals) = getData(dataFile)
	
	# Get penalty parameters.
	penalties = numpy.loadtxt(penaltyFile, dtype=numpy.float64)
	
	# Determine matrix sizes and set up G-matrix. - 3 columns for x,y,z coord
	numParams = impVals.shape[0] 
	numObs = dataVals.shape[1] * dataVals.shape[2]  #generalized for 2D or 3D
	aMat = respVals.reshape((numParams, numObs)).transpose()
	
	
	parDiag = numpy.eye(numParams, dtype=numpy.float64)	
	penMat = penalty * parDiag
	designMat = np.vstack((aMat, penMat))
	designMatTrans = designMat.transpose()
	
	# Data vector is a flattened version of the dataVals, plus the a priori
	# values of the parameters (assumed to be zero).
	constraints = numpy.zeros(numParams, dtype=numpy.float64)
	# NOTE: 'a priori parameters values' equivalent to linear constraint, such that to fix the first fault patch slip:
	#contraints[0] = 1
	dataVec = numpy.concatenate((dataVals.flatten(), constraints))
	
	# Fault-slip (model) read from output file
	
	# Compute predicted results and residual.
	model_surface_displacements = np.dot(aMat, solution)

	return model_surface_displacements



def plot_surface_displacements(dataFile, dispLS='disp_LS.txt'):
	"""plot predicted surface displacements from inversion against real or synthetic data
	"""
	# Load surface point data
	(dataCoords, dataVals) = load_h5(dataFile)
	
	# Load displacements from inverted slip
	x,y,z,ux,uy,uz = np.loadtxt(dispLS,unpack=True)
	
	data = dataVals[0]
	#model = predicted.reshape(data.shape)
	#x,y,z = dataCoords.T * 1e-3
	x,y,z = np.array([x,y,z])*1e-3
	model = np.vstack( [ux,uy,uz] ).T
	
	ms = 200
	labels = ['X','Y','Z']
	for dof in [0,1,2]:
		fig, (ax,ax1,ax2) = plt.subplots(1,3, 
										subplot_kw=dict(aspect=1.0, adjustable='box-forced'), 
										sharex=True, sharey=True, figsize=(17,8.5))
	
		sc = ax.scatter(x,y, c=data[:,dof], s=ms, norm=MidpointNormalize(midpoint=0), cmap=plt.cm.bwr)
		ax.set_title(labels[dof] + ' Displacement')
		ax.set_xlabel('X [km]')
		ax.set_ylabel('Y [km]')
		cb = plt.colorbar(sc,ax=ax, orientation='horizontal', pad=0.1, ticks=MaxNLocator(nbins=5)) #5 ticks only)
		cb.set_label('m')
	
		sc1 = ax1.scatter(x,y, c=model[:,dof], s=ms, norm=MidpointNormalize(midpoint=0), cmap=plt.cm.bwr)
		ax1.set_title('Model')
		cb1 = plt.colorbar(sc1, ax=ax1, orientation='horizontal', pad=0.1,ticks=MaxNLocator(nbins=5))
		cb1.set_label('m')
		
		sc2 = ax2.scatter(x,y, c=data[:,dof]-model[:,dof], s=ms, norm=MidpointNormalize(midpoint=0), cmap=plt.cm.bwr)
		cb2 = plt.colorbar(sc2, ax=ax2, orientation='horizontal', pad=0.1,ticks=MaxNLocator(nbins=5))
		cb2.set_label('m')
		ax2.set_title('Residual')

	plt.show()


def plot_contours(dataFile, dispLS='disp_LS.txt'):
	"""Add contours to scatter plot
	"""
	# Load surface point data
	(dataCoords, dataVals) = load_h5(dataFile)
	
	# Load displacements from inverted slip
	x,y,z,ux,uy,uz = np.loadtxt(dispLS,unpack=True)
	
	data = dataVals[0]
	#model = predicted.reshape(data.shape)
	#x,y,z = dataCoords.T * 1e-3
	x,y,z = np.array([x,y,z])*1e-3
	model = np.vstack( [ux,uy,uz] ).T
	
	# contour grid
	npts = 100
	xi = np.linspace(x.min(), x.max(), npts)
	yi = np.linspace(y.min(), y.max(), npts)	
	
	ms = 100
	labels = ['X','Y','Z']
	for dof in [0,1,2]:
		# Contour sets
		datai = griddata(x,y,data[:,dof], xi,yi,interp='nn')
		modeli = griddata(x,y,model[:,dof], xi,yi,interp='nn')
		residuali = datai - modeli
		
		
		fig, (ax,ax1,ax2) = plt.subplots(1,3, 
										subplot_kw=dict(aspect=1.0, adjustable='box-forced'), 
										sharex=True, sharey=True, figsize=(17,8.5))
	
		cf = ax.contourf(xi,yi,datai)
		cl = ax.contour(xi,yi,datai)
		# NOTE: make sure same color norm for scatter and contours!
		#sc = ax.scatter(x,y, c=data[:,dof], s=ms, norm=MidpointNormalize(midpoint=0)) #use default Jet
		sc = ax.scatter(x,y, c='k', s=ms) #use default Jet
		ax.set_title(labels[dof] + ' Displacement')
		ax.set_xlabel('X [km]')
		ax.set_ylabel('Y [km]')
		cb = plt.colorbar(cf,ax=ax, orientation='horizontal', pad=0.1, ticks=MaxNLocator(nbins=5)) #5 ticks only)
		cb.set_label('m')
	
		cf1 = ax1.contourf(xi,yi,modeli)#, norm=cf.norm)
		cl1 = ax1.contour(xi,yi,modeli)
		#sc1 = ax1.scatter(x,y, c=model[:,dof], s=ms, norm=MidpointNormalize(midpoint=0))
		sc1 = ax1.scatter(x,y,c='k',s=ms)
		ax1.set_title('Model')
		cb1 = plt.colorbar(cf1, ax=ax1, orientation='horizontal', pad=0.1,ticks=MaxNLocator(nbins=5))
		cb1.set_label('m')
		
		cf2 = ax2.contourf(xi,yi,residuali)
		cl2 = ax2.contour(xi,yi,residuali)
		#sc2 = ax2.scatter(x,y, c=data[:,dof]-model[:,dof], s=ms, norm=MidpointNormalize(midpoint=0), cmap=plt.cm.bwr)
		sc2 = ax2.scatter(x,y,c='k',s=ms)
		cb2 = plt.colorbar(cf2, ax=ax2, orientation='horizontal', pad=0.1,ticks=MaxNLocator(nbins=5))
		cb2.set_label('m')
		ax2.set_title('Residual')

	plt.show()


def main():
	"""Read input arguments from command line, and call inverstion function  
	"""
	parser = argparse.ArgumentParser(description="Plot results of FEM-genereated green's function inversion against data")
	parser.add_argument('outdir', help='pylith output directory')
	parser.add_argument("component", type=int, help="Slip component (0=X, 1=Y, 2=Z)")
	
	parser.add_argument("-i", "--impulses", action="store", type=str, help="HDF5 file with fault GF info")
	parser.add_argument("-r", "--responses", action="store", type=str,  help="HDF5 file with GF responses")
	parser.add_argument("-d", "--data", action="store", type=str, help="HDF5 file with surface point displacement data")
	parser.add_argument("-l", "--lcurve", action="store", type=str, help="text file with residuals for L-curve")
	parser.add_argument("-f", "--forward", action="store", type=str, help="synthetic slip solution from forward model")
	parser.add_argument("-o", "--output", action="store", type=str, help="text file with estimated slip")
	parser.add_argument('--version', action='version', version='0.1')
	
	parser.set_defaults(component = 0,
						impulses = 'fault-inverse.h5',
						responses = 'points-inverse.h5',
						data = 'points-forward.h5',
						forward = 'fault-forward.h5',
						lcurve = 'Lcurve.txt',
						output = 'slip_inverted.txt')
	
	# Read from command line
	#args = parser.parse_args()
	# Testing defaults (%run invert_slip3d) from ipython terminal
	#args = parser.parse_args(['output/step01', 0])
	args = parser.parse_args(['output/step04', '2'])
	
	for f in [args.impulses, args.responses, args.data, args.output, args.forward]:
		path = os.path.join(args.outdir, f) 
		if not os.path.isfile( path ):
			raise ValueError("'{0}' does not exist".format(path))
	
	pwd = os.getcwd()
	os.chdir(args.outdir) 
	
	# Plots (comment ones you don't want to see)
	#plot_Lcurve(args.lcurve)
	#plot_surface_displacements(args.data)
	plot_contours(args.data)
	#plot_fault_slip_magnitude(args.forward, args.output, args.lcurve, dampVal=0.001)
	
	os.chdir(pwd)


if __name__ == '__main__':
	main()