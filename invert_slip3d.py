#!/usr/bin/env python
"""Damped least squares inversion for 3D okada-type pylith problems

Modified from examples/2d/greensfncs/strikeslip

Usage: invert_greensfncs_3d.py output_directory

Author: Scott Henderson (sth54@cornell.edu)

Wishlist:
* Try a different penalty function for the inversion. One simple option
   would be a 1D Laplacian approximation (sets of values of [1, -2, 1])
   centered along the diagonal.
   
* SVD Inversion

* find non-zero components automatically, instead of specifying on command line

* save inversion outputs as h5 (then could use paraview or matlab) or ascii
"""

# The code requires the numpy and h5py packages.
from __future__ import print_function
import numpy as np
import h5py
import os
import argparse


def load_h5(h5File, field='displacement', components=(0,1,2)):
  """get data values and coordinates from h5 file
  """
  print('Reading {0} (components {1}) from {2}...'.format(field, components, h5File))
  
  data = h5py.File(h5File, "r", driver="sec2")
  dataC = data['geometry/vertices'][:]
  dataV = data['vertex_fields/{0}'.format(field)][:,:,components]
  data.close()
  
  return (dataC, dataV)



def cull_impulses(impC, impV):
  """extract non-zero fault impulses for use in inversion
  
  Only vertices with non zero impulses are included in inversion matrices.
  Extract these coordinates, impulse values, and corresponding responses.
  """
  impInds = np.nonzero(impV != 0.0) 
  impCUsed = impC[impInds[1]] 
  impVUsed = impV[impInds[0], impInds[1]]

  return  (impCUsed, impVUsed)


def invert_ls(impulseFile, responseFile, dataFile, penaltyFile, outputFile, components=0):
  """regular least squares, no damping or linear constraints
  """
  print('Least Squares Inversion\n==========\n')
  # Load data - displacements at surface points
  (dataCoords, dataVals) = load_h5(dataFile)
  
  # Impulse comp=0 should match dof in step01_greensfns.cfg
  data = load_h5(impulseFile, 'slip', components=components)
  (impCoords, impVals) = cull_impulses(*data)
  (respCoords, respVals) = load_h5(responseFile)
  
  # Set up inversion
  numParams = impVals.shape[0] 
  numObs = dataVals.shape[1] * dataVals.shape[2]  
  G = respVals.reshape((numParams, numObs)).transpose()
  dataVec = dataVals.flatten()
  
  # Initialize Solution Arrays
  invResults = np.zeros((numParams, 4)) #column for each coordinate, plus slip column
  invResults[:,0:3] = impCoords
  # note, fixed below, save to 49x6 array (just stack on columns)
  #surfaceResults = np.zeros((dataCoords.shape[0], 6))
  #surfaceResults[:,0:3] = respCoords # surface solution only found where GF responses were determined, not necessary the same as output points on surface (user could use different files,,, unsure why you would though)
  
  # Form generalized inverse matrix.
  genInv = np.dot( np.linalg.inv( np.dot(G.T, G) ), G.T)
  
  # Slip solution is matrix-vector product of generalized inverse with data vector.
  modelSlip = np.dot(genInv, dataVec)
  invResults[:, -1] = modelSlip
  
  # Surface displacements from model slip
  surfaceDisp = np.dot(G, modelSlip)
  #surfaceResults[:, -3:] = surfaceDisp
  surfaceResults = np.hstack( [respCoords, surfaceDisp.reshape(respCoords.shape)] )
  residual = dataVals.flatten() - surfaceDisp
  
  # Keep track of norms for L-curve.
  residualNorm = np.linalg.norm(residual)
  modelNorm = np.linalg.norm(modelSlip)
  print("residual norm = {0}\nmodel norm = {1}".format(residualNorm, modelNorm))

  # Output is coordinates of slip vertex and single component of slip for each damping parameter
  np.savetxt('slip_LS.txt',
             invResults,
             header="X_Coord Y_Coord Z_Coord {}_Slip".format(components),
             fmt="%14.6e")
  
  # Also save surface displacements from model slip, since we have them
  # NOTE: could save this again as an h5 file?
  # NOTE: instead of sum of squared residual, could also save residual at every output point.
  np.savetxt('disp_LS.txt',
             surfaceResults,
             header="X_Coord Y_Coord Z_Coord Ux Uy Uz",
             fmt="%14.6e")
  
  print('DONE!, saved results to slip_LS.txt and disp_LS.txt')
  


def invert_dls(impulseFile, responseFile, dataFile, penaltyFile, outputFile, components=0):
  """Damped least squares inversion with optional linear constraints
  """
  print('Damped Least Squares Inversion with Linear Constraints\n==========\n')
  
  # Load data - displacements at surface points
  (dataCoords, dataVals) = load_h5(dataFile)
  
  # Impulse comp=0 should match dof in step01_greensfns.cfg
  data = load_h5(impulseFile, 'slip', components=components)
  (impCoords, impVals) = cull_impulses(*data)
  (respCoords, respVals) = load_h5(responseFile)
  
  # Get penalty parameters
  penalties = np.loadtxt(penaltyFile, dtype=np.float64)
  
  # Set up inversion
  numParams = impVals.shape[0] 
  numObs = dataVals.shape[1] * dataVals.shape[2]  
  aMat = respVals.reshape((numParams, numObs)).transpose()
  
  # Create diagonal matrix to use as damping parameters
  parDiag = np.eye(numParams, dtype=np.float64)
  # Data vector is a flattened version of the dataVals, plus the a priori values of the parameters (constraints)
  constraints = np.zeros(numParams, dtype=np.float64)
  # NOTE: to fix the first fault patch slip:
  #contraints[0] = 1
  dataVec = np.concatenate((dataVals.flatten(), constraints))
  
  # Determine number of inversions and create empty array to hold results.
  numInv = penalties.shape[0] 
  invResults = np.zeros((numParams, 3 + numInv))
  invResults[:,0:3] = impCoords
  head = "# X_Coord Y_Coord Z_Coord"
  
  # Do inversion
  # ---------
  modelNorms = []
  residualNorms = []
  print('penalty_parameter   residual_norm')
  for inversion in range(numInv):
      penalty = penalties[inversion]
      head += " Penalty=%g" % penalty
      
      # Scale diagonal by penalty parameter, and stack G-matrix with penalty matrix.
      penMat = penalty * parDiag
      designMat = np.vstack((aMat, penMat))
      designMatTrans = designMat.transpose()
      
      # Form generalized inverse matrix.
      genInv = np.dot(np.linalg.inv(np.dot(designMatTrans, designMat)), designMatTrans)
      
      # Solution is matrix-vector product of generalized inverse with data vector.
      solution = np.dot(genInv, dataVec)
      invResults[:, 3 + inversion] = solution
      
      # Compute predicted results and residual.
      predicted = np.dot(aMat, solution)
      residual = dataVals.flatten() - predicted
      
      # Keep track of norms for L-curve.
      residualNorm = np.linalg.norm(residual)
      modelNorm = np.linalg.norm(solution)
      residualNorms.append(residualNorm)
      modelNorms.append(modelNorm)
      print("{0}\t{1}".format(penalty,residualNorm))
      
  head += "\n"


  # Output for L-curve
  lcurveFile ='Lcurve.txt'
  np.savetxt(lcurveFile,
             np.array([penalties, modelNorms, residualNorms]).T,
             header='penalty\tmodel_norm\tresidual_norm',
             fmt="%14.6e")

  # Output is coordinates of slip vertex and single component of slip for each damping parameter
  np.savetxt(outputFile,
             invResults,
             header=head,
             fmt="%14.6e")
  
  print('DONE!, saved results to {0} and {1}'.format(outputFile,lcurveFile))



def main():
  """Read input arguments from command line, and call inverstion function
  """
  parser = argparse.ArgumentParser(description="perform inversion with FEM generated green's functions")
  parser.add_argument('outdir', help='pylith output directory')
  parser.add_argument("component", type=int, help="Slip component (0=X, 1=Y, 2=Z)")
  
  parser.add_argument("-i", "--impulses", help="HDF5 file with fault GF info") #action="store", type=str,
  parser.add_argument("-r", "--responses", help="HDF5 file with GF responses")
  parser.add_argument("-d", "--data", help="HDF5 file with surface point displacement data")
  parser.add_argument("-p", "--penalty", help="text file with penalty parameters")
  parser.add_argument("-o", "--output", help="text file with estimated slip")

  parser.add_argument('--version', action='version', version='0.1')
  
  parser.set_defaults(component = 0,
                      impulses = 'fault-inverse.h5',
                      responses = 'points-inverse.h5',
                      data = 'points-forward.h5',
                      penalty = 'penalty_params.txt',
                      output = 'slip_inverted.txt')
  
  # Read from command line
  #args = parser.parse_args()
  # Testing defaults (%run invert_slip3d) from ipython terminal
  #args = parser.parse_args(['output/step01', '0'])
  args = parser.parse_args(['output/step04', '2'])
  
  for f in [args.impulses, args.responses, args.data, args.penalty]:
      path = os.path.join(args.outdir, f) 
      if not os.path.isfile( path ):
          raise ValueError("'{0}' does not exist".format(path))
  
  pwd = os.getcwd()
  os.chdir(args.outdir) 
  invert_ls(args.impulses, args.responses, args.data, args.penalty, args.output, args.component)
  #invert_dls(args.impulses, args.responses, args.data, args.penalty, args.output, args.component)
  os.chdir(pwd)


if __name__ == '__main__':
  main()