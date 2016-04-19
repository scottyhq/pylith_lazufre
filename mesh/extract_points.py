#!/usr/bin/env python
'''
Read EXODUSII mesh and output ascii text points at surface nodes

Usage: ./extract_points.py mesh.exo nodesetID skip

For the nodeset ID number, look int the cubit.jou file:
e.g. 'nodeset 17 group face_zpos'. If for some reason, you
want to coordinates of a side, other id's should work.

Example ./extract_poins.py box_hex8_2km.exo 17 5
--> Gets surface nodes, writes every 5th location to file
'''
import sys
import netCDF4
import numpy as np

# Read input from command line
meshFile = sys.argv[1]
nodesetID = int(sys.argv[2])
n = int(sys.argv[3])

# Defaults for ascii file Output
outName = meshFile.replace('.exo','_points.txt')
fmt = '%12.4f'
header = 'Automatically generated with gen_points.py\nX\tY\tZ'

# Read Coordinates in EXODUSII Mesh file
exodus = netCDF4.Dataset(meshFile, 'r')
# NOTE: .copy() critical to re-ordering in memory for later sorting by column
#http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
coords = exodus.variables['coord'][:].transpose()

#NOTE: careful of python index starting at 0
# a, = np.nonzero(ids == 17)[0] #kept as numpy array
ids = list(exodus.variables['ns_prop1'][:])
nsID = ids.index(nodesetID) + 1
# corresponding names:
#print [''.join(x) for x in exodus.variables['ns_names'][:]]
nsName = 'node_ns{}'.format(nsID)
nodes = exodus.variables[nsName][:] -1
surface = coords[nodes]

# Sort by x-coordinate for readability
#surface.sort(axis=0) #gotcha,,, works elementwise
#output = surface[surface[:,0].argsort()] # dirty method to sort by 1st column
# sort by 1st, then 2nd column by converting to record array:
#surface.view('f8,f8,f8').sort(order=['f0','f1'], axis=0)
# Confusing!!! Perhaps use pandas, but here is the short answer, NOTE: first column
# to sort on comes last in list!
ind = np.lexsort( (surface[:,1], surface[:,0]) )
output = surface[ind]

# Only output every n'th point:
#n=5
output = output[::n]

# Export to file
np.savetxt(outName, output, fmt=fmt, header=header)

print 'Saved ', outName
