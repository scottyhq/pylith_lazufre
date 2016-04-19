# -*- coding: utf-8 -*-
"""
Facilitation of creating spatialdb files to specify nodal displacements on fault

NOTE: for now just simple distributions of uniform opening... but could also
write values at specific locations and then use pylith linear interpolation

@author: scott
"""

import numpy as np
import pandas as pd

# SETTINGS HERE
savename = 'sill_opening_30x15km.spatialdb'
length = 30e3
width = 15e3
depth = -10e3

# Prescribe displacement at every node in mesh
Le = 2 #km
xx = np.arange(-100,101,Le)*1e3 #Note 201 critical because includes exactly 0
yy = np.arange(-100,101,Le)*1e3
X,Y = np.meshgrid(xx,yy)

U = np.zeros_like(xx)
slip = (X >=-width/2) & (X <=width/2) & (Y >=-length/2) & (X <=length/2)
U[slip] = 1

header = """//Template for SimpleDB 3D material properties from tomography
#SPATIAL.ascii 1
SimpleDB {
  num-values = 3
  value-names =  left-lateral-slip  reverse-slip  fault-opening
  value-units =  m m m
  num-locs = NLOC 
  data-dim = 2
  space-dim = 3
  cs-data = cartesian {
    to-meters = 1.0
    space-dim = 3
  }
}
// Columns are
// (1) x coordinate (km)
// (2) y coordinate (km)
// (3) z coordinate (km)
// (4) left-lateral-slip (m) (right-lateral is negative)
// (5) reverse-slip (m)
// (6) fault-opening (m)

"""

df = pd.DataFrame(dict(x=X.flatten(), y=Y.flatten(), u=U.flatten()))
# Fill in missing values
df['Z'] = 0

with open('savename', 'w') as f:
    f.write(header.replace('NLOC',str(dfS.shape[0])))
    dfW.to_csv(f,columns=['x','y','z','z','z','u'], 
          index=False,
          #header=header.replace('NLOC',str(dfS.shape[0])), #doesn't work
          header=False,
          float_format='%5.1f', #cleaner presentation
          sep=' ')
          
          
print 'Saved' + savename 


