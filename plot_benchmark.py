# -*- coding: utf-8 -*-
"""
Benchmark Lazufre Mesh against Okada & Fialko 'Crack' solutions

Created on Sun Dec 28 20:11:05 2014

@author: scott
"""
import roipy as rp
import pylithTools as pt
import numpy as np
import matplotlib.pyplot as plt

# Compare to analytic solutions
import roipy.models.okada as o
import roipy.models.mogi as m

# Plot Okada Solution
n = 1 # analytic grid spacing [km]
xx = np.arange(-100,101,n)*1e3 #Note 201 critical because includes exactly 0
yy = np.arange(-100,101,n)*1e3
X,Y = np.meshgrid(xx,yy)

# NOTE: no young's modulus specified?...
# NOTE: Length and Width should be changed to match mesh size. In particular, 
# for single 'node' opening, patch tapers to zero at adjacent nodes, so 
# for 2km element length, opening is more like pyramid w/ base length=2km
xcen=0
ycen=0

U = -0.1 # [m] negative --> opening
# x,y are the observation points
d = 10e3 #[m] # d is depth (positive down)
nu = 0.25 # nu is Poisson ratio
delta = 0.0 # [degrees] delta is dip angle, 90.0 exactly might cause numerical issues?
strike = 0.0 # [degrees] counter clockwise from north

# for 4km mesh
#length = 4e3 #[m] # len,W are the fault length and width, resp. 
#width = 4e3 #[m] #Profile Shifts w/ increasing width
# NOTE: that code is set up so that patch is aligned on western side
# by length is automatically centered
#ycen = -length/2
#xcen = -width/2
Le = 2e3 #element length [m]
nodes = 5 # number of nodes in sill w/ prescribed opening of U (assume symmetrical)
length = Le*nodes
width = Le*nodes
xcen = -width/2

fault_type = 3 # fault_type is 1 2 3 for strike, dip, and opening
tp = np.zeros_like(X) # tp is the optional topo vector (size(x))

# Run the model
#calc_okada(U,x,y,nu,delta,d,length,W,fault_type,strike,tp)
ux,uy,uz = o.calc_okada(U,X-xcen,Y-ycen,nu,delta,d,length,width,fault_type,strike,tp)  
print 'Okada Uz_max={:.3f} [m]'.format(uz.max()) 
print 'Okada Ur_max/Uz_max={:.2f}'.format(ux.max() / uz.max()) 

# make up viewing geometry:
inc = 22.0
ald = -77.0
wavelength = 5.66
data = np.dstack([ux, uy, uz])
cart2los = o.get_cart2los(inc,ald,X) #NOTE: not sure why 'x' being passed...
los = np.sum(data * cart2los, axis=2)

# Create dictionary of parameters for plotting routine
params=dict(xcen=xcen,ycen=ycen,U=U,d=d,nu=nu,delta=delta,strike=strike,length=length,width=width,fault_type=fault_type,inc=inc,ald=ald,wavelength=wavelength)
o.plot_components(X,Y,ux,uy,uz,los,params)
o.plot_los(X,Y,ux,uy,uz,los,params)
ind = np.nonzero(xx==0)[0][0] #easier way?
#ind = 100
o.plot_profile(ind,ux,uy,uz,los) #NOTE: have to change 100 if meshgrid changed...


# Vertical and Radial Profile
# NOTE: could compare 'point' crack and 'point' sphere in plot...
# NOTE: re-run FEM with just profile output (with higher density (interpolated points)

# For MOGI Solution use order of magnitude dV/yr 0.01 km^3/yr, but plot as normalized solution...
#dV=0.01*1e9 #order of magnitude
# NOTE: check volume equivalence for mogi or sill opening:
# NOTE: no xcen shift b/c code already set-up to be centered
dV = -1*length*width*U #NOTE: opposite sign convention, units m^3
print '%e [km^3]' % (dV*1e-9)
urM,uzM = m.calc_mogi(X,Y-ycen,dV=dV,d=10e3,nu=0.25)
print 'Mogi Uz_max={:.2f}'.format(uzM.max())
print 'Mogi Ur_max/Uz_max={:.2f}'.format(urM.max() / uzM.max())

# Compare analytical solutions with FEM output:
path = '/Volumes/OptiHDD/data/pylith/3d/lazufre'
#nodes,data,elmts = pt.util.load_h5('output/benchmark_convergence/step01_5km/points-forward.h5')
#nodes,data,elmts = pt.util.load_h5('output/benchmark_convergence/step01_4km/points-forward.h5')
#Note - profile, for benchmark, could output at exactly same points (not nec node points (interpolated))
# NOTE also, 2.5km means that nodes don't necessarily line up along x=0
#nodes,data,elmts = pt.util.load_h5('output/benchmark_convergence/step01_2.5km/points-forward.h5') 

#nodes,data,elmts = pt.util.load_h5('output/benchmark_convergence/step01_2.5km/points-profile.h5') 
#nodes,data,elmts = pt.util.load_h5('output/benchmark_convergence/step01_2.5km/points-profile-refined.h5') 
#nodes,data,elmts = pt.util.load_h5('output/benchmark_convergence/step01_2km/points-profile.h5') 

# NOTE: instead, use soft-link to make specific directory active
#nodes,data,elmts = pt.util.load_h5('output/step01/points-forward.h5') 
nodes,data,elmts = pt.util.load_h5('output/step01/points-profile.h5') 
#nodes,data,elmts = pt.util.load_h5('output/step01_delnegro2009_2km_sill20x20km/points-profile.h5')

# Plot x-axis FEM profile, raw output, no normalization:
x,y,z = nodes.T
uxF,uyF,uzF = data[0].T
ind = (nodes[:,0] >= 0) & (nodes[:,1] == 0)
print 'FEM Uz_max={:.2f}'.format(uzF.max())
print 'FEM Ur_max/Uz_max={:.2f}'.format(uxF.max() / uzF.max())

plt.style.use('presentation')
plt.figure(figsize=(11,8.5))
plt.plot(x[ind]/1e3, uzF[ind]*1e2, 'ko', ls='none', ms=5, zorder=10, label='uz_fem')
plt.plot(x[ind]/1e3, uxF[ind]*1e2, 'ks', ls='none', ms=5, zorder=10, label='ur_fem')

xpos = (X >=0) & (Y == 0)
plt.plot(X[xpos]/1e3, uzM[xpos]*1e2,  'b-', lw=1, label='Mogi')
plt.plot(X[xpos]/1e3, urM[xpos]*1e2,  'b--', lw=1)

plt.plot(X[xpos]/1e3, uz[xpos]*1e2, 'g-', label='Okada')
plt.plot(X[xpos]/1e3, ux[xpos]*1e2, 'g--')

plt.legend(loc='upper right')
#plt.title('FEM Ur/Uz={:.3f}'.format(uxF.max() / uzF.max())) #Just print out
plt.ylabel('Displacement [cm]')
plt.xlabel('Distance [km]')
plt.axhline(color='k',lw=1,ls='--')

plt.savefig('lazufre_fem_benchmark.pdf',bbox_inches='tight')

# Plot Normalized Displacements:
plt.style.use('presentation')
plt.figure(figsize=(11,8.5))
plt.plot(x[ind]/1e3, uzF[ind]/uzF.max(), 'ko-', ms=6, label='FEM')
plt.plot(x[ind]/1e3, uxF[ind]/uzF.max(), 'ks--', ms=6)

xpos = (X >=0) & (Y == 0)
plt.plot(X[xpos]/1e3, uzM[xpos]/uzM.max(), 'b-', label='Mogi')
plt.plot(X[xpos]/1e3, urM[xpos]/uzM.max(), 'b--')

plt.plot(X[xpos]/1e3, uz[xpos]/uz.max(), 'g-', label='Okada')
plt.plot(X[xpos]/1e3, ux[xpos]/uz.max(), 'g--')

plt.legend(loc='upper right')
#plt.title('Ur/Uz={:.2f}'.format(ux.max() / uz.max()))
plt.ylabel('Normalized Displacement')
plt.xlabel('Distance [km]')
plt.axhline(color='k',lw=1,ls='--')

plt.savefig('lazufre_fem_benchmark_normalized.pdf',bbox_inches='tight')

plt.show()
# NOTE: could calculate residual at FEM node points to show convergence of mesh


