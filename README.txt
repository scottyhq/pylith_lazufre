----------------------------------------------------------------------
PROBLEM DESCRIPTION
----------------------------------------------------------------------

These examples are modified from /hex8/step21 and expand on calculating
Greens Functions for 3D problems. Accompanying scripts plot pylith output
and demonstrate performing inversions with FEM-generated Greens Functions. 

To run a forward model (for synthetic data to test subsequent inversion):

    pylith stepXX.cfg

To generate greens functions for a particular step:

    pylith stepXX.cfg stepXX_greensfns.cfg 


To generate full output, add full_output.cfg to the end:

    pylith stepXX.cfg full_output.cfg


To change the output folder when experimenting (default=output/FULL)

    sed '-s/FULL/NEW_NAME/' full_output.cfg new_name_output.cfg


To perform inversion:

    ./invert_slip3d.py


To visualize results:

    ./plot_inv_results3d.py



----------------------------------------------------------------------
DESCRIPTION OF FILES
----------------------------------------------------------------------

pylithapp.cfg - PyLith configuration file containing parameters common to
all simulations.

step01: Benchmark for Okada 'point sill' opening
step02: 1D Uturuncu Velocity Model applied to Lazufre
step03: 3D Tomography from Spica



----------------------------------------------------------------------
DESCRIPTION OF MESH
----------------------------------------------------------------------
The mesh is 200 km x 200 km x 100 km with linear hexahedral cells that have
edges 1.0 km long.

The box spans the volume:

  -100 km <= x <= +100 km
  -100 km <= y <= +100 km
  -100 km <= z <= 0  km.

The mesh is generated using CUBIT. Journal files are included in the mesh
directory and are annotated to guide you through the GUI to replicate the
commands in the journal files should you prefer to use the GUI.

NOTE: Importing Exodus files into PyLith requires the netcdf
library. This is included in the PyLith binary distribution. If you
are compiling from the source code, you will want to use the
--enable-cubit option to turn on importing meshes from CUBIT (you must
have the netcdf library and header files installed when configuring
PyLith).

You can examine the Exodus file exported from CUBIT using the ncdump
command.


----------------------------------------
DESCRIPTION OF INDIVIDUAL FILES
----------------------------------------

box_hex8_1000m.exo

  	Exodus file containing mesh exported from Cubit.

geometry.jou

  	Cubit journal file (script) to generate solid model geometry

mesh_hex8_1000m.jou

  	Cubit journal file (script) for vertical fault

mesh_hex8_250m.jou

	finer mesh to allow more detailed material property layering

geometry_sill.jou
	buried sill, rather than horizontal plane

mesh_hex8_sill_1000m.jou

	switch fault nodes to horizontal plane, rather than vertical fault for sill problem


----------------------------------------
./spatialdb
----------------------------------------

impulses_all  
	impulses to be applied for GF generation (all fault notes)

impulses_subset
	impulses to generated for subset of fault nodes only

mat_elastic
	uniform elastic properties
	
mat_layered
	weak layer at 1-2km depth

mat_halves
	weak X<0 side of domain

right_lateral
	right lateral slip on upper central sections of fault

reverse
	reverse motion on same fault

opening
	identical to dyke-opening on hex8/step21 example
