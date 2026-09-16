## Load libraries
import openpmd_api as io
import numpy as np
import re

## Define functions
def open_maps(address, skiplines=4, minline=0, maxline=1, lenline=2):
    ## Open files and read data
    with open(address, 'r') as file:
        lines = file.readlines()
    data = np.loadtxt(lines[skiplines:])
    r, z, EBrphi, EBz = data[:,0], data[:,1], data[:,2], data[:,3]
    ## Evaluate field limits and mesh size
    minvals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[minline]) # Find minimum values
    Rmin, Xmin = float(minvals[0]), float(minvals[1])
    maxvals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[maxline]) # Find maximum values
    Rmax, Xmax = float(maxvals[0]), float(maxvals[1])
    lenvals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[lenline]) # Find number of points
    Rlen, Xlen = int(lenvals[0])+1, int(lenvals[1])+1
    spacing = r[1] - r[0]
    ## Reshape data
    EBrphi_arr = np.asarray([EBrphi.reshape(Xlen, Rlen).T])
    EBz_arr = np.asarray([EBz.reshape(Xlen, Rlen).T])
    return EBrphi_arr, EBz_arr, Rlen, Xlen, spacing, Rmin, Rmax, Xmin, Xmax

## Constants
num_nodes = 1
address = "convert_maps/rz_geometry/input/"  
    
'''
## Example usage
# number of grid points
# nr*nz = the number of lines in B*_rz.txt
num_nodes = 1
nr = 47
nz = 47
# read data
Br_data = np.loadtxt('Br_rz.txt').reshape(num_nodes, nr, nz)
Bz_data = np.loadtxt('Bz_rz.txt').reshape(num_nodes, nr, nz)
'''

'''
## Upload field maps
Br_static, Bz_static, nr, nz, spacing, rmin, rmax, zmin, zmax = open_maps(f"{address}ECRIPAC_solenoid_initial.txt")
print(f"Static field maps loaded: Br_static shape = {Br_static.shape}, Bz_static shape = {Bz_static.shape}, nr = {nr}, nz = {nz}, spacing = {spacing}, rmin = {rmin}, rmax = {rmax}, zmin = {zmin}, zmax = {zmax}")

# create openpmd file for static fields
name = f"{address}ECRIPAC_staticfields_thetaMode.h5"
series = io.Series(name,io.Access.create)
# only 1 iteratiion needed
it = series.iterations[1]

# set meta information
B = it.meshes["B"]
B.grid_spacing = np.array([spacing, spacing])
B.grid_global_offset = [rmin, zmin]
B.axis_labels = ['r', 'z']
B.geometry = io.Geometry.thetaMode
B.geometry_parameters = "m=" + str(num_nodes) + ";imag=+"

B.unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.I: -1,
    io.Unit_Dimension.T: -2
}

# label
B_r = B["r"]
B_r.position = [0,0,0]
B_t = B["t"]
B_t.position = [0,0,0]
B_z = B["z"]
B_z.position = [0,0,0]

dataset = io.Dataset(Br_static.dtype, Br_static.shape)
B_r.reset_dataset(dataset)
B_t.reset_dataset(dataset)
B_z.reset_dataset(dataset)

B_r.store_chunk(Br_static)
B_t.make_constant(0.0)
B_z.store_chunk(Bz_static)

series.flush()
del series
'''
#'''
## Upload pulsed field maps
Br_pulsed, Bz_pulsed, nr_p, nz_p, spacing_p, rmin_p, rmax_p, zmin_p, zmax_p = open_maps(f"{address}ECRIPAC_solenoid_pulsed.txt")
Et_induced, Ez_induced, nr_p, nz_p, spacing_p, rmin_p, rmax_p, zmin_p, zmax_p = open_maps(f"{address}ECRIPAC_induced_field.txt")

## Create openpmd file for pulsed fields
name = f"{address}ECRIPAC_pulsedfields_thetaMode.h5"
series = io.Series(name,io.Access.create)
# only 1 iteratiion needed
it = series.iterations[1]

# set meta information
B = it.meshes["B"]
B.grid_spacing = np.array([spacing_p, spacing_p])
B.grid_global_offset = [rmin_p, zmin_p]
B.axis_labels = ['r', 'z']
B.geometry = io.Geometry.thetaMode
B.geometry_parameters = "m=" + str(num_nodes) + ";imag=+"

B.unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.I: -1,
    io.Unit_Dimension.T: -2
}

# label
B_r = B["r"]
B_r.position = [0,0,0]
B_t = B["t"]
B_t.position = [0,0,0]
B_z = B["z"]
B_z.position = [0,0,0]

dataset = io.Dataset(Br_pulsed.dtype, Br_pulsed.shape)
B_r.reset_dataset(dataset)
B_t.reset_dataset(dataset)
B_z.reset_dataset(dataset)

B_r.store_chunk(Br_pulsed)
B_t.make_constant(0.0)
B_z.store_chunk(Bz_pulsed)

# set meta information
E = it.meshes["E"]
E.grid_spacing = np.array([spacing_p, spacing_p])
E.grid_global_offset = [rmin_p, zmin_p]
E.axis_labels = ['r', 'z']
E.geometry = io.Geometry.thetaMode
E.geometry_parameters = "m=" + str(num_nodes) + ";imag=+"
E.unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.L:  1,
    io.Unit_Dimension.I: -1,
    io.Unit_Dimension.T: -3
}

# label
E_r = E["r"]
E_r.position = [0,0,0]
E_t = E["t"]
E_t.position = [0,0,0]
E_z = E["z"]
E_z.position = [0,0,0]

dataset = io.Dataset(Br_pulsed.dtype, Br_pulsed.shape)
E_r.reset_dataset(dataset)
E_t.reset_dataset(dataset)
E_z.reset_dataset(dataset)

E_r.make_constant(0.0)
E_t.store_chunk(Et_induced)
E_z.make_constant(0.0)

series.flush()
del series
#'''