import openpmd_api as io
import numpy as np
import scipy.constants as sciconst
import scipy.special as scisp
import sys
sys.path.append("/home/cernuschi/coding/smilei/ecripac-pic/functions/")
import fieldmaps as fm
import TEMfields as tf

## Import field maps function for 3d field
address = "convert_maps/3d_geometry/"
Bstatic_path = f"{address}input/ECRIPAC_solenoid_initial.txt"
Bpulsed_path = f"{address}input/ECRIPAC_solenoid_pulsed.txt"
Einduced_path = f"{address}input/ECRIPAC_induced_field.txt"
fr_pul_SI = 200000 # Frequency of the pulsed field [Hz]
omega_pul_SI = 2*np.pi*fr_pul_SI  # Pulsed field angular frequency [rad/s]
phase_pul = 1./fr_pul_SI/4  # Phase of the pulsed field [code units]
Bstatic = fm.EBmapsAM(4, 0, 1, 2, Bstatic_path, 1, 0) # Poisson static B field (33, 27, 28, 29 for poisson generated and 1e-4 norm)
Bpulsed = fm.EBmapsAM(4, 0, 1, 2, Bpulsed_path, 1, 1, omega=omega_pul_SI, ph=phase_pul) # COMSOL pulsed B field
Einduced = fm.EBmapsAM(4, 0, 1, 2, Einduced_path, 0, 1, omega=omega_pul_SI, ph=phase_pul) # COMSOL induced E field

def z_maps(x, y, z):
    r = (x**2 + y**2)**0.5
    ir, irp, ix, ixp, dr, idr, dx, idx = Bstatic.find_grid(z, r)
    B_static = Bstatic.EBarr[Bstatic.offset+ix*Bstatic.Rlen+ir]*dr*dx + Bstatic.EBarr[Bstatic.offset+ix*Bstatic.Rlen+irp]*idr*dx + Bstatic.EBarr[Bstatic.offset+ixp*Bstatic.Rlen+ir]*dr*idx + Bstatic.EBarr[Bstatic.offset+ixp*Bstatic.Rlen+irp]*idr*idx # Bilinear interpolation for axial field component
    B_pulsed = Bpulsed.EBarr[Bpulsed.offset+ix*Bpulsed.Rlen+ir]*dr*dx + Bpulsed.EBarr[Bpulsed.offset+ix*Bpulsed.Rlen+irp]*idr*dx + Bpulsed.EBarr[Bpulsed.offset+ixp*Bpulsed.Rlen+ir]*dr*idx + Bpulsed.EBarr[Bpulsed.offset+ixp*Bpulsed.Rlen+irp]*idr*idx # Bilinear interpolation for axial field component
    return B_static, B_pulsed

def x_maps(x, y, z):
    r = (x**2 + y**2)**0.5
    ir, irp, ix, ixp, dr, idr, dx, idx = Bstatic.find_grid(z, r)
    B_static = Bstatic.EBarr[ix*Bstatic.Rlen+ir]*dr*dx + Bstatic.EBarr[ix*Bstatic.Rlen+irp]*idr*dx + Bstatic.EBarr[ixp*Bstatic.Rlen+ir]*dr*idx + Bstatic.EBarr[ixp*Bstatic.Rlen+irp]*idr*idx # Bilinear interpolation for radial field component
    B_pulsed = Bpulsed.EBarr[ix*Bpulsed.Rlen+ir]*dr*dx + Bpulsed.EBarr[ix*Bpulsed.Rlen+irp]*idr*dx + Bpulsed.EBarr[ixp*Bpulsed.Rlen+ir]*dr*idx + Bpulsed.EBarr[ixp*Bpulsed.Rlen+irp]*idr*idx # Bilinear interpolation for radial field component
    E_induced = Einduced.EBarr[ix*Einduced.Rlen+ir]*dr*dx + Einduced.EBarr[ix*Einduced.Rlen+irp]*idr*dx + Einduced.EBarr[ixp*Einduced.Rlen+ir]*dr*idx + Einduced.EBarr[ixp*Einduced.Rlen+irp]*idr*idx # Bilinear interpolation for induced field component
    return B_static*x/r, B_pulsed*x/r, -E_induced*y/r

def y_maps(x, y, z):
    r = (x**2 + y**2)**0.5
    ir, irp, ix, ixp, dr, idr, dx, idx = Bstatic.find_grid(z, r)
    B_static = Bstatic.EBarr[ix*Bstatic.Rlen+ir]*dr*dx + Bstatic.EBarr[ix*Bstatic.Rlen+irp]*idr*dx + Bstatic.EBarr[ixp*Bstatic.Rlen+ir]*dr*idx + Bstatic.EBarr[ixp*Bstatic.Rlen+irp]*idr*idx # Bilinear interpolation for radial field component
    B_pulsed = Bpulsed.EBarr[ix*Bpulsed.Rlen+ir]*dr*dx + Bpulsed.EBarr[ix*Bpulsed.Rlen+irp]*idr*dx + Bpulsed.EBarr[ixp*Bpulsed.Rlen+ir]*dr*idx + Bpulsed.EBarr[ixp*Bpulsed.Rlen+irp]*idr*idx # Bilinear interpolation for radial field component
    E_induced = Einduced.EBarr[ix*Einduced.Rlen+ir]*dr*dx + Einduced.EBarr[ix*Einduced.Rlen+irp]*idr*dx + Einduced.EBarr[ixp*Einduced.Rlen+ir]*dr*idx + Einduced.EBarr[ixp*Einduced.Rlen+irp]*idr*idx # Bilinear interpolation for induced field component
    return B_static*y/r, B_pulsed*y/r, E_induced*x/r

def evaluate_field_3d(xymin, xymax, zmin, zmax, spacing):
    x_arr = np.arange(xymin, xymax+spacing, spacing)
    y_arr = np.arange(xymin, xymax+spacing, spacing)
    z_arr = np.arange(zmin, zmax+spacing, spacing)
    nx = len(x_arr)
    ny = len(y_arr)
    nz = len(z_arr)
    Bx_static = np.zeros((nx, ny, nz))
    Bx_pulsed = np.zeros((nx, ny, nz))
    Ex_induced = np.zeros((nx, ny, nz))
    By_static = np.zeros((nx, ny, nz))
    By_pulsed = np.zeros((nx, ny, nz))
    Ey_induced = np.zeros((nx, ny, nz))
    Bz_static = np.zeros((nx, ny, nz))
    Bz_pulsed = np.zeros((nx, ny, nz))
    for i, z in enumerate(z_arr):
        for j, y in enumerate(y_arr):
            Bx_static[:, j, i], Bx_pulsed[:, j, i], Ex_induced[:, j, i] = x_maps(x_arr, y*np.ones_like(x_arr), z*np.ones_like(x_arr))
            By_static[:, j, i], By_pulsed[:, j, i], Ey_induced[:, j, i] = y_maps(x_arr, y*np.ones_like(x_arr), z*np.ones_like(x_arr))
            Bz_static[:, j, i], Bz_pulsed[:, j, i] = z_maps(x_arr, y*np.ones_like(x_arr), z*np.ones_like(x_arr))
    return Bx_static, Bx_pulsed, Ex_induced, By_static, By_pulsed, Ey_induced, Bz_static, Bz_pulsed

## Functions to generate TE111 mode field maps
E_PW = 1.e5 # TEM field intensity [V/m]
L_cavity = 0.08763  # Cavity length [m]
R_cavity = 0.05  # Cavity radius [m]
z0_sim = -L_cavity/2. # Displacement of the cavity with respect to the simulation box [code units]
TEf = tf.TE111fieldsAM(L_cavity, R_cavity, z0_sim, E_PW, c=sciconst.c)

def TEmaps(x, y, z_raw):
    ## Precompute coordinates
    r = (x**2 + y**2)**0.5
    z = z_raw - z0_sim # Axial position in the cavity frame
    r[np.where(r==0)] = 1e-10 # Avoid division by zero
    theta = np.arctan2(y, x) # Azimuthal angle
    ## Precompute TEM fields quantities
    krr = TEf.k_mn*r
    jpr = scisp.j0(krr)-scisp.j1(krr)/krr
    ## Compute cylindrical field components
    Br = TEf.kxB0aix*jpr*np.cos(TEf.k_x*z)*np.cos(TEf.m*theta)
    Bth = -TEf.kxB0ma2ix2*scisp.j1(TEf.k_mn*r)/r*np.cos(TEf.k_x*z)*np.sin(TEf.m*theta)
    Bz = TEf.B0*scisp.j1(TEf.k_mn*r)*np.sin(TEf.k_x*z)*np.cos(TEf.m*theta)
    Er = -TEf.omegaB0ma2ix2*scisp.j1(TEf.k_mn*r)/r*np.sin(TEf.k_x*z)*np.sin(TEf.m*theta)
    Eth = -TEf.omegaB0aix*jpr*np.sin(TEf.k_x*z)*np.cos(TEf.m*theta)
    ## Convert to Cartesian coordinates
    Bx = Br*np.cos(theta) - Bth*np.sin(theta)
    By = Br*np.sin(theta) + Bth*np.cos(theta)
    Ex = Er*np.cos(theta) - Eth*np.sin(theta)
    Ey = Er*np.sin(theta) + Eth*np.cos(theta)
    return Bx, By, Bz, Ex, Ey

def evaluate_TEfield_3d(xymin, xymax, zmin, zmax, spacing):
    x_arr = np.arange(xymin, xymax+spacing, spacing)
    y_arr = np.arange(xymin, xymax+spacing, spacing)
    z_arr = np.arange(zmin, zmax+spacing, spacing)
    nx = len(x_arr)
    ny = len(y_arr)
    nz = len(z_arr)
    Bx = np.zeros((nx, ny, nz))
    By = np.zeros((nx, ny, nz))
    Bz = np.zeros((nx, ny, nz))
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    for i, z in enumerate(z_arr):
        for j, y in enumerate(y_arr):
            Bx[:, j, i], By[:, j, i], Bz[:, j, i], Ex[:, j, i], Ey[:, j, i] = TEmaps(x_arr, y*np.ones_like(x_arr), z*np.ones_like(x_arr))
    return Bx, By, Bz, Ex, Ey
    
'''
## Example usage
nx = 47
ny = 47
nz = 47
# read data
Bx_data = np.loadtxt('Bx_3d.txt').reshape(nx, ny, nz)
By_data = np.loadtxt('By_3d.txt').reshape(nx, ny, nz)
Bz_data = np.loadtxt('Bz_3d.txt').reshape(nx, ny, nz)
'''

## Upload field maps
num_nodes = 1
xymin = -0.05
xymax = 0.05
zmin = -0.025
zmax = 0.025
spacing = 0.00025
'''
Bx_static, Bx_pulsed, Ex_induced, By_static, By_pulsed, Ey_induced, Bz_static, Bz_pulsed = evaluate_field_3d(xymin, xymax, zmin, zmax, spacing)


# create openpmd file for pulsed fields
name = f"{address}ECRIPAC_staticfields_3d.h5"
series = io.Series(name,io.Access.create)
# only 1 iteratiion needed
it = series.iterations[1]

# set meta information
B = it.meshes["B"]
B.grid_spacing = np.array([spacing, spacing, spacing])
B.grid_global_offset = [xymin, xymin, zmin]
B.axis_labels = ['x', 'y', 'z']
B.geometry = io.Geometry.cartesian
B.unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.I: -1,
    io.Unit_Dimension.T: -2
}

# label
B_x = B["x"]
B_x.position = [0,0,0]
B_y = B["y"]
B_y.position = [0,0,0]
B_z = B["z"]
B_z.position = [0,0,0]

dataset = io.Dataset(Bx_static.dtype, Bx_static.shape)
B_x.reset_dataset(dataset)
B_y.reset_dataset(dataset)
B_z.reset_dataset(dataset)

B_x.store_chunk(Bx_static)
B_y.store_chunk(By_static)
B_z.store_chunk(Bz_static)

series.flush()
del series


## Create openpmd file for pulsed fields
name = f"{address}ECRIPAC_pulsedfields_3d.h5"
series = io.Series(name,io.Access.create)
# only 1 iteratiion needed
it = series.iterations[1]

# set meta information
B = it.meshes["B"]
B.grid_spacing = np.array([spacing, spacing, spacing])
B.grid_global_offset = [xymin, xymin, zmin]
B.axis_labels = ['x', 'y', 'z']
B.geometry = io.Geometry.cartesian
B.unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.I: -1,
    io.Unit_Dimension.T: -2
}

# label
B_x = B["x"]
B_x.position = [0,0,0]
B_y = B["y"]
B_y.position = [0,0,0]
B_z = B["z"]
B_z.position = [0,0,0]

dataset = io.Dataset(Bx_pulsed.dtype, Bx_pulsed.shape)
B_x.reset_dataset(dataset)
B_y.reset_dataset(dataset)
B_z.reset_dataset(dataset)

B_x.store_chunk(Bx_pulsed)
B_y.store_chunk(By_pulsed)
B_z.store_chunk(Bz_pulsed)

# set meta information
E = it.meshes["E"]
E.grid_spacing = np.array([spacing, spacing, spacing])
E.grid_global_offset = [xymin, xymin, zmin]
E.axis_labels = ['x', 'y', 'z']
E.geometry = io.Geometry.cartesian
E.unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.L:  1,
    io.Unit_Dimension.I: -1,
    io.Unit_Dimension.T: -3
}

# label
E_x = E["x"]
E_x.position = [0,0,0]
E_y = E["y"]
E_y.position = [0,0,0]
E_z = E["z"]
E_z.position = [0,0,0]

dataset = io.Dataset(Bx_pulsed.dtype, Bx_pulsed.shape)
E_x.reset_dataset(dataset)
E_y.reset_dataset(dataset)
E_z.reset_dataset(dataset)

E_x.store_chunk(Ex_induced)
E_y.store_chunk(Ey_induced)
E_z.make_constant(0.0)

series.flush()
del series
'''

## Create openpmd file for TE111 fields
Bx, By, Bz, Ex, Ey = evaluate_TEfield_3d(xymin, xymax, zmin, zmax, spacing)

name = f"{address}ECRIPAC_TE111_3d.h5"
series = io.Series(name,io.Access.create)
# only 1 iteratiion needed
it = series.iterations[1]

# set meta information
B = it.meshes["B"]
B.grid_spacing = np.array([spacing, spacing, spacing])
B.grid_global_offset = [xymin, xymin, zmin]
B.axis_labels = ['x', 'y', 'z']
B.geometry = io.Geometry.cartesian
B.unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.I: -1,
    io.Unit_Dimension.T: -2
}

# label
B_x = B["x"]
B_x.position = [0,0,0]
B_y = B["y"]
B_y.position = [0,0,0]
B_z = B["z"]
B_z.position = [0,0,0]

dataset = io.Dataset(Bx.dtype, Bx.shape)
B_x.reset_dataset(dataset)
B_y.reset_dataset(dataset)
B_z.reset_dataset(dataset)

B_x.store_chunk(Bx)
B_y.store_chunk(By)
B_z.store_chunk(Bz)

# set meta information
E = it.meshes["E"]
E.grid_spacing = np.array([spacing, spacing, spacing])
E.grid_global_offset = [xymin, xymin, zmin]
E.axis_labels = ['x', 'y', 'z']
E.geometry = io.Geometry.cartesian
E.unit_dimension = {
    io.Unit_Dimension.M:  1,
    io.Unit_Dimension.L:  1,
    io.Unit_Dimension.I: -1,
    io.Unit_Dimension.T: -3
}

# label
E_x = E["x"]
E_x.position = [0,0,0]
E_y = E["y"]
E_y.position = [0,0,0]
E_z = E["z"]
E_z.position = [0,0,0]

dataset = io.Dataset(Bx.dtype, Bx.shape)
E_x.reset_dataset(dataset)
E_y.reset_dataset(dataset)
E_z.reset_dataset(dataset)

E_x.store_chunk(Ex)
E_y.store_chunk(Ey)
E_z.make_constant(0.0)

series.flush()
del series