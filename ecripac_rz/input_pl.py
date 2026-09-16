## RZ simulation of PLEIADE phase starting from particle distribution of 3D simulation

######################
###### Libraries #####
######################
from pywarpx import picmi
import numpy as np
import scipy.constants as sciconst

###################################
##### Constants and variables #####
###################################
##### Physical constants #####
c = picmi.constants.c  # Speed of light in vacuum [m/s]
eps0 = picmi.constants.ep0  # Vacuum permittivity [F/m]
e = picmi.constants.q_e  # Elementary charge [C]
me = picmi.constants.m_e  # Electron mass [kg]
mp = picmi.constants.m_p  # Proton mass [kg]
mn = sciconst.m_n  # Neutron mass [kg]
electron_mass_MeV = sciconst.physical_constants["electron mass energy equivalent in MeV"][0]  # Electron mass in MeV

##### Plasma chamber geometry #####
R_cavity = 0.05  # Cavity radius [m]
L_cavity = 1.8  # Cavity length [m]
neg_zdispl = 0.1  # Negative z displacement [m]

##### Plasma quantities #####
Ld_min = 2.22e-4 # [m] keep Debye length consistent with previous simulation - Minimum Debye length calculated for ions at the end of 3D simulation approx 1 mm (for electrons 10 cm)

#########################
##### Picmi objects #####
#########################
##### Grid #####
## Define grid quantities
dz = 0.9128*Ld_min  # Longitudinal resolution [m]
dr = 0.781*Ld_min  # Radial resolution [m]
Lz = L_cavity+neg_zdispl  # Longitudinal size of the simulation window [m]
Lr = R_cavity  # Radial size of the simulation window [m]
nr = int(Lr/dr)  # Number of grid points in the radial direction
nz = int(Lz/dz)  # Number of grid points in the longitudinal direction
rmin = 0.  # Minimum radius [m]
rmax = Lr  # Maximum radius [m]
zmin = -neg_zdispl  # Minimum longitudinal position [m]
zmax = L_cavity  # Maximum longitudinal position [m]
max_grid_size = 64 # Nominal value used in the example
blocking_factor = 32 # Nominal value used in the example

## Create grid object
grid = picmi.CylindricalGrid(
    ## Standard input parameters
    number_of_cells = [nr, nz],
    lower_bound = [rmin, zmin],
    upper_bound = [rmax, zmax],
    lower_boundary_conditions = ["none", "absorbing_silver_mueller"],
    upper_boundary_conditions = ["absorbing_silver_mueller", "absorbing_silver_mueller"],
    lower_boundary_conditions_particles = ["none", "absorbing"],
    upper_boundary_conditions_particles = ["absorbing", "absorbing"],
    ## Parameters for WarpX compiled with USE_RZ = TRUE
    warpx_max_grid_size = max_grid_size,
    warpx_blocking_factor = blocking_factor,
)

##### Electromagnetic solver #####
## Define solver quantities
method_em = "Yee"  # Electromagnetic solver method
cfl = 0.95 # CFL condition
dt = cfl/c/np.sqrt(1./dr**2+1./dr**2+1./dz**2)
max_steps = 400000 # Number of time steps to run the simulation

## Create solver object
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method = method_em,
    cfl = cfl
)

##### Binomial filter #####
#filter = picmi.BinomialSmoother(n_pass=2)

##### Applied fields #####
## Field maps
phase = np.pi/2. # Phase of the pulsed field
omega_pul = 2*np.pi*2e5 # Angular frequency of the pulsed field [rad/s]
t_pc = 1.16726417984e-06

# Applied fields don't work with azimuthal modes
Bstatic = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_staticfields_thetaMode.h5",
    load_E=False,
    load_B=True,
)
Bpulsed = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_pulsedfields_thetaMode.h5",
    load_E=False,
    load_B=True,
    warpx_B_time_function=f"sin({omega_pul}*(t+{t_pc}) + {phase})",
)
Einduced = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_pulsedfields_thetaMode.h5",
    load_E=True,
    load_B=False,
    warpx_E_time_function=f"cos({omega_pul}*(t+{t_pc}) + {phase})",
)

##### Particles #####
## Restart electron species
electrons = picmi.Species(
    name=  "electrons",
    initial_distribution=picmi.FromFileDistribution(file_path="chkpoints/electrons.h5"),
)

## Create ion species
alphas = picmi.Species(
    name=  "alphas",
    initial_distribution=picmi.FromFileDistribution(file_path="chkpoints/alphas.h5"),
)


##### Diagnostics #####
## Particle diagnostics
particle_period = int(max_steps/5) # Number of time steps between two outputs
smallparticle_period = int(max_steps/50) # Number of time steps between two outputs for the small particle diagnostic
diag_particle_list = ["position", "momentum", "weighting"]
particle_diag = picmi.ParticleDiagnostic(
    name = "particle_diag",
    period = particle_period,
    species = [electrons, alphas],
    data_list = diag_particle_list,
    warpx_format = "openpmd",
    warpx_openpmd_backend="h5",
    warpx_dump_last_timestep = True
)
smallparticle_diag = picmi.ParticleDiagnostic(
    name = "smallparticle_diag",
    period = smallparticle_period,
    species = [electrons, alphas],
    data_list = diag_particle_list,
    warpx_format = "openpmd",
    warpx_openpmd_backend="h5",
    warpx_dump_last_timestep = True,
    warpx_random_fraction = 1.e-2, # Only dump 1% of the particles
)

## Field diagnostics
field_period = int(max_steps/50) # Number of time steps between two outputs
diag_field_list = ["B", "E", "J", "rho"]
field_diag = picmi.FieldDiagnostic(
    name = "field_diag",
    grid = grid,
    period = field_period,
    data_list = diag_field_list,
    number_of_cells = [int(nr/4), int(nz/4)],
    warpx_format = "openpmd",
    warpx_openpmd_backend="h5",
    warpx_dump_last_timestep = True
)

## Checkpoint
T_checkpoint = int(max_steps/2.)
checkpoint = picmi.Checkpoint(
    name = "chkpoint",
    period = T_checkpoint,
    write_dir = "./chkpoints",
    warpx_file_prefix="Python_restart_runtime_components_chk",
)

############################
##### Simulation setup #####
############################
## Define simulation quantities
particle_shape = "quadratic" # Particle shape

## Create simulation object
sim = picmi.Simulation(
    solver = solver,
    max_steps = max_steps,
    verbose = 1,
    particle_shape = particle_shape,
    warpx_load_balance_intervals = 10,
)

## Add species to the simulation
sim.add_species(
    electrons,
    layout = None,
    initialize_self_field = True,
)
sim.add_species(
    alphas,
    layout = None,
    initialize_self_field = True,
)

## Add applied fields to the simulation
sim.add_applied_field(Bstatic)
sim.add_applied_field(Bpulsed)
sim.add_applied_field(Einduced)

## Add diagnostics to the simulation
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(smallparticle_diag)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(checkpoint)

# Write input file that can be used to run with the compiled version
sim.write_input_file(file_name="inputs_pl_rz_picmi")

# Initialize inputs and WarpX instance
#sim.initialize_inputs()
#sim.initialize_warpx()

# Advance simulation until last time step
#sim.step(max_steps)