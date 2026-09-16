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
omega0 = 2*np.pi*2.45e9  # Reference angular frequency: microwave frequency [rad/s]
k = omega0/c  # Reference wavenumber [1/m]
n_crit = eps0*omega0**2*me/e**2  # Plasma critical number density [m-3]

##### Plasma chamber geometry #####
R_cavity = 0.05  # Cavity radius [m]
L_cavity = 0.08763  # Cavity length [m]

##### Plasma quantities #####
T_e_ev = 20 # Electron temperature [eV]
T_ion_ev = 10 # Helium ion temperature [eV]
n_plasma = 0.15*n_crit # Plasma density [m^-3]
Ld_min = (eps0/(n_plasma*e**2)*T_ion_ev*e)**0.5 # Approximated debye length [m]

#########################
##### Picmi objects #####
#########################
##### Grid #####
## Define grid quantities
dz = 0.878*Ld_min*2  # z resolution [m]
dx = 0.878*Ld_min*2  # x resolution [m]
dy = 0.878*Ld_min*2  # y resolution [m]
Lz = 20.e-2  # Longitudinal size of the simulation window [m]
Lx = R_cavity*2  # x size of the simulation window [m]
Ly = R_cavity*2  # y size of the simulation window [m]
nz = int(Lz/dz)  # Number of grid points in the longitudinal direction
nx = int(Lx/dx)  # Number of grid points in the x direction
ny = int(Ly/dy)  # Number of grid points in the y direction
xmin = -Lx/2.  # Minimum x position [m]
xmax = Lx/2.  # Maximum x position [m]
ymin = -Ly/2.  # Minimum y position [m]
ymax = Ly/2.  # Maximum y position [m]
zmin = -2.e-2  # Minimum longitudinal position [m]
zmax = Lz + zmin # Maximum longitudinal position [m]
max_grid_size = 64 # Nominal value used in the example
blocking_factor = 32 # Nominal value used in the example
T_sim = 1.25e-6  # Simulation time [s]
T_ga = 0.3e-6  # GA phase total duration [s]
T_trans = 0.3e-6  # Time for the field to transit from max to 0 [s]
T_mw = 1.2e-6  # Time to start moving window [s]

##### Electromagnetic solver #####
## Define solver quantities
method_em = "Yee"  # Electromagnetic solver method
cfl = 0.95 # CFL condition
dt = cfl/c/np.sqrt(1./dx**2+1./dy**2+1./dz**2)
max_steps = 400000 # Number of time steps to run the simulation

## Create grid object
grid = picmi.Cartesian3DGrid(
    ## Standard input parameters
    number_of_cells = [nx, ny, nz],
    lower_bound = [xmin, ymin, zmin],
    upper_bound = [xmax, ymax, zmax],
    lower_boundary_conditions = ["absorbing_silver_mueller", "absorbing_silver_mueller", "absorbing_silver_mueller"],
    upper_boundary_conditions = ["absorbing_silver_mueller", "absorbing_silver_mueller", "absorbing_silver_mueller"],
    lower_boundary_conditions_particles = ["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles = ["absorbing", "absorbing", "absorbing"],
    warpx_max_grid_size = max_grid_size,
    warpx_blocking_factor = blocking_factor,
    #moving_window_velocity = [0., 0., 3e5],
    #warpx_start_moving_window_step = int(T_mw/dt)
)

## Create solver object
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method = method_em,
    cfl = cfl
)

##### Applied fields #####
## Field maps
omega_pul = 2*np.pi*2e5 # Angular frequency of the pulsed field [rad/s]
phase = np.pi/2. # Phase of the pulsed field
t_pc = 1.16726417984e-06

Bstatic = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_staticfields_3d_pl2.h5",
    load_E=False,
    load_B=True,
)
Bpulsed = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_pulsedfields_3d_pl2.h5",
    load_E=False,
    load_B=True,
    warpx_B_time_function=f"sin({omega_pul}*(t+{t_pc}) + {phase})",
)
Einduced = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_pulsedfields_3d_pl2.h5",
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
    warpx_dump_last_timestep = True
)
smallparticle_diag = picmi.ParticleDiagnostic(
    name = "smallparticle_diag",
    period = smallparticle_period,
    species = [electrons, alphas],
    data_list = diag_particle_list,
    warpx_format = "openpmd",
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
    number_of_cells = [int(nx/4), int(ny/4), int(nz/4)],
    warpx_format = "openpmd",
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
)
sim.add_species(
    alphas,
    layout = None,
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
sim.write_input_file(file_name="inputs_3d_picmi_pl")

# Initialize inputs and WarpX instance
#sim.initialize_inputs()
#sim.initialize_warpx()

# Advance simulation until last time step
#sim.step(max_steps)