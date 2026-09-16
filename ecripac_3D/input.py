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
dz = 0.936*Ld_min  # z resolution [m]
dx = 0.878*Ld_min  # x resolution [m]
dy = 0.878*Ld_min  # y resolution [m]
Lz = 4.e-2  # Longitudinal size of the simulation window [m]
Lx = R_cavity*2  # x size of the simulation window [m]
Ly = R_cavity*2  # y size of the simulation window [m]
nz = int(Lz/dz)  # Number of grid points in the longitudinal direction
nx = int(Lx/dx)  # Number of grid points in the x direction
ny = int(Ly/dy)  # Number of grid points in the y direction
xmin = -Lx/2.  # Minimum x position [m]
xmax = Lx/2.  # Maximum x position [m]
ymin = -Ly/2.  # Minimum y position [m]
ymax = Ly/2.  # Maximum y position [m]
zmin = -Lz/2.  # Minimum longitudinal position [m]
zmax = Lz/2.  # Maximum longitudinal position [m]
max_grid_size = 64 # Nominal value used in the example
blocking_factor = 32 # Nominal value used in the example
T_sim = 1.25e-6  # Simulation time [s]
T_ga = 0.3e-6  # GA phase total duration [s]
T_trans = 0.3e-6  # Time for the field to transit from max to 0 [s]

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
)

##### Electromagnetic solver #####
## Define solver quantities
method_em = "Yee"  # Electromagnetic solver method
cfl = 0.95 # CFL condition
dt = cfl/c/np.sqrt(1./dx**2+1./dy**2+1./dz**2)
max_steps = 400000 # Number of time steps to run the simulation
#max_steps = int(T_ga/dt) 

## Create solver object
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method = method_em,
    cfl = cfl
)

solver_es = picmi.ElectrostaticSolver(
    grid=grid
)

##### Binomial filter #####
#filter = picmi.BinomialSmoother(n_pass=2)

##### Applied fields #####
## Field maps
omega_pul = 2*np.pi*2e5 # Angular frequency of the pulsed field [rad/s]
phase = np.pi/2. # Phase of the pulsed field
E0 = 1e5 # Amplitude of the electric field [V/m]

Bstatic = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_staticfields_3d.h5",
    load_E=False,
    load_B=True,
)
Bpulsed = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_pulsedfields_3d.h5",
    load_E=False,
    load_B=True,
    warpx_B_time_function=f"sin({omega_pul}*t + {phase})",
)
Einduced = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_pulsedfields_3d.h5",
    load_E=True,
    load_B=False,
    warpx_E_time_function=f"cos({omega_pul}*t + {phase})",
)
Etem = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_TE111_3d.h5",
    load_E=True,
    load_B=False,
    warpx_E_time_function=f"sin({omega0}*t)*20*(1. - exp((t-3.6476604499999995e-07)/5e-9))",
)
Btem = picmi.LoadAppliedField(
    read_fields_from_path="../res/ECRIPAC_TE111_3d.h5",
    load_E=False,
    load_B=True,
    warpx_B_time_function=f"cos({omega0}*t)*20*(1.-(t-{T_ga})/{T_trans})",
)

## Plane waves and analytical fields
#pw_field = picmi.AnalyticAppliedField( # Induced field is missing
#    Ex_expression = f"{E0}*cos({omega0}*t - {k}*z)",
#    Ey_expression = f"{E0}*sin({omega0}*t - {k}*z)",
#    Ez_expression = "0"
#)

##### Particles #####
## Define particle quantities
R_plasma = 10.e-3 # Plasma radius [m]
L_plasma = 10.e-3 # Plasma length [m]
plasma_xmin = -R_plasma
plasma_ymin = -R_plasma
plasma_zmin = -L_plasma/2.
plasma_xmax = R_plasma
plasma_ymax = R_plasma
plasma_zmax = L_plasma/2.
n_ppc = [2, 2, 2] # Number of particles per cell in each direction
Mi = 2*(mp + mn)
gamma_e = 1+T_e_ev/electron_mass_MeV/1e6
v_e = c*(1-1/gamma_e**2)**0.5
gamma_i = 1+T_ion_ev/electron_mass_MeV/1e6*me/Mi
v_i = c*(1-1/gamma_i**2)**0.5 

## Define particle distributions
#electrons_uniform_distribution = picmi.UniformDistribution( # Uniform distribution, need to modify later on
#    density = n_plasma,
#    lower_bound = [plasma_xmin, plasma_ymin, plasma_zmin],
#    upper_bound = [plasma_xmax, plasma_ymax, plasma_zmax],
#    #fill_in = True,
#    rms_velocity = [v_e/2., v_e/2., v_e/2.], # Approximate thermal velocity, need to modify later on
#)
#alpha_uniform_distribution = picmi.UniformDistribution( # Uniform distribution, need to modify later on
#    density = n_plasma/2., # Assuming a fully ionized helium plasma, the ion density is half the electron density
#    lower_bound = [plasma_xmin, plasma_ymin, plasma_zmin],
#    upper_bound = [plasma_xmax, plasma_ymax, plasma_zmax],
#    #fill_in = True,
#    rms_velocity = [v_i/2., v_i/2., v_i/2.], # Approximate thermal velocity, need to modify later on
#)

electrons_uniform_distribution = picmi.AnalyticDistribution(
    density_expression = f"(x**2 + y**2 <= {R_plasma}**2)*{n_plasma}",
    lower_bound = [plasma_xmin, plasma_ymin, plasma_zmin],
    upper_bound = [plasma_xmax, plasma_ymax, plasma_zmax],
    #fill_in = True,
    rms_velocity = [v_e/2., v_e/2., v_e/2.], # Approximate thermal velocity, need to modify later on
)
alpha_uniform_distribution = picmi.AnalyticDistribution(
    density_expression = f"(x**2 + y**2 <= {R_plasma}**2)*{n_plasma}/2.", # Assuming a fully ionized helium plasma, the ion density is half the electron density
    lower_bound = [plasma_xmin, plasma_ymin, plasma_zmin],
    upper_bound = [plasma_xmax, plasma_ymax, plasma_zmax],
    #fill_in = True,
    rms_velocity = [v_i/2., v_i/2., v_i/2.], # Approximate thermal velocity, need to modify later on
)

## Define particles layout
particles_layout = picmi.GriddedLayout(
    grid = grid,
    n_macroparticle_per_cell = n_ppc
)

## Create electron species
electrons = picmi.Species(
    particle_type = "electron",
    name=  "electrons",
    initial_distribution=electrons_uniform_distribution,
)

## Create ion species
alphas = picmi.Species(
    name=  "alphas",
    charge = 2*e, # Assuming a fully ionized helium plasma
    mass = Mi,
    initial_distribution=alpha_uniform_distribution,
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
    #time_step_size = dt, # Only if CFL not provided
)

## Add species to the simulation
sim.add_species(
    electrons,
    layout = particles_layout,
)
sim.add_species(
    alphas,
    layout = particles_layout,
)

## Add applied fields to the simulation
sim.add_applied_field(Bstatic)
sim.add_applied_field(Bpulsed)
sim.add_applied_field(Einduced)
sim.add_applied_field(Etem)
sim.add_applied_field(Btem)
#sim.add_applied_field(pw_field)

## Add diagnostics to the simulation
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(smallparticle_diag)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(checkpoint)

# Write input file that can be used to run with the compiled version
sim.write_input_file(file_name="inputs_3d_picmi")

# Initialize inputs and WarpX instance
#sim.initialize_inputs()
#sim.initialize_warpx()

# Advance simulation until last time step
#sim.step(max_steps)