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
T_e_ev = 200 # Electron temperature [eV]
T_ion_ev = 10 # Helium ion temperature [eV]
n_plasma = 0.15*n_crit # Plasma density [m^-3]
Ld_min = (eps0/(n_plasma*e**2)*T_ion_ev*e)**0.5 # Approximated debye length [m]

#########################
##### Picmi objects #####
#########################
##### Grid #####
## Define grid quantities
n_AM_modes = 2  # Number of azimuthal modes
dz = 0.936*Ld_min  # Longitudinal resolution [m]
dr = 0.878*Ld_min  # Radial resolution [m]
Lz = 4.e-2  # Longitudinal size of the simulation window [m]
Lr = R_cavity  # Radial size of the simulation window [m]
nr = int(Lr/dr)  # Number of grid points in the radial direction
nz = int(Lz/dz)  # Number of grid points in the longitudinal direction
rmin = 0.  # Minimum radius [m]
rmax = Lr  # Maximum radius [m]
zmin = -Lz/2.  # Minimum longitudinal position [m]
zmax = Lz/2.  # Maximum longitudinal position [m]
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
    n_azimuthal_modes = n_AM_modes,
    ## Parameters for WarpX compiled with USE_RZ = TRUE
    warpx_max_grid_size = max_grid_size,
    warpx_blocking_factor = blocking_factor,
)

##### Electromagnetic solver #####
## Define solver quantities
method_em = "Yee"  # Electromagnetic solver method
cfl = 0.95 # CFL condition

## Create solver object
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method = method_em,
    cfl = cfl
)

##### Binomial filter #####
filter = picmi.BinomialSmoother(n_pass=2)

##### Applied fields #####
## Field maps
phase = np.pi/2. # Phase of the pulsed field

# Applied fields don't work with azimuthal modes
#Bstatic = picmi.LoadAppliedField(
#    read_fields_from_path="res/ECRIPAC_staticfields_thetaMode.h5",
#    load_E=False,
#    load_B=True,
#)
#Bpulsed = picmi.LoadAppliedField(
#    read_fields_from_path="res/ECRIPAC_pulsedfields_thetaMode.h5",
#    load_E=False,
#    load_B=True,
#    warpx_B_time_function=f"sin({omega0}*t + {phase})",
#)
#Einduced = picmi.LoadAppliedField(
#    read_fields_from_path="res/ECRIPAC_pulsedfields_thetaMode.h5",
#    load_E=True,
#    load_B=False,
#    warpx_E_time_function=f"cos({omega0}*t + {phase})",
#)

## Plane waves and analytical fields
E0 = 1e5 # Amplitude of the electric field [V/m]
omega_pul = 2*np.pi*2e5 # Angular frequency of the pulsed field [rad/s]
phase_pul = 1./2e5/4
Rs2 = 0.2**2
zl = -0.08
zr = 0.08
Bpul_const = -4.93/2.*(Rs2+zr**2)**1.5 # Constant mu0*I*Rsol^2/2 for the field expressions
c1_static = -6.39030891e-05
c2_static = 5.00171222e+00

pw_field = picmi.AnalyticAppliedField( # Induced field is missing
    Bx_expression = f"(-{c1_static}*(x**2+y**2)**0.5/2. + {Bpul_const}*3*(x**2+y**2)**0.5/2.*((z-{zl})*(1/({Rs2}**2+(z-{zl})**2)**2.5-5*(x**2+y**2)/4/({Rs2}**2+(z-{zl})**2)**3.5) + (z-{zr})*(1/({Rs2}**2+(z-{zr})**2)**2.5-5*(x**2+y**2)/4/({Rs2}**2+(z-{zr})**2)**3.5))*sin({omega_pul}*(t+{phase_pul})))*x/(x**2+y**2)**0.5",
    By_expression = f"(-{c1_static}*(x**2+y**2)**0.5/2. + {Bpul_const}*3*(x**2+y**2)**0.5/2.*((z-{zl})*(1/({Rs2}**2+(z-{zl})**2)**2.5-5*(x**2+y**2)/4/({Rs2}**2+(z-{zl})**2)**3.5) + (z-{zr})*(1/({Rs2}**2+(z-{zr})**2)**2.5-5*(x**2+y**2)/4/({Rs2}**2+(z-{zr})**2)**3.5))*sin({omega_pul}*(t+{phase_pul})))*y/(x**2+y**2)**0.5",
    Bz_expression = f"({c1_static}*z + {c2_static}) + {Bpul_const}*(1./({Rs2}**2+(z-{zl})**2)**1.5 - 3.*(x**2+y**2)/2./({Rs2}**2+(z-{zl})**2)**2.5 + 1./({Rs2}**2+(z-{zr})**2)**1.5 - 3.*(x**2+y**2)/2./({Rs2}**2+(z-{zr})**2)**2.5)*sin({omega_pul}*(t+{phase_pul}))",
    Ex_expression = f"{E0}*cos({omega0}*t - {k}*z)",
    Ey_expression = f"{E0}*sin({omega0}*t - {k}*z)",
    Ez_expression = "0"
)

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
n_ppc = [2, 8, 2] # Number of particles per cell in each direction
Mi = 2*(mp + mn)
gamma_e = 1+T_e_ev/electron_mass_MeV/1e6
v_e = c*(1-1/gamma_e**2)**0.5
gamma_i = 1+T_ion_ev/electron_mass_MeV/1e6/Mi
v_i = c*(1-1/gamma_i**2)**0.5 

## Define particle distributions
electrons_uniform_distribution = picmi.UniformDistribution( # Uniform distribution, need to modify later on
    density = n_plasma,
    lower_bound = [plasma_xmin, plasma_ymin, plasma_zmin],
    upper_bound = [plasma_xmax, plasma_ymax, plasma_zmax],
    #fill_in = True,
    rms_velocity = [v_e/2., v_e/2., v_e/2.], # Approximate thermal velocity, need to modify later on
)

alpha_uniform_distribution = picmi.UniformDistribution( # Uniform distribution, need to modify later on
    density = n_plasma/2., # Assuming a fully ionized helium plasma, the ion density is half the electron density
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
particle_period = 100 # Number of time steps between two outputs
diag_particle_list = ["position", "momentum", "weighting"]
particle_diag = picmi.ParticleDiagnostic(
    name = "particle_diag",
    period = particle_period,
    species = [electrons, alphas],
    data_list = diag_particle_list,
    warpx_format = "openpmd"
)

## Field diagnostics
field_period = 100 # Number of time steps between two outputs
diag_field_list = ["B", "E", "J", "rho"]
field_diag = picmi.FieldDiagnostic(
    name = "field_diag",
    grid = grid,
    period = field_period,
    data_list = diag_field_list,
    warpx_dump_rz_modes = 1,
    warpx_format = "openpmd"
)

############################
##### Simulation setup #####
############################
## Define simulation quantities
max_steps = 10000 # Number of time steps to run the simulation
particle_shape = "cubic" # Particle shape

## Create simulation object
sim = picmi.Simulation(
    solver = solver,
    max_steps = max_steps,
    verbose = 1,
    particle_shape = particle_shape,
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
#sim.add_applied_field(Bstatic)
#sim.add_applied_field(Bpulsed)
#sim.add_applied_field(Einduced)
sim.add_applied_field(pw_field)

## Add diagnostics to the simulation
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

# Write input file that can be used to run with the compiled version
sim.write_input_file(file_name="inputs_rz_picmi")

# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

# Advance simulation until last time step
sim.step(max_steps)