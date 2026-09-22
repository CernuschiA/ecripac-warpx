############################################################
######################### Libraries ########################
############################################################
## System libraries
import sys
import pickle
import numpy as np
import scipy.constants as sciconst
import matplotlib.pyplot as plt
from pprint import pprint
from openpmd_viewer import OpenPMDTimeSeries
plt.rcParams.update({'font.size': 28})

## Custom libraries
sys.path.append("/home/cernuschi/coding/warpx/ecripac/functions/")
sys.path.append("//wsl$/Ubuntu/home/cernuschi/coding/warpx/ecripac/functions/")
import particle_class as pc
import field_class as fc
import extfield_class as efc

############################################################
###################### Pre-processing ######################
############################################################

##### Define variables #####
sys_path = "/home/cernuschi/pole_acc_storage/"
#sys_path = "//lpscdata12.in2p3.fr/data11-pole/CERNUSCHI/"
## Define single path
address_folder = "/home/cernuschi/coding/warpx/ecripac2/diags/"
#address_folder = "/home/cernuschi/coding/warpx/ecripac/ecripac_rz/diags/"
#address_folder = sys_path + "PhD/Simulation results/WarpX/TE111_200kHz/test7_tr1/diags/"
particle_path = f"{address_folder}particle_diag/"
field_path = f"{address_folder}field_diag/"
#path_save = "/home/cernuschi/coding/warpx/ecripac/diagnostics/"
path_save = sys_path + "PhD/Pictures/temp_warpx/"

## Define path lists
common_path = sys_path + "PhD/Simulation results/WarpX/TE111_200kHz/"
path_list = [f"{common_path}test{i}/diags/" for i in range(4,8)]
labels = ["2.5 kV/cm", "3.5 kV/cm", "6 kV/cm", "20 kV/cm"]
diag_type = "smallparticle_diag/"

## Define constants
dmult = 1#e2
t_disp = 6.5658610116e-07
grid_refinement = None#[4, 4, 4] # Standard grid refinement used in all 3D simulations

## Debug
#warpx = OpenPMDTimeSeries(particle_path)
#field, field_info = warpx.get_field(field="E", iteration=0)
#pprint(vars(warpx))

##### External fields #####
## Analytical pulsed field behaviour
Bmax = 5 # [T]
B0 = 0.0875 # [T]
omega_pul = 2*np.pi*2e5 # [rad/s]
def Bfield(t):
    return Bmax*(1-(Bmax-B0)/Bmax*np.sin(omega_pul*t+np.pi/2.))/B0

#'''
## Field maps -- Static 1.6 and pulsed 6.6 for 3D simulations, Static 1.1 and pulsed 4.4 for cylindrical simulations, Static 1.8 and pulsed 6.7 for pleiade 3D
Bstatic_path = sys_path + "PhD/Simulation results/Field maps/ECRIPAC/Static (Poisson)/Conf1.1/ECRIPAC_solenoid_initial.txt"
Bpulsed_path = sys_path + "PhD/Simulation results/Field maps/ECRIPAC/Pulsed (COMSOL)/Conf1_f200kHz/Test4.4/ECRIPAC_solenoid_pulsed.txt"
Einduced_path = sys_path + "PhD/Simulation results/Field maps/ECRIPAC/Pulsed (COMSOL)/Conf1_f200kHz/Test4.4/ECRIPAC_induced_field.txt"
f_pul = 2e5 # Frequency of the pulsed field [Hz]

## TE fields
E_PW = 20.e5 # TEM field intensity [V/m]
L_cavity = 0.08763  # Cavity length [m]
R_cavity = 0.05  # Cavity radius [m]
z0_sim = -L_cavity/2. # Displacement of the cavity with respect to the simulation box [code units]
fHF = 2.45e9 # Angular frequency of the HF fields (equal to self.omega) [rad/s]
T_ga = 3e-7 # Duration of the GA phase [s]
T_trans = 0.75e-7 # Duration of the transition phase for turning of the HF fields [s]
TE111 = efc.TEfields(1, 1, 1, L_cavity, R_cavity, z0_sim, E_PW, sciconst.c, fHF, T_ga, T_trans)
#TE111 = efc.PWfields(0., E_PW, sciconst.c, fHF, T_ga, T_trans)

## External fields class
#extfield = efc.extfields(Bstatic_path, Bpulsed_path, Einduced_path, f_pul, TE111)
#'''
############################################################
########################### Main ###########################
############################################################

''' ---------------- '''
''' Open diagnostics '''
''' ---------------- '''
## Fields
#fields = fc.field_diagnostic.from_path(field_path, t_disp=t_disp, save_address=path_save)#, extfield=extfield, save_address=path_save)

## Electrons
electrons = pc.track_particles(field_path, particle_path, "electrons", dmult=dmult, grid_refinement=grid_refinement)#, save_address=path_save, extfield=extfield
#electrons = pc.track_lost_particles(field_path, particle_path, "electrons", grid_refinement=grid_refinement)
#electrons = pc.track_filtered(field_path, particle_path, "electrons", "gamma", ll=2.75, grid_refinement=grid_refinement)
#electrons_t2 = pc.track_filtered(field_path, particle_path, "electrons", "gamma", ll=13, ul=14.7, save_address=path_save, grid_refinement=grid_refinement)
#electrons_t3 = pc.track_filtered(field_path, particle_path, "electrons", "gamma", ll=14.7, save_address=path_save, grid_refinement=grid_refinement)

## Ions
#ions = pc.track_particles(field_path, particle_path, "alphas", dmult=dmult, grid_refinement=grid_refinement)#, save_address=path_save
#ions = pc.track_lost_particles(field_path, particle_path, "alphas", grid_refinement=grid_refinement, ts=-11, save_address=path_save)

## Multiple species
#electrons_pop = [pc.track_particles(path+"field_diag/", path+diag_type, "electrons", dmult=dmult, grid_refinement=grid_refinement) for path in path_list]
#for lab, elec in zip(labels, electrons_pop):
#    elec.label = lab

## Save class with pickle
#with open(path_save+"electrons_class.pkl", "wb") as f:
#    pickle.dump(electrons, f)
#with open(path_save+"ions_class.pkl", "wb") as f:
#    pickle.dump(ions, f)
#with open(path_save+"fields_class.pkl", "wb") as f:
#    pickle.dump(fields, f)
    
## Load class with pickle
#with open(path_save+"electrons_class.pkl", "rb") as f:
#    electrons = pickle.load(f)
#with open(path_save+"ions_class.pkl", "rb") as f:
#    ions = pickle.load(f)
#with open(path_save+"fields_class.pkl", "rb") as f:
#    fields = pickle.load(f)

''' ----------------- '''
''' Field diagnostics '''
''' ----------------- '''
## Field plots
## 2D
#fields.plot_field2D("E", comp="x")
#fields.plot_field2D("E", comp="y")#, vmin=True, vmax=True)
#fields.plot_field2D("E", comp="r")#, vmin=True, vmax=True)
#fields.plot_field2D("E", comp="t")#, vmin=True, vmax=True)
#fields.plot_field2D("E", comp="z")#, vmin=True, vmax=True)
#fields.plot_field2D("E")
#fields.plot_field2D("B", comp="x")
#fields.plot_field2D("B", comp="y")
#fields.plot_field2D("B", comp="r")
#fields.plot_field2D("B", comp="t")
#fields.plot_field2D("B", comp="z")
#fields.plot_field2D("J", comp="x")
#fields.plot_field2D("J", comp="y")
#fields.plot_field2D("J", comp="r")
#fields.plot_field2D("J", comp="t")
#fields.plot_field2D("J", comp="z")
#fields.plot_field2D("rho")
## Save plots
#fields.screenshot_2D("E", ts=1000000, comp="r", vmin=-8.3e5, vmax=8.3e5)
#fields.screenshot_2D("rho", ts=776000, vmin=-0.0055, vmax=0.0055)
#fields.screenshot_2D("rho", ts=840000, vmin=-0.0055, vmax=0.0055)
#fields.screenshot_2D("rho", ts=880000, vmin=-0.0055, vmax=0.0055)

## 1D
#fields.plot_field1D("E", comp="x")#, vmin=True, vmax=True)
#fields.plot_field1D("E", comp="y")#, vmin=True, vmax=True)
#fields.plot_field1D("E", comp="r")#, vmin=True, vmax=True)
#fields.plot_field1D("E", comp="t")#, vmin=True, vmax=True)
#fields.plot_field1D("E", comp="z")#, vmin=True, vmax=True)
#fields.plot_field1D("B", comp="x")
#fields.plot_field1D("B", comp="y")
#fields.plot_field1D("B", comp="r")
#fields.plot_field1D("B", comp="t")
#fields.plot_field1D("B", comp="z")
#fields.plot_field1D("J", comp="x")
#fields.plot_field1D("J", comp="y")
#fields.plot_field1D("J", comp="r")
#fields.plot_field1D("J", comp="t")
#fields.plot_field1D("J", comp="z")
#fields.plot_field1D("rho")#, vmin=True, vmax=True)

## Fourier
#fields.fourier_plot("E", comp="x", logscale=False, electrons=electrons, ions=ions)
#fields.fourier_plot("E", comp="y", logscale=False, electrons=electrons, ions=ions)
#fields.fourier_plot("E", comp="z", logscale=False, electrons=electrons, ions=ions)
#fields.polar_fourier_plot("E", comp="r", logscale=False, electrons=electrons, ions=ions)
#fields.polar_fourier_plot("E", comp="theta", logscale=False, electrons=electrons, ions=ions)


''' --------------------------- '''
''' Track Particles diagnostics '''
''' --------------------------- '''
## Position and density plots
## Electrons diagnostics
#electrons.particle_position(q_color=electrons.gamma, clabel="gamma")
#electrons.particle_position(q_color=electrons.w, clabel="weight")
#electrons.position_distribution_slider(w=electrons.w)
#electrons.lost_histogram(flag_notintegral=True)
## Trapped electrons diagnostics
#electrons_t1.particle_position(q_color=electrons_t1.gamma, clabel="gamma")
#electrons_t2.particle_position(q_color=electrons_t2.gamma, clabel="gamma")
#electrons_t3.particle_position(q_color=electrons_t3.gamma, clabel="gamma")
#electrons_t1.position_distribution_slider(w=electrons_t1.w)
#electrons_t2.position_distribution_slider(w=electrons_t2.w)
#electrons_t3.position_distribution_slider(w=electrons_t3.w)
## Ions diagnostics
#ions.particle_position(q_color=ions.gamma, clabel="gamma")
#ions.position_distribution_slider(w=ions.w)
#ions.lost_histogram(flag_notintegral=True)
## Multiple species diagnostics
#electrons.c = "red"
#ions.c = "blue"
#electrons_t3.c = "green"
#electrons.population_position([electrons, ions], alpha=0.3)

## Density plots
## Electrons diagnostics
#electrons.densityplot1D_slider("evaluate_density", "n [m^3]")
#electrons.densityplot2D_slider("evaluate_density", "n [m^3]")
#electrons.densityplot2D_slider("evaluate_temperature", "T [keV]")
#electrons.densityplot2D_slider("evaluate_debyelength", r"$\lambda_D$ [m]")
electrons.densityplot2D_slider("evaluate_plasmapar", r"N$_D^3$")
#electrons.densityplot2D_slider("evaluate_plasmafreq", r"$\omega_{p}$ [rad/s]")
#electrons.densityplot2D_slider("evaluate_beta", r"$\beta$")
## Ions diagnostics
#ions.densityplot1D_slider("evaluate_density", "n [m^3]")
#ions.densityplot2D_slider("evaluate_density", "n [m^3]")
#ions.densityplot2D_slider("evaluate_temperature", "T [keV]")
#ions.densityplot2D_slider("evaluate_beta", r"$\beta$")
#ions.densityplot2D_slider("evaluate_plasmafreq", r"$\omega_{p}$ [rad/s]")
#ions.densityplot2D_slider("evaluate_debyelength", r"$\lambda_D$ [m]")
## Save plots
#electrons.screenshot_densityplot2D("evaluate_density", r"n$_e$ [m$^{-3}$]", ts=2880000, zpos=0.00104, vmax=4.4e16)
#ions.screenshot_densityplot2D("evaluate_density", r"n$_i$ [m$^{-3}$]", ts=1000000)
#ions.screenshot_densityplot2D("evaluate_density", r"n$_i$ [m$^{-3}$]", ts=1600000)

## Energy plots
## Electrons diagnostics
#electrons.momentum_distribution_slider()
#electrons.edf_slider()
#electrons.gamma_B_plot(B=Bfield)
#electrons.emittance_slider()
## Trapped electrons diagnostics
#electrons_t.edf_slider()
## Ions diagnostics
#ions.momentum_distribution_slider()
#ions.edf_slider(flag_keV=True, bins=1000)
#ions.gamma_B_plot(flag_keV=True)
#ions.emittance_slider()
## Multiple species diagnostics
#electrons_pop[0].population_edf(electrons_pop, bins=100, flag_keV=True)

## Other analysis
## Electrons diagnostics
#electrons.classattribute_slider("ga_phase", "Electron velocity phase [deg]", xlim=[0., 180.])
#electrons.classattribute_slider("pitch_angle", "Pitch angle [deg]", xlim=[0., 180.])
#electrons.evaluate_average("r", "r [cm]")
#electrons.evaluate_average("gamma", "γ")
## Ions diagnostics
#ions.classattribute_slider("ga_phase", "Ion velocity phase [deg]", xlim=[0., 180.])
#ions.classattribute_slider("pitch_angle", "Ion pitch angle [deg]", xlim=[0., 180.])
#ions.evaluate_average("pz", r"px [γv/c]")
#ions.evaluate_average("r", r"r$_i$ [cm]", mf=1e2)
#ions.evaluate_average("gamma", r"W$_i$ [keV]")

## Pickle dump
#pickle.dump(electrons.time, open(f"{path_save}/time.pkl", "wb"))
#pickle.dump(([np.average(gamma) for gamma in electrons.gamma], [np.std(gamma) for gamma in electrons.gamma]), open(f"{path_save}/gamma_e_pc.pkl", "wb"))
#pickle.dump(([np.average(r) for r in electrons.r], [np.std(r) for r in electrons.r]), open(f"{path_save}/r_e_pc.pkl", "wb"))
#with open(path_save+"ions_gamma_20.pkl", "wb") as f: pickle.dump(([np.average(pc.kin_en_from_gamma(gamma, ions.K_ref_keV)) for gamma in ions.gamma], [np.std(pc.kin_en_from_gamma(gamma, ions.K_ref_keV)) for gamma in ions.gamma]), f)
#We = pc.kin_en_from_gamma(electrons.gamma[18], electrons.K_ref_keV)
#print(f"Trapped electrons average energy: {np.average(We[We>2900])} keV, std: {np.std(We[We>2900])} keV")
#pickle.dump(We, open(f"{path_save}/electrons_edf_pc.pkl", "wb"))
#Wi = pc.kin_en_from_gamma(ions.gamma[-11], ions.K_ref_keV)
#print(f"Ions average energy: {np.average(Wi)} keV, std: {np.std(Wi)} keV")
#pickle.dump(Wi, open(f"{path_save}/ions_edf_pc.pkl", "wb"))
#print(np.asarray(ions.lost_part)/ions.px[0].shape[0])
#pp_e = [np.average(np.sqrt(px**2+py**2)) for px, py in zip(electrons.px, electrons.py)]
#pz_e = [np.average(pz) for pz in electrons.pz]
#pp_i = [np.average(np.sqrt(px**2+py**2)) for px, py in zip(ions.px, ions.py)]
#pz_i = [np.average(pz) for pz in ions.pz]

#plt.figure()
#plt.plot(electrons.time, pp_e, label="Electrons")
#plt.grid()
#plt.figure()
#plt.plot(electrons.time, pz_e, label="Electrons")
#plt.grid()
#plt.figure()
#plt.plot(ions.time, pp_i, label="Ions")
#plt.grid()
#plt.figure()
#plt.plot(ions.time, pz_i, label="Ions")
#plt.grid()



plt.show()