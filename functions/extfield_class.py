############################################################
######################### LIBRARIES ########################
############################################################

import numpy as np
import scipy.special as scisp
import re

############################################################
########################### CLASS ##########################
############################################################

'''------------------------------------------------ '''
''' Superclass for electric and magnetic field maps '''
'''------------------------------------------------ '''
class EBmaps:
    def __init__(self, skipline, minline, maxline, lenline, path, flag_magn, flag_puls, omega=0, ph=0, normalization=1, space_norm=1, time_norm=1, debug_flag=False):
        ''' Initialize field map object '''
        ## Main initialization
        self.path = path # Path to the file
        self.normalization = normalization
        self.space_norm = space_norm # Space normalization factor
        self.time_norm = time_norm # Time normalization factor
        self.ph = ph # Phase of pulsed field
        self.flag_magn = flag_magn # 1 if magnetic field, 0 if electric field
        self.flag_puls = flag_puls # 1 if pulsed field, 0 if static field
        self.debug = debug_flag # Print debug information
        with open(path, "r") as file: # Open field map
            lines = file.readlines()
        minvals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[minline]) # Find minimum values
        self.Rmin, self.Xmin = float(minvals[0])*space_norm, float(minvals[1])*space_norm
        maxvals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[maxline]) # Find maximum values
        self.Rmax, self.Xmax = float(maxvals[0])*space_norm, float(maxvals[1])*space_norm
        lenvals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[lenline]) # Find number of points
        self.Rlen, self.Xlen = int(lenvals[0])+1, int(lenvals[1])+1
        ## Further initialization
        self.readmap(lines[skipline:]) # Read field map
        #self.readmap_doubleindex(lines[skipline:]) # Read field map
        self.omega = omega/time_norm # Omega*time normalization factor (to have omega[1/s]*t[s])
        self.Rstep = (self.Rmax-self.Rmin)/(self.Rlen-1.) # Radial step
        self.Xstep = (self.Xmax-self.Xmin)/(self.Xlen-1.) # Axial step
        #self.setup_bicubic_interpolation() # Set up bicubic interpolation for field map
        ## Debug
        if debug_flag is True:
            print(f"Field map read from {path}")
            print(f"Field map size: {self.Rlen}x{self.Xlen}")
            print(f"Field map range: R = [{self.Rmin}, {self.Rmax}], X = [{self.Xmin}, {self.Xmax}]")
            print(f"Field map range in SI units: R = [{self.Rmin/self.space_norm}, {self.Rmax/self.space_norm}], X = [{self.Xmin/self.space_norm}, {self.Xmax/self.space_norm}]")
            print(f"Field map step: R = {self.Rstep}, X = {self.Xstep}")
            print(f"Normalization factor: {self.normalization}")
            print(f"Phase: {self.ph}")
            print(f"Omega: {self.omega}")
            print(f"Flag magnetic: {self.flag_magn}")
            print(f"Flag pulsed: {self.flag_puls}")
    
    def readmap(self, lines):
        ''' Read field map from file '''
        EBarray = np.loadtxt(lines)[:, 2:4]*self.normalization # Read field in a 2D array
        self.offset = self.Rlen*self.Xlen # Offset for indexing in 1d array
        ## Single array for field
        self.EBarr = np.zeros(2*self.offset) # Initialize 1D array for field
        for i in range(self.Xlen):
            for j in range(self.Rlen):
                self.EBarr[i*self.Rlen+j] = EBarray[i*self.Rlen+j, 0] # Assign radial field
                self.EBarr[self.offset+i*self.Rlen+j] = EBarray[i*self.Rlen+j, 1] # Assign axial field
                
    def readmap_doubleindex(self, lines):
        ''' Read field map from file into a 2D array '''
        EBarray = np.loadtxt(lines)[:, 2:4]*self.normalization # Read field in a 2D array
        self.EBarr = np.zeros((self.Xlen, self.Rlen, 2)) # Initialize 3D array for field
        for i in range(self.Xlen):
            for j in range(self.Rlen):
                self.EBarr[i, j, 0] = EBarray[i*self.Rlen+j, 0] # Assign radial field
                self.EBarr[i, j, 1] = EBarray[i*self.Rlen+j, 1] # Assign axial field
    

''' ------------------------------------------------------------------- '''
''' Subclass for electric and magnetic field maps in azimuthal symmetry '''
''' ------------------------------------------------------------------- '''
class EBmapsAM(EBmaps):
    def find_grid(self, x, r):
        ''' Find grid positions for all particles using numpy arrays x and r '''
        ## Check if map limits are respected
        if np.any(r < self.Rmin) or np.any(r > self.Rmax):
            raise ValueError(f"Radial position out of bounds: r = {r}, Rmin = {self.Rmin}, Rmax = {self.Rmax}")
        if np.any(x < self.Xmin) or np.any(x > self.Xmax):
            raise ValueError(f"Axial position out of bounds: x = {x}, Xmin = {self.Xmin}, Xmax = {self.Xmax}")
        ## Compute indexes and interpolation parameters
        rt = r - self.Rmin # Relative particle radial positions with respect to field map
        xt = x - self.Xmin # Relative particle axial positions with respect to field map
        ir = (rt/self.Rstep).astype(np.int32) # Lower index for r
        irp = ir+1
        ix = (xt/self.Xstep).astype(np.int32) # Lower index for x
        ixp = ix+1
        dr = 1.-(rt/self.Rstep - ir) # Radial parameter for interpolation
        idr = 1.-dr # Radial parameter for interpolation
        dx = 1.-(xt/self.Xstep - ix) # Axial parameter for interpolation
        idx = 1.-dx # Axial parameter for interpolation
        return ir, irp, ix, ixp, dr, idr, dx, idx

''' ---------------------------------------------- '''
''' Class for Transverse Electric (TE) mode fields '''
''' ---------------------------------------------- '''   
class TEfields:
    ## Initialize the class
    def __init__(self, m, n, p, L, R, z0, E0, c, fHF, T_ga, T_trans):
        ''' Initialize TE fields object '''
        ## Main initialization
        self.m = m # Order of TEM mode (mnp)
        self.n = n # Order of TEM mode (mnp)
        self.p = p # Order of TEM mode (mnp)
        self.L = L # Cavity length
        self.R = R # Cavity radius
        self.z0 = z0 # Axial displacement of the cavity
        self.E0 = E0 # Electric field amplitude
        self.B0 = E0/c # Magnetic field amplitude
        self.omegaHF = 2.*np.pi*fHF # Angular frequency of the HF fields (equal to self.omega)
        self.x_mn_pr = scisp.jnp_zeros(m, n)[0] # Zeros of Bessel function derivative
        self.T_ga = T_ga # Duration of the GA phase [s]
        self.T_trans = T_trans # Duration of the transition phase for turning of the HF fields [s]
        self.pw_flag = False
        ## Constants to speed up calculations
        self.k_x = np.pi*self.p/self.L # Longitudinal wave number
        self.k_mn = self.x_mn_pr/self.R # Transverse wave number
        self.omega = np.sqrt((self.k_x**2+self.k_mn**2)*c**2) # Angular frequency
        self.kxB0aix = self.k_x*self.B0*self.R/self.x_mn_pr # kx*B0*R/J'
        self.omegaB0aix = self.omega*self.B0*self.R/self.x_mn_pr # omega*B0*R/J'
        self.kxB0ma2ix2 = self.k_x*self.B0*self.m*self.R**2/self.x_mn_pr**2 # kx*B0*m*R^2/J'^2
        self.omegaB0ma2ix2 = self.omega*self.B0*self.m*self.R**2/self.x_mn_pr**2 # omega*B0*m*R^2/J'^2
    
    def evaluate_cartesian_components(self, x, y, z_raw):
        ''' Evaluate Cartesian components of TE fields at given coordinates '''
        ## Precompute coordinates
        r = (x**2 + y**2)**0.5
        z = z_raw - self.z0 # Axial position in the cavity frame
        r[np.where(r==0)] = 1e-10 # Avoid division by zero
        theta = np.arctan2(y, x) # Azimuthal angle
        ## Precompute TEM fields quantities
        krr = self.k_mn*r
        jpr = scisp.j0(krr)-scisp.j1(krr)/krr
        ## Compute cylindrical field components
        Br = self.kxB0aix*jpr*np.cos(self.k_x*z)*np.cos(self.m*theta)
        Bth = -self.kxB0ma2ix2*scisp.j1(self.k_mn*r)/r*np.cos(self.k_x*z)*np.sin(self.m*theta)
        Bz = self.B0*scisp.j1(self.k_mn*r)*np.sin(self.k_x*z)*np.cos(self.m*theta)
        Er = -self.omegaB0ma2ix2*scisp.j1(self.k_mn*r)/r*np.sin(self.k_x*z)*np.sin(self.m*theta)
        Eth = -self.omegaB0aix*jpr*np.sin(self.k_x*z)*np.cos(self.m*theta)
        ## Alternative forumlation
        #Er = self.E0*scisp.j1(self.k_mn*r)/self.k_mn/r*np.cos(self.k_x*z)*np.sin(self.m*theta)
        #Eth = self.E0*jpr*np.cos(self.k_x*z)*np.cos(self.m*theta)
        ## Convert to Cartesian coordinates
        Bx = Br*np.cos(theta) - Bth*np.sin(theta)
        By = Br*np.sin(theta) + Bth*np.cos(theta)
        Ex = Er*np.cos(theta) - Eth*np.sin(theta)
        Ey = Er*np.sin(theta) + Eth*np.cos(theta)
        return Bx, By, Bz, Ex, Ey
    
    def ramp_function(self, t):
        ''' Compute time behavior for the HF field '''
        if t < self.T_ga:
            return 1.
        elif t < self.T_ga + self.T_trans:
            return 1.-(t-self.T_ga)/self.T_trans
        else:
            return 0.
    
    def E_time_dependence(self, t): ## To modify for PC phase
        ''' Compute time dependence of E fields '''
        return np.sin(self.omegaHF*t)*self.ramp_function(t)
    
    def B_time_dependence(self, t): ## To modify for PC phase
        ''' Compute time dependence of B fields '''
        return np.cos(self.omegaHF*t)*self.ramp_function(t)

''' --------------------------------- '''
''' Class for Plane Waves (PW) fields '''
''' --------------------------------- '''
class PWfields:
    ## Initialize the class
    def __init__(self, z0, E0, c, fHF, T_ga, T_trans):
        ''' Initialize TE fields object '''
        ## Main initialization
        self.z0 = z0 # Axial displacement of the cavity
        self.E0 = E0 # Electric field amplitude
        self.omegaHF = 2.*np.pi*fHF # Angular frequency of the HF fields (equal to self.omega)
        self.k = self.omegaHF/c # Wave number
        self.T_ga = T_ga # Duration of the GA phase [s]
        self.T_trans = T_trans # Duration of the transition phase for turning of the HF fields [s]
        self.E_time_dependence = self.B_time_dependence = lambda t: 0. # Set variables to avoid errors when evaluating fields with plane wave class
        self.pw_flag = True # Flag to identify plane wave fields when evaluating total fields in extfields class
    
    def evaluate_cartesian_components(self, x, y, z):
        ''' Return null arrays for generality of the code, as plane wave fields cannot be evaluated without time dependence '''
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    
    def ramp_function(self, t):
        ''' Compute time behavior for the HF field '''
        if t < self.T_ga:
            return 1.
        elif t < self.T_ga + self.T_trans:
            return 1.-(t-self.T_ga)/self.T_trans
        else:
            return 0.
    
    def evaluate_E(self, z, t):
        ''' Evaluate E field at given axial coordinate and time '''
        z_cav = z - self.z0 # Axial position in the cavity frame
        Ex = self.E0*np.cos(self.omegaHF*t-self.k*z_cav)*self.ramp_function(t)
        Ey = self.E0*np.sin(self.omegaHF*t-self.k*z_cav)*self.ramp_function(t)
        return Ex, Ey

''' ----------------------------------------------- '''
''' Class for complete external fields contribution '''
''' ----------------------------------------------- '''
class extfields:
    def __init__(self, path_static, path_pulsed, path_induced, f_pul, hf_class, ph_pul=np.pi/2.):
        ''' Initialize external fields object '''
        ## Initialize static and pulsed field maps
        self.omega_pul = 2*np.pi*f_pul
        self.phase_pul = ph_pul
        self.Bstatic = EBmapsAM(4, 0, 1, 2, path_static, 1, 0)
        self.Bpulsed = EBmapsAM(4, 0, 1, 2, path_pulsed, 1, 1, omega=self.omega_pul, ph=self.phase_pul) # COMSOL pulsed B field
        self.Einduced = EBmapsAM(4, 0, 1, 2, path_induced, 0, 1, omega=self.omega_pul, ph=self.phase_pul) # COMSOL induced E field
        self.hf_class = hf_class # High frequency fields class (e.g. TEfields) for additional field contributions (e.g. from cavity modes)
        ## Initialize index flag to avoid unnecessary index evaluations
        self.index_flag = True # Flag to evaluate indexes
    
    def Bpulsed_time_dependence(self, t):
        ''' Compute time dependence of pulsed field '''
        return np.sin(self.omega_pul*t+self.phase_pul)
    
    def Einduced_time_dependence(self, t):
        ''' Compute time dependence of induced field '''
        return np.cos(self.omega_pul*t+self.phase_pul)
    
    ''' ---------------------------------- '''
    ''' Cylindrical components evaluation  '''
    ''' ---------------------------------- '''
    def evaluate_maps_cylindrical_components(self, r, z, evaluate_indexes=False):
        ''' Compute field maps cylindrical components at given coordinates '''
        ## Evaluate indexes
        if evaluate_indexes or self.index_flag:
            self.ir, self.irp, self.ix, self.ixp, self.dr, self.idr, self.dx, self.idx = self.Bstatic.find_grid(z, r)
            self.index_flag = False
        ## Evaluate cylindrical field components
        B_static_z = self.Bstatic.EBarr[self.Bstatic.offset+self.ix*self.Bstatic.Rlen+self.ir]*self.dr*self.dx + self.Bstatic.EBarr[self.Bstatic.offset+self.ix*self.Bstatic.Rlen+self.irp]*self.idr*self.dx + self.Bstatic.EBarr[self.Bstatic.offset+self.ixp*self.Bstatic.Rlen+self.ir]*self.dr*self.idx + self.Bstatic.EBarr[self.Bstatic.offset+self.ixp*self.Bstatic.Rlen+self.irp]*self.idr*self.idx # Bilinear interpolation for axial field component
        B_pulsed_z = self.Bpulsed.EBarr[self.Bpulsed.offset+self.ix*self.Bpulsed.Rlen+self.ir]*self.dr*self.dx + self.Bpulsed.EBarr[self.Bpulsed.offset+self.ix*self.Bpulsed.Rlen+self.irp]*self.idr*self.dx + self.Bpulsed.EBarr[self.Bpulsed.offset+self.ixp*self.Bpulsed.Rlen+self.ir]*self.dr*self.idx + self.Bpulsed.EBarr[self.Bpulsed.offset+self.ixp*self.Bpulsed.Rlen+self.irp]*self.idr*self.idx # Bilinear interpolation for axial field component
        B_static_r = self.Bstatic.EBarr[self.ix*self.Bstatic.Rlen+self.ir]*self.dr*self.dx + self.Bstatic.EBarr[self.ix*self.Bstatic.Rlen+self.irp]*self.idr*self.dx + self.Bstatic.EBarr[self.ixp*self.Bstatic.Rlen+self.ir]*self.dr*self.idx + self.Bstatic.EBarr[self.ixp*self.Bstatic.Rlen+self.irp]*self.idr*self.idx # Bilinear interpolation for radial field component
        B_pulsed_r = self.Bpulsed.EBarr[self.ix*self.Bpulsed.Rlen+self.ir]*self.dr*self.dx + self.Bpulsed.EBarr[self.ix*self.Bpulsed.Rlen+self.irp]*self.idr*self.dx + self.Bpulsed.EBarr[self.ixp*self.Bpulsed.Rlen+self.ir]*self.dr*self.idx + self.Bpulsed.EBarr[self.ixp*self.Bpulsed.Rlen+self.irp]*self.idr*self.idx # Bilinear interpolation for radial field component
        E_induced_r = self.Einduced.EBarr[self.ix*self.Einduced.Rlen+self.ir]*self.dr*self.dx + self.Einduced.EBarr[self.ix*self.Einduced.Rlen+self.irp]*self.idr*self.dx + self.Einduced.EBarr[self.ixp*self.Einduced.Rlen+self.ir]*self.dr*self.idx + self.Einduced.EBarr[self.ixp*self.Einduced.Rlen+self.irp]*self.idr*self.idx # Bilinear interpolation for induced field component
        return B_static_r, B_pulsed_r, E_induced_r, B_static_z, B_pulsed_z

    def evaluate_cylindrical_components(self, r, z, key, evaluate_indexes=False):
        ''' Compute cylindrical components of complete external fields at given coordinates '''
        static, pulsed = {}, {}
        ## Evaluate components from each field contribution -- HF fields not considered because simulation when either TE or PW fields are present is not cylindrically symmetric
        static["Br"], pulsed["Br"], pulsed["Et"], static["Bz"], pulsed["Bz"] = self.evaluate_maps_cylindrical_components(r, z, evaluate_indexes=evaluate_indexes)
        ## Set to zero other components
        static["Bt"], static["Er"], static["Et"], static["Ez"] = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)
        pulsed["Bt"], pulsed["Er"], pulsed["Ez"] = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)
        ## Set fields time behavior
        pulsed["Br_tb"] = pulsed["Bz_tb"] = self.Bpulsed_time_dependence
        pulsed["Et_tb"] = self.Einduced_time_dependence
        pulsed["Bt_tb"] = pulsed["Er_tb"] = pulsed["Ez_tb"] = lambda t: 0
        ## Return field component corresponding to key
        return static[key], pulsed[key], pulsed[key+"_tb"]
        
    ''' -------------------------------- '''
    ''' Cartesian components evaluation  '''
    ''' -------------------------------- '''
    def evaluate_maps_cartesian_components(self, x, y, z, evaluate_indexes=False):
        ''' Compute field maps cartesian components at given coordinates '''
        ## Evaluate indexes and cylindrical components
        r = (x**2 + y**2)**0.5
        B_static_r, B_pulsed_r, E_induced_r, B_static_z, B_pulsed_z = self.evaluate_maps_cylindrical_components(r, z, evaluate_indexes=evaluate_indexes)
        ## Return Cartesian components
        return B_static_r*x/r, B_pulsed_r*x/r, -E_induced_r*y/r, B_static_r*y/r, B_pulsed_r*y/r, E_induced_r*x/r, B_static_z, B_pulsed_z
    
    def evaluate_cartesian_components(self, x, y, z, key, evaluate_indexes=False):
        ''' Compute Cartesian components of complete external fields at given coordinates '''
        static, pulsed, hf = {}, {}, {}
        ## Evaluate components from each field contribution
        static["Bx"], pulsed["Bx"], pulsed["Ex"], static["By"], pulsed["By"], pulsed["Ey"], static["Bz"], pulsed["Bz"] = self.evaluate_maps_cartesian_components(x, y, z, evaluate_indexes=evaluate_indexes)
        hf["Bx"], hf["By"], hf["Bz"], hf["Ex"], hf["Ey"] = self.hf_class.evaluate_cartesian_components(x, y, z)
        ## Set to zero other components
        static["Ex"], static["Ey"], static["Ez"] = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        pulsed["Ez"] = np.zeros_like(x)
        hf["Ez"] = np.zeros_like(x)
        ## Set fields time behavior
        pulsed["Bx_tb"] = pulsed["By_tb"] = pulsed["Bz_tb"] = self.Bpulsed_time_dependence
        pulsed["Ex_tb"] = pulsed["Ey_tb"] = self.Einduced_time_dependence
        hf["Ex_tb"] = hf["Ey_tb"] = self.hf_class.E_time_dependence
        hf["Bx_tb"] = hf["By_tb"] = hf["Bz_tb"] = self.hf_class.B_time_dependence
        pulsed["Ez_tb"] = hf["Ez_tb"] = lambda t: 0
        ## Return field component corresponding to key
        return static[key], pulsed[key], pulsed[key+"_tb"], hf[key], hf[key+"_tb"]
    
    def evaluate_Bfield(self, x, y, z, t, evaluate_indexes=False):
        ''' Compute total B field at given coordinates and time '''
        Bx_s, Bx_p, Bx_p_tb, Bx_hf, Bx_hf_tb = self.evaluate_cartesian_components(x, y, z, "Bx", evaluate_indexes=evaluate_indexes)
        By_s, By_p, By_p_tb, By_hf, By_hf_tb = self.evaluate_cartesian_components(x, y, z, "By", evaluate_indexes=evaluate_indexes)
        Bz_s, Bz_p, Bz_p_tb, Bz_hf, Bz_hf_tb = self.evaluate_cartesian_components(x, y, z, "Bz", evaluate_indexes=evaluate_indexes)
        Bx = Bx_s + Bx_p*Bx_p_tb(t) + Bx_hf*Bx_hf_tb(t)
        By = By_s + By_p*By_p_tb(t) + By_hf*By_hf_tb(t)
        Bz = Bz_s + Bz_p*Bz_p_tb(t) + Bz_hf*Bz_hf_tb(t)
        return (Bx, By, Bz)
    
    def evaluate_Efield(self, x, y, z, t, evaluate_indexes=False):
        ''' Compute total E field at given coordinates and time '''
        Ex_s, Ex_p, Ex_p_tb, Ex_hf, Ex_hf_tb = self.evaluate_cartesian_components(x, y, z, "Ex", evaluate_indexes=evaluate_indexes)
        Ey_s, Ey_p, Ey_p_tb, Ey_hf, Ey_hf_tb = self.evaluate_cartesian_components(x, y, z, "Ey", evaluate_indexes=evaluate_indexes)
        Ex = Ex_p*Ex_p_tb(t) + Ex_hf*Ex_hf_tb(t)
        Ey = Ey_p*Ey_p_tb(t) + Ey_hf*Ey_hf_tb(t)
        return (Ex, Ey, np.zeros_like(Ex))
    
    def evaluate_Efield_pw(self, x, y, z, t, evaluate_indexes=False):
        ''' Compute total E field at given coordinates and time for plane wave fields only '''
        Ex_s, Ex_p, Ex_p_tb, Ex_hf, Ex_hf_tb = self.evaluate_cartesian_components(x, y, z, "Ex", evaluate_indexes=evaluate_indexes)
        Ey_s, Ey_p, Ey_p_tb, Ey_hf, Ey_hf_tb = self.evaluate_cartesian_components(x, y, z, "Ey", evaluate_indexes=evaluate_indexes)
        Ex_pw, Ey_pw = self.hf_class.evaluate_E(z, t)
        Ex = Ex_p*Ex_p_tb(t) + Ex_pw
        Ey = Ey_p*Ey_p_tb(t) + Ey_pw
        return (Ex, Ey, np.zeros_like(Ex))