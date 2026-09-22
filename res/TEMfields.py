############################################################
######################### LIBRARIES ########################
############################################################

import numpy as np
import scipy.special as scisp

############################################################
########################### CLASS ##########################
############################################################

class TEMfields:
    ## Initialize the class
    def __init__(self, m, n, p, L, R, x0, E0, c):
        ## Main initialization
        self.m = m # Order of TEM mode (mnp)
        self.n = n # Order of TEM mode (mnp)
        self.p = p # Order of TEM mode (mnp)
        self.L = L # Cavity length
        self.R = R # Cavity radius
        self.x0 = x0 # Axial displacement of the cavity
        self.B0 = E0/c # Magnetic field amplitude
        self.x_mn_pr = scisp.jnp_zeros(m, n)[0] # Zeros of Bessel function derivative
        ## Constants to speed up calculations
        self.k_x = np.pi*self.p/self.L # Longitudinal wave number
        self.k_mn = self.x_mn_pr/self.R # Transverse wave number
        self.omega = np.sqrt((self.k_x**2+self.k_mn**2)*c**2) # Angular frequency
        self.kxB0aix = self.k_x*self.B0*self.R/self.x_mn_pr # kx*B0*R/J'
        self.omegaB0aix = self.omega*self.B0*self.R/self.x_mn_pr # omega*B0*R/J'
        self.kxB0ma2ix2 = self.k_x*self.B0*self.m*self.R**2/self.x_mn_pr**2 # kx*B0*m*R^2/J'^2
        self.omegaB0ma2ix2 = self.omega*self.B0*self.m*self.R**2/self.x_mn_pr**2 # omega*B0*m*R^2/J'^2
        
#

class TE111fieldsAM(TEMfields):
    def __init__(self, L, R, x0, E0, c=1):
        super().__init__(1, 1, 1, L, R, x0, E0, c)
        
    ## Calculate fields
    def evaluateBx(self, x_raw, r, t):
        x = x_raw-self.x0 # Axial position in the cavity frame
        Bx = self.B0*scisp.j1(self.k_mn*r)*np.sin(self.k_x*x)*np.cos(self.omega*t)
        return Bx
    
    def evaluateBr(self, x_raw, r, t):
        x = x_raw-self.x0 # Axial position in the cavity frame
        zero_ind_r = np.where(r==0) # In theory it's useless since r=0 is a BC
        r[zero_ind_r] = 1e-10 # Avoid division by zero
        krr = self.k_mn*r
        jpr = scisp.j0(krr)-scisp.j1(krr)/krr
        Br = self.kxB0aix*jpr*np.cos(self.k_x*x)*np.cos(self.omega*t)
        return Br
    
    def evaluateBt(self, x_raw, r, t):
        x = x_raw-self.x0 # Axial position in the cavity frame
        zero_ind_r = np.where(r==0) # In theory it's useless since r=0 is a BC
        r[zero_ind_r] = 1e-10 # Avoid division by zero
        Bth = -self.kxB0ma2ix2*scisp.j1(self.k_mn*r)/r*np.cos(self.k_x*x)*np.cos(self.omega*t)
        return Bth
    
    def evaluateEr(self, x_raw, r, t):
        x = x_raw-self.x0 # Axial position in the cavity frame
        zero_ind_r = np.where(r==0) # In theory it's useless since r=0 is a BC
        r[zero_ind_r] = 1e-10 # Avoid division by zero
        Er = -self.omegaB0ma2ix2*scisp.j1(self.k_mn*r)/r*np.sin(self.k_x*x)*np.sin(self.omega*t)
        return Er
    
    def evaluateEt(self, x_raw, r, t):
        x = x_raw-self.x0 # Axial position in the cavity frame
        zero_ind_r = np.where(r==0) # In theory it's useless since r=0 is a BC
        r[zero_ind_r] = 1e-10 # Avoid division by zero
        krr = self.k_mn*r
        jpr = scisp.j0(krr)-scisp.j1(krr)/krr
        Eth = -self.omegaB0aix*jpr*np.sin(self.k_x*x)*np.sin(self.omega*t)
        return Eth
    
    def debug_func(self, x_raw, r, t):
        x = x_raw-self.x0 # Axial position in the cavity frame
        zero_ind_r = np.where(r==0) # In theory it's useless since r=0 is a BC
        r[zero_ind_r] = 1e-10 # Avoid division by zero
        krr = self.k_mn*r
        jpr = scisp.j0(krr)-scisp.j1(krr)/krr
        print(f"kzz = {self.k_x*x[0]}, kr = {self.k_mn*r[0]}, omega*t = {self.omega*t}")
        print(f"cos(kzz) = {np.cos(self.k_x*x)[0]}, cosomega*t = {np.cos(self.omega*t)}, J1(kr) = {scisp.j1(self.k_mn*r)[0]}, J'(kr) = {jpr[0]}")