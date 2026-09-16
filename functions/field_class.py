import sys
sys.path.append("/home/cernuschi/coding/warpx/ecripac/functions/")
from warpx_class import *
from scipy.ndimage import map_coordinates
       
''' --------------------------- '''
''' Field diagnostic superclass '''
''' --------------------------- '''
class field_diagnostic(warpx_namelist):
    ''' ------------------------ '''
    ''' Initialization functions '''
    ''' ------------------------ '''
    @classmethod
    def from_path(cls, path_data, extfield=False, save_address=None, t_disp=0):
        """ Factory method selecting the correct field diagnostic subclass """
        ## Construct OpenPMD object once
        warpx = OpenPMDTimeSeries(path_data)
        # Determine geometry
        _, field_info = warpx.get_field(field="rho", iteration=0)
        geometry = field_info.component_attrs["geometry"]
        # Select subclass
        print(f"Geometry is {geometry}")
        if geometry == "cartesian":
            field_class = cartesian_field
        elif geometry == "thetaMode":
            field_class = cylindrical_field
        else:
            raise ValueError(f"Unknown geometry '{geometry}'")
        # Instantiate selected subclass
        return field_class(warpx, extfield=extfield, save_address=save_address, t_disp=t_disp)
    
    def __init__(self, warpx, extfield=False, save_address=None, t_disp=0):
        """ Common initialization for all field diagnostics """
        ## Initialize superclass
        super().__init__(warpx, save_address)
        if save_address is not None:
            self.save_address = save_address
        ## Extract attributes from WarpX data
        self.timesteps = warpx.iterations # List of timesteps available
        self.time = warpx.t+t_disp # List of time values corresponding to timesteps
        self.dt = self.time[1] / self.timesteps[1] # Timestep size
        self.T = warpx.tmax+t_disp # Maximum time
        ## Get class attributes
        self.class_attributes(warpx, extfield)
    
    def get_warpx_attributes(self, warpx, extfield, field, comp=None):
        ''' Function to get attributes from WarpX data '''
        if extfield is False:
            return np.asarray([warpx.get_field(field=field, coord=comp, iteration=it)[0] for it in self.timesteps]) # Axis order: (z, y, x) for cartesian, (z, r) for cylindrical
        elif self.geometry == "cartesian":
            zgrid, ygrid, xgrid = np.meshgrid(self.zgrid, self.ygrid, self.xgrid, indexing="ij")
            ef_st, ef_pu, ef_pu_tb, ef_hf, ef_hf_tb = extfield.evaluate_cartesian_components(xgrid, ygrid, zgrid, field+comp)
            return np.asarray([warpx.get_field(field=field, coord=comp, iteration=it)[0] - (ef_st + ef_pu*ef_pu_tb(t) + ef_hf*ef_hf_tb(t)) for it, t in zip(self.timesteps,self.time)]) # Axis order: (z, y, x)
        elif self.geometry == "thetaMode":
            zgrid, rgrid = np.meshgrid(self.zgrid, np.abs(self.rgrid), indexing="ij")
            ef_st, ef_pu, ef_pu_tb = extfield.evaluate_cylindrical_components(rgrid, zgrid, field+comp)
            return np.asarray([warpx.get_field(field=field, coord=comp, iteration=it)[0] - (ef_st + ef_pu*ef_pu_tb(t)) for it, t in zip(self.timesteps,self.time)]) # Axis order: (z, r)


class cartesian_field(field_diagnostic):
    ''' ------------------------ '''
    ''' Initialization functions '''
    ''' ------------------------ '''
    def class_attributes(self, warpx, extfield): 
        ''' Function to create class attributes '''
        self.rho = self.get_warpx_attributes(warpx, False, "rho")
        self.Jx = self.get_warpx_attributes(warpx, False, "j", comp="x")
        self.Jy = self.get_warpx_attributes(warpx, False, "j", comp="y")
        self.Jz = self.get_warpx_attributes(warpx, False, "j", comp="z")
        self.Ex = self.get_warpx_attributes(warpx, extfield, "E", comp="x")
        self.Ey = self.get_warpx_attributes(warpx, extfield, "E", comp="y")
        self.Ez = self.get_warpx_attributes(warpx, extfield, "E", comp="z")
        self.Bx = self.get_warpx_attributes(warpx, extfield, "B", comp="x")
        self.By = self.get_warpx_attributes(warpx, extfield, "B", comp="y")
        self.Bz = self.get_warpx_attributes(warpx, extfield, "B", comp="z")
        self.J = np.asarray([np.sqrt(Jx**2 + Jy**2 + Jz**2) for Jx, Jy, Jz in zip(self.Jx, self.Jy, self.Jz)])
        self.E = np.asarray([np.sqrt(Ex**2 + Ey**2 + Ez**2) for Ex, Ey, Ez in zip(self.Ex, self.Ey, self.Ez)])
        self.B = np.asarray([np.sqrt(Bx**2 + By**2 + Bz**2) for Bx, By, Bz in zip(self.Bx, self.By, self.Bz)])
    

    ''' -------------------------------------------- '''
    ''' Subroutines for cartesian plotting functions '''
    ''' -------------------------------------------- '''
    def double_fourier_transform(self, field, comp, nx, ny, nz, ts, flag_mean=True, flag_window=True, flag_pomega=True, logscale=True):
        ''' Function to compute the double Fourier transform of a given field component along a given axis '''
        ## Extract field component
        f = getattr(self, field)[:ts, nz, ny, nx].copy()
        Nt, Ns = f.shape
        ## Subtract mean value to remove k=0 component
        if flag_mean:
            f -= np.mean(f, axis=1, keepdims=True)
        ## Apply window function to reduce spectral leakage
        if flag_window:
            wt = np.hanning(Nt)[:, None]
            ws = np.hanning(Ns)[None, :]
            f *= wt*ws
        ## 2D FFT
        F = np.fft.fft2(f)
        P = np.abs(F)**2
        ## Shift zero frequency component to center of spectrum
        P = np.fft.fftshift(P)
        ## Obtain axes
        k = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Ns, d=getattr(self, "d"+comp)))
        omega = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Nt, d=self.time[1]-self.time[0]))
        ## Set only positive omega
        if flag_pomega:
            mask = omega > 0
            omega = omega[mask]
            P = P[mask, :]
        ## Apply log scale
        if logscale:
            P = np.log10(P + 1e-30)
        return k, omega, P
    
    def plot_longitudinal_dispersion(self, ts, k, omega, electrons, ions, ax, lines=None):
        ''' Plot relevant longitudinal dispersion relations for the plasma '''
        ## Obtain relevant plasma frequencies and dispersion relations
        omega_pe = plasma_frequency(electrons, ts)
        omega_pi = plasma_frequency(ions, ts)
        omega_uh = omega_upper_hybrid(electrons, ts)
        omega_lh = omega_lower_hybrid(electrons, ions, ts)
        dr_bohm_gross = bohm_gross_dr(electrons, ts, k)
        print(f"Plasma frequencies at time {self.time[ts]}: omega_pe = {omega_pe:.2e} rad/s, omega_pi = {omega_pi:.2e} rad/s")
        print(f"Plasma frequencies at time {self.time[ts]}: omega_uh = {omega_uh:.2e} rad/s, omega_lh = {omega_lh:.2e} rad/s")
        ## Create lines dictionary if not provided
        if lines is None:
            lines = {
                "pe": ax.plot([], [], color="r", linestyle="--")[0],
                "pi": ax.plot([], [], color="r", linestyle="--")[0],
                "uh": ax.plot([], [], color="r", linestyle="--")[0],
                "lh": ax.plot([], [], color="r", linestyle="--")[0],
                "bg": ax.plot([], [], color="r", linestyle="-")[0],
            }
        ## Update dispersion relations in place
        if omega_pe < omega[-1]:
            lines["pe"].set_data(k, np.ones_like(k) * omega_pe)
        else:
            lines["pe"].set_data([], [])
        if omega_pi < omega[-1]:
            lines["pi"].set_data(k, np.ones_like(k) * omega_pi)
        else:
            lines["pi"].set_data([], [])
        if omega_uh < omega[-1]:
            lines["uh"].set_data(k, np.ones_like(k) * omega_uh)
        else:
            lines["uh"].set_data([], [])
        if omega_lh < omega[-1]:
            lines["lh"].set_data(k, np.ones_like(k) * omega_lh)
        else:
            lines["lh"].set_data([], [])
        mask = dr_bohm_gross < omega[-1]
        lines["bg"].set_data(k[mask], dr_bohm_gross[mask])
        return lines
        
    def plot_transversal_dispersion(self, ts, k, omega, electrons, ions, ax, lines=None):
        ''' Plot relevant transversal dispersion relations for the plasma '''
        ## Obtain relevant plasma frequencies and dispersion relations
        Omega_e = gyrofrequency(electrons, ts)
        Omega_i = gyrofrequency(ions, ts)
        omega_pe = plasma_frequency(electrons, ts)
        omega_pi = plasma_frequency(ions, ts)
        omega_r = omega_righthand(electrons, ts)
        omega_l = omega_lefthand(electrons, ts)
        ca = alfven_velocity(ions, ts)
        ca_dr = np.abs(ca*k)
        void = np.abs(sciconst.c*k)
        print(f"Gyrofrequencies at time {self.time[ts]}: Omega_e = {Omega_e:.2e} rad/s, Omega_i = {Omega_i:.2e} rad/s")
        print(f"Cutoff frequencies at time {self.time[ts]}: omega_r = {omega_r:.2e} rad/s, omega_l = {omega_l:.2e} rad/s")
        print(f"Alfven frequency at time {self.time[ts]}: ca = {ca:.2e} m/s")
        ## Create lines dictionary if not provided
        if lines is None:
            lines = {
                "omega_pe": ax.plot([], [], color="r", linestyle="--")[0],
                "omega_pi": ax.plot([], [], color="r", linestyle="--")[0],
                "Omega_e": ax.plot([], [], color="r", linestyle="--")[0],
                "Omega_i": ax.plot([], [], color="r", linestyle="--")[0],
                "omega_r": ax.plot([], [], color="r", linestyle="--")[0],
                "omega_l": ax.plot([], [], color="r", linestyle="--")[0],
                "ca": ax.plot([], [], color="r", linestyle="-")[0],
                "void": ax.plot([], [], color="r", linestyle="-")[0],
            }
        ## Update dispersion relations in place
        if omega_pe < omega[-1]:
            lines["omega_pe"].set_data(k, np.ones_like(k) * omega_pe)
        else:
            lines["omega_pe"].set_data([], [])
        if omega_pi < omega[-1]:
            lines["omega_pi"].set_data(k, np.ones_like(k) * omega_pi)
        else:
            lines["omega_pi"].set_data([], [])
        if Omega_e < omega[-1]:
            lines["Omega_e"].set_data(k, np.ones_like(k) * Omega_e)
        else:
            lines["Omega_e"].set_data([], [])
        if Omega_i < omega[-1]:
            lines["Omega_i"].set_data(k, np.ones_like(k) * Omega_i)
        else:
            lines["Omega_i"].set_data([], [])
        if omega_r < omega[-1]:
            lines["omega_r"].set_data(k, np.ones_like(k) * omega_r)
        else:
            lines["omega_r"].set_data([], [])
        if omega_l < omega[-1]:
            lines["omega_l"].set_data(k, np.ones_like(k) * omega_l)
        else:
            lines["omega_l"].set_data([], [])
        mask_ca = ca_dr < omega[-1]
        lines["ca"].set_data(k[mask_ca], ca_dr[mask_ca])
        mask_void = void < omega[-1]
        lines["void"].set_data(k[mask_void], void[mask_void])
        return lines
    
    
    ''' ---------------------------------------------- '''
    ''' Subroutines for cylindrical plotting functions '''
    ''' ---------------------------------------------- '''
    def _interp_polar(self, field, nz, x_coords, y_coords, ts, interp_order=3):
        ''' Interpolate a stored (t, z, y, x) field array onto a set of (x, y) points at fixed z, for every stored timestep at once '''
        ## Extract the field data at the specified z index and the relative information
        data = getattr(self, field)[:ts, nz, :, :] # shape (Nt, Ny, Nx)
        Nt = data.shape[0]
        ## Determine the spatial indexes for the circle
        Np = x_coords.size
        x_idx = (x_coords - self.xgrid[0]) / self.dx
        y_idx = (y_coords - self.ygrid[0]) / self.dy
        ## Build spatial and temporal arrays and extract the field values at the specified points
        t_idx = np.repeat(np.arange(Nt), Np)
        x_idx_full = np.tile(x_idx, Nt)
        y_idx_full = np.tile(y_idx, Nt)
        f = map_coordinates(data, [t_idx, y_idx_full, x_idx_full], order=interp_order, mode='nearest')
        return f.reshape(Nt, Np)
        
    def polar_fourier_transform(self, field, comp, direction, nz, ts, r0=0.02, theta0=0.0, rmin=0., rmax=None, nr=128, ntheta=256, x0=0., y0=0., interp_order=3, flag_mean=True, flag_window=True, flag_pomega=True, logscale=True):
        '''
        Double Fourier transform of a field component in polar coordinates around (x0, y0), at fixed z. Mirrors double_fourier_transform but replaces the Cartesian line/axis with:
          direction="r"     -> a line at fixed angle theta0, r in [rmin, rmax]
                                -> returns (kr, omega, P)
          direction="theta" -> a ring at fixed radius r0, theta in [0, 2*pi)
                                -> returns (m, omega, P), m the integer azimuthal mode number
        comp: "r" or "theta" projects (field+"x", field+"y") onto the local radial/azimuthal unit vectors -- this is what you want for E, since a raw Ex or Ey mixes the two physical components in a theta-dependent way. comp="" samples the named scalar field (e.g. density, potential) directly, with no projection.
        '''
        ## Set default values for rmax if not provided
        if rmax is None:
            rmax = self.xmax
        ## Create the coordinates for the direction of interest
        if direction == "theta":
            theta = np.linspace(0.0, 2*np.pi, ntheta, endpoint=False)
            x_coords = x0 + r0*np.cos(theta)
            y_coords = y0 + r0*np.sin(theta)
            Ns = ntheta
        elif direction == "r":
            r_grid = np.linspace(rmin, rmax, nr)
            theta = theta0
            x_coords = x0 + r_grid*np.cos(theta0)
            y_coords = y0 + r_grid*np.sin(theta0)
            Ns = nr
        else:
            raise ValueError('direction must be "r" or "theta"')
        ## Interpolate field onto the ring/line, projecting onto the local radial/azimuthal unit vectors if requested
        if comp in ("r", "theta"):
            fx = self._interp_polar(field+"x", nz, x_coords, y_coords, ts, interp_order)
            fy = self._interp_polar(field+"y", nz, x_coords, y_coords, ts, interp_order)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            f = (fx*cos_t + fy*sin_t) if comp == "r" else (-fx*sin_t + fy*cos_t)
        else:
            f = self._interp_polar(field, nz, x_coords, y_coords, ts, interp_order)
        Nt = f.shape[0]
        ## Subtract mean to remove the k=0 / m=0 component
        if flag_mean:
            f = f - np.mean(f, axis=1, keepdims=True)
        ## Window function: theta is exactly periodic over 2*pi, so only window in time for the azimuthal case -- windowing a periodic signal in theta only costs you resolution for no benefit. The radial line is a genuine finite, non-periodic cut, so window both axes there, as in the Cartesian version.
        if flag_window:
            wt = np.hanning(Nt)[:, None]
            f = f*wt if direction == "theta" else f*wt*np.hanning(Ns)[None, :]
        ## 2D FFT
        F = np.fft.fft2(f)
        P = np.fft.fftshift(np.abs(F)**2)
        if direction == "theta":
            ## d=1/Ns makes fftfreq return the integer mode number m directly
            k = np.round(np.fft.fftshift(np.fft.fftfreq(Ns, d=1.0/Ns))).astype(int)
        else:
            dr = r_grid[1] - r_grid[0]
            k = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Ns, d=dr))
        omega = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Nt, d=self.time[1]-self.time[0]))
        ## Keep only positive omega
        if flag_pomega:
            mask = omega > 0
            omega = omega[mask]
            P = P[mask, :]
        ## Apply log scale to enhance visibility
        if logscale:
            P = np.log10(P + 1e-30)
        ## Return dispersion relation
        return k, omega, P
    
    def plot_harmonic_overlay(self, ax, mmax, Omega=0., lines=None, electrons=None, ts=None):
        ''' Overlay the rigid-rotation harmonic lines omega = |m|*Omega expected from a bunch orbiting at angular frequency Omega, for direct comparison with the (m, omega) spectrum '''
        ## Set real gyrofrequency if available
        if electrons is not None:
            Omega = gyrofrequency(electrons, ts)
        ## Plot the harmonic lines
        m_line = np.arange(0, mmax+1)
        omega_line = m_line*Omega
        if lines is None:
            line_pos, = ax.plot(m_line, omega_line, 'w--', lw=1.2, label=r'$\omega=m\Omega$')
            line_neg, = ax.plot(-m_line, omega_line, 'w--', lw=1.2)
            ax.legend(loc='upper right', fontsize=8)
            return (line_pos, line_neg)
        else:
            lines[0].set_data(m_line, omega_line)
            lines[1].set_data(-m_line, omega_line)
            return lines
        
    ''' ---------------------------- '''
    ''' Cartesian plotting functions '''
    ''' ---------------------------- '''
    def plot_field2D(self, field, comp="", vmin=None, vmax=None):
        ''' Create a slider plot for a given field component '''
        ## Create slider function for spatial translation
        def update_space(val, attr, ind, plot, cbar, array):
            setattr(self, attr, list(array).index(val))
            field_array = self.fgrid[ind(getattr(self, attr))]
            if attr != "z_gridind":
                field_array = np.transpose(field_array)
            update_2Ddensityplot(field_array, plot, cbar, vmin=vmin, vmax=vmax)
            plt.draw()
        ## Create slider function for time evolution
        def update_time(val):
            ind = list(self.timesteps).index(val)
            self.fgrid = getattr(self, field+comp)[ind]
            update_2Ddensityplot(self.fgrid[self.z_gridind, :, :], xy_den, xy_cbar, vmin=vmin, vmax=vmax)
            update_2Ddensityplot(np.transpose(self.fgrid[:, self.y_gridind, :]), xz_den, xz_cbar, vmin=vmin, vmax=vmax)
            update_2Ddensityplot(np.transpose(self.fgrid[:, :, self.x_gridind]), yz_den, yz_cbar, vmin=vmin, vmax=vmax)
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
            plt.draw()
            print(f"{field}{comp} at t = {self.time[ind]:.2e} s: min = {np.amin(self.fgrid):.2e}, max = {np.amax(self.fgrid):.2e}")
        ## Create initial density plots
        vmin = np.amin(getattr(self, field+comp)) if vmin is True else vmin
        vmax = np.amax(getattr(self, field+comp)) if vmax is True else vmax
        self.fgrid, unit = getattr(self, field+comp)[0], getattr(self, f"{field}_unit")
        self.x_gridind, self.y_gridind, self.z_gridind = self.ncell_x//2, self.ncell_y//2, self.ncell_z//2
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        xy_den, xy_cbar = imshowplot(np.flipud(self.fgrid[self.z_gridind, :, :]), [self.xmin, self.xmax, self.ymin, self.ymax], field+comp+unit, "x [m]", "y [m]", ax=ax[0], lmin=vmin, lmax=vmax)
        xz_den, xz_cbar = imshowplot(np.flipud(np.transpose(self.fgrid[:, self.y_gridind, :])), [self.zmin, self.zmax, self.xmin, self.xmax], field+comp+unit, "z [m]", "x [m]", ax=ax[1], lmin=vmin, lmax=vmax)
        yz_den, yz_cbar = imshowplot(np.flipud(np.transpose(self.fgrid[:, :, self.x_gridind])), [self.zmin, self.zmax, self.ymin, self.ymax], field+comp+unit, "z [m]", "y [m]", ax=ax[2], lmin=vmin, lmax=vmax)
        plt.suptitle(f"t = {self.time[0]:.2e} s")
        ## Add slider for spatial translation
        plt.subplots_adjust(bottom=0.5)
        slider_xy = create_slider('z [m]', self.zgrid[0], self.zgrid[-1], self.zgrid[self.ncell_z//2], self.zgrid, loc=[0.12, 0.32, 0.2, 0.03])
        slider_xz = create_slider('y [m]', self.ygrid[0], self.ygrid[-1], self.ygrid[self.ncell_y//2], self.ygrid, loc=[0.40, 0.32, 0.2, 0.03])
        slider_yz = create_slider('x [m]', self.xgrid[0], self.xgrid[-1], self.xgrid[self.ncell_x//2], self.xgrid, loc=[0.68, 0.32, 0.2, 0.03])
        slider_xy.on_changed(lambda val: update_space(val, 'z_gridind', lambda ind: (ind, slice(None), slice(None)), xy_den, xy_cbar, self.zgrid))
        slider_xz.on_changed(lambda val: update_space(val, 'y_gridind', lambda ind: (slice(None), ind, slice(None)), xz_den, xz_cbar, self.ygrid))
        slider_yz.on_changed(lambda val: update_space(val, 'x_gridind', lambda ind: (slice(None), slice(None), ind), yz_den, yz_cbar, self.xgrid))
        ## Either add slider
        if self.save_address is None:
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, loc=[0.25, 0.1, 0.45, 0.03])
            slider.on_changed(update_time)
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, field+comp+"_2D_anim", update_time, fargs=())
            
    def screenshot_2D(self, field, ts, comp="", vmin=None, vmax=None, xpos=None, ypos=None, zpos=None):
        ''' Create a 2D screenshot of the field '''
        ## Set position and time and fix colorbar limits
        x_gridind, y_gridind, z_gridind = self.set_position(xpos, ypos, zpos)
        vmin = np.amin(getattr(self, field+comp)) if vmin is True else vmin
        vmax = np.amax(getattr(self, field+comp)) if vmax is True else vmax
        ## Evaluate time/timestep and the corresponding field component
        ts_ind = self.find_timestep(ts)
        fgrid, unit = getattr(self, field+comp)[ts_ind], getattr(self, f"{field}_unit")
        ## Create plots
        imshowplot(np.flipud(fgrid[z_gridind, :, :]), [self.xmin, self.xmax, self.ymin, self.ymax], field+comp+unit, "x [m]", "y [m]", lmin=vmin, lmax=vmax, savepath=f"{self.save_address}{field+comp}_xy_{ts_ind}.png")
        imshowplot(np.flipud(np.transpose(fgrid[:, y_gridind, :])), [self.zmin, self.zmax, self.xmin, self.xmax], field+comp+unit, "z [m]", "x [m]", lmin=vmin, lmax=vmax, savepath=f"{self.save_address}{field+comp}_xz_{ts_ind}.png")
        imshowplot(np.flipud(np.transpose(fgrid[:, :, x_gridind])), [self.zmin, self.zmax, self.ymin, self.ymax], field+comp+unit, "z [m]", "y [m]", lmin=vmin, lmax=vmax, savepath=f"{self.save_address}{field+comp}_yz_{ts_ind}.png")

    def plot_field1D(self, field, comp="", vmin=None, vmax=None):
        ''' Create a slider plot for 1D density distribution ''' ## TO UPDATE FOR WARPX
        ## Create slider function for spatial translation
        def update_space(val, attr, array):
            setattr(self, attr, list(array).index(val))
            update_cartesian2Dplot_slider(x_plot, y_plot, z_plot, self.fgrid[self.z_gridind, self.y_gridind, :], self.fgrid[self.z_gridind, :, self.x_gridind], self.fgrid[:, self.y_gridind, self.x_gridind], ax[0], ax[1], ax[2])
            plt.draw()
        ## Create slider function for time evolution
        def update_time(val):
            ind = list(self.timesteps).index(val)
            self.fgrid = getattr(self, field+comp)[ind]
            update_cartesian2Dplot_slider(x_plot, y_plot, z_plot, self.fgrid[self.z_gridind, self.y_gridind, :], self.fgrid[self.z_gridind, :, self.x_gridind], self.fgrid[:, self.y_gridind, self.x_gridind], ax[0], ax[1], ax[2])
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
            plt.draw()
        ## Create initial plots
        vmin = np.amin(getattr(self, field+comp)) if vmin is True else vmin
        vmax = np.amax(getattr(self, field+comp)) if vmax is True else vmax
        self.fgrid, unit = getattr(self, field+comp)[0], getattr(self, f"{field}_unit")
        self.x_gridind, self.y_gridind, self.z_gridind = self.ncell_x//2, self.ncell_y//2, self.ncell_z//2
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        x_plot, = plot2D(self.xgrid, self.fgrid[self.z_gridind, self.y_gridind, :], "x [m]", field+comp+unit, ax=ax[0])
        ax[0].set_ylim(vmin, vmax)
        y_plot, = plot2D(self.ygrid, self.fgrid[self.z_gridind, :, self.x_gridind], "y [m]", field+comp+unit, ax=ax[1])
        ax[1].set_ylim(vmin, vmax)
        z_plot, = plot2D(self.zgrid, self.fgrid[:, self.y_gridind, self.x_gridind], "z [m]", field+comp+unit, ax=ax[2])
        ax[2].set_ylim(vmin, vmax)
        plt.suptitle(f"t = {self.time[0]:.2e} s")
        ## Add slider for spatial translation
        plt.subplots_adjust(bottom=0.5)
        slider_z = create_slider('z [m]', self.zgrid[0], self.zgrid[-1], self.zgrid[self.ncell_z//2], self.zgrid, loc=[0.12, 0.32, 0.2, 0.03])
        slider_y = create_slider('y [m]', self.ygrid[0], self.ygrid[-1], self.ygrid[self.ncell_y//2], self.ygrid, loc=[0.40, 0.32, 0.2, 0.03])
        slider_x = create_slider('x [m]', self.xgrid[0], self.xgrid[-1], self.xgrid[self.ncell_x//2], self.xgrid, loc=[0.68, 0.32, 0.2, 0.03])
        slider_z.on_changed(lambda val: update_space(val, 'z_gridind', self.zgrid))
        slider_y.on_changed(lambda val: update_space(val, 'y_gridind', self.ygrid))
        slider_x.on_changed(lambda val: update_space(val, 'x_gridind', self.xgrid))
        ## Either add slider
        if self.save_address is None:
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, loc=[0.25, 0.1, 0.45, 0.03])
            slider.on_changed(update_time)
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, field+comp+"_1D_anim", update_time, fargs=())
            
    def screenshot_1D(self, field, ts, comp="", vmin=None, vmax=None, xpos=None, ypos=None, zpos=None):
        ''' Create a 1D screenshot of the field '''
        ## Set position and time and fix colorbar limits
        x_gridind, y_gridind, z_gridind = self.set_position(xpos, ypos, zpos)
        vmin = np.amin(getattr(self, field+comp)) if vmin is True else vmin
        vmax = np.amax(getattr(self, field+comp)) if vmax is True else vmax
        ## Evaluate time/timestep and the corresponding field component
        ts_ind = self.find_timestep(ts)
        fgrid, unit = getattr(self, field+comp)[ts_ind], getattr(self, f"{field}_unit")
        ## Create plots
        plot2D(self.xgrid, fgrid[z_gridind, y_gridind, :], "x [m]", field+comp+unit, ylim=[vmin, vmax], savepath=f"{self.save_address}{field+comp}_1d_x_{ts_ind}.png")
        plot2D(self.ygrid, fgrid[z_gridind, :, x_gridind], "y [m]", field+comp+unit, ylim=[vmin, vmax], savepath=f"{self.save_address}{field+comp}_1d_y_{ts_ind}.png")
        plot2D(self.zgrid, fgrid[:, y_gridind, x_gridind], "z [m]", field+comp+unit, ylim=[vmin, vmax], savepath=f"{self.save_address}{field+comp}_1d_z_{ts_ind}.png")
        
       
    def fourier_plot(self, field, comp="", vmin=None, vmax=None, logscale=True, electrons=None, ions=None):
        ''' Create a slider plot for a given field component '''
        ## General function for plot update
        def update_plots():
            clear_field_axes(ax)
            ## Evaluate Fourier transform
            self.kx, self.omegax, self.Px = self.double_fourier_transform(field+comp, "x", slice(None), self.y_gridind, self.z_gridind, self.ts_fp+1, logscale=logscale)
            self.ky, self.omegay, self.Py = self.double_fourier_transform(field+comp, "y", self.x_gridind, slice(None), self.z_gridind, self.ts_fp+1, logscale=logscale)
            self.kz, self.omegaz, self.Pz = self.double_fourier_transform(field+comp, "z", self.x_gridind, self.y_gridind, slice(None), self.ts_fp+1, logscale=logscale)
            ## Create plots
            pcolormesh_plot(self.kx, self.omegax, self.Px, cbar_label, r'k$_x$ [m$^{-1}$]', r'$\omega$ [rad/s]', ax=ax[0], lmin=vmin, lmax=vmax)
            pcolormesh_plot(self.ky, self.omegay, self.Py, cbar_label, r'k$_y$ [m$^{-1}$]', r'$\omega$ [rad/s]', ax=ax[1], lmin=vmin, lmax=vmax)
            pcolormesh_plot(self.kz, self.omegaz, self.Pz, cbar_label, r'k$_z$ [m$^{-1}$]', r'$\omega$ [rad/s]', ax=ax[2], lmin=vmin, lmax=vmax)
            plt.suptitle(f"t = {self.time[self.ts_fp]:.2e} s")
            plt.draw()
        ## Create slider function for spatial translation
        def update_space(val, attr, array):
            setattr(self, attr, list(array).index(val))
            update_plots()
        ## Create slider function for time evolution
        ## To select time slice
        def update_time(val):
            self.ts_fp = list(self.timesteps).index(val)
            update_plots()
        ## To update the dispersion relation plots
        def update_time_dr(val):
            ind = list(electrons.timesteps).index(val)
            self.dispersion_x = self.plot_transversal_dispersion(ind, self.kx, self.omegax, electrons, ions, ax[0], lines=self.dispersion_x)
            self.dispersion_y = self.plot_transversal_dispersion(ind, self.ky, self.omegay, electrons, ions, ax[1], lines=self.dispersion_y)
            self.dispersion_z = self.plot_longitudinal_dispersion(ind, self.kz, self.omegaz, electrons, ions, ax[2], lines=self.dispersion_z)
            plt.draw()
        ## Evaluate quantities for Fourier transform
        self.x_gridind, self.y_gridind, self.z_gridind = self.ncell_x//2, self.ncell_y//2, self.ncell_z//2
        self.ts_fp = len(self.timesteps)-1
        ## Create initial density plots
        cbar_label = r'$\log_{10} P(k,\omega)$' if logscale else r'$P(k,\omega)$'
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        update_plots()
        ## Add slider for spatial translation
        plt.subplots_adjust(bottom=0.5)
        slider_x = create_slider('x [m]', self.xgrid[0], self.xgrid[-1], self.xgrid[self.ncell_x//2], self.xgrid, loc=[0.12, 0.32, 0.2, 0.03])
        slider_y = create_slider('y [m]', self.ygrid[0], self.ygrid[-1], self.ygrid[self.ncell_y//2], self.ygrid, loc=[0.40, 0.32, 0.2, 0.03])
        slider_z = create_slider('z [m]', self.zgrid[0], self.zgrid[-1], self.zgrid[self.ncell_z//2], self.zgrid, loc=[0.68, 0.32, 0.2, 0.03])
        slider_x.on_changed(lambda val: update_space(val, 'x_gridind', self.xgrid))
        slider_y.on_changed(lambda val: update_space(val, 'y_gridind', self.ygrid))
        slider_z.on_changed(lambda val: update_space(val, 'z_gridind', self.zgrid))
        ## Add temporal slider and dispersion relation if plasma is given
        slider = create_slider('Timestep', self.timesteps[4], self.timesteps[-1], self.timesteps[-1], self.timesteps, loc=[0.25, 0.1, 0.45, 0.03])
        slider.on_changed(update_time)
        if electrons is not None and ions is not None:
            self.dispersion_x, self.dispersion_y, self.dispersion_z = None, None, None
            slider_dr = create_slider('Timestep_dr', electrons.timesteps[0], electrons.timesteps[-1], electrons.timesteps[0], electrons.timesteps, loc=[0.25, 0.02, 0.45, 0.03])
            slider_dr.on_changed(update_time_dr)
            update_time_dr(electrons.timesteps[0])
        ## Show plot
        plt.show(block=True)
        
    
    ''' ------------------------------ '''
    ''' Cylindrical plotting functions '''
    ''' ------------------------------ '''  
    def polar_fourier_plot(self, field, comp="", vmin=None, vmax=None, logscale=True, electrons=None, ions=None, Omega=None, mmax=8, r0=0.02, theta0=0.0, rmax=None, nr=128, ntheta=256, x0=0.0, y0=0.0):
        ''' Create a slider plot of the radial (kr, omega) and azimuthal (m, omega) spectra around (x0, y0), analogous to fourier_plot '''
        ## General functions for plot update
        def update_plots():
            clear_field_axes(ax)
            ## Evaluate Fourier transform
            self.kr, self.omega_r, self.Pr = self.polar_fourier_transform(field, comp, "r", self.z_gridind, self.ts_fp+1, r0=r0, theta0=theta0, rmax=rmax, nr=nr, x0=x0, y0=y0, logscale=logscale)
            self.m, self.omega_t, self.Pt = self.polar_fourier_transform(field, comp, "theta", self.z_gridind, self.ts_fp+1, r0=r0, theta0=theta0, ntheta=ntheta,x0=x0, y0=y0, logscale=logscale)
            ## Create plots
            pcolormesh_plot(self.kr, self.omega_r, self.Pr, cbar_label, r'k$_r$ [m$^{-1}$]', r'$\omega$ [rad/s]', ax=ax[0], lmin=vmin, lmax=vmax)
            pcolormesh_plot(self.m, self.omega_t, self.Pt, cbar_label, r'azimuthal mode $m$', r'$\omega$ [rad/s]', ax=ax[1], lmin=vmin, lmax=vmax)
            plt.suptitle(f"t = {self.time[self.ts_fp]:.2e} s")
            plt.draw()
        ## Create slider function for spatial translation
        def update_space(val, attr):
            setattr(self, attr, val)
            update_plots()
        ## Create slider function for time evolution
        def update_time(val):
            self.ts_fp = list(self.timesteps).index(val)
            update_plots()
        def update_time_dr(val):
            ind = list(electrons.timesteps).index(val)
            self.dispersion_r = self.plot_transversal_dispersion(ind, self.kr, self.omega_r, electrons, ions, ax[0], lines=self.dispersion_r)
            self.dispersion_t = self.plot_transversal_dispersion(ind, self.m, self.omega_t, electrons, ions, ax[1], lines=self.dispersion_t)
            plt.draw()
        ## Set spatial and temporal coordinates
        self.ts_fp = len(self.timesteps)-1
        self.z_gridind = self.ncell_z // 2
        self.r0, self.theta0 = r0, theta0
        if rmax is None:
            rmax = self.xmax
        ## Create density plots
        cbar_label = r'$\log_{10} P(k,\omega)$' if logscale else r'$P(k,\omega)$'
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        update_plots()
        ## Add sliders for spatial translation
        plt.subplots_adjust(bottom=0.35)
        slider_r0 = create_slider('r0 [m]', 0.0, rmax, r0, np.linspace(0.0, rmax, 200), loc=[0.15, 0.18, 0.3, 0.03])
        slider_theta0 = create_slider('theta0 [rad]', 0.0, 2*np.pi, theta0, np.linspace(0.0, 2*np.pi, 200), loc=[0.55, 0.18, 0.3, 0.03])
        slider_r0.on_changed(lambda val: update_space(val, 'r0'))
        slider_theta0.on_changed(lambda val: update_space(val, 'theta0'))
        ## Add temporal slider
        slider = create_slider('Timestep', self.timesteps[4], self.timesteps[-1], self.timesteps[-1], self.timesteps, loc=[0.25, 0.1, 0.45, 0.03])
        slider.on_changed(update_time)
        ## Overlap disperion relation
        if electrons is not None and ions is not None:
            self.dispersion_r, self.dispersion_t = None, None
            slider_dr = create_slider('Timestep', electrons.timesteps[0], electrons.timesteps[-1], electrons.timesteps[0], electrons.timesteps, loc=[0.25, 0.02, 0.45, 0.03])
            slider_dr.on_changed(update_time_dr)
            update_time_dr(electrons.timesteps[0])
        if Omega is not None:
            self.plot_harmonic_overlay(ax[1], Omega, mmax)
        ## Show plot
        plt.show(block=True)
       


class cylindrical_field(field_diagnostic):
    ''' ------------------------ '''
    ''' Initialization functions '''
    ''' ------------------------ '''
    def class_attributes(self, warpx, extfield): 
        ''' Function to create class attributes '''
        self.rho = self.get_warpx_attributes(warpx, False, "rho")
        self.Jr = self.get_warpx_attributes(warpx, False, "j", comp="r")
        self.Jt = self.get_warpx_attributes(warpx, False, "j", comp="t")
        self.Jz = self.get_warpx_attributes(warpx, False, "j", comp="z")
        self.Er = self.get_warpx_attributes(warpx, extfield, "E", comp="r")
        self.Et = self.get_warpx_attributes(warpx, extfield, "E", comp="t")
        self.Ez = self.get_warpx_attributes(warpx, extfield, "E", comp="z")
        self.Br = self.get_warpx_attributes(warpx, extfield, "B", comp="r")
        self.Bt = self.get_warpx_attributes(warpx, extfield, "B", comp="t")
        self.Bz = self.get_warpx_attributes(warpx, extfield, "B", comp="z")
        self.J = np.asarray([np.sqrt(Jr**2 + Jt**2 + Jz**2) for Jr, Jt, Jz in zip(self.Jr, self.Jt, self.Jz)])
        self.E = np.asarray([np.sqrt(Er**2 + Et**2 + Ez**2) for Er, Et, Ez in zip(self.Er, self.Et, self.Ez)])
        self.B = np.asarray([np.sqrt(Br**2 + Bt**2 + Bz**2) for Br, Bt, Bz in zip(self.Br, self.Bt, self.Bz)])
    
           
    ''' ------------------------------ '''
    ''' Cylindrical plotting functions '''
    ''' ------------------------------ '''
    def plot_field2D(self, field, comp="", vmin=None, vmax=None):
        ''' Create a slider plot for a given field component '''
        ## Create slider function for time evolution
        def update_time(val):
            ind = list(self.timesteps).index(val)
            self.fgrid = getattr(self, field+comp)[ind]
            update_2Ddensityplot(np.transpose(self.fgrid), den, cbar, vmin=vmin, vmax=vmax)
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
            plt.draw()
            print(f"{field}{comp} at t = {self.time[ind]:.2e} s: min = {np.amin(self.fgrid):.2e}, max = {np.amax(self.fgrid):.2e}")
        ## Create initial density plots
        vmin = np.amin(getattr(self, field+comp)) if vmin is True else vmin
        vmax = np.amax(getattr(self, field+comp)) if vmax is True else vmax
        self.fgrid, unit = getattr(self, field+comp)[0], getattr(self, f"{field}_unit")
        fig, ax = plt.subplots()
        den, cbar = imshowplot(np.flipud(np.transpose(self.fgrid)), [self.zmin, self.zmax, self.rmin, self.rmax], field+comp+unit, "z [m]", "r [m]", ax=ax, lmin=vmin, lmax=vmax)
        plt.suptitle(f"t = {self.time[0]:.2e} s")
        ## Add slider for spatial translation
        plt.subplots_adjust(bottom=0.25)
        ## Either add slider
        if self.save_address is None:
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, loc=[0.25, 0.1, 0.45, 0.03])
            slider.on_changed(update_time)
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, field+comp+"_2D_anim", update_time, fargs=())
            
    def screenshot_2D(self, field, ts, comp="", vmin=None, vmax=None):
        ''' Create a 2D screenshot of the field '''
        ## Set position and time and fix colorbar limits
        vmin = np.amin(getattr(self, field+comp)) if vmin is True else vmin
        vmax = np.amax(getattr(self, field+comp)) if vmax is True else vmax
        ## Evaluate time/timestep and the corresponding field component
        ts_ind = self.find_timestep(ts)
        fgrid, unit = getattr(self, field+comp)[ts_ind], getattr(self, f"{field}_unit")
        ## Create plots
        imshowplot(np.flipud(np.transpose(fgrid)), [self.zmin, self.zmax, self.rmin, self.rmax], field+comp+unit, "z [m]", "r [m]", lmin=vmin, lmax=vmax, savepath=f"{self.save_address}{field+comp}_rz_{ts_ind}.png")
    
    def plot_field1D(self, field, comp="", vmin=None, vmax=None):
        ''' Create a slider plot for 1D density distribution ''' ## TO UPDATE FOR WARPX
        ## Create slider function for spatial translation
        def update_space(val, attr, array):
            setattr(self, attr, list(array).index(val))
            update_2Dplot_slider(r_plot, self.fgrid[self.z_gridind, :], ax[0])
            update_2Dplot_slider(z_plot, self.fgrid[:, self.r_gridind], ax[1])
            plt.draw()
        ## Create slider function for time evolution
        def update_time(val):
            ind = list(self.timesteps).index(val)
            self.fgrid = getattr(self, field+comp)[ind]
            update_2Dplot_slider(r_plot, self.fgrid[self.z_gridind, :], ax[0])
            update_2Dplot_slider(z_plot, self.fgrid[:, self.r_gridind], ax[1])
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
            plt.draw()
        ## Create initial plots
        vmin = np.amin(getattr(self, field+comp)) if vmin is True else vmin
        vmax = np.amax(getattr(self, field+comp)) if vmax is True else vmax
        self.fgrid, unit = getattr(self, field+comp)[0], getattr(self, f"{field}_unit")
        self.r_gridind, self.z_gridind = self.ncell_r//2, self.ncell_z//2
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        r_plot, = plot2D(self.rgrid, self.fgrid[self.z_gridind, :], "r [m]", field+comp+unit, ax=ax[0])
        ax[0].set_ylim(vmin, vmax)
        z_plot, = plot2D(self.zgrid, self.fgrid[:, self.r_gridind], "z [m]", field+comp+unit, ax=ax[1])
        ax[1].set_ylim(vmin, vmax)
        plt.suptitle(f"t = {self.time[0]:.2e} s")
        ## Add slider for spatial translation
        plt.subplots_adjust(bottom=0.5)
        slider_z = create_slider('z [m]', self.zgrid[0], self.zgrid[-1], self.zgrid[self.ncell_z//2], self.zgrid, loc=[0.2, 0.32, 0.2, 0.03])
        slider_r = create_slider('r [m]', self.rgrid[0], self.rgrid[-1], self.rgrid[self.ncell_r//2], self.rgrid, loc=[0.60, 0.32, 0.2, 0.03])
        slider_z.on_changed(lambda val: update_space(val, 'z_gridind', self.zgrid))
        slider_r.on_changed(lambda val: update_space(val, 'r_gridind', self.rgrid))
        ## Either add slider
        if self.save_address is None:
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, loc=[0.25, 0.1, 0.45, 0.03])
            slider.on_changed(update_time)
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, field+comp+"_1D_anim", update_time, fargs=())