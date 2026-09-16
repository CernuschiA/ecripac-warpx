import sys
sys.path.append("/home/cernuschi/coding/warpx/ecripac/functions/")
from warpx_class import *

''' ------------------------- '''
''' Track particle superclass '''
''' ------------------------- '''
class track_particles(warpx_namelist):
    ''' ------------------------ '''
    ''' Initialization functions '''
    ''' ------------------------ '''
    def __init__(self, path_field, path_data, species_name, dmult=1., flag_superclass=True, save_address=None, extfield=None, grid_refinement=None):
        ''' Function to initialize the track_particles superclass '''
        ## Initialize superclass
        super().__init__(path_field, save_address=save_address, grid_refinement=grid_refinement, pclass_flag=True)
        if save_address is not None:
            self.save_address = save_address+species_name+"_" 
        ## Open diagnostics and extract raw data
        self.name = species_name
        warpx = OpenPMDTimeSeries(path_data)
        self.timesteps = warpx.iterations # List of timesteps available
        self.time = warpx.t # List of time values corresponding to timesteps
        self.dt, self.T = self.time[1]/self.timesteps[1], warpx.tmax # Timestep size and maximum time
        self.mass = warpx.get_particle(var_list=['mass'], iteration=0, species=species_name)[0][0] # Particle mass [SI]
        self.charge = warpx.get_particle(var_list=['charge'], iteration=0, species=species_name)[0][0] # Particle charge [SI]
        self.density_multiplier = dmult # Multiplier for particle density, based on number of particles tracked   
        self.extfield = extfield # External fields class, if available   
        ## Obtain reference quantities for SI conversion
        self.K_ref_keV = self.mass*sciconst.physical_constants["electron mass energy equivalent in MeV"][0]*1e3/sciconst.m_e # Reference kinetic energy [keV]
        ## Initialize classes attributes if you use superclass
        if flag_superclass is True:
            self.superclass_attributes(warpx)
            if extfield is not None:
                self.evaluate_extfield_quantities(extfield)

    def superclass_attributes(self, warpx): 
        ''' Function to create class attributes '''   
        self.x = self.get_warpx_attributes(warpx, "x")
        self.y = self.get_warpx_attributes(warpx, "y")
        self.z = self.get_warpx_attributes(warpx, "z")
        self.px = self.get_warpx_attributes(warpx, "ux") # gamma*vx/c
        self.py = self.get_warpx_attributes(warpx, "uy") # gamma*vy/c
        self.pz = self.get_warpx_attributes(warpx, "uz") # gamma*vz/c
        self.w = self.get_warpx_attributes(warpx, "w")
        self.gamma = tuple(gamma_from_p(uxn, uyn, uzn) for uxn, uyn, uzn in zip(self.px, self.py, self.pz))
        self.lost_part = [len(self.x[0])-len(self.x[i]) for i in range(len(self.x))] # Number of lost particles at each timestep
        self.r = tuple(np.sqrt(x**2 + y**2) for x, y in zip(self.x, self.y)) # Radial position of particles
        ## Print information on particle tracking
        print(f"Total number of {self.name} tracked: {self.px[0].shape[0]}")
        print(f"Total number of confined {self.name} tracked: {self.px[-1].shape[0]} ({(self.px[-1].shape[0])/self.px[0].shape[0]*100:.2f} %)")
        print(f"Total number of lost {self.name}: {self.px[0].shape[0]-self.px[-1].shape[0]} ({(self.px[0].shape[0]-self.px[-1].shape[0])/self.px[0].shape[0]*100:.2f} %)")
        
    def get_warpx_attributes(self, warpx, var, mask=False):
        ''' Function to get attributes from WarpX data '''
        if mask is False:
            mask = [slice(None)]*len(self.timesteps)
        return tuple(warpx.get_particle(var_list=[var], iteration=it, species=self.name)[0][mask[i]] for i, it in enumerate(self.timesteps))
    
    def evaluate_extfield_quantities(self, extfield):
        ''' Function to evaluate field related quantities for each particle at each timestep, based on the external fields class '''
        ## Save external fields attribute
        Bfield = tuple(extfield.evaluate_Bfield(self.x[i], self.y[i], self.z[i], self.time[i], evaluate_indexes=True) for i in range(len(self.x)))
        self.Bx = tuple(Bfield[i][0] for i in range(len(Bfield)))
        self.By = tuple(Bfield[i][1] for i in range(len(Bfield)))
        self.Bz = tuple(Bfield[i][2] for i in range(len(Bfield)))
        if extfield.hf_class.pw_flag is False:
            Efield = tuple(extfield.evaluate_Efield(self.x[i], self.y[i], self.z[i], self.time[i], evaluate_indexes=True) for i in range(len(self.x)))
        else:
            Efield = tuple(extfield.evaluate_Efield_pw(self.x[i], self.y[i], self.z[i], self.time[i], evaluate_indexes=True) for i in range(len(self.x)))
        self.Ex = tuple(Efield[i][0] for i in range(len(Efield)))
        self.Ey = tuple(Efield[i][1] for i in range(len(Efield)))
        self.Ez = tuple(Efield[i][2] for i in range(len(Efield)))
        ## Pitch angle and gyromagnetic autoresonance phase
        self.pitch_angle = tuple(angle_vector(self.px[i], self.py[i], self.pz[i], self.Bx[i], self.By[i], self.Bz[i])*180/np.pi for i in range(len(self.px)))
        self.ga_phase = tuple(angle_vector(self.px[i], self.py[i], self.pz[i], self.Ex[i], self.Ey[i], self.Ez[i])*180/np.pi for i in range(len(self.px)))
        
    ''' ------------------------ '''
    ''' Plasma physics functions '''
    ''' ------------------------ '''
    def evaluate_density(self, ts):
        ''' Evaluate the 3D density distribution of particles '''
        density_grid = evaluate_density3D_numba(self.x[ts], self.y[ts], self.z[ts], self.w[ts], self.dx, self.dy, self.dz, self.ncell_x, self.ncell_y, self.ncell_z, np.abs(self.xmin), np.abs(self.ymin), np.abs(self.zmin))
        return density_grid*self.density_multiplier
    
    def evaluate_temperature(self, ts, kinen=True):
        ''' Evaluate the 3D temperature distribution of particles '''
        temperature_grid, p_grid = evaluate_temperature3D_numba(self.x[ts], self.y[ts], self.z[ts], self.gamma[ts], self.dx, self.dy, self.dz, self.ncell_x, self.ncell_y, self.ncell_z, np.abs(self.xmin), np.abs(self.ymin), np.abs(self.zmin))
        temperature_grid[p_grid != 0.] /= p_grid[p_grid != 0.]
        temperature_grid[temperature_grid == 0.] = 1.
        return kin_en_from_gamma(temperature_grid, self.K_ref_keV) if kinen else temperature_grid
    
    def evaluate_debyelength(self, ts):
        ''' Evaluate the 3D Debye length distribution of particles '''
        density_grid = self.evaluate_density(ts)
        temperature_grid = self.evaluate_temperature(ts)
        debye_grid = np.zeros_like(density_grid)
        debye_grid[density_grid != 0.] = np.sqrt(sciconst.epsilon_0*temperature_grid[density_grid != 0.]*1e3*sciconst.e/self.charge**2/density_grid[density_grid != 0.])
        return debye_grid
    
    def evaluate_plasmapar(self, ts):
        ''' Evaluate the 3D plasma parameter distribution of particles '''
        density_grid = self.evaluate_density(ts)
        debye_grid = self.evaluate_debyelength(ts)
        return (4/3)*np.pi*density_grid*debye_grid**3
    
    def evaluate_plasmafreq(self, ts):
        ''' Evaluate the 3D plasma frequency distribution of particles '''
        density_grid = self.evaluate_density(ts)
        gamma_grid, p_grid = evaluate_temperature3D_numba(self.x[ts], self.y[ts], self.z[ts], self.gamma[ts], self.dx, self.dy, self.dz, self.ncell_x, self.ncell_y, self.ncell_z, np.abs(self.xmin), np.abs(self.ymin), np.abs(self.zmin))
        gamma_grid[p_grid != 0.] /= p_grid[p_grid != 0.]
        gamma_grid[gamma_grid == 0.] = 1.
        return np.sqrt(density_grid*self.charge**2/(gamma_grid*sciconst.epsilon_0*self.mass))
    
    def evaluate_Bnorm(self, ts):
        ''' Evaluate the 3D normalized magnetic field distribution at the particle positions '''
        x_mesh, y_mesh, z_mesh = np.meshgrid(self.xgrid, self.ygrid, self.zgrid, indexing='ij')
        B_field = self.extfield.evaluate_Bfield(x_mesh, y_mesh, z_mesh, self.time[ts], evaluate_indexes=True)
        return np.sqrt(B_field[0]**2 + B_field[1]**2 + B_field[2]**2)
    
    def evaluate_beta(self, ts):
        ''' Evaluate the 3D beta (plasma pressure / magnetic pressure) distribution of particles '''
        density_grid = self.evaluate_density(ts)
        temperature_grid = self.evaluate_temperature(ts)*1e3*sciconst.e
        Bnorm = self.evaluate_Bnorm(ts)
        B_pressure = Bnorm**2/(2*sciconst.mu_0)
        return density_grid*temperature_grid/B_pressure
    
    def plasma_average(self, ts, func, args=()):
        ''' Evaluate the average of a given function over the plasma distribution at a given timestep '''
        density_grid = self.evaluate_density(ts)
        func_grid = func(ts, *args)
        return np.sum(density_grid*func_grid)/np.sum(density_grid)
    
    ''' ------------------------ '''
    ''' Pre-processing functions '''
    ''' ------------------------ '''
    def return_units_gamma(self, flag_keV):
        ''' Return the units and rescale factor for either Lorentz factor or kinetic energy in keV '''
        if flag_keV is True:
            return lambda x:  kin_en_from_gamma(x, self.K_ref_keV), " [keV]", "K"
        else:
            return lambda x: x, " [code units]", f"{gamma_unicode}"
    
    def cartesian_histplot_slider(self, attr, bins, ylab, xlim, ylim, zlim, plim, unit_label, w=None, rescale=1):
        ''' Generic function to create cartesian histogram plots with a slider '''
        ## Setup initial plot and setup limits
        xdata, ydata, zdata = getattr(self, attr+"x"), getattr(self, attr+"y"), getattr(self, attr+"z")
        if (xlim is not None and not isinstance(xlim, list)) or (ylim is not None and not isinstance(ylim, list)) or (zlim is not None and not isinstance(zlim, list)) or (plim is not None and not isinstance(plim, list)):
            xlim, ylim, zlim, plim = find_cartesian_lim_tuples(xdata, ydata, zdata, attr, self.timesteps, unit_label, rescale=rescale)
        fig, ax = cartesian_histplot(xdata[0]*rescale, ydata[0]*rescale, zdata[0]*rescale, bins, bins, bins, attr, unit_label, ylab, xlim, ylim, zlim, plim, suptitle=f"t = {self.time[0]:.2e} s", w=w[0] if isinstance(w, tuple) else None)
        ## Either add slider
        if self.save_address is None:
            slider = create_slider("Timestep", self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, adjust_plot=True)
            slider.on_changed(lambda val: update_cartesian_histplot(val, ax, xdata, ydata, zdata, bins, bins, bins, attr, unit_label, ylab, xlim, ylim, zlim, plim, self.timesteps, rescale, self.time, w=w))
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, f"{attr}_cartesian_histplot_anim", update_cartesian_histplot, fargs = (ax, xdata, ydata, zdata, bins, bins, bins, attr, unit_label, ylab, xlim, ylim, zlim, plim, self.timesteps, rescale, self.time, w))
    
    ''' --------------- '''
    ''' Print functions '''
    ''' --------------- '''
    def print_attr_info(self, attr, flag_keV):
        ''' Print minimum and maximum values of a given attribute '''
        minx, minx_ind = tuple_minvalue(getattr(self, attr))
        maxx, maxx_ind = tuple_maxvalue(getattr(self, attr))
        if attr == "gamma":
            rescale_func, unit, attr = self.return_units_gamma(flag_keV)
            minx, maxx = rescale_func(minx), rescale_func(maxx)
        else:
            unit = getattr(self, f"{attr}_unit")
        print(f"{attr}: min = {minx:.2e}{unit} (timestep {self.timesteps[minx_ind]} - t = {self.time[minx_ind]:.2e} s), max = {maxx:.2e}{unit} (timestep {self.timesteps[maxx_ind]} - t = {self.time[maxx_ind]:.2e} s)")
        
    def print_minmax(self, flag_keV=False):
        ''' Print minimum and maximum values of all class attributes'''
        print(f"Tracked {self.name} minimum and maximum values:")
        for attr in ["x", "y", "z", "px", "py", "pz", "gamma"]:
            self.print_attr_info(attr, flag_keV)
        
    ''' ----------------------------------------------------------- '''
    ''' Functions for analysis of position and density distribution '''
    ''' ----------------------------------------------------------- '''
    def particle_position(self, q_color=None, clabel=None, n_part=10000):
        ''' Plot particle position distribution in 2D and 3D '''
        ## Evaluate indexes mask at t=0
        if len(self.x[0]) > n_part:
            ind_p = sample_random_indexes(self.x[0], n_part)
        else:
            ind_p = slice(None)
        ## 2D slider function
        def update2D(val):
            ind = list(self.timesteps).index(val)
            if len(self.x[ind]) > n_part:
                ind_pu = sample_random_indexes(self.x[ind], n_part)
            else:
                ind_pu = slice(None)
            update_2Dscatterplot(self.z[ind][ind_pu], self.x[ind][ind_pu], plZX, q_color[ind][ind_pu], cbarZX)
            update_2Dscatterplot(self.z[ind][ind_pu], self.y[ind][ind_pu], plZY, q_color[ind][ind_pu], cbarZY)
            update_2Dscatterplot(self.x[ind][ind_pu], self.y[ind][ind_pu], plXY, q_color[ind][ind_pu], cbarXY)
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
            plt.draw()
        ## Setup 2D plot
        self.plasma_chamber()
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        plZX, cbarZX = plot2D(self.z[0][ind_p], self.x[0][ind_p], "z [m]", "x [m]", xlim=[self.zmin, self.zmax], ylim=[self.xmin, self.xmax], col=q_color[0][ind_p], clabel=clabel, ax=ax[0], plot_type="scatter")
        ax[0].add_artist(self.rectZY)
        plZY, cbarZY = plot2D(self.z[0][ind_p], self.y[0][ind_p], "z [m]", "y [m]", xlim=[self.zmin, self.zmax], ylim=[self.ymin, self.ymax], col=q_color[0][ind_p], clabel=clabel, ax=ax[1], plot_type="scatter")
        ax[1].add_artist(self.rectZX)
        plXY, cbarXY = plot2D(self.x[0][ind_p], self.y[0][ind_p], "x [m]", "y [m]", xlim=[self.xmin, self.xmax], ylim=[self.ymin, self.ymax], col=q_color[0][ind_p], clabel=clabel, ax=ax[2], plot_type="scatter")
        ax[2].add_artist(self.circle)
        plt.suptitle(f"t = {self.time[0]:.2e} s")
        plt.subplots_adjust(bottom=0.25, wspace=0.6, hspace=0.6)
        ## Either add slider
        if self.save_address is None:
            slider2D = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps)
            slider2D.on_changed(update2D)
        ## Or save animation
        else:
            self.save_animation(fig, "position_anim", update2D, fargs=())
        ## 3D slider function
        def update3D(val):
            ind = list(self.timesteps).index(val)
            if len(self.x[ind]) > n_part:
                ind_pu = sample_random_indexes(self.x[ind], n_part)
            else:
                ind_pu = slice(None)
            sc3D._offsets3d = (self.z[ind][ind_pu], self.x[ind][ind_pu], self.y[ind][ind_pu])
            sc3D.set_array(q_color[ind][ind_pu])
            cbar3D.mappable.set_clim(vmin=np.amin(q_color[ind][ind_pu]), vmax=np.amax(q_color[ind][ind_pu]))
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
        ## Setup 3D plot
        ax3D = plt.figure().add_subplot(111, projection='3d')
        ax3D.set_xlabel("z [m]", labelpad=10)
        ax3D.set_ylabel("x [m]", labelpad=10)
        ax3D.set_zlabel("y [m]", labelpad=10)
        ax3D.plot_surface(self.z_grid, self.x_grid, self.y_grid, alpha=0.2)
        sc3D = ax3D.scatter(self.z[0][ind_p], self.x[0][ind_p], self.y[0][ind_p], s=scatterdim, c=q_color[0][ind_p], cmap=colormap)
        cbar3D = plt.colorbar(sc3D, label=clabel, pad=0.15, shrink=0.5)
        plt.suptitle(f"t = {self.time[0]:.2e} s")
        ## Either add slider
        if self.save_address is None:
            slider3D = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, adjust_plot=True)
            slider3D.on_changed(update3D)
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, "position_anim_3D", update3D, fargs=())

    def position_distribution_slider(self, bins=100, unit_label=" [m]", w=None, rescale=1.):
        ''' Create a slider plot for position distribution in longitudinal and transversal planes '''
        self.cartesian_histplot_slider("", bins, 'Number of particles', [self.xmin*rescale, self.xmax*rescale], [self.ymin*rescale, self.ymax*rescale], [self.zmin*rescale, self.zmax*rescale], [0., self.xmax*rescale], unit_label, w=w, rescale=rescale)
    
    def lost_histogram(self, flag_notintegral=False):
        ''' Create a histogram of lost particles over time '''
        ## Create time array and convert to not integral number of lost particles if needed
        if flag_notintegral is True:
            self.lost_part = np.diff([self.lost_part[0]]+self.lost_part)
            ylabel = f"Number of lost {self.name}"
        else:
            ylabel = f"Integral number of lost {self.name}"
        #ylabel = r"dN$_{i,loss}$/dt"
        ## Create histogram plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(self.time*1e9, self.lost_part, width=np.asarray([*np.diff(self.time), self.time[-1]-self.time[-2]])*1e9, align='edge', facecolor='none', edgecolor='blue', linewidth=1.5)
        ax.set_xlabel("t [ns]")
        ax.set_ylabel(ylabel)
        plt.grid()
        plt.tight_layout()
        ## Save figure if needed
        if self.save_address is not None:
            plt.savefig(self.save_address+f"lost_{self.name}_histogram.png", bbox_inches='tight', dpi=dpi)
    
    ''' --------------------------- '''
    ''' Functions for density plots '''
    ''' --------------------------- '''
    def densityplot2D_slider(self, attr, label, vmin=None, vmax=None):
        ''' Create a slider density plot for several 2D plane of a given plasma quantity, based on a given function '''
        ## Determine function
        func = getattr(self, attr)
        ## Create slider function for spatial translation
        def update_space(val, attr, ind, plot, cbar, array):
            setattr(self, attr, list(array).index(val))
            update_2Ddensityplot(self.dgrid[ind(getattr(self, attr))], plot, cbar, vmin=vmin, vmax=vmax)
            plt.draw()
        ## Create slider function for time evolution
        def update_time(val):
            ind = list(self.timesteps).index(val)
            self.dgrid = func(ind)
            update_2Ddensityplot(self.dgrid[:, :, self.z_gridind], xy_den, xy_cbar, vmin=vmin, vmax=vmax)
            update_2Ddensityplot(self.dgrid[:, self.y_gridind, :], xz_den, xz_cbar, vmin=vmin, vmax=vmax)
            update_2Ddensityplot(self.dgrid[self.x_gridind, :, :], yz_den, yz_cbar, vmin=vmin, vmax=vmax)
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
            plt.draw()
            print(f"{label} at t = {self.time[ind]:.2e} s: min = {np.amin(self.dgrid[self.dgrid > 0]):.2e}, max = {np.amax(self.dgrid):.2e}, average = {self.plasma_average(ind, func):.2e}")
        ## Create initial grid and 2D density plots
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        self.dgrid = func(0)
        self.x_gridind, self.y_gridind, self.z_gridind = self.ncell_x//2, self.ncell_y//2, self.ncell_z//2
        xy_den, xy_cbar = imshowplot(np.flipud(self.dgrid[:, :, self.z_gridind]), [self.xmin, self.xmax, self.ymin, self.ymax], label, "x [m]", "y [m]", ax=ax[0], lmin=vmin, lmax=vmax)
        xz_den, xz_cbar = imshowplot(np.flipud(self.dgrid[:, self.y_gridind, :]), [self.zmin, self.zmax, self.xmin, self.xmax], label, "z [m]", "x [m]", ax=ax[1], lmin=vmin, lmax=vmax)
        yz_den, yz_cbar = imshowplot(np.flipud(self.dgrid[self.x_gridind, :, :]), [self.zmin, self.zmax, self.ymin, self.ymax], label, "z [m]", "y [m]", ax=ax[2], lmin=vmin, lmax=vmax)
        plt.suptitle(f"t = {self.time[0]:.2e} s")
        ## Add slider for spatial translation
        plt.subplots_adjust(bottom=0.5)
        slider_xy = create_slider('z [m]', self.zgrid[0], self.zgrid[-1], self.zgrid[self.ncell_z//2], self.zgrid, loc=[0.12, 0.32, 0.2, 0.03])
        slider_xz = create_slider('y [m]', self.ygrid[0], self.ygrid[-1], self.ygrid[self.ncell_y//2], self.ygrid, loc=[0.40, 0.32, 0.2, 0.03])
        slider_yz = create_slider('x [m]', self.xgrid[0], self.xgrid[-1], self.xgrid[self.ncell_x//2], self.xgrid, loc=[0.68, 0.32, 0.2, 0.03])
        slider_xy.on_changed(lambda val: update_space(val, 'z_gridind', lambda ind: (slice(None), slice(None), ind), xy_den, xy_cbar, self.zgrid))
        slider_xz.on_changed(lambda val: update_space(val, 'y_gridind', lambda ind: (slice(None), ind, slice(None)), xz_den, xz_cbar, self.ygrid))
        slider_yz.on_changed(lambda val: update_space(val, 'x_gridind', lambda ind: (ind, slice(None), slice(None)), yz_den, yz_cbar, self.xgrid))
        ## Either add slider
        if self.save_address is None:
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, loc=[0.25, 0.1, 0.45, 0.03])
            slider.on_changed(update_time)
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, "densityplot2D_anim", update_time, fargs=())
            
    def screenshot_densityplot2D(self, attr, label, ts, vmin=None, vmax=None, xpos=None, ypos=None, zpos=None):
        ''' Create a screenshot of 2D density plot for several 2D plane of a given plasma quantity at a given position '''
        ## Determine function, position and timestep, set colorbar limits
        x_gridind, y_gridind, z_gridind = self.set_position(xpos, ypos, zpos)
        ts_ind = self.find_timestep(ts)
        dgrid = getattr(self, attr)(ts_ind)
        vmin = np.amin(dgrid) if vmin is True else vmin
        vmax = np.amax(dgrid) if vmax is True else vmax
        ## Create plots
        imshowplot(np.flipud(dgrid[:, :, z_gridind]), [self.xmin, self.xmax, self.ymin, self.ymax], label, "x [m]", "y [m]", lmin=vmin, lmax=vmax, savepath=f"{self.save_address}{attr}_xy_{ts_ind}.png")
        imshowplot(np.flipud(dgrid[:, y_gridind, :]), [self.zmin, self.zmax, self.xmin, self.xmax], label, "z [m]", "x [m]", lmin=vmin, lmax=vmax, savepath=f"{self.save_address}{attr}_xz_{ts_ind}.png")
        imshowplot(np.flipud(dgrid[x_gridind, :, :]), [self.zmin, self.zmax, self.ymin, self.ymax], label, "z [m]", "y [m]", lmin=vmin, lmax=vmax, savepath=f"{self.save_address}{attr}_yz_{ts_ind}.png")
        

    def densityplot1D_slider(self, attr, label):
        ''' Create a slider density plot for several 1D plane of a given plasma quantity, based on a given function '''
        ## Determine function
        func = getattr(self, attr)
        ## Create slider function for spatial translation
        def update_space(val, attr, array):
            setattr(self, attr, list(array).index(val))
            update_cartesian2Dplot_slider(x_plot, y_plot, z_plot, self.dgrid[:, self.y_gridind, self.z_gridind], self.dgrid[self.x_gridind, :, self.z_gridind], self.dgrid[self.x_gridind, self.y_gridind, :], ax[0], ax[1], ax[2])
            plt.draw()
        ## Create slider function for time evolution
        def update_time(val):
            ind = list(self.timesteps).index(val)
            self.dgrid = func(ind)
            update_cartesian2Dplot_slider(x_plot, y_plot, z_plot, self.dgrid[:, self.y_gridind, self.z_gridind], self.dgrid[self.x_gridind, :, self.z_gridind], self.dgrid[self.x_gridind, self.y_gridind, :], ax[0], ax[1], ax[2])
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
            plt.draw()
        ## Create initial grid and 2D density plots
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        self.dgrid = func(0)
        self.x_gridind, self.y_gridind, self.z_gridind = self.ncell_x//2, self.ncell_y//2, self.ncell_z//2
        x_plot, = plot2D(self.xgrid, self.dgrid[:, self.y_gridind, self.z_gridind], "x [m]", label, ax=ax[0])
        y_plot, = plot2D(self.ygrid, self.dgrid[self.x_gridind, :, self.z_gridind], "y [m]", label, ax=ax[1])
        z_plot, = plot2D(self.zgrid, self.dgrid[self.x_gridind, self.y_gridind, :], "z [m]", label, ax=ax[2])
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
            self.save_animation(fig, "densityplot1D_anim", update_time, fargs=())
    
    ''' ---------------------------------------------------------- '''
    ''' Functions for analysis of velocity and energy distribution '''
    ''' ---------------------------------------------------------- '''
    def momentum_distribution_slider(self, bins=100, xlim=None, ylim=None, zlim=None, plim=None, unit_label=f" [{gamma_unicode}v/c]", rescale=1.):
        ''' Create a slider plot for momentum distribution in longitudinal and transversal planes '''
        ## Create slider plot
        self.cartesian_histplot_slider("p", bins, f'Number of {self.name}', xlim, ylim, zlim, plim, unit_label, rescale=rescale)

    def gamma_B_plot(self, flag_max=False, B=None, flag_keV=False):
        ''' Create a plot of average B field and Lorentz factor over time '''
        ## Convert gamma to keV if needed
        rescale, gamma_unit, gamma_name = self.return_units_gamma(flag_keV)
        ## Process data for the plot
        n_part = [len(x) for x in self.x]
        if flag_max is False:
            gamma = [rescale(np.average(gamma)) for gamma in self.gamma]
        else:
            gamma = [rescale(np.max(gamma)) for gamma in self.gamma]
        ## Create the plot
        fig, ax = plt.subplots()
        if B is not None:
            ax.plot(self.time, B(self.time), color='r')
        ax.plot(self.time, gamma, color='b')
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f'B/B0 (red) and {gamma_name}{gamma_unit} (blue)')
        ax2 = ax.twinx()
        ax2.plot(self.time, n_part, color='g')
        ax2.set_ylabel(f'# of {self.name} (green)')
        plt.grid()
        if self.save_address is not None:
            plt.savefig(self.save_address+"gamma_B_plot.png", bbox_inches='tight', dpi=dpi)

    def edf_slider(self, bins=100, xlim=None, flag_keV=False):
        ''' Create a slider plot for energy distribution functions '''
        ## Convert gamma to keV if needed
        rescale, gamma_unit, gamma_name = self.return_units_gamma(flag_keV)
        ## Set initial histogram
        def update_func(ind):
            return rescale(self.gamma[ind])
        if xlim is not None and not isinstance(xlim, list):
            max_x, max_ind = tuple_maxvalue(self.gamma, rescale=rescale)
            xlim = [0.95 if not flag_keV else 0., max_x]
            print(f"Maximum {gamma_name}{gamma_unit} over all timesteps at timestep {self.timesteps[max_ind]}: {max_x:.2e}")
        fig, ax = histplot(update_func(0), bins, gamma_name+gamma_unit, f'Number of {self.name}', xlim=xlim, suptitle=f"t = {self.time[0]:.2e} s")
        ## Either add slider
        if self.save_address is None:
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, adjust_plot=True)
            slider.on_changed(lambda val: update_histogram_slider(val, ax, update_func, bins, gamma_name+gamma_unit, f'Number of {self.name}', xlim, self.timesteps, self.time))
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, "edf_anim", update_histogram_slider, fargs=(ax, update_func, bins, gamma_name+gamma_unit, f'Number of {self.name}', xlim, self.timesteps, self.time))

    def emittance_slider(self, bins=100, vmin=None, vmax=None, xrange=None, yrange=None, zrange=None, prange=None):
        ''' Create a slider density plot for several 2D plane of a given plasma quantity, based on a given function '''
        ## Create slider function for time evolution
        def update_time(val):
            ind = list(self.timesteps).index(val)
            hist2D_plot(self.x[ind], self.px[ind], bins, f"Number of {self.name}", "x [m]", "px [normalized]", ax=ax[0], vmin=vmin, vmax=vmax, range=[xrange, prange], cbar=x_cbar)
            hist2D_plot(self.y[ind], self.py[ind], bins, f"Number of {self.name}", "y [m]", "py [normalized]", ax=ax[1], vmin=vmin, vmax=vmax, range=[yrange, prange], cbar=y_cbar)
            hist2D_plot(self.z[ind], self.pz[ind], bins, f"Number of {self.name}", "z [m]", "pz [normalized]", ax=ax[2], vmin=vmin, vmax=vmax, range=[zrange, prange], cbar=z_cbar)
            plt.suptitle(f"t = {self.time[ind]:.2e} s")
            plt.draw()
        ## Set geometry limits if requested
        xrange = [self.xmin, self.xmax] if xrange is True else xrange
        yrange = [self.ymin, self.ymax] if yrange is True else yrange
        zrange = [self.zmin, self.zmax] if zrange is True else zrange
        ## Create initial grid and 2D density plots
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        self.x_gridind, self.y_gridind, self.z_gridind = self.ncell_x//2, self.ncell_y//2, self.ncell_z//2
        x_im, x_cbar = hist2D_plot(self.x[0], self.px[0], bins, f"Number of {self.name}", "x [m]", "px [normalized]", ax=ax[0], vmin=vmin, vmax=vmax, range=[xrange, prange])
        y_im, y_cbar = hist2D_plot(self.y[0], self.py[0], bins, f"Number of {self.name}", "y [m]", "py [normalized]", ax=ax[1], vmin=vmin, vmax=vmax, range=[yrange, prange])
        z_im, z_cbar = hist2D_plot(self.z[0], self.pz[0], bins, f"Number of {self.name}", "z [m]", "pz [normalized]", ax=ax[2], vmin=vmin, vmax=vmax, range=[zrange, prange])
        plt.suptitle(f"t = {self.time[0]:.2e} s")
        ## Either add slider
        if self.save_address is None:
            plt.subplots_adjust(bottom=0.3)
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, loc=[0.25, 0.1, 0.45, 0.03])
            slider.on_changed(update_time)
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, "densityplot2D_anim", update_time, fargs=())
    
    ''' ------------------------------ '''
    ''' Functions for general analysis '''
    ''' ------------------------------ '''
    def classattribute_slider(self, attr, xlab, bins=100, xlim=None):
        ''' Create a slider plot for any class attribute distribution ''' 
        ## Set initial histogram
        def update_func(ind):
            return getattr(self, attr)[ind]
        fig, ax = histplot(update_func(0), bins, xlab, f'Number of {self.name}', xlim=xlim, suptitle=f"t = {self.time[0]:.2e} s")
        ## Either add slider
        if self.save_address is None:
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, adjust_plot=True)
            slider.on_changed(lambda val: update_histogram_slider(val, ax, update_func, bins, xlab, f'Number of {self.name}', xlim, self.timesteps, self.time))
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, f"{attr}_anim", update_histogram_slider, fargs=(ax, update_func, bins, xlab, f'Number of {self.name}', xlim, self.timesteps, self.time))
        
    def evaluate_average(self, attr, ylab, xlim=None, ylim=None, mf=1):
        ''' Function to evaluate the average of a given attribute '''
        ## Evaluate average and standard deviation of a given attribute over time
        avg = [np.average(getattr(self, attr)[ind]*mf) for ind in range(len(self.timesteps))]
        std = [np.std(getattr(self, attr)[ind]*mf) for ind in range(len(self.timesteps))]
        ## Create the plot
        plot2D(self.time*1e9, avg, "t [ns]", ylab, xlim=xlim, ylim=ylim)
        plt.errorbar(self.time*1e9, avg, yerr=std, fmt=" ", lw=0.5, color="blue")
        if self.save_address is not None:
            plt.savefig(self.save_address+f"average_{attr}_{self.name}.png", bbox_inches='tight', dpi=dpi)
        

''' --------------------------------------------------------- '''   
''' Subclass to filter a part of tracked particles population '''
''' --------------------------------------------------------- '''
class track_filtered(track_particles):
    ''' ------------------------ '''
    ''' Initialization functions '''
    ''' ------------------------ '''
    def __init__(self, path_field, path_data, species_name, filter_quant, ll=-np.inf, ul=np.inf, ts=False, save_address=None, extfield=None, grid_refinement=None):
        ''' Function to initialize the subclass '''
        ## Initialize superclass
        super().__init__(path_field, path_data, species_name, save_address=save_address, flag_superclass=False, grid_refinement=grid_refinement)
        ## Create dictionary to map filter names to class attributes
        warpx = OpenPMDTimeSeries(path_data)
        x = self.get_warpx_attributes(warpx, "x")
        y = self.get_warpx_attributes(warpx, "y")
        px = self.get_warpx_attributes(warpx, "ux")
        py = self.get_warpx_attributes(warpx, "uy")
        pz = self.get_warpx_attributes(warpx, "uz")
        filter_dict = {"x": x,
                       "y": y,
                       "z": self.get_warpx_attributes(warpx, "z"),
                       "r": tuple(np.sqrt(xx**2 + yy**2) for xx, yy in zip(x, y)),
                       "px": px,
                       "py": py,
                       "pz": pz,
                       "w": self.get_warpx_attributes(warpx, "w"),
                       "gamma": tuple(gamma_from_p(uxn, uyn, uzn) for uxn, uyn, uzn in zip(px, py, pz))}
        ## Find timestep for filtering and create mask for id to track filtered particles at each timestep
        ts_ind = self.find_timestep(ts)
        id = self.get_warpx_attributes(warpx, "id")
        id_filt = id[ts_ind][np.argwhere((np.array(filter_dict[filter_quant][ts_ind])<ul) & (np.array(filter_dict[filter_quant][ts_ind])>ll))][:,0]
        id_mask = tuple(np.where(np.isin(id_n, id_filt))[0] for id_n in id)
        ## Create filtered class attributes
        self.x = tuple(x[mask] for x, mask in zip(filter_dict["x"], id_mask))
        self.y = tuple(y[mask] for y, mask in zip(filter_dict["y"], id_mask))
        self.z = tuple(z[mask] for z, mask in zip(filter_dict["z"], id_mask))
        self.px = tuple(px[mask] for px, mask in zip(filter_dict["px"], id_mask))
        self.py = tuple(py[mask] for py, mask in zip(filter_dict["py"], id_mask))
        self.pz = tuple(pz[mask] for pz, mask in zip(filter_dict["pz"], id_mask))
        self.w = tuple(w[mask] for w, mask in zip(filter_dict["w"], id_mask))
        self.gamma = tuple(gamma[mask] for gamma, mask in zip(filter_dict["gamma"], id_mask))
        self.lost_part = [len(self.x[0])-len(self.x[i]) for i in range(len(self.x))] # Number of lost particles at each timestep
        self.r = tuple(np.sqrt(x**2 + y**2) for x, y in zip(self.x, self.y)) # Radial position of particles
        if extfield is not None:
            self.evaluate_extfield_quantities(extfield)
        ## Print diagnostics on filtered population
        print(f"Total number of filtered {self.name} tracked: {self.px[0].shape[0]}")
        print(f"Total number of filtered {self.name} tracked: {self.x[0].shape[0]} ({self.x[0].shape[0]/x[0].shape[0]*100:.2f}% of total)")
        print(f"Total number of confined {self.name} tracked (within filtered population): {self.px[-1].shape[0]} ({(self.px[-1].shape[0])/(self.px[0].shape[0] if self.px[0].shape[0] != 0 else 1)*100:.2f} %)")
        print(f"Total number of lost {self.name} (within filtered population): {self.px[0].shape[0]-self.px[-1].shape[0]} ({(self.px[0].shape[0]-self.px[-1].shape[0])/(self.px[0].shape[0] if self.px[0].shape[0] != 0 else 1)*100:.2f} %)")
        


''' ------------------------------------ '''
''' Subclass for tracking lost particles '''
''' ------------------------------------ '''
class track_lost_particles(track_particles):
    ''' ------------------------ '''
    ''' Initialization functions '''
    ''' ------------------------ '''
    def __init__(self, path_field, path_data, species_name, save_address=None, extfield=None, ts=-1, grid_refinement=None):
        ''' Function to initialize the subclass '''
        ## Initialize superclass
        super().__init__(path_field, path_data, species_name, save_address=save_address, flag_superclass=False, grid_refinement=grid_refinement)
        ## Create mask for id to track filtered particles at each timestep
        warpx = OpenPMDTimeSeries(path_data)
        id = self.get_warpx_attributes(warpx, "id")
        id_mask = tuple(np.where(~np.isin(id_n, id[ts]))[0] for id_n in id)
        ## Create lost particles class attributes
        self.x = self.get_warpx_attributes(warpx, "x", mask=id_mask)
        self.y = self.get_warpx_attributes(warpx, "y", mask=id_mask)
        self.z = self.get_warpx_attributes(warpx, "z", mask=id_mask)
        self.px = self.get_warpx_attributes(warpx, "ux", mask=id_mask) # gamma*vx/c
        self.py = self.get_warpx_attributes(warpx, "uy", mask=id_mask) # gamma*vy/c
        self.pz = self.get_warpx_attributes(warpx, "uz", mask=id_mask) # gamma*vz/c
        self.w = self.get_warpx_attributes(warpx, "w", mask=id_mask)
        self.gamma = tuple(gamma_from_p(uxn, uyn, uzn) for uxn, uyn, uzn in zip(self.px, self.py, self.pz))
        self.r = tuple(np.sqrt(x**2 + y**2) for x, y in zip(self.x, self.y)) # Radial position of particles
        if extfield is not None:
            self.evaluate_extfield_quantities(extfield)
        ## Print diagnostics on lost population
        print(f"Total number of lost {self.name} tracked: {self.px[0].shape[0]} ({(self.px[0].shape[0])/id[0].shape[0]*100:.2f} %)")