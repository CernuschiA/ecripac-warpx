import sys
sys.path.append("/home/cernuschi/coding/warpx/ecripac/functions/")
from base_funcs import *

############################################################
######################### Classes ##########################
############################################################

''' ----------------- '''
''' Warpx superclass '''
''' ----------------- '''
class warpx_namelist:
    ''' ------------------------ '''
    ''' Initialization functions '''
    ''' ------------------------ '''
    def __init__(self, path_data, save_address=None, fps=60, grid_refinement=None, pclass_flag=False):
        ''' Function to initialize the Warpx superclass '''
        ## Delete and recreate folder to save figures
        if save_address is not None:
            if os.path.isdir(save_address):
                shutil.rmtree(save_address)
            os.makedirs(save_address)
            self.save_address = save_address
            self.fps = fps
        else:
            self.save_address = None
        ## Open field diagnostics (if not already provided) and extract raw data
        if isinstance(path_data, str):
            warpx = OpenPMDTimeSeries(path_data)
        else:
            warpx = path_data
        _, field_info = warpx.get_field(field="rho", iteration=0)
        self.geometry = field_info.component_attrs["geometry"]
        # Set data for theta mode (cylindrical)
        if field_info.component_attrs["geometry"] == "thetaMode":
            self.rmin, self.rmax, self.dr = field_info.rmin, field_info.rmax, field_info.dr
            self.rgrid = field_info.r
            self.ncell_r = len(self.rgrid)-1 # Number of cells in radial direction
            self.xmin, self.xmax, self.dx = self.rmin, self.rmax, self.dr
            self.ymin, self.ymax, self.dy = self.rmin, self.rmax, self.dr
            self.xgrid, self.ygrid = self.rgrid.copy(), self.rgrid.copy()
        # Set data for cartesian geometry (3D)
        elif field_info.component_attrs["geometry"] == "cartesian":
            self.xmin, self.xmax, self.dx = field_info.xmin, field_info.xmax, field_info.dx
            self.ymin, self.ymax, self.dy = field_info.ymin, field_info.ymax, field_info.dy
            self.xgrid, self.ygrid = field_info.x, field_info.y
        # Set common data for both geometries
        self.zmin, self.zmax, self.dz = field_info.zmin, field_info.zmax, field_info.dz
        self.zgrid = field_info.z
        self.Lx, self.Ly, self.Lz = self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin # Domain size
        # Compensate for field coarsening when initializing particle class
        if isinstance(grid_refinement, (list, tuple, np.ndarray)) and pclass_flag:
            self.dx /= grid_refinement[0]
            self.dy /= grid_refinement[1]
            self.dz /= grid_refinement[2]
            self.xgrid = np.arange(self.xmin, self.xmax+self.dx, self.dx)
            self.ygrid = np.arange(self.ymin, self.ymax+self.dy, self.dy)
            self.zgrid = np.arange(self.zmin, self.zmax+self.dz, self.dz)
        self.ncell_x, self.ncell_y, self.ncell_z = len(self.xgrid)-1, len(self.ygrid)-1, len(self.zgrid)-1 # Number of cells in each direction
        ## Obtain reference quantities for SI conversion
        self.Q_unit = " [C]" # Charge units
        self.m_unit = " [kg]" # Mass units
        self.p_unit = f" [{gamma_unicode}{beta_unicode}]" # Momentum units
        self.x_unit = " [m]" # Length units
        self.t_unit = " [s]" # Time units
        self.n_unit = " [m^-3]" # Density units
        self.E_unit = " [V/m]" # Electric field units
        self.B_unit = " [T]" # Magnetic field units
        self.rho_unit = " [C/m^3]" # Charge density units
        self.J_unit = " [A/m^2]" # Current density units

    ''' ------------------------------------------- '''
    ''' General diagnostic postprocessing functions '''
    ''' ------------------------------------------- '''
    def plasma_chamber(self):
        ''' Create geometry of plasma chamber for plotting purposes '''
        ## Initialize axes and grids
        z_pc = np.linspace(self.zmin, self.zmax, 100)
        self.theta = np.linspace(0, 2*np.pi, 100)
        self.t_grid, self.z_grid = np.meshgrid(self.theta, z_pc)
        self.x_grid = self.xmax*np.cos(self.t_grid)
        self.y_grid = self.ymax*np.sin(self.t_grid)
        ## Create shapes for plotting
        self.circle = plt.Circle((0 , 0), self.xmax, alpha=0.1)
        self.rectZX = plt.Rectangle((self.zmin, self.xmin), self.Lx, self.Lx, alpha=0.1)
        self.rectZY = plt.Rectangle((self.zmin, self.ymin), self.Lx, self.Ly, alpha=0.1)
        
    def set_position(self, x, y, z):
        ''' Retrieve indexes of the Yee grid corresponding to a given position '''
        x_gridind = self.ncell_x//2 if x is None else int((x-self.xmin)//self.dx)
        y_gridind = self.ncell_y//2 if y is None else int((y-self.ymin)//self.dy)
        z_gridind = self.ncell_z//2 if z is None else int((z-self.zmin)//self.dz)
        return x_gridind, y_gridind, z_gridind
    
    def find_timestep(self, ts):
        ''' Find the index of a given timestep or time in the simulation '''
        if abs(ts) < 1:
            ts_ind = min(range(len(self.time)), key=lambda i: abs(self.time[i] - ts))
        else:
            ts_ind = list(self.timesteps).index(ts)
        return ts_ind

    def save_animation(self, fig, ani_name, func, fargs):
        ''' Save animation to file '''
        ani = matani.FuncAnimation(fig, func, self.timesteps, interval=1000/self.fps, fargs=fargs)
        FFwriter = matani.FFMpegWriter(fps=self.fps)
        ani.save(f"{self.save_address}{ani_name}.mp4", writer=FFwriter)
    
    
    ''' ----------------------------------------- '''
    ''' Multiple populations diagnostic functions '''
    ''' ----------------------------------------- '''
    def population_position(self, pop, n_part=10000, alpha=1.):
        ''' Plot multiple populations position distributions in 3D '''
        ## Create mask function
        def pop_random_indexes(arr):
            if len(arr) > n_part:
                ind_p = sample_random_indexes(arr, n_part)
            else:
                ind_p = slice(None)
            return ind_p
        dict = [{"c": p.c} for p in pop]
        for i, p in enumerate(pop):
            dict[i]["mask"] = pop_random_indexes(p.x[0])
        ## 3D slider function
        def update3D(val):
            ind = list(pop[0].timesteps).index(val)
            for d, p in zip(dict, pop):
                d["mask"] = pop_random_indexes(p.x[ind])
                d["sc"]._offsets3d = (p.z[ind][d["mask"]], p.x[ind][d["mask"]], p.y[ind][d["mask"]])
            plt.suptitle(f"t = {pop[0].time[ind]:.2e} s")
        ## Setup 3D plot
        self.plasma_chamber()
        ax3D = plt.figure().add_subplot(111, projection='3d')
        ax3D.set_xlabel("z [m]", labelpad=10)
        ax3D.set_ylabel("x [m]", labelpad=10)
        ax3D.set_zlabel("y [m]", labelpad=10)
        ax3D.plot_surface(self.z_grid, self.x_grid, self.y_grid, alpha=0.2)
        for d, p in zip(dict, pop):
            d["sc"] = ax3D.scatter(p.z[0][d["mask"]], p.x[0][d["mask"]], p.y[0][d["mask"]], s=scatterdim, c=d["c"], alpha=alpha)
        plt.suptitle(f"t = {pop[0].time[0]:.2e} s")
        ## Add slider to 2D plots
        slider3D = create_slider('Timestep', pop[0].timesteps[0], pop[0].timesteps[-1], pop[0].timesteps[0], pop[0].timesteps, adjust_plot=True)
        slider3D.on_changed(update3D)
        ## 2D slider function
        def update2D(val):
            ind = list(self.timesteps).index(val)
            for d, p in zip(dict, pop):
                d["mask"] = pop_random_indexes(p.x[ind])
                update_2Dscatterplot(p.z[ind][d["mask"]], p.x[ind][d["mask"]], d["plZX"])
                update_2Dscatterplot(p.z[ind][d["mask"]], p.y[ind][d["mask"]], d["plZY"])
                update_2Dscatterplot(p.x[ind][d["mask"]], p.y[ind][d["mask"]], d["plXY"])
            plt.suptitle(f"t = {pop[0].time[ind]:.2e} s")
            plt.draw()
        ## Setup 2D plot
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].add_artist(self.rectZY)
        ax[1].add_artist(self.rectZX)
        ax[2].add_artist(self.circle)
        for d, p in zip(dict, pop):
            d["plZX"] = plot2D(p.z[0][d["mask"]], p.x[0][d["mask"]], "z [m]", "x [m]", xlim=[self.zmin, self.zmax], ylim=[self.xmin, self.xmax], col=d["c"], ax=ax[0], alpha=alpha, plot_type="scatter")
            d["plZY"] = plot2D(p.z[0][d["mask"]], p.y[0][d["mask"]], "z [m]", "y [m]", xlim=[self.zmin, self.zmax], ylim=[self.ymin, self.ymax], col=d["c"], ax=ax[1], alpha=alpha, plot_type="scatter")
            d["plXY"] = plot2D(p.x[0][d["mask"]], p.y[0][d["mask"]], "x [m]", "y [m]", xlim=[self.xmin, self.xmax], ylim=[self.ymin, self.ymax], col=d["c"], ax=ax[2], alpha=alpha, plot_type="scatter")
        plt.suptitle(f"t = {pop[0].time[0]:.2e} s")
        plt.subplots_adjust(bottom=0.25, wspace=0.6, hspace=0.6)
        ## Either add slider
        if self.save_address is None:
            slider2D = create_slider('Timestep', pop[0].timesteps[0], pop[0].timesteps[-1], pop[0].timesteps[0], pop[0].timesteps)
            slider2D.on_changed(update2D)
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, "position_anim", update2D, fargs=())
    
    def population_edf(self, pop, bins=100, xlim=None, flag_keV=False, norm=False):
        ''' Create a slider plot for energy distribution functions '''
        ## Convert gamma to keV if needed
        _, gamma_unit, gamma_name = pop[0].return_units_gamma(flag_keV)
        ## Define update function for slider
        def update_func(ind):
            return [p.return_units_gamma(flag_keV)[0](p.gamma[ind]) for p in pop]
        ## Set xlim if flag active
        if xlim is not None and not isinstance(xlim, list):
            max_x = max([tuple_maxvalue(p.gamma, rescale=p.return_units_gamma(flag_keV)[0])[0] for i, p in enumerate(pop)])
            xlim = [0.95 if not flag_keV else 0., max_x]
            print(f"Maximum {gamma_name}{gamma_unit} over all timesteps: {max_x:.2e}")
        ymax = None
        ## Setup initial plot
        fig, ax = histplot(update_func(0)[0], bins, gamma_name+gamma_unit, 'Number of particles', xlim=xlim, suptitle=f"t = {self.time[0]:.2e} s", norm=norm, label=pop[0].label)
        for i in range(1, len(pop)):
            hval, _ = histplot(update_func(0)[i], bins, gamma_name+gamma_unit, 'Number of particles', xlim=xlim, ylim=[0, ymax], suptitle=f"t = {self.time[0]:.2e} s", ax=ax, norm=norm, label=pop[i].label)
            ymax = np.max([ymax, np.max(hval*1.1)]) if ymax is not None else np.max(hval)
        ## Either add slider
        if self.save_address is None:
            slider = create_slider('Timestep', self.timesteps[0], self.timesteps[-1], self.timesteps[0], self.timesteps, adjust_plot=True)
            slider.on_changed(lambda val: update_multihistogram_slider(val, ax, update_func, [p.label for p in pop], bins, gamma_name+gamma_unit, 'Number of particles', xlim, self.timesteps, self.time, norm))
            plt.show(block=True)
        ## Or save animation
        else:
            self.save_animation(fig, "edf_anim", update_multihistogram_slider, fargs=(ax, update_func, [p.label for p in pop], bins, gamma_name+gamma_unit, 'Number of particles', xlim, self.timesteps, self.time, norm))