############################################################
######################### Libraries ########################
############################################################

## System libraries
import shutil
import os
from openpmd_viewer import OpenPMDTimeSeries

## Scientific and plotting libraries
import numpy as np
import scipy.constants as sciconst
import matplotlib.animation as matani
import matplotlib.pyplot as plt
import matplotlib.widgets as plt_w
from matplotlib import colors, colormaps
from numba import njit

############################################################
######################### Constants ########################
############################################################

## Symbols unicode
gamma_unicode = '\u03B3'
beta_unicode = '\u03B2'
perpendicular_unicode = '\u22A5'

## Plotting settings
axcolor = 'lightgoldenrodyellow'
colormap = colormaps.get_cmap('magma')
norma = colors.LogNorm()
scatterdim = 1
dpi = 300
rng = np.random.default_rng()

############################################################
######################### Functions ########################
############################################################

''' ------------------------- '''
''' Data processing functions '''
''' ------------------------- '''
def max_triplenestedlist(list, rescale=1):
    ''' Return the maximum value among all arrays in a triply nested list '''
    maxval = -np.inf
    ind = (0, 0, 0)
    for i, sublist in enumerate(list):
        for j, arr in enumerate(sublist):
            for k, val in enumerate(arr):
                if val > maxval:
                    maxval = val
                    ind = (i, j, k)
    if callable(rescale):
        return (rescale(maxval)), ind
    return maxval*rescale, ind

def min_triplenestedlist(list, rescale=1):
    ''' Return the minimum value among all arrays in a triply nested list '''
    minval = np.inf
    ind = (0, 0, 0)
    for i, sublist in enumerate(list):
        for j, arr in enumerate(sublist):
            for k, val in enumerate(arr):
                if val < minval:
                    minval = val
                    ind = (i, j, k)
    if callable(rescale):
        return (rescale(minval)), ind
    return minval*rescale, ind

def tuple_maxvalue(tup, rescale=1):
    ''' Return the maximum value among all arrays in a tuple '''
    maxval = -np.inf
    ind = 0
    for i, arr in enumerate(tup):
        if arr.size > 0 and np.nanmax(arr) > maxval:
                maxval = np.nanmax(arr)
                ind = i
    if callable(rescale):
        return (rescale(maxval)), ind
    return maxval*rescale, ind

def tuple_minvalue(tup, rescale=1):
    ''' Return the minimum value among all arrays in a tuple '''
    minval = np.inf
    ind = 0
    for i, arr in enumerate(tup):
        if arr.size > 0 and np.nanmin(arr) < minval:
            minval = np.nanmin(arr)
            ind = i
    if callable(rescale):
        return (rescale(minval)), ind
    return minval*rescale, ind

def find_cartesian_lim_tuples(xdata, ydata, zdata, q, timesteps, unit_label, rescale=1):
    ''' Find tuple maximum and minimum values for cartesian components '''
    xmax, xmax_ind = tuple_maxvalue(xdata, rescale)
    ymax, ymax_ind = tuple_maxvalue(ydata, rescale)
    zmax, zmax_ind = tuple_maxvalue(zdata, rescale)
    xmin, xmin_ind = tuple_minvalue(xdata, rescale)
    ymin, ymin_ind = tuple_minvalue(ydata, rescale)
    zmin, zmin_ind = tuple_minvalue(zdata, rescale)
    xlim = [xmin, xmax]
    ylim = [ymin, ymax]
    zlim = [zmin, zmax]
    plim = [0., np.hypot(ymax, zmax)]
    ## Print results
    print(f"Maximum {q}x = {xmax:.2e}{unit_label} at timesteps {timesteps[xmax_ind]}, minimum {q}x = {xmin:.2e}{unit_label} at timesteps {timesteps[xmin_ind]}")
    print(f"Maximum {q}y = {ymax:.2e}{unit_label} at timesteps {timesteps[ymax_ind]}, minimum {q}y = {ymin:.2e}{unit_label} at timesteps {timesteps[ymin_ind]}")
    print(f"Maximum {q}z = {zmax:.2e}{unit_label} at timesteps {timesteps[zmax_ind]}, minimum {q}z = {zmin:.2e}{unit_label} at timesteps {timesteps[zmin_ind]}")
    return xlim, ylim, zlim, plim

def angle_vector(vx1, vy1, vz1, vx2, vy2, vz2):
    ''' Calculate the angle between two 3D vectors '''
    dot_product = vx1*vx2 + vy1*vy2 + vz1*vz2
    mag1 = np.sqrt(vx1**2 + vy1**2 + vz1**2)
    mag2 = np.sqrt(vx2**2 + vy2**2 + vz2**2)
    cos_angle = dot_product / (mag1 * mag2)
    angle = np.arccos(cos_angle)
    return angle

def perpendicular_component(x, y):
    ''' Calculate the perpendicular component from transversal coordinates '''
    return (np.asarray(x)**2+np.asarray(y)**2)**0.5

def sample_random_indexes(arr, n):
    ''' Return n random indexes from an array '''
    return rng.choice(arr.shape[0], n, replace=False)

''' -------------------- '''
''' Scientific functions '''
''' -------------------- '''
def kin_en_from_gamma(gamma, mc2):
    ''' Calculate kinetic energy from Lorentz factor '''
    return (gamma - 1.)*mc2

def gamma_from_p(px, py, pz):
    ''' Calculate Lorentz factor from normalized momentum '''
    return (1.+px**2+py**2+pz**2)**0.5

def cic_weighting(x, dx, disp):
        ''' Cloud-in-cell weighting scheme for 1D arrays '''
        ind = int(x/dx + 0.5 + disp) # Nearest grid point index
        if ind < 0:
            ind = 0
        xx = x/dx  + disp - ind # Relative position to nearest grid point
        xx2 = xx**2
        Nm1 = 0.5*(xx2 - xx + 0.25) # Shape function area at ind-1
        N0 = 0.75 - xx2 # Shape function area at ind
        Np1 = 0.5*(xx2 + xx + 0.25) # Shape function area at ind+1
        return ind, Nm1, N0, Np1

@njit
def evaluate_density3D_numba(x, y, z, w, dx, dy, dz, ncell_x, ncell_y, ncell_z, xmin, ymin, zmin):
    ''' Evaluate density on a 3D grid using cloud-in-cell weighting scheme with Numba optimization '''
    ## Precompute inverse of cell sizes and grid displacements
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz
    disp_x = xmin // dx
    disp_y = ymin // dy
    disp_z = zmin // dz
    ## Initialize grid
    grid = np.zeros((ncell_x+1, ncell_y+1, ncell_z+1))
    n = x.shape[0]
    ## Loop over particles and deposit weights on grid
    for i in range(n):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        wi = w[i]
        # ----- X -----
        gx = xi * inv_dx
        ix = int(gx + 0.5 + disp_x)
        if ix < 0:
            ix = 0
        xx = gx + disp_x - ix
        xx2 = xx * xx
        xNm1 = 0.5 * (xx2 - xx + 0.25)
        xN0  = 0.75 - xx2
        xNp1 = 0.5 * (xx2 + xx + 0.25)
        # ----- Y -----
        gy = yi * inv_dy
        iy = int(gy + 0.5 + disp_y)
        if iy < 0:
            iy = 0
        yy = gy + disp_y - iy
        yy2 = yy * yy
        yNm1 = 0.5 * (yy2 - yy + 0.25)
        yN0  = 0.75 - yy2
        yNp1 = 0.5 * (yy2 + yy + 0.25)
        # ----- Z -----
        gz = zi * inv_dz
        iz = int(gz + 0.5 + disp_z)
        if iz < 0:
            iz = 0
        zz = gz + disp_z - iz
        zz2 = zz * zz
        zNm1 = 0.5 * (zz2 - zz + 0.25)
        zN0  = 0.75 - zz2
        zNp1 = 0.5 * (zz2 + zz + 0.25)
        # ----- Clamp once -----
        xm1 = ix-1 if ix > 0 else 0
        xp1 = ix+1 if ix < ncell_x-1 else ncell_x-1
        ym1 = iy-1 if iy > 0 else 0
        yp1 = iy+1 if iy < ncell_y-1 else ncell_y-1
        zm1 = iz-1 if iz > 0 else 0
        zp1 = iz+1 if iz < ncell_z-1 else ncell_z-1
        # ----- Deposition -----
        grid[xm1, ym1, zm1] += wi*xNm1*yNm1*zNm1
        grid[xm1, ym1, iz ] += wi*xNm1*yNm1*zN0
        grid[xm1, ym1, zp1] += wi*xNm1*yNm1*zNp1

        grid[xm1, iy , zm1] += wi*xNm1*yN0*zNm1
        grid[xm1, iy , iz ] += wi*xNm1*yN0*zN0
        grid[xm1, iy , zp1] += wi*xNm1*yN0*zNp1

        grid[xm1, yp1, zm1] += wi*xNm1*yNp1*zNm1
        grid[xm1, yp1, iz ] += wi*xNm1*yNp1*zN0
        grid[xm1, yp1, zp1] += wi*xNm1*yNp1*zNp1

        grid[ix , ym1, zm1] += wi*xN0*yNm1*zNm1
        grid[ix , ym1, iz ] += wi*xN0*yNm1*zN0
        grid[ix , ym1, zp1] += wi*xN0*yNm1*zNp1

        grid[ix , iy , zm1] += wi*xN0*yN0*zNm1
        grid[ix , iy , iz ] += wi*xN0*yN0*zN0
        grid[ix , iy , zp1] += wi*xN0*yN0*zNp1

        grid[ix , yp1, zm1] += wi*xN0*yNp1*zNm1
        grid[ix , yp1, iz ] += wi*xN0*yNp1*zN0
        grid[ix , yp1, zp1] += wi*xN0*yNp1*zNp1

        grid[xp1, ym1, zm1] += wi*xNp1*yNm1*zNm1
        grid[xp1, ym1, iz ] += wi*xNp1*yNm1*zN0
        grid[xp1, ym1, zp1] += wi*xNp1*yNm1*zNp1

        grid[xp1, iy , zm1] += wi*xNp1*yN0*zNm1
        grid[xp1, iy , iz ] += wi*xNp1*yN0*zN0
        grid[xp1, iy , zp1] += wi*xNp1*yN0*zNp1

        grid[xp1, yp1, zm1] += wi*xNp1*yNp1*zNm1
        grid[xp1, yp1, iz ] += wi*xNp1*yNp1*zN0
        grid[xp1, yp1, zp1] += wi*xNp1*yNp1*zNp1
    grid /= (dx * dy * dz)
    return grid

@njit
def evaluate_temperature3D_numba(x, y, z, g, dx, dy, dz, ncell_x, ncell_y, ncell_z, xmin, ymin, zmin):
    ''' Evaluate average temperature on a 3D grid with Numba optimization '''
    ## Precompute inverse of cell sizes and grid displacements
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz
    disp_x = xmin // dx
    disp_y = ymin // dy
    disp_z = zmin // dz
    ## Initialize grid
    grid = np.zeros((ncell_x+1, ncell_y+1, ncell_z+1))
    grid_p = np.zeros((ncell_x+1, ncell_y+1, ncell_z+1))
    n = x.shape[0]
    ## Loop over particles and deposit weights on grid
    for i in range(n):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        gi = g[i]
        # ----- X -----
        gx = xi * inv_dx
        ix = int(gx + 0.5 + disp_x)
        if ix < 0:
            ix = 0
        # ----- Y -----
        gy = yi * inv_dy
        iy = int(gy + 0.5 + disp_y)
        if iy < 0:
            iy = 0
        # ----- Z -----
        gz = zi * inv_dz
        iz = int(gz + 0.5 + disp_z)
        if iz < 0:
            iz = 0
        # ----- Deposition -----
        grid[ix, iy, iz] += gi
        grid_p[ix, iy, iz] += 1
    return grid, grid_p

''' --------------------------------------------------------- '''
''' Dispersion relations and plasma characteristic velocities '''
''' --------------------------------------------------------- '''
def plasma_frequency(pop, ts):
    ''' Evaluate the average plasma frequency for a given population at a given timestep '''
    return pop.plasma_average(ts, pop.evaluate_plasmafreq)

def gyrofrequency(pop, ts):
    ''' Evaluate the average gyrofrequency for a given population at a given timestep '''
    B_avg = pop.plasma_average(ts, pop.evaluate_Bnorm)
    gamma_avg = pop.plasma_average(ts, pop.evaluate_temperature, args=[False])
    return np.abs(pop.charge*B_avg/pop.mass/gamma_avg)

def omega_lefthand(electrons, ts):
    ''' Evaluate the average cut-off frequency omega lefthand for the plasma at a given timestep '''
    Omega_e = gyrofrequency(electrons, ts)
    omega_p = plasma_frequency(electrons, ts)
    return 0.5*(np.sqrt(Omega_e**2 + 4*omega_p**2) - Omega_e)

def omega_righthand(electrons, ts):
    ''' Evaluate the average cut-off frequency omega righthand for the plasma at a given timestep '''
    Omega_e = gyrofrequency(electrons, ts)
    omega_p = plasma_frequency(electrons, ts)
    return 0.5*(np.sqrt(Omega_e**2 + 4*omega_p**2) + Omega_e)

def omega_upper_hybrid(electrons, ts):
    ''' Evaluate the average upper hybrid frequency for the plasma at a given timestep '''
    Omega_e = gyrofrequency(electrons, ts)
    omega_p = plasma_frequency(electrons, ts)
    return np.sqrt(Omega_e**2 + omega_p**2)

def omega_lower_hybrid(electrons, ions, ts):
    Omega_i = gyrofrequency(ions, ts)
    Omega_e = gyrofrequency(electrons, ts)
    omega_p = plasma_frequency(electrons, ts)
    return np.sqrt(Omega_i*Omega_e/(1 + Omega_e**2/omega_p**2))

def alfven_velocity(ions, ts):
    ''' Evaluate the average Alfven velocity for the plasma at a given timestep '''
    rho = np.average(ions.evaluate_density(ts))
    gamma_i_avg = ions.plasma_average(ts, ions.evaluate_temperature, args=[False])
    B_avg = ions.plasma_average(ts, ions.evaluate_Bnorm)
    ca = max(B_avg/np.sqrt(sciconst.mu_0*rho*ions.mass*gamma_i_avg), sciconst.c)
    return ca

def bohm_gross_dr(electrons, ts, k):
    ''' Evaluate the average Bohm-Gross dispersion relation for the plasma at a given timestep '''
    omega_p = plasma_frequency(electrons, ts)
    We = electrons.plasma_average(ts, electrons.evaluate_temperature)*1e3
    v_th = min(np.sqrt(2*sciconst.e*We/electrons.mass), sciconst.c)
    return np.sqrt(omega_p**2 + 3*v_th**2*k**2)

''' ------------------ '''
''' Plotting functions '''
''' ------------------ '''

def plot2D(x, y, xlab, ylab, xlim=None, ylim=None, xlog=False, ylog=False, col=None, clabel=None, vx=None, vy=None, ax=None, alpha=1., plot_type="plot", savepath=None):
    ''' Generic function to create 2D plots of various types '''
    ## Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 8))
    ## Select plot type
    if plot_type == "plot":
        pl2D = ax.plot(x, y)
    elif plot_type == "scatter":
        pl2D = ax.scatter(x, y, s=scatterdim, c=col, cmap=colormap, alpha=alpha)
    elif plot_type == "quiver":
        pl2D = ax.quiver(x, y, vx, vy, col)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()
    ## Possible log scales and colorbar
    if xlog is not False:
        ax.set_xscale('log')
    if ylog is not False:
        ax.set_yscale('log')
    if (col is not None) and (clabel is not None):
        cbar = plt.colorbar(pl2D, label=clabel)
        return pl2D, cbar
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return pl2D

def triple_plot(x, y1, y2, y3, xlab, ylab, labels, xlim=None, ylim=None, ax=None, save_address=None):
    ''' Generic function to create triple plots '''
    ## Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1)
    ## Plot data
    ax.plot(x, y1, label=labels[0])
    ax.plot(x, y2, label=labels[1])
    ax.plot(x, y3, label=labels[2])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()
    ax.legend()
    if save_address is not None:
        plt.savefig(save_address) 

def histplot(data, bins, xlab, ylab, w=None, xlim=[None, None], ylim=[None, None], norm=False, label=None, suptitle=None, ax=None, fig=None):
    ''' Generic function to create histogram plots '''
    ## Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1)
    ## Plot histogram
    hvals, hbins, _ = ax.hist(data, bins=bins, weights=w, histtype="step", density=norm, label=label)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True)
    ax.set_xlim(xlim)
    if type(ylim) is list:
        ax.set_ylim(ylim)
    else:
        upper = np.max(hvals) * 1.1
        ax.set_ylim(0, upper if ylim is None else max(upper, float(ylim)))
    plt.suptitle(suptitle)
    if label is not None:
        ax.legend()
    if fig is not None:
        return fig, ax
    return hvals, hbins

def cartesian_histplot(xdata, ydata, zdata, xbins, ybins, zbins, xlab, xunits, ylab, xlim, ylim, zlim, plim, w=None, ax=None, suptitle=None, fig=None):
    ''' Generic function to create histogram plots for all cartesian components '''
    ## Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=(18,5))
        plt.subplots_adjust(wspace=0.4)
    ## Plot histograms
    histplot(xdata, xbins, f"{xlab}x{xunits}", ylab, xlim=xlim, ax=ax[0], w=w)
    histplot(ydata, ybins, f"{xlab}y{xunits}", ylab, xlim=ylim, ax=ax[1], w=w)
    histplot(zdata, zbins, f"{xlab}z{xunits}", ylab, xlim=zlim, ax=ax[2], w=w)
    histplot(perpendicular_component(xdata, ydata), zbins, f"{xlab}{perpendicular_unicode}{xunits}", ylab, xlim=plim, ax=ax[3], w=w)
    if suptitle is not None:
        plt.suptitle(suptitle)
    if fig is not None:
        return fig, ax

def imshowplot(quantity, limits, colorbar_label, xlab, ylab, ax=None, norma=None, lmin=None, lmax=None, asp='auto', savepath=None):
    ''' Generic function to create imshow plots of 2D quantities '''
    ## Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12, 8))
    ## Plot imshow
    imrw = ax.imshow(quantity, interpolation='bilinear', extent=limits, cmap=colormap, aspect=asp, norm=norma, vmin=lmin, vmax=lmax)
    cbar = plt.colorbar(imrw, label=colorbar_label)
    ax.set(xlabel=xlab, ylabel=ylab)
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return imrw, cbar

def pcolormesh_plot(xedges, yedges, quantity, colorbar_label, xlab, ylab, ax=None, norma=None, lmin=None, lmax=None):
    ''' Generic function to create pcolormesh plots of 2D quantities '''
    ## Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1)
    ## Plot pcolormesh
    pcm = ax.pcolormesh(xedges, yedges, quantity, cmap=colormap, norm=norma, vmin=lmin, vmax=lmax)
    cbar = plt.colorbar(pcm, label=colorbar_label)
    ax.set(xlabel=xlab, ylabel=ylab)
    return pcm, cbar

def hist2D_plot(xdata, ydata, bins, colorbar_label, xlab, ylab, ax=None, range=None, dens=False, vmin=None, vmax=None, cbar=None):
    ''' Generic function to create 2D histogram plots '''
    ## Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        ax.cla()
    ## Plot 2D histogram
    h2d, xedges, yedges, im = ax.hist2d(xdata, ydata, bins=bins, range=range, density=dens, cmap=colormap, vmin=vmin, vmax=vmax)
    if range is not None:
        ax.set_xlim(range[0])
        ax.set_ylim(range[1])
    ax.set(xlabel=xlab, ylabel=ylab)
    if cbar is None:
        cbar = plt.colorbar(im, label=colorbar_label)
    else:
        vmin = np.amin(h2d) if vmin is None else vmin
        vmax = np.amax(h2d) if vmax is None else vmax
        cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
    return im, cbar

''' --------------------------- '''
''' Slider and update functions '''
''' --------------------------- '''

def create_slider(label, sl_st, sl_end, sl_init, sl_step, loc=[0.25, 0.1, 0.65, 0.03], adjust_plot=False):
    ''' Generic function to create a slider '''
    ## Adjust plot to make space for slider
    if adjust_plot is True:
        plt.subplots_adjust(bottom=0.25)
    ## Create slider
    axslider = plt.axes(loc, facecolor=axcolor)
    slider = plt_w.Slider(axslider, label, sl_st, sl_end, valinit=sl_init, valstep=sl_step)
    return slider

def update_2Dplot_slider(plot, data, ax, lim=True):
    ''' Generic function to update a 2D plot with a slider '''
    plot.set_ydata(data)
    if lim is True:
        ax.set_ylim(np.min(data), np.max(data)*1.1)

def update_cartesian2Dplot_slider(plotx, ploty, plotz, xdata, ydata, zdata, xax, yax, zax, lim=True):
    ''' Generic function to update a triple cartesian 2D plots with a slider '''
    update_2Dplot_slider(plotx, xdata, xax, lim)
    update_2Dplot_slider(ploty, ydata, yax, lim)
    update_2Dplot_slider(plotz, zdata, zax, lim)

def update_histogram_slider(val, ax, func, bins, xlab, ylab, xlim, timesteps, time):
    ''' Generic function to update histogram plots with a slider '''
    ## Update histogram
    ax.cla()
    ind = list(timesteps).index(val)
    data = func(ind)
    histplot(data, bins, xlab, ylab, xlim=xlim, ax=ax, suptitle=f"t = {time[ind]:.2e} s")
    plt.draw()
    ## Print information on timestep
    print(f"Maximum {xlab} at timestep {timesteps[ind]}: {np.max(data):.2e}")
    print(f"Minimum {xlab} at timestep {timesteps[ind]}: {np.min(data):.2e}")
    print(f"Average {xlab} at timestep {timesteps[ind]}: {np.average(data):.2e}")

def update_multihistogram_slider(val, ax, funcs, labels, bins, xlab, ylab, xlim, timesteps, time, norm):
    ''' Generic function to update multiple histogram plots with a slider '''
    ## Update histogram
    ax.cla()
    ymax = 0
    ind = list(timesteps).index(val)
    for data, lab in zip(funcs(ind), labels):
        hval, _ = histplot(data, bins, xlab, ylab, xlim=xlim, ylim=ymax, ax=ax, suptitle=f"t = {time[ind]:.2e} s", norm=norm, label=lab)
        ymax = np.max([ymax, np.max(hval*1.1)]) if ymax is not None else np.max(hval)
    plt.draw()
    
def update_2Dscatterplot(x, y, plot, q=None, cbar=None):
    ''' Generic function to update 2D scatter plots '''
    xy = np.vstack((x, y)).T
    plot.set_offsets(xy)
    if q is not None and cbar is not None:
        plot.set_array(q)
        cbar.mappable.set_clim(vmin=np.amin(q), vmax=np.amax(q))
    
def update_2Ddensityplot(data, plot, cbar, vmin=None, vmax=None):
    ''' Generic function to update 2D density plots '''
    vmin = np.amin(data) if vmin is None else vmin
    vmax = np.amax(data) if vmax is None else vmax
    plot.set_array(np.flipud(data))
    cbar.mappable.set_clim(vmin=vmin, vmax=vmax)

def update_cartesian_pcolormesh(xplot, yplot, zplot, xdata, ydata, zdata, xcbar, ycbar, zcbar, vmin, vmax):
    ''' Generic function to update cartesian pcolormesh plots '''
    xvmin = np.amin(xdata) if vmin is None else vmin
    xvmax = np.amax(xdata) if vmax is None else vmax
    yvmin = np.amin(ydata) if vmin is None else vmin
    yvmax = np.amax(ydata) if vmax is None else vmax
    zvmin = np.amin(zdata) if vmin is None else vmin
    zvmax = np.amax(zdata) if vmax is None else vmax
    xplot.set_array(xdata)
    yplot.set_array(ydata)
    zplot.set_array(zdata)
    xcbar.mappable.set_clim(vmin=xvmin, vmax=xvmax)
    ycbar.mappable.set_clim(vmin=yvmin, vmax=yvmax)
    zcbar.mappable.set_clim(vmin=zvmin, vmax=zvmax)

def update_cartesian_histplot(val, ax, xdata, ydata, zdata, xbins, ybins, zbins, xlab, xunits, ylab, xlim, ylim, zlim, plim, timesteps, rescale, time, w=None):
    ''' Generic function to update cartesian histogram plots with a slider '''
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()
    ax[3].cla()
    ind = list(timesteps).index(val)
    cartesian_histplot(xdata[ind]*rescale, ydata[ind]*rescale, zdata[ind]*rescale, xbins, ybins, zbins, xlab, xunits, ylab, xlim, ylim, zlim, plim, ax=ax, suptitle=f"t = {time[ind]:.2e} s", w=w[ind] if isinstance(w, tuple) else None)
    plt.draw()
    
def clear_field_axes(ax):
    ''' Clear all field axes '''
    for a in ax: # remove any colorbar(s) attached to mappables on this axis
        for coll in list(a.collections):
            if coll.colorbar is not None:
                coll.colorbar.remove()
        a.cla()