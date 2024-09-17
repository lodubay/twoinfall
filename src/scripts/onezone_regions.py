"""
This script plots the outputs of one-zone models over the APOGEE data at 
three regions of the galaxy.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vice
from multizone.src.models import twoinfall, twoinfall_sf_law, equilibrium_mass_loading
import paths
from multizone.src.yields import W23
from multizone.src import models, dtds
from apogee_sample import APOGEESample
from _globals import END_TIME, ONEZONE_DEFAULTS, TWO_COLUMN_WIDTH, ZONE_WIDTH
from colormaps import paultol
from track_and_mdf import setup_axes, plot_vice_onezone
from utils import get_bin_centers

ONSET = 3.5
XLIM = (-1.4, 0.6)
YLIM = (-0.12, 0.48)

def main():
    plt.style.use(paths.styles / "paper.mplstyle")
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.36*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(7, 22, wspace=0.)
    subfigs = [fig.add_subfigure(gs[:,i:i+w]) for i, w in zip((0, 8, 15), (8, 7, 7))]
    # Outflow mass-loading factor
    eta_func = equilibrium_mass_loading(alpha_h_eq=0.1, tau_sfh=15., tau_star=2.)
    axs0 = plot_region(subfigs[0], 4, eta=eta_func, 
                       color=paultol.highcontrast.colors[2],
                       xlim=XLIM, ylim=YLIM)
    axs1 = plot_region(subfigs[1], 8, eta=eta_func,
                       color=paultol.highcontrast.colors[1],
                       xlim=XLIM, ylim=YLIM, ylabel=False)
    axs2 = plot_region(subfigs[2], 12, eta=eta_func,
                       color=paultol.highcontrast.colors[0],
                       xlim=XLIM, ylim=YLIM, ylabel=False)
    plt.subplots_adjust(bottom=0.13, top=0.98, left=0.16, right=0.98, wspace=0.5)
    fig.savefig(paths.figures / "onezone_regions", dpi=300)
    plt.close()
    
def plot_region(fig, radius, dr=2., eta=equilibrium_mass_loading(), 
                output_dir=paths.data/"onezone"/"regions", 
                color=None, **kwargs):
    axs = setup_axes(fig, title="", **kwargs)
    # Plot underlying APOGEE contours
    apogee_data = APOGEESample.load()
    apogee_solar = apogee_data.region(galr_lim=(radius - dr/2, radius + dr/2), 
                                      absz_lim=(0, 2))
    pcm = axs[0].hexbin(apogee_solar("FE_H"), apogee_solar("O_FE"),
                  gridsize=50, bins="log",
                  extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
                  cmap="Greys", linewidths=0.2)
    cax = axs[0].inset_axes([0.05, 0.05, 0.05, 0.8])
    fig.colorbar(pcm, cax=cax, orientation="vertical")
    
    # APOGEE abundance distributions
    feh_df, bin_edges = apogee_solar.mdf(col="FE_H", range=XLIM, 
                                         smoothing=0.2)
    axs[1].plot(get_bin_centers(bin_edges), feh_df / max(feh_df), "k-")
    ofe_df, bin_edges = apogee_solar.mdf(col="O_FE", range=YLIM, 
                                         smoothing=0.05)
    axs[2].plot(ofe_df / max(ofe_df), get_bin_centers(bin_edges), "k-")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    name = str(output_dir / f"radius{radius:02d}")
    
    simtime = np.arange(0, 13.21, 0.01)
    
    area = np.pi * ((radius + ZONE_WIDTH)**2 - radius**2)
    
    sz = vice.singlezone(
        name = name,
        func = twoinfall(
            radius, 
            first_timescale=1., 
            second_timescale=10., 
            onset=ONSET),
        mode = "ifr",
        **ONEZONE_DEFAULTS
    )
    sz.eta = eta(radius)
    sz.tau_star = twoinfall_sf_law(area, onset=ONSET)
    print(sz.eta)
    print(sz.func.second.timescale)
    sz.run(simtime, overwrite=True)
    
    plot_vice_onezone(name, fig=fig, axs=axs, markers=[], color=color)
    # Weight by SFR
    hist = vice.history(name)
    axs[0].scatter(hist["[fe/h]"][::10], hist["[o/fe]"][::10], 
                   s=[20*h for h in hist["sfr"][::10]],
                   c=color)
    # Mark every Gyr
    axs[0].scatter(hist["[fe/h]"][::100], hist["[o/fe]"][::100], 
                   s=[5*h for h in hist["sfr"][::100]], c="w", zorder=10)
    
    # Label axes
    axs[0].text(0.95, 0.95, r"$R_{\rm gal}=%s$ kpc" % radius,
                va="top", ha="right", transform=axs[0].transAxes)
    
    return axs


if __name__ == "__main__":
    main()