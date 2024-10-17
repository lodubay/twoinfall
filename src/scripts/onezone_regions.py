"""
This script plots the outputs of one-zone models over the APOGEE data at 
three regions of the galaxy.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vice
import paths
from multizone.src.yields import W23
# from vice.yields.presets import JW20
# from multizone.src.yields import J21
# from multizone.src.yields import F04
from multizone.src import models, dtds
from multizone.src.models.gradient import gradient
from apogee_sample import APOGEESample
from _globals import END_TIME, ONEZONE_DEFAULTS, TWO_COLUMN_WIDTH, ZONE_WIDTH
from colormaps import paultol
from track_and_mdf import setup_axes, plot_vice_onezone
from utils import get_bin_centers, twoinfall_gradient

ONSET = 4.2
XLIM = (-1.4, 0.6)
YLIM = (-0.12, 0.48)

def main():
    plt.style.use(paths.styles / "paper.mplstyle")
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.36*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(7, 22, wspace=0.)
    subfigs = [fig.add_subfigure(gs[:,i:i+w]) for i, w in zip((0, 8, 15), (8, 7, 7))]
    # Outflow mass-loading factor
    eta_func = models.equilibrium_mass_loading(equilibrium=0.2, tau_sfh=15., tau_star=2.)
    # eta_func = vice.milkyway.default_mass_loading
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
    
def plot_region(fig, radius, dr=2., eta=models.equilibrium_mass_loading(), 
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
    axs[1].plot(get_bin_centers(bin_edges), feh_df / max(feh_df), 
                color="gray", linestyle="-", marker=None)
    ofe_df, bin_edges = apogee_solar.mdf(col="O_FE", range=YLIM, 
                                         smoothing=0.05)
    axs[2].plot(ofe_df / max(ofe_df), get_bin_centers(bin_edges), 
                color="gray", linestyle="-", marker=None)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    name = str(output_dir / f"radius{radius:02d}")
    
    simtime = np.arange(0, 13.21, 0.01)
    
    area = np.pi * ((radius + ZONE_WIDTH)**2 - radius**2)
    
    sz = vice.singlezone(
        name = name,
        # func = twoinfall_gradient(
        #     radius, 
        #     first_timescale=1., 
        #     second_timescale=models.insideout.timescale(radius), 
        #     onset=ONSET,
        #     mass_loading=eta),
        func = insideout_gradient(radius),
        mode = "ifr",
        **ONEZONE_DEFAULTS
    )
    sz.eta = eta(radius)
    # sz.tau_star = models.twoinfall_sf_law(area, onset=ONSET)
    sz.tau_star = models.fiducial_sf_law(area)
    sz.run(simtime, overwrite=True)
    
    plot_vice_onezone(name, fig=fig, axs=axs, markers=[], color=color)
    # Weight by SFR
    hist = vice.history(name)
    axs[0].scatter(hist["[fe/h]"][::10], hist["[o/fe]"][::10], 
                   s=[10*h/max(hist["sfr"]) for h in hist["sfr"][::10]],
                   c=color)
    # Mark every Gyr
    axs[0].scatter(hist["[fe/h]"][::100], hist["[o/fe]"][::100], 
                   s=[2*h/max(hist["sfr"]) for h in hist["sfr"][::100]], 
                   c="w", zorder=10)
    
    # Label axes
    axs[0].text(0.95, 0.95, r"$R_{\rm gal}=%s$ kpc" % radius,
                va="top", ha="right", transform=axs[0].transAxes)
    # axs[0].text(0.95, 0.85, r"$\tau_2=%s$ Gyr" % round(sz.func.second.timescale, 1),
    #             va="top", ha="right", transform=axs[0].transAxes)
    
    return axs

class insideout_gradient(models.insideout):
    def __init__(self, radius, dr=0.1, **kwargs):
        super().__init__(radius, dr=dr, **kwargs)
        area = np.pi * ((radius + dr/2.)**2 - (radius - dr/2.)**2)
        self.norm *= gradient(radius) * area


if __name__ == "__main__":
    main()
