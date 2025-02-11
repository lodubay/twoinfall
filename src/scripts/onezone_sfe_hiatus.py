"""
This script plots a one-zone VICE model with a short hiatus in the SFE.
"""

import numpy as np
import matplotlib.pyplot as plt

import vice

from multizone.src.yields import yZ2
vice.yields.sneia.settings['fe'] *= 10**-0.1
from multizone.src.models.utils import exponential
from multizone.src import outflows
from track_and_mdf import setup_figure, plot_vice_onezone
from apogee_sample import APOGEESample
import paths
from utils import get_bin_centers
from _globals import ONEZONE_DEFAULTS
from colormaps import paultol

RADIUS = 8.
ZONE_WIDTH = 2.
HIATUS_ONSET = 1.4
HIATUS_DURATION = 0.2
TAU_STAR_BASE = 2.0
TAU_STAR_ENHANCEMENT = 10
FEH_LIM = (-1.6, 0.6)
OFE_LIM = (-0.08, 0.48)

def main(verbose=True):
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    fig, axs = setup_figure(xlim=FEH_LIM, ylim=OFE_LIM)

    # Plot underlying APOGEE contours
    apogee_data = APOGEESample.load()
    apogee_solar = apogee_data.region(galr_lim=(7, 9), absz_lim=(0, 2))
    # apogee_solar.plot_kde2D_contours(axs[0], 'FE_H', 'O_FE', c='k', lw=1,
    #                                  plot_kwargs={'zorder': 1})
    pcm = axs[0].hexbin(apogee_solar('FE_H'), apogee_solar('O_FE'),
                  gridsize=50, #bins='log',
                  extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]],
                  cmap='binary', linewidths=0.2)
    cax = axs[0].inset_axes([0.05, 0.05, 0.05, 0.8])
    fig.colorbar(pcm, cax=cax, orientation='vertical', label='# APOGEE Stars')
    
    # APOGEE abundance distributions
    feh_df, bin_edges = apogee_solar.mdf(col='FE_H', range=FEH_LIM, 
                                         smoothing=0.2)
    axs[1].plot(get_bin_centers(bin_edges), feh_df / max(feh_df), 
                color='gray', linestyle='-', marker=None)
    ofe_df, bin_edges = apogee_solar.mdf(col='O_FE', range=OFE_LIM, 
                                         smoothing=0.05)
    axs[2].plot(ofe_df / max(ofe_df), get_bin_centers(bin_edges), 
                color='gray', linestyle='-', marker=None)
    
    # Set up output directory
    output_dir = paths.data / 'onezone' / 'sfe_hiatus'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # One-zone model parameters
    simtime = np.arange(0, 13.21, 0.01)
    eta = outflows.equilibrium()
    if verbose:
        print('Eta = ', eta(RADIUS))

    # Reference: constant tau_star
    name = str(output_dir / 'constant')
    ifr = exponential(norm=1, timescale=15)
    sz = vice.singlezone(
        name = name,
        func = ifr,
        mode = 'ifr',
        **ONEZONE_DEFAULTS
    )
    sz.tau_star = TAU_STAR_BASE
    sz.eta = eta(RADIUS)
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, fig=fig, axs=axs, label='Fiducial', 
                      linestyle='--', marker_labels=True)

    # SFE hiatus model
    name = str(output_dir / 'onset13_width02')
    tau_star = tau_star_burst(onset=HIATUS_ONSET, duration=HIATUS_DURATION)
    sz = vice.singlezone(
        name = name,
        func = ifr,
        mode = 'ifr',
        **ONEZONE_DEFAULTS
    )
    sz.tau_star = tau_star
    sz.eta = eta(RADIUS)
    sz.run(simtime, overwrite=True)
    plot_vice_onezone(name, fig=fig, axs=axs, label='SFE Hiatus')
    axs[0].legend(loc='upper right')
    
    plt.savefig(paths.figures / 'onezone_sfe_hiatus')
    plt.close()


class tau_star_burst:
    def __init__(self, base=TAU_STAR_BASE, onset=HIATUS_ONSET, 
                 duration=HIATUS_DURATION, enhancement=TAU_STAR_ENHANCEMENT):
        self.base = base
        self.onset = onset
        self.duration = duration
        self.enhancement = enhancement

    def __call__(self, time):
        if time < self.onset:
            return self.base #* PRE_BURST_ENHANCEMENT
        elif time < self.onset + self.duration:
            return self.base * self.enhancement
        else:
            return self.base
    

if __name__ == '__main__':
    main()
