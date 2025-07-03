"""
This script plots 2D density histograms of multi-zone models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from apogee_sample import APOGEESample
from _globals import ONE_COLUMN_WIDTH, ZONE_WIDTH
import paths
from colormaps import paultol
from utils import twoinfall_onezone, insideout_onezone
from multizone.src.yields import yZ1
from multizone.src import models, outflows
from _globals import ONEZONE_DEFAULTS

FEH_LIM = (-1.1, 1.1)
OFE_LIM = (-0.15, 0.55)
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 2)
GRIDSIZE = 30
ZONE_WIDTH = 2


def main(style='paper'):
    # Setup figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.highcontrast.colors)
    fig, ax = plt.subplots(
        sharex=True, sharey=True, 
        figsize=(ONE_COLUMN_WIDTH, 0.6 * ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0.},
        tight_layout=True
    )
    ax.set_xlim(FEH_LIM)
    ax.set_ylim(OFE_LIM)
    
    # Plot APOGEE data
    apogee_sample = APOGEESample.load()
    apogee_solar = apogee_sample.region(galr_lim=(7, 9), absz_lim=(0, 2))
    cmap_name = 'binary'
    pcm = ax.hexbin(
        apogee_solar('FE_H'), apogee_solar('O_FE'),
        gridsize=GRIDSIZE,
        extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]],
        cmap=cmap_name, linewidths=0.2, 
    )
    # fig.subplots_adjust(right=0.75)
    # cax = fig.add_axes([0.75, 0.11, 0.05, 0.77])
    fig.colorbar(pcm, ax=ax, orientation='vertical', label='# APOGEE Stars')

    # Set up onezone model params
    output_dir = paths.data / 'onezone' / 'smooth_vs_twoinfall'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    simtime = np.arange(0, 13.21, 0.01)

    # Plot smooth SFH predictions
    vice.yields.sneia.settings['fe'] *= 10**-0.1
    eta_func = outflows.yZ1
    for radius in [4, 8, 12]:
        name = str(output_dir / f'smooth_radius{radius:02d}')
        area = np.pi * ((radius + ZONE_WIDTH/2)**2 - (radius - ZONE_WIDTH/2)**2)
        sfr = insideout_onezone(
            radius,
            dr=ZONE_WIDTH
        )
        sz = vice.singlezone(
            name = name,
            func = sfr,
            mode = 'sfr',
            **ONEZONE_DEFAULTS
        )
        sz.eta = eta_func(radius)
        sz.tau_star = models.fiducial_sf_law(area)
        sz.run(simtime, overwrite=True)
        hist = vice.history(name)
        ax.plot(hist['[fe/h]'], hist['[o/fe]'], color='w', linewidth=2)
        ax.plot(hist['[fe/h]'], hist['[o/fe]'], linewidth=1, linestyle='--',
                    label=r'%s kpc' % radius)

    # Plot twoinfall predictions
    vice.yields.sneia.settings['fe'] *= 10**0.1
    eta_func = outflows.yZ1
    radius = 8
    name = str(output_dir / f'twoinfall_radius{radius:02d}')
    area = np.pi * ((radius + ZONE_WIDTH/2)**2 - (radius - ZONE_WIDTH/2)**2)
    ifr = twoinfall_onezone(
        radius, 
        # first_timescale=.1,
        second_timescale=models.twoinfall_expvar.timescale(radius), 
        # onset=3.,
        mass_loading=eta_func,
        dr=ZONE_WIDTH
    )
    sz = vice.singlezone(
        name = name,
        func = ifr,
        mode = 'ifr',
        **ONEZONE_DEFAULTS
    )
    sz.eta = eta_func(radius)
    sz.tau_star = models.twoinfall_sf_law(area, onset=ifr.onset)
    sz.run(simtime, overwrite=True)
    hist = vice.history(name)
    ax.plot(hist['[fe/h]'], hist['[o/fe]'], color='w', linewidth=2)
    ax.plot(hist['[fe/h]'], hist['[o/fe]'], linewidth=1, color='k',
            label=r'%s kpc' % radius)

    handles, labels = ax.get_legend_handles_labels()
    legend1 = plt.legend(handles[:-1], labels[:-1], title='Smooth SFH',
                         handletextpad=0.5)
    legend2 = plt.legend(handles[-1:], labels[-1:], title='Two-Infall', 
                         loc='upper right', bbox_to_anchor=(0.97, 0.53),
                         handletextpad=0.5)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('[O/Fe]')

    plt.savefig(paths.figures / 'smooth_vs_twoinfall')
    plt.close()


if __name__ == '__main__':
    main()
