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
from multizone.src.yields import yZ2

FEH_LIM = (-1.1, 0.9)
OFE_LIM = (-0.15, 0.55)
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 2)
GRIDSIZE = (30, 12)


def main(style='paper'):
    # Setup figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.highcontrast.colors[::-1])
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
    fig.colorbar(pcm, ax=ax, orientation='vertical', label='# APOGEE Stars', 
                 pad=0.02, aspect=15)

    # Plot smooth SFH predictions
    output_name = 'yZ2-insideout/diskmodel'
    multioutput = vice.output(str(paths.multizone / output_name))
    for radius in [4, 8, 12]:
        zone = int(radius / ZONE_WIDTH)
        hist = multioutput.zones[f'zone{zone}'].history
        ax.plot(hist['[fe/h]'], hist['[o/fe]'], color='w', linewidth=2)
        ax.plot(hist['[fe/h]'], hist['[o/fe]'], linewidth=1, linestyle='--',
                    label=r'%s kpc' % radius)

    # Plot twoinfall predictions
    output_name = 'yZ2-earlyonset/diskmodel'
    radius = 8
    zone = int(radius / ZONE_WIDTH)
    multioutput = vice.output(str(paths.multizone / output_name))
    hist = multioutput.zones[f'zone{zone}'].history
    ax.plot(hist['[fe/h]'], hist['[o/fe]'], color='w', linewidth=2)
    ax.plot(hist['[fe/h]'], hist['[o/fe]'], linewidth=1, color='k',
            label=r'%s kpc' % radius)

    handles, labels = ax.get_legend_handles_labels()
    legend1 = plt.legend(handles[:-1], labels[:-1], title='Smooth SFH',
                         handletextpad=0.5)
    legend2 = plt.legend(handles[-1:], labels[-1:], title='Two-Infall', 
                         loc='upper right', bbox_to_anchor=(0.68, 1.),
                         handletextpad=0.5)
    ax.add_artist(legend1)
    # ax.add_artist(legend2)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('[O/Fe]')

    plt.savefig(paths.figures / 'smooth_vs_twoinfall')
    plt.close()


if __name__ == '__main__':
    main()
