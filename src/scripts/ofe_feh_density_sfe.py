"""
This script plots 2D density histograms of multi-zone models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from _globals import ONE_COLUMN_WIDTH, ZONE_WIDTH
import paths
from colormaps import paultol

OUTPUT_NAMES = [
    'sfe_tests/tmol/diskmodel',
    'sfe_tests/tstep/diskmodel',
    'sfe_tests/fiducial/diskmodel'
]
LABELS = [
    r'$\propto\tau_{\rm mol}$',
    'Step-function',
    'Both'
]
FEH_LIM = (-1.2, 0.6)
OFE_LIM = (-0.15, 0.55)
GALR_LIM = (7, 9)
ABSZ_LIM = (0.5, 1)
GRIDSIZE = 50


def main(style='paper', cmap='Blues'):
    # Setup figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(len(OUTPUT_NAMES), 1,
                            figsize=(0.67 * ONE_COLUMN_WIDTH, 2 * ONE_COLUMN_WIDTH),
                            sharex=True, sharey=True)
    plt.subplots_adjust(right=0.98, left=0.06, bottom=0.08, top=0.95,
                        wspace=0., hspace=0.)
    
    apogee_sample = APOGEESample.load()
    for i, ax in enumerate(axs):
        mzs = MultizoneStars.from_output(OUTPUT_NAMES[i])
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        ax.set_title(LABELS[i], y=0.85)
        subset = mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
        pcm = ax.hexbin(
            subset('[fe/h]'), subset('[o/fe]'),
            C=100 * subset('mstar') / subset('mstar').sum(),
            reduce_C_function=np.sum, vmax=1.3,
            gridsize=GRIDSIZE, cmap=cmap, linewidths=0.1,
            extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]],
            marginals=True
        )
        cax = axs[i].inset_axes([1.05, 0.05, 0.05, 0.9])
        cbar = fig.colorbar(pcm, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('Percent of stellar mass', labelpad=4)
        # Gas abundance track, weighted by SFR
        galr_mean = (GALR_LIM[1] + GALR_LIM[0]) / 2.
        zone = int(galr_mean / ZONE_WIDTH)
        multioutput = vice.output(str(subset.fullpath))
        hist = multioutput.zones[f'zone{zone}'].history
        axs[i].plot(hist['[fe/h]'], hist['[o/fe]'], color='k', marker='none', linewidth=2)
        for tstart in np.arange(1, 15, 2):
            axs[i].plot(
                hist['[fe/h]'][100*tstart:min(100*(tstart+1),len(hist['[o/fe]'])-1)], 
                hist['[o/fe]'][100*tstart:min(100*(tstart+1),len(hist['[o/fe]'])-1)], 
                color='w', marker='none', linewidth=1
            )
    
    # Axes limits
    axs[0].set_xlim(FEH_LIM)
    axs[0].set_ylim(OFE_LIM)
    # Axis labels
    axs[-1].set_xlabel('[Fe/H]', labelpad=2)
    for i, ax in enumerate(axs):
        ax.set_ylabel('[O/Fe]', labelpad=2)
    # Set x-axis ticks
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))

    fullpath = paths.extra / 'sfe_tests' / 'ofe_feh_density.png'
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath)
    plt.close()


if __name__ == '__main__':
    main()
