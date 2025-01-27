"""
This script plots 2D density histograms of multi-zone models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from scatter_plot_grid import setup_axes, setup_colorbar, plot_gas_abundance
from _globals import TWO_COLUMN_WIDTH
import paths

OUTPUT_NAMES = [
    'yields/yZ1/diskmodel',
    'yields/yZ2/diskmodel'
]
LABELS = [
    r'(a) $y/Z_\odot=1$',
    r'(b) $y/Z_\odot=2$',
    '(c) APOGEE'
]
FEH_LIM = (-1.3, 0.7)
OFE_LIM = (-0.15, 0.55)
GALR_BINS = [3, 5, 7, 9, 11, 13]
ABSZ_LIM = (0.5, 1)
GRIDSIZE = 50


def main(style='paper', cmap='Greys'):
    # Setup figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(len(OUTPUT_NAMES)+1, len(GALR_BINS)-1,
                            figsize=(TWO_COLUMN_WIDTH, 0.75 * TWO_COLUMN_WIDTH),
                            sharex=True, sharey=True)
    plt.subplots_adjust(right=0.98, left=0.06, bottom=0.08, top=0.95,
                        wspace=0., hspace=0.2)
    
    apogee_sample = APOGEESample.load()
    apogee_index = len(OUTPUT_NAMES)
    for i, output_name in enumerate(OUTPUT_NAMES):
        mzs = MultizoneStars.from_output(output_name)
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        axs[i,2].set_title(LABELS[i])
        for j, ax in enumerate(axs[i,:]):
            galr_lim = (GALR_BINS[j], GALR_BINS[j+1])
            subset = mzs.region(galr_lim=galr_lim, absz_lim=ABSZ_LIM)
            pcm = axs[i,j].hexbin(subset('[fe/h]'), subset('[o/fe]'),
                                  C=subset('mstar') / subset('mstar').sum(),
                                  reduce_C_function=np.sum, #bins='log',
                                  gridsize=GRIDSIZE, cmap=cmap, linewidths=0.1,
                                  extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]])

    axs[apogee_index,2].set_title(LABELS[apogee_index])
    for j, ax in enumerate(axs[apogee_index]):
        galr_lim = (GALR_BINS[j], GALR_BINS[j+1])
        subset = apogee_sample.region(galr_lim=galr_lim, absz_lim=ABSZ_LIM)
        pcm = axs[apogee_index,j].hexbin(subset('FE_H'), subset('O_FE'),
                              C=np.ones(subset.nstars) / subset.nstars,
                              reduce_C_function=np.sum, #bins='log',
                              gridsize=GRIDSIZE, cmap=cmap, linewidths=0.2,
                              extent=[FEH_LIM[0], FEH_LIM[1], OFE_LIM[0], OFE_LIM[1]])
    
    # Axes limits
    axs[0,0].set_xlim(FEH_LIM)
    axs[0,0].set_ylim(OFE_LIM)
    # Axis labels
    for ax in axs[-1]:
        ax.set_xlabel('[Fe/H]', labelpad=2)
    for i, ax in enumerate(axs[:,0]):
        ax.set_ylabel('[O/Fe]', labelpad=2)
    # Label bins in Rgal
    for i, ax in enumerate(axs[0]):
        galr_lim = (GALR_BINS[i], GALR_BINS[i+1])
        ax.text(0.5, 0.9, r'$%s\leq R_{\rm{Gal}} < %s$ kpc' % galr_lim,
                transform=ax.transAxes, ha='center', va='top')
    # Set x-axis ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.savefig(paths.figures / 'ofe_feh_density')
    plt.close()


if __name__ == '__main__':
    main()
