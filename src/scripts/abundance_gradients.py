"""
This script compares radial abundance gradients between VICE multi-zone
model outputs and APOGEE data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from multizone_stars import MultizoneStars
from utils import radial_gradient, weighted_quantile, get_bin_centers
from colormaps import paultol
from _globals import ONE_COLUMN_WIDTH, GALR_BINS, MAX_SF_RADIUS, ZONE_WIDTH
import paths

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.vibrant.colors)
    fig, axs = plt.subplots(3, 1, figsize=(ONE_COLUMN_WIDTH, 2 * ONE_COLUMN_WIDTH),
                            tight_layout=True, sharex=True)
    
    output_names = [
        'gaussian/outflow/no_gasflow/pristine/J21/twoinfall/diskmodel',
        'gaussian/outflow/no_gasflow/pristine/W24/twoinfall/diskmodel'
    ]
    labels = ['J21', 'W24mod']
    colors = [paultol.vibrant.colors[0], paultol.vibrant.colors[1]]
    
    for i, output_name in enumerate(output_names):
        # Gas
        mout = vice.output(str(paths.multizone / output_name))
        xarr = np.arange(0, MAX_SF_RADIUS, ZONE_WIDTH)
        axs[0].plot(xarr, radial_gradient(mout, '[o/h]'), 
                    color=colors[i], linestyle='-', 
                    label='%s gas (present-day)' % labels[i])
        axs[1].plot(xarr, radial_gradient(mout, '[fe/h]'), 
                    color=colors[i], linestyle='-')
        axs[2].plot(xarr, radial_gradient(mout, '[o/fe]'), 
                    color=colors[i], linestyle='-')
        # Stars
        mzs = MultizoneStars.from_output(output_name)
        median_abundances = np.zeros((3, len(GALR_BINS)-1))
        for j in range(len(GALR_BINS)-1):
            galr_lim = GALR_BINS[j:j+2]
            subset = mzs.filter({'galr_final': tuple(galr_lim), 
                                'zfinal': (0, 0.5),
                                'age': (0, 0.1)})
            median_abundances[:,j] = [
                weighted_quantile(subset.stars, '[o/h]', 'mstar', quantile=0.5),
                weighted_quantile(subset.stars, '[fe/h]', 'mstar', quantile=0.5),
                weighted_quantile(subset.stars, '[o/fe]', 'mstar', quantile=0.5),
            ]
        axs[0].plot(get_bin_centers(GALR_BINS), median_abundances[0], 
                    color=colors[i], marker='o', linestyle='none', 
                    label='Stars (<100 Myr old)')
        axs[1].plot(get_bin_centers(GALR_BINS), median_abundances[1], 
                    color=colors[i], marker='o', linestyle='none')
        axs[2].plot(get_bin_centers(GALR_BINS), median_abundances[2], 
                    color=colors[i], marker='o', linestyle='none')
    # Reference gradient and sun
    axs[0].plot(xarr, -0.08 * (xarr - 8.0), 'k--', label='Reference (-0.08 dex/kpc)')
    axs[0].scatter([8], [0], marker='+', color='k')
    axs[1].scatter([8], [0], marker='+', color='k')
    axs[2].scatter([8], [0], marker='+', color='k')
    # Configure axes
    axs[0].set_xlim((-1, 17))
    axs[0].set_ylim((-0.7, 0.7))
    axs[0].xaxis.set_major_locator(MultipleLocator(4))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].set_ylabel('[O/H]')
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].set_ylim((-0.7, 0.7))
    axs[1].set_ylabel('[Fe/H]')
    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[2].set_ylim((-0.12, 0.12))
    axs[2].set_ylabel('[O/Fe]')
    axs[2].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.02))
    axs[2].set_xlabel('Radius [kpc]')    
    axs[0].legend()

    # Save
    plt.savefig(paths.figures / 'abundance_gradients', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
