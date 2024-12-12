"""
This script plots the distributions of migration distance as a function of
stellar population age and formation radius.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable
from multizone_stars import MultizoneStars
from utils import box_smooth, get_color_list, discrete_colormap
import paths
from _globals import TWO_COLUMN_WIDTH

MIGRATION_SCHEMES = ['gaussian', 'gaussian_strong']
ROW_LABELS = ['Fiducial Gaussian', r'Gaussian ($\sigma_{\rm RM8} = 3.6$ kpc)']#r'Gaussian ($\tau^{0.5}$)']
RFORM_BINS = [3, 5, 7, 9, 11, 13]
AGE_BINS = [0, 2, 4, 6, 8, 10, 12]
CMAP_NAME = 'jet'

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    rfinal_bins = np.arange(0, 20.01, 0.01)
    
    # pick discrete colors
    cmap, norm = discrete_colormap(CMAP_NAME, AGE_BINS)
    colors = get_color_list(cmap, AGE_BINS)
    
    fig, axs = plt.subplots(len(MIGRATION_SCHEMES), len(RFORM_BINS)-1, 
                            figsize=(TWO_COLUMN_WIDTH, 3), 
                            sharex=True, sharey=True)
    plt.subplots_adjust(left=0.06, right=0.93, top=0.92, wspace=0., hspace=0.)
    
    for m, row in enumerate(axs):
        mig = MIGRATION_SCHEMES[m]
        output_name = '/'.join((mig, 'outflow/no_gasflow/pristine/J21/insideout/diskmodel'))
        mzs = MultizoneStars.from_output(output_name)
        row[2].text(0.5, 0.85, ROW_LABELS[m], 
                    ha='center', transform=row[2].transAxes,
                    bbox={
                        'facecolor': 'w',
                        'edgecolor': 'none',
                        'boxstyle': 'round',
                        'pad': 0.15,
                        'alpha': 1.,
                    })
        for i, ax in enumerate(row):
            rform_lim = tuple(RFORM_BINS[i:i+2])
            ax.vlines(rform_lim, ymin=0, ymax=0.6, linestyle=':', color='k')
            if m == 0:
                ax.set_title(r'$%d\leq R_{\rm{form}} < %d$ kpc' % rform_lim)
            for j, color in enumerate(colors):
                age_lim = tuple(AGE_BINS[j:j+2])
                # bin data by formation radius and age
                subset = mzs.filter({'age': age_lim, 'galr_origin': rform_lim})
                # for analogue migration, limit to bins with >100 stars
                nodups = subset.stars.copy().drop_duplicates('analog_id')
                if nodups.shape[0] < 100 and mig == 'diffusion':
                    continue
                hist, _ = np.histogram(subset('galr_final'), rfinal_bins, 
                                       density=True)
                # apply boxcar smoothing with a width of 0.5 kpc
                hist_smooth = box_smooth(hist, rfinal_bins, 0.5)
                bin_centers = (rfinal_bins[:-1] + rfinal_bins[1:]) / 2
                ax.plot(bin_centers, hist_smooth, c=color, ls='-', 
                        label=r'$%d - %d$' % age_lim, zorder=10-j)
    
    axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.02))
    axs[0,0].set_xlim((-2, 22))
    axs[0,0].set_ylim((0, 0.46))
    for ax in axs[-1]:
        ax.set_xlabel(r'$R_{\rm{final}}$ [kpc]')
    for ax in axs[:,0]:
        ax.set_ylabel('PDF')
    # axs[0,0].legend(loc='upper right', frameon=False, 
    #                 handlelength=1, handletextpad=0.5, title='Age [Gyr]')
    
    # Add colorbar on right side
    height = fig.subplotpars.top - fig.subplotpars.bottom
    cax = plt.axes([0.94, fig.subplotpars.bottom, 0.01, height])
    cbar = fig.colorbar(ScalarMappable(norm, cmap), cax)
    cbar.set_label('Age [Gyr]')
    
    plt.savefig(paths.figures / 'radial_migration', dpi=300)

if __name__ == '__main__':
    main()
