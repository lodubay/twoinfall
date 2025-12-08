"""
This script plots a comparison between the age catalogs of Leung et al. (2023)
(i.e., the NN ages), and the [C/N] ages of Roberts et al. (2025).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize

from utils import get_bin_centers
from apogee_sample import APOGEESample
import paths
from _globals import TWO_COLUMN_WIDTH

FEH_LIM = (-0.8, 0.6)
AGE_LIM = (0, 14)
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)

def main(style='paper'):
    # Import apogee sample
    full_sample = APOGEESample.load()
    local_sample = full_sample.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)

    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.36*TWO_COLUMN_WIDTH))
    gs = fig.add_gridspec(2, 2, hspace=0., wspace=0.25)
    ax00 = fig.add_subplot(gs[:,0])
    ax01 = fig.add_subplot(gs[0,1])
    ax01.tick_params(axis='x', labelbottom=False)
    ax11 = fig.add_subplot(gs[1,1], sharex=ax01, sharey=ax01)

    # Big panel: compare NN and [C/N] ages
    all_shared_stars = full_sample.filter({'L23_AGE': (0, 15), 'CN_AGE': (0, 15)})
    pcm00 = ax00.hexbin(
        all_shared_stars('L23_AGE'), all_shared_stars('CN_AGE'),
        gridsize=50, cmap='Spectral_r', linewidths=0.2,
        extent=[AGE_LIM[0], AGE_LIM[1], AGE_LIM[0], AGE_LIM[1]]
    )
    fig.colorbar(pcm00, ax=ax00, label='Number of stars')
    # one-to-one line
    ax00.plot(AGE_LIM, AGE_LIM, 'k--')
    # Rolling median
    sorted_ages = all_shared_stars.data.sort_values('L23_AGE')[['L23_AGE', 'CN_AGE']]
    rolling_medians = sorted_ages.rolling(1000, min_periods=100, step=100).median()
    ax00.plot(rolling_medians['L23_AGE'], rolling_medians['CN_AGE'], 'k-')
    # Binned median uncertainty
    big_age_bins = np.linspace(AGE_LIM[0], AGE_LIM[1], 4)
    median_age_errors = all_shared_stars.binned_intervals(
        'L23_AGE_ERR', 'L23_AGE', big_age_bins, quantiles=[0.5]
    )
    ax00.errorbar(
        get_bin_centers(big_age_bins), 12 * np.ones(median_age_errors[0.5].shape), 
        xerr=median_age_errors[0.5], yerr=all_shared_stars('CN_AGE_ERR').median(),
        c='w', marker='.', ms=0, linestyle='none', elinewidth=0.5, capsize=0,
    )
    ax00.set_xlabel('NN age [Gyr]')
    ax00.set_ylabel('[C/N] age [Gyr]')

    # Small panels: age-metallilcity relations
    local_shared_stars = local_sample.filter({'L23_AGE': (0, 15), 'CN_AGE': (0, 15)})
    axs = [ax01, ax11]
    labels = ['NN ages', '[C/N] ages']
    norm = Normalize(vmax=100)
    for j, age_col in enumerate(['L23_AGE', 'CN_AGE']):
        pcm = axs[j].hexbin(
            local_shared_stars(age_col), local_shared_stars('FE_H'),
            gridsize=(50, 15), cmap='Spectral_r', norm=norm, linewidths=0.2,
            extent=[AGE_LIM[0], AGE_LIM[1], FEH_LIM[0], FEH_LIM[1]]
        )
        # fig.colorbar(pcm, ax=axs[j], aspect=10, shrink=0.9)
        sorted_ages = local_shared_stars.data.sort_values(age_col)[[age_col, 'FE_H']]
        rolling_medians = sorted_ages.rolling(300, min_periods=100, step=100).median()
        axs[j].plot(rolling_medians[age_col], rolling_medians['FE_H'], 'k-')
        axs[j].set_ylabel('[Fe/H]')
        axs[j].set_title(labels[j], y=0.75, x=0.95, color='w', ha='right')
    ax11.set_xlabel('Age [Gyr]')
    fig.colorbar(pcm, ax=axs, label='Number of stars', extend='max')

    ax00.set_xlim(AGE_LIM)
    ax00.set_ylim(AGE_LIM)
    ax00.xaxis.set_major_locator(MultipleLocator(5))
    ax00.xaxis.set_minor_locator(MultipleLocator(1))
    ax00.yaxis.set_major_locator(MultipleLocator(5))
    ax00.yaxis.set_minor_locator(MultipleLocator(1))
    ax01.set_xlim(AGE_LIM)
    ax01.set_ylim(FEH_LIM)
    ax01.yaxis.set_major_locator(MultipleLocator(0.5))
    ax01.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax11.xaxis.set_major_locator(MultipleLocator(5))
    ax11.xaxis.set_minor_locator(MultipleLocator(1))
    ax11.yaxis.set_major_locator(MultipleLocator(0.5))
    ax11.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.savefig(paths.figures / 'compare_age_catalogs')
    plt.close()


if __name__ == '__main__':
    main()
