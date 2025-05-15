"""
This script compares the abundance evolution of multi-zone models with 
different yield sets and outflow settings.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from abundance_evolution_yields import compare_abundance_evolution
from _globals import ONE_COLUMN_WIDTH
import paths

OUTPUT_NAMES = [
    'yZ1/fiducial/diskmodel',
    'yZ1/pre_enrichment/mh05_alpha00_eta06/diskmodel',
    'yZ2/fiducial/diskmodel'
]
LABELS = [
    'Pristine infall',
    'Pre-enriched infall',
    'Higher yields\n' + r'$(y/Z_\odot=2)$',
]
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)


def main(verbose=False, uncertainties=True, style='poster', cmap='winter_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = compare_abundance_evolution(
        OUTPUT_NAMES, 
        LABELS,
        (1.5 * ONE_COLUMN_WIDTH, 1.6 * ONE_COLUMN_WIDTH),
        verbose=verbose,
        uncertainties=uncertainties,
        cmap=cmap,
        galr_lim=GALR_LIM,
        absz_lim=ABSZ_LIM,
        cbar_orientation='horizontal',
        age_col='CN_AGE'
    )
    axs[2,0].set_ylim((-0.1, 0.5))
    fig.suptitle(r'Low yields & outflows $(y/Z_\odot=1)$', 
                 x=0.38, y=0.945, fontsize=plt.rcParams['axes.labelsize'])
    fig.savefig(paths.extra / 'poster' / 'abundance_evolution_poster')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='abundance_evolution_poster.py',
        description='Compare age-abundance relations between multi-zone \
outputs with different parameters and APOGEE data.'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument(
        '--style', 
        metavar='STYLE', 
        type=str,
        default='poster',
        help='Plot style to use (default: poster).'
    )
    parser.add_argument(
        '--cmap', 
        metavar='COLORMAP', 
        type=str,
        default='winter_r',
        help='Name of colormap for color-coding model output (default: winter_r).'
    )
    args = parser.parse_args()
    main(**vars(args))
