"""
Similar to stellar_abundance_evolution.py but for the y/Zsun=2 models.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from stellar_abundance_evolution import compare_abundance_evolution
from _globals import TWO_COLUMN_WIDTH
import paths

OUTPUT_NAMES = [
    'yZ2-fiducial/diskmodel',
    'yZ2-earlyonset/diskmodel',
    'yZ2-diskratio/diskmodel',
    'yZ2-preenrich/diskmodel',
]
LABELS = [
    '(a)\nFiducial',
    '(b)\n' + r'$t_{\rm max}=2.2$ Gyr',
    '(c)\n' + r'$f_\Sigma(R_\odot)=0.5$',
    '(d)\n' + r'${\rm [X/H]}_{\rm CGM}=-0.5$',
]
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)
CMAP = 'viridis_r'


def main(verbose=False, uncertainties=True, style='paper', cmap=CMAP, ages='L23'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    figsize = (TWO_COLUMN_WIDTH, 0.8 * TWO_COLUMN_WIDTH)
    titley = 0.98
    fig, axs = compare_abundance_evolution(
        OUTPUT_NAMES, 
        LABELS,
        figsize,
        verbose=verbose,
        uncertainties=uncertainties,
        cmap=cmap,
        galr_lim=GALR_LIM,
        absz_lim=ABSZ_LIM,
        age_col='%s_AGE' % ages
    )
    fig.suptitle(r'$y/Z_\odot=2$', y=titley)
    fig.savefig(paths.extra / 'stellar_abundance_evolution_yZ2.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='stellar_abundance_evolution.py',
        description='Compare stellar age-abundance relations between multi-zone \
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
        default='paper',
        help='Plot style to use (default: paper).'
    )
    parser.add_argument(
        '--cmap', 
        metavar='COLORMAP', 
        type=str,
        default=CMAP,
        help='Name of colormap for color-coding model output (default: %s).' % CMAP
    )
    parser.add_argument(
        '--ages', 
        metavar='AGE-SOURCE', 
        type=str,
        default='L23',
        choices=['L23', 'CN'],
        help='Type of age estimate to use (default: L23).'
    )
    args = parser.parse_args()
    main(**vars(args))
