"""
This script compares the abundance evolution of multi-zone models with 
different yield sets and outflow settings.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from abundance_evolution_yields import compare_abundance_evolution
from _globals import TWO_COLUMN_WIDTH
import paths

OUTPUT_NAMES = [
    'yZ1/fiducial/diskmodel',
    'yZ1/migration_strength/strength50/diskmodel',
    'yZ1/pre_enrichment/mh05_alpha00/diskmodel',
    'yZ1/thick_thin_ratio/solar050/diskmodel'
]
LABELS = [
    '(a)\nFiducial',
    '(b)\n' + r'$\sigma_{\rm RM8}=5.0$ kpc',
    '(c)\n' + r'${\rm [X/H]}_{\rm CGM}=-0.5$',
    '(d)\n' + r'$f_\Sigma(R_\odot)=0.5$'
]
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)


def main(verbose=False, uncertainties=True, style='paper', cmap='winter_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = compare_abundance_evolution(
        OUTPUT_NAMES, 
        LABELS,
        (TWO_COLUMN_WIDTH, 0.7 * TWO_COLUMN_WIDTH),
        verbose=verbose,
        uncertainties=uncertainties,
        cmap=cmap,
        galr_lim=GALR_LIM,
        absz_lim=ABSZ_LIM
    )
    fig.suptitle(r'$y/Z_\odot=1$')
    fig.savefig(paths.figures / 'abundance_evolution_params')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='abundance_evolution_params.py',
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
        default='paper',
        help='Plot style to use (default: paper).'
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
