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
    'sfe_tests/tmol/diskmodel',
    'sfe_tests/tstep/diskmodel',
    'sfe_tests/tstep4/diskmodel',
    'sfe_tests/fiducial/diskmodel'
]
LABELS = [
    r'$\propto\tau_{\rm mol}$',
    'Step-function (x2)',
    'Step-function (x4)',
    'Both (x2)'
]
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)


def main(verbose=False, uncertainties=True, style='paper', cmap='winter_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = compare_abundance_evolution(
        OUTPUT_NAMES, 
        LABELS,
        (TWO_COLUMN_WIDTH, 0.6 * TWO_COLUMN_WIDTH),
        verbose=verbose,
        uncertainties=uncertainties,
        cmap=cmap,
        galr_lim=GALR_LIM,
        absz_lim=ABSZ_LIM
    )
    fullpath = paths.extra / 'sfe_tests' / 'abundance_evolution.png'
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath)
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
