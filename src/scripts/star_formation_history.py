"""
This script plots the infall rate, star formation rate, gas mass, and 
star formation efficiency timescale over time for one multi-zone model.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from utils import get_color_list, get_bin_centers
from _globals import ONE_COLUMN_WIDTH, ZONE_WIDTH, GALR_BINS
import paths

OUTPUT_NAME = 'yZ1-fiducial/diskmodel'
CMAP = 'viridis_r'

def main(output_name=OUTPUT_NAME, style='paper', cmap=CMAP, verbose=False):
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        4, 1, 
        sharex=True, 
        # tight_layout=True,
        figsize=(ONE_COLUMN_WIDTH, 2.5 * ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0.}
    )
    multioutput = vice.output(str(paths.multizone / output_name))
    axs[0].set_ylabel(r'$\log \dot \Sigma_{\rm in}$ [M$_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    # axs[0].set_yscale('log')
    axs[0].set_ylim((-7.5, 0.5))
    axs[0].yaxis.set_major_locator(MultipleLocator(2))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.5))
    axs[1].set_ylabel(r'$\log \dot \Sigma_\star$ [M$_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    # axs[1].set_yscale('log')
    axs[1].set_ylim((-5, -0.5))
    axs[1].yaxis.set_major_locator(MultipleLocator(1))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.25))
    axs[2].set_ylabel(r'$\log \Sigma_g$ [M$_{\odot}\,\rm{kpc}^{-2}$]')
    # axs[2].set_yscale('log')
    axs[2].set_ylim((5, 8.5))
    axs[2].yaxis.set_major_locator(MultipleLocator(1))
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.25))
    axs[3].set_ylabel(r'$\tau_\star\equiv\Sigma_g/\dot\Sigma_\star$ [Gyr]')
    axs[3].set_ylim((0, 13))
    axs[3].yaxis.set_major_locator(MultipleLocator(5))
    axs[3].yaxis.set_minor_locator(MultipleLocator(1))
    axs[3].set_xlabel('Time [Gyr]')
    axs[3].set_xlim((-1, 14))
    axs[3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[3].xaxis.set_major_locator(MultipleLocator(5))
    # Axes labels
    labels = ['(a)', '(b)', '(c)', '(d)']
    for ax, label in zip(axs, labels):
        ax.text(0.5, 0.95, label, va='top', ha='center', 
                transform=ax.transAxes, 
                size=plt.rcParams['axes.labelsize'])

    # Annotations in top panel
    axs[0].annotate(
        r'$\tau_1$', (0.5, -0.5), xytext=(2, -3), 
        arrowprops={'arrowstyle': '<-'}, size=plt.rcParams['axes.labelsize']
    )
    axs[0].text(
        3.5, -1, r'$t_{\rm max}$', ha='left', va='bottom',
        size=plt.rcParams['axes.labelsize']
    )
    axs[0].annotate(
        '', (3.2, -1.3), xytext=(3.2, 0.2), arrowprops={'arrowstyle': '->'}
    )
    axs[0].annotate(
        r'$\tau_2$', (5.5, -0.8), xytext=(11, -1.3),
        arrowprops={'arrowstyle': '<-'}, size=plt.rcParams['axes.labelsize']
    )

    zones = [int(galr / ZONE_WIDTH) for galr in get_bin_centers(GALR_BINS)]
    colors = get_color_list(plt.get_cmap(cmap), GALR_BINS)
    for zone, color in zip(zones, colors):
        radius = (zone + 0.5) * ZONE_WIDTH
        area = np.pi * ((radius + ZONE_WIDTH)**2 - radius**2)
        label = f'{zone * ZONE_WIDTH:.0f}'
        kwargs = dict(label=label, color=color, ls='-')
        history = multioutput.zones[f'zone{zone}'].history
        time = np.array(history['time'])
        infall_surface = np.array(history['ifr']) / area
        axs[0].plot(time, np.log10(infall_surface), **kwargs)
        sf_surface = np.array(history['sfr']) / area
        axs[1].plot(time[1:], np.log10(sf_surface[1:]), **kwargs)
        gas_surface = np.array(history['mgas']) / area
        axs[2].plot(time, np.log10(gas_surface), **kwargs)
        tau_star = [history['mgas'][i+1] / history['sfr'][i+1] * 1e-9 for i in range(
                    len(history['time']) - 1)]
        axs[3].plot(history['time'][1:], tau_star, **kwargs)
    axs[0].legend(title=r'$R_{\rm gal}$ [kpc]', ncols=3, loc='lower right')
    # Save
    savedir = {
        'paper': paths.figures,
        'poster': paths.extra / 'poster'
    }[style]
    if not savedir.is_dir():
        savedir.mkdir()
    plt.savefig(savedir / 'star_formation_history')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='star_formation_history.py',
        description='Plot the star formation history of a multizone output.'
    )
    parser.add_argument(
        '-o', '--output-name', 
        metavar='NAME',
        type=str,
        default=OUTPUT_NAME,
        help='Name of VICE multizone output located within src/data/multizone.\
 (Default: yZ1/fiducial/diskmodel).'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument(
        '--style', 
        type=str,
        default='paper',
        choices=['paper', 'poster', 'presentation'],
        help='Plot style to use (default: paper).'
    )
    parser.add_argument(
        '--cmap', 
        metavar='COLORMAP', 
        type=str,
        default=CMAP,
        help='Name of colormap for color-coding model output (default: %s).' % CMAP
    )
    args = parser.parse_args()
    main(**vars(args))
