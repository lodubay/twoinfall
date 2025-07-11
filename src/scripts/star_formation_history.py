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

OUTPUT_NAME = 'yZ1/fiducial/diskmodel'
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
    axs[0].set_ylabel(r'$\dot \Sigma_{\rm in}$ [M$_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    axs[0].set_yscale('log')
    axs[0].set_ylim((5e-6, 0.8))
    axs[1].set_ylabel(r'$\dot \Sigma_\star$ [M$_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    axs[1].set_yscale('log')
    axs[1].set_ylim((1e-5, 0.3))
    axs[2].set_ylabel(r'$\Sigma_g$ [M$_{\odot}\,\rm{kpc}^{-2}$]')
    axs[2].set_yscale('log')
    axs[2].set_ylim((1e5, 3e8))
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
        r'$\tau_1$', (1, 5e-2), xytext=(3, 5e-3), 
        arrowprops={'arrowstyle': '<-'}, size=plt.rcParams['axes.labelsize']
    )
    axs[0].text(
        2.1, 0.13, r'$t_{\rm max}$', ha='center', va='bottom',
        size=plt.rcParams['axes.labelsize']
    )
    axs[0].annotate(
        '', (0, 0.1), xytext=(4.2, 0.1), arrowprops={'arrowstyle': '<->'}
    )
    axs[0].annotate(
        r'$\tau_2$', (6, 0.1), xytext=(11, 4e-2),
        arrowprops={'arrowstyle': '<-'}, size=plt.rcParams['axes.labelsize']
    )

    zones = [int(galr / ZONE_WIDTH) for galr in get_bin_centers(GALR_BINS)]
    colors = get_color_list(plt.get_cmap(cmap), GALR_BINS)
    linestyle = '-'
    for zone, color in zip(zones, colors):
        radius = (zone + 0.5) * ZONE_WIDTH
        area = np.pi * ((radius + ZONE_WIDTH)**2 - radius**2)
        label = f'{zone * ZONE_WIDTH:.0f}'
        history = multioutput.zones[f'zone{zone}'].history
        time = np.array(history['time'])
        infall_surface = np.array(history['ifr']) / area
        axs[0].plot(time, infall_surface, color=color, label=label, ls=linestyle)
        sf_surface = np.array(history['sfr']) / area
        axs[1].plot(time[1:], sf_surface[1:], color=color, ls=linestyle, label=label)
        gas_surface = np.array(history['mgas']) / area
        axs[2].plot(time, gas_surface, color=color, ls=linestyle, label=label)
        tau_star = [history['mgas'][i+1] / history['sfr'][i+1] * 1e-9 for i in range(
                    len(history['time']) - 1)]
        axs[3].plot(history['time'][1:], tau_star, color=color, ls=linestyle,
                    label=label)
    axs[0].legend(title=r'$R_{\rm gal}$ [kpc]', ncols=3, loc='lower right')
    # leg = axs[0].legend(
        # loc='center left', bbox_to_anchor=(0., 0.6), frameon=True, ncols=1, 
        #                   handlelength=0, columnspacing=1.5, handletextpad=0,
        #                   facecolor='#ffffff', fancybox=False, framealpha=0.,
        #                   edgecolor='none', labelspacing=0.1)
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.9)
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
