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

def main(output_name=OUTPUT_NAME, style='paper', cmap='plasma_r', verbose=False):
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        2, 2, 
        sharex=True, 
        # tight_layout=True,
        figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0.25, 'wspace': 0.3}
    )
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    multioutput = vice.output(str(paths.multizone / output_name))
    axs[0,0].set_title(r'$\dot \Sigma_{\rm in}$ [M$_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    axs[0,0].text(0.1, 0.95, '(a)', va='top', transform=axs[0,0].transAxes)
    axs[0,0].set_yscale('log')
    axs[0,0].set_ylim((5e-6, 0.1))
    axs[0,1].set_title(r'$\dot \Sigma_\star$ [M$_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    axs[0,1].text(0.1, 0.95, '(b)', va='top', transform=axs[0,1].transAxes)
    axs[0,1].set_yscale('log')
    axs[0,1].set_ylim((1e-5, 1e-1))
    axs[1,0].set_title(r'$\Sigma_g$ [M$_{\odot}\,\rm{kpc}^{-2}$]')
    axs[1,0].text(0.1, 0.95, '(c)', va='top', transform=axs[1,0].transAxes)
    axs[1,0].set_yscale('log')
    axs[1,0].set_ylim((1e5, 2e8))
    axs[1,1].set_title(r'$\tau_\star\equiv\Sigma_g/\dot\Sigma_\star$ [Gyr]')
    axs[1,1].text(0.1, 0.95, '(d)', va='top', transform=axs[1,1].transAxes)
    axs[1,1].set_ylim((0, 12))
    axs[1,1].yaxis.set_major_locator(MultipleLocator(5))
    axs[1,1].yaxis.set_minor_locator(MultipleLocator(1))
    for ax in axs[1,:]:
        ax.set_xlabel('Time [Gyr]')
        ax.set_xlim((-1, 14))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
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
        axs[0,0].plot(time, infall_surface, color=color, label=label, ls=linestyle)
        sf_surface = np.array(history['sfr']) / area
        axs[0,1].plot(time[1:], sf_surface[1:], color=color, ls=linestyle, label=label)
        gas_surface = np.array(history['mgas']) / area
        axs[1,0].plot(time, gas_surface, color=color, ls=linestyle, label=label)
        tau_star = [history['mgas'][i+1] / history['sfr'][i+1] * 1e-9 for i in range(
                    len(history['time']) - 1)]
        axs[1,1].plot(history['time'][1:], tau_star, color=color, ls=linestyle,
                    label=label)
    axs[0,0].legend(
        frameon=False,
        title=r'$R_{\rm gal}$ [kpc]', 
        ncols=2, 
        loc='lower right', 
        borderpad=0.2, labelspacing=0.2, columnspacing=0.5, 
        handlelength=1.0, handletextpad=0.5, 
    )
    # Save
    plt.savefig(paths.figures / 'star_formation_history')
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
    args = parser.parse_args()
    main(**vars(args))
