"""
This script creates multiple plots from the given multizone output for diagnostic purposes.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import vice

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from age_abundance_grid import plot_age_abundance_grid
from feh_distribution import plot_feh_distribution
from ofe_distribution import plot_ofe_distribution
# from ofe_bimodality import plot_bimodality_comparison
from ofe_feh_grid import plot_ofe_feh_grid
from utils import get_bin_centers, get_color_list
import paths
from _globals import TWO_COLUMN_WIDTH, ZONE_WIDTH, GALR_BINS

def main(output_name, verbose=False, tracks=False, log_age=False, 
         uncertainties=False, apogee_data=False, style='paper'):
    # Import APOGEE data
    apogee_sample = APOGEESample.load()
    # Import multizone stars data
    mzs = MultizoneStars.from_output(output_name)
    parent_dir = paths.extra / mzs.name.replace('diskmodel', '')
    # Forward-model APOGEE uncertainties
    if uncertainties:
        mzs.model_uncertainty(inplace=True, apogee_data=apogee_sample.data)
    # Age vs [O/H]
    plot_age_abundance_grid(mzs, '[o/h]', color_by='galr_origin', cmap='plasma_r', 
                            apogee_sample=apogee_sample,
                            style=style, log=log_age, verbose=verbose,
                            medians=apogee_data, tracks=tracks)
    # Age vs [Fe/H]
    plot_age_abundance_grid(mzs, '[fe/h]', color_by='galr_origin', cmap='plasma_r', 
                            apogee_sample=apogee_sample,
                            style=style, log=log_age, verbose=verbose,
                            medians=apogee_data, tracks=tracks)
    # Age vs [O/Fe]
    plot_age_abundance_grid(mzs, '[o/fe]', color_by='[fe/h]', cmap='viridis', 
                            apogee_sample=apogee_sample,
                            style=style, log=log_age, verbose=verbose,
                            medians=apogee_data, tracks=tracks)
    # Abundance distributions
    plot_feh_distribution(mzs, apogee_sample, style=style)
    plot_ofe_distribution(mzs, apogee_sample, style=style)
    # [O/Fe] vs [Fe/H]
    plot_ofe_feh_grid(mzs, apogee_sample, tracks=tracks, cmap='plasma_r',
                      apogee_contours=apogee_data, style=style)
    # [O/Fe] vs [Fe/H], color-coded by age
    plot_ofe_feh_grid(mzs, apogee_sample, tracks=tracks,
                      apogee_contours=apogee_data, style=style,
                      color_by='age', cmap='Spectral_r', 
                      fname='ofe_feh_age_grid.png')
    # Star formation history
    plot_sfh(output_name, style=style)
    print('Done! Plots are located at %s' % str(parent_dir))


def plot_sfh(output_name, style='paper', cmap='plasma_r', fname='sfh.png'):
    """
    Plot the history of gas infall, star formation, gas mass, and star 
    formation efficiency timescale for the given multizone output.
    """
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(1, 4, sharex=True, tight_layout=True,
                            figsize=(TWO_COLUMN_WIDTH, 0.3 * TWO_COLUMN_WIDTH))
    fig.suptitle(output_name)
    multioutput = vice.output(str(paths.multizone / output_name))
    axs[0].set_ylabel(r'$\dot \Sigma_{\rm in}$ [$M_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    axs[0].set_yscale('log')
    # axs[0].set_ylim((3e-4, 0.1))
    axs[1].set_ylabel(r'$\dot \Sigma_\star$ [$M_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    axs[1].set_yscale('log')
    # axs[1].set_ylim((3e-5, 0.1))
    axs[2].set_ylabel(r'$\Sigma_g$ [$M_{\odot}\,\rm{kpc}^{-2}$]')
    axs[2].set_yscale('log')
    # axs[2].set_ylim((5e5, 2e8))
    axs[3].set_ylabel(r'$\tau_\star$ [Gyr]')
    axs[3].set_ylim((0, 12))
    for ax in axs:
        ax.set_xlabel('Time [Gyr]')
        ax.set_xlim((-1, 14))
    zones = [int(galr / ZONE_WIDTH) for galr in get_bin_centers(GALR_BINS)]
    colors = get_color_list(plt.get_cmap(cmap), GALR_BINS)
    linestyle = '-'
    for zone, color in zip(zones, colors):
        radius = (zone + 0.5) * ZONE_WIDTH
        area = np.pi * ((radius + ZONE_WIDTH)**2 - radius**2)
        label = f'{zone * ZONE_WIDTH} kpc'
        history = multioutput.zones[f'zone{zone}'].history
        time = np.array(history['time'])
        infall_surface = np.array(history['ifr']) / area
        axs[0].plot(time, infall_surface, color=color, label=label, ls=linestyle)
        sf_surface = np.array(history['sfr']) / area
        axs[1].plot(time, sf_surface, color=color, ls=linestyle, label=label)
        gas_surface = np.array(history['mgas']) / area
        axs[2].plot(time, gas_surface, color=color, ls=linestyle, label=label)
        tau_star = [history['mgas'][i+1] / history['sfr'][i+1] * 1e-9 for i in range(
                    len(history['time']) - 1)]
        axs[3].plot(history['time'][1:], tau_star, color=color, ls=linestyle,
                    label=label)
    # Save
    fullpath = paths.extra / output_name.replace('diskmodel', fname)
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='extra_plots.py',
        description='Generate multiple diagnostic plots for the given multizone output.'
    )
    parser.add_argument(
        'output_name', 
        metavar='NAME',
        type=str,
        help='Name of VICE multizone output located within src/data/multizone.'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument(
        '-t', '--tracks', 
        action='store_true',
        help='Plot ISM tracks in addition to stellar abundances.'
    )
    parser.add_argument(
        '-l', '--log-age', 
        action='store_true',
        help='Plot age on a log scale.'
    )
    parser.add_argument(
        '-u', '--uncertainties', 
        action='store_true',
        help='Forward-model APOGEE uncertainties in VICE output.'
    )
    parser.add_argument(
        '-a', '--apogee-data', 
        action='store_true',
        help='Plot APOGEE data for comparison.'
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
