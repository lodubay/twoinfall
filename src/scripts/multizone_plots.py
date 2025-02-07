"""
This script creates multiple plots from the given multizone output for diagnostic purposes.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from multizone_stars import MultizoneStars
from apogee_sample import APOGEESample
from age_abundance_grid import plot_age_abundance_grid
from feh_distribution import plot_feh_distribution
from ofe_distribution import plot_ofe_distribution
from mdf_by_age import plot_mdf_by_age
from mdf_widths import plot_mdf_widths
# from ofe_bimodality import plot_bimodality_comparison
from ofe_feh_grid import plot_ofe_feh_grid
from density_gradient import plot_density_gradient
from utils import get_bin_centers, get_color_list, radial_gradient, weighted_quantile
import paths
from _globals import TWO_COLUMN_WIDTH, ZONE_WIDTH, GALR_BINS, ONE_COLUMN_WIDTH, MAX_SF_RADIUS

def main(output_names, verbose=False, tracks=False, log_age=False, 
         uncertainties=False, apogee_data=False, style='paper'):
    # Import APOGEE data
    apogee_sample = APOGEESample.load()
    for output_name in output_names:
        print(output_name)
        # Import multizone stars data
        try:
            mzs = MultizoneStars.from_output(output_name)
        except: 
            print('No multizone output found.')
            continue
        parent_dir = paths.extra / 'multizone' / mzs.name.replace('diskmodel', '')
        # Forward-model APOGEE uncertainties
        mzs_copy = mzs.copy() # Version without forward-modeled uncertainties
        if uncertainties:
            mzs.model_uncertainty(inplace=True, apogee_data=apogee_sample.data)
        # Abundance gradients
        plot_abundance_gradients(mzs, uncertainties=uncertainties, style=style)
        # Age vs [O/H]
        plot_age_abundance_grid(mzs, '[o/h]', color_by='galr_origin', cmap='winter_r', 
                                apogee_sample=apogee_sample,
                                style=style, log=log_age, verbose=verbose,
                                medians=apogee_data, tracks=tracks)
        if apogee_data:
            plot_age_abundance_grid(mzs, '[o/h]', color_by='galr_origin', cmap='winter_r', 
                                    apogee_sample=apogee_sample, bin_by_age=True,
                                    style=style, log=log_age, verbose=verbose,
                                    medians=apogee_data, tracks=tracks,
                                    fname='age_oh_grid_by_age.png')
        # Age vs [Fe/H]
        plot_age_abundance_grid(mzs, '[fe/h]', color_by='galr_origin', cmap='winter_r', 
                                apogee_sample=apogee_sample,
                                style=style, log=log_age, verbose=verbose,
                                medians=apogee_data, tracks=tracks)
        if apogee_data:
            plot_age_abundance_grid(mzs, '[fe/h]', color_by='galr_origin', cmap='winter_r', 
                                    apogee_sample=apogee_sample, bin_by_age=True,
                                    style=style, log=log_age, verbose=verbose,
                                    medians=apogee_data, tracks=tracks,
                                    fname='age_feh_grid_by_age.png')
        # Age vs [O/Fe]
        plot_age_abundance_grid(mzs, '[o/fe]', color_by='[fe/h]', cmap='viridis', 
                                apogee_sample=apogee_sample,
                                style=style, log=log_age, verbose=verbose,
                                medians=apogee_data, tracks=tracks)
        if apogee_data:
            plot_age_abundance_grid(mzs, '[o/fe]', color_by='[fe/h]', cmap='viridis', 
                                    apogee_sample=apogee_sample, bin_by_age=True,
                                    style=style, log=log_age, verbose=verbose,
                                    medians=apogee_data, tracks=tracks,
                                    fname='age_ofe_grid_by_age.png')
        # Abundance distributions
        plot_feh_distribution(mzs, apogee_sample, style=style)
        plot_ofe_distribution(mzs, apogee_sample, style=style)
        # Abundance distributions as a function of age
        plot_mdf_by_age(mzs_copy, col='[fe/h]', xlim=(-1.2, 0.7), style=style)
        plot_mdf_by_age(mzs_copy, col='[o/h]', xlim=(-1.0, 0.7), style=style)
        # plot_mdf_by_age(mzs, col='[o/fe]', xlim=(-0.15, 0.55), smoothing=0.02)
        # MDF width as a function of age
        plot_mdf_widths(mzs_copy, col='[fe/h]', style=style)
        plot_mdf_widths(mzs_copy, col='[o/h]', style=style)
        # [O/Fe] vs [Fe/H]
        plot_ofe_feh_grid(mzs, apogee_sample, tracks=tracks, cmap='winter_r',
                        apogee_contours=apogee_data, style=style)
        # [O/Fe] vs [Fe/H], color-coded by age
        plot_ofe_feh_grid(mzs, apogee_sample, tracks=tracks,
                        apogee_contours=apogee_data, style=style,
                        color_by='age', cmap='Spectral_r', 
                        fname='ofe_feh_age_grid.png')
        # Star formation history
        plot_sfh(output_name, style=style)
        # Stellar density gradient
        plot_density_gradient(mzs, components=True)
        # Mass-loading factor
        plot_mass_loading(output_name)
        print('Done! Plots are located at %s' % str(parent_dir))


def plot_sfh(output_name, style='paper', cmap='plasma_r', fname='sfh.png'):
    """
    Plot the history of gas infall, star formation, gas mass, and star 
    formation efficiency timescale for the given multizone output.
    """
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(1, 4, sharex=True, tight_layout=True,
                            figsize=(TWO_COLUMN_WIDTH, 0.33 * TWO_COLUMN_WIDTH))
    fig.suptitle(output_name)
    multioutput = vice.output(str(paths.multizone / output_name))
    axs[0].set_title(r'$\dot \Sigma_{\rm in}$ [M$_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    axs[0].set_yscale('log')
    # axs[0].set_ylim((3e-4, 0.1))
    axs[1].set_title(r'$\dot \Sigma_\star$ [M$_{\odot}\,\rm{yr}^{-1}\,\rm{kpc}^{-2}$]')
    axs[1].set_yscale('log')
    axs[1].set_ylim((1e-5, 1e-1))
    axs[2].set_title(r'$\Sigma_g$ [M$_{\odot}\,\rm{kpc}^{-2}$]')
    axs[2].set_yscale('log')
    axs[2].set_ylim((1e5, 3e8))
    axs[3].set_title(r'$\tau_\star$ [Gyr]')
    axs[3].set_ylim((0, 12))
    axs[3].yaxis.set_major_locator(MultipleLocator(5))
    axs[3].yaxis.set_minor_locator(MultipleLocator(1))
    for ax in axs:
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
        axs[0].plot(time, infall_surface, color=color, label=label, ls=linestyle)
        sf_surface = np.array(history['sfr']) / area
        axs[1].plot(time[1:], sf_surface[1:], color=color, ls=linestyle, label=label)
        gas_surface = np.array(history['mgas']) / area
        axs[2].plot(time, gas_surface, color=color, ls=linestyle, label=label)
        tau_star = [history['mgas'][i+1] / history['sfr'][i+1] * 1e-9 for i in range(
                    len(history['time']) - 1)]
        axs[3].plot(history['time'][1:], tau_star, color=color, ls=linestyle,
                    label=label)
    axs[0].legend(frameon=False, title=r'$R_{\rm gal}$ [kpc]', ncols=2, 
                  loc='lower right', borderpad=0.2, labelspacing=0.2, columnspacing=1.2)
    # Save
    fullpath = paths.extra / 'multizone' / output_name.replace('diskmodel', fname)
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


def plot_mass_loading(output_name, style='paper', fname='mass_loading.png'):
    """
    Plot the outflow mass-loading factor as a function of radius.
    """
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH))

    multioutput = vice.output(str(paths.multizone / output_name))
    eta_list = radial_gradient(multioutput, 'eta_0')
    radii = [i * ZONE_WIDTH for i in range(len(eta_list))]

    ax.plot(radii, eta_list, "k-")
    
    ax.axvline(8, color="r", ls="--")
    ax.axhline(eta_list[80], color="r", ls="--")
    
    ax.set_xlabel(r"$R_{\rm gal}$ [kpc]")
    ax.set_ylabel(r"$\eta\equiv\dot\Sigma_{\rm out}/\dot\Sigma_\star$")
    # Save
    fullpath = paths.extra / 'multizone' / output_name.replace('diskmodel', fname)
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


def plot_abundance_gradients(mzs, uncertainties=False, style='paper'):
    """
    Plot the radial gas and stellar abundance gradient for a multizone output.
    """
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(3, 1, figsize=(ONE_COLUMN_WIDTH, 2 * ONE_COLUMN_WIDTH),
                            tight_layout=True, sharex=True)
    fig.suptitle(mzs.name)
    # Gas
    mout = vice.output(str(paths.multizone / mzs.name))
    xarr = np.arange(0, MAX_SF_RADIUS, ZONE_WIDTH)
    axs[0].plot(xarr, radial_gradient(mout, '[o/h]'), 'k-', label='Gas (present-day)')
    axs[1].plot(xarr, radial_gradient(mout, '[fe/h]'), 'k-')
    axs[2].plot(xarr, radial_gradient(mout, '[o/fe]'), 'k-')
    # Stars
    median_abundances = np.zeros((3, len(GALR_BINS)-1))
    for i in range(len(GALR_BINS)-1):
        galr_lim = GALR_BINS[i:i+2]
        subset = mzs.filter({'galr_final': tuple(galr_lim), 
                             'zfinal': (0, 0.5),
                             'age': (0, 0.1)})
        median_abundances[:,i] = [
            weighted_quantile(subset.stars, '[o/h]', 'mstar', quantile=0.5),
            weighted_quantile(subset.stars, '[fe/h]', 'mstar', quantile=0.5),
            weighted_quantile(subset.stars, '[o/fe]', 'mstar', quantile=0.5),
        ]
    axs[0].plot(get_bin_centers(GALR_BINS), median_abundances[0], 'ko', label='Stars (<100 Myr old)')
    axs[1].plot(get_bin_centers(GALR_BINS), median_abundances[1], 'ko')
    axs[2].plot(get_bin_centers(GALR_BINS), median_abundances[2], 'ko')
    # Reference gradient and sun
    axs[0].plot(xarr, -0.06 * (xarr - 8.0), 'k--', label='Reference (-0.06 dex/kpc)')
    axs[0].scatter([8], [0], marker='+', color='k')
    axs[1].plot(xarr, -0.06 * (xarr - 8.0), 'k--', label='Reference (-0.06 dex/kpc)')
    axs[1].scatter([8], [0], marker='+', color='k')
    axs[2].scatter([8], [0], marker='+', color='k')
    # Configure axes
    axs[0].set_xlim((-1, 17))
    axs[0].set_ylim((-0.7, 0.7))
    axs[0].xaxis.set_major_locator(MultipleLocator(4))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].set_ylabel('[O/H]')
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].set_ylim((-0.7, 0.7))
    axs[1].set_ylabel('[Fe/H]')
    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[2].set_ylim((-0.12, 0.12))
    axs[2].set_ylabel('[O/Fe]')
    axs[2].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.02))
    axs[2].set_xlabel('Radius [kpc]')    
    axs[0].legend()
    # Save
    fullpath = paths.extra / 'multizone' / mzs.name.replace('diskmodel', 'abundance_gradients.png')
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='multizone_plots.py',
        description='Generate multiple diagnostic plots for the given multizone output.'
    )
    parser.add_argument(
        'output_names', 
        metavar='NAME',
        type=str,
        nargs='+',
        help='Name(s) of VICE multizone output located within src/data/multizone.'
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
