"""
This script plots stellar density as a function of Galactocentric radius.
"""

import argparse
import math as m

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from multizone_stars import MultizoneStars
import vice

from utils import get_bin_centers, radial_gradient
from multizone.src.models.diskmodel import two_component_disk
import paths
from _globals import MAX_SF_RADIUS, ZONE_WIDTH


def main(output_name, components=False, style='paper', origin=False):
    # Import multioutput stars data
    mzs = MultizoneStars.from_output(output_name)
    plot_density_gradient(mzs, components=components, style=style, origin=origin)


def plot_density_gradient(mzs, components=False, style='paper', label='VICE',
                          color='r', fname='density_gradient.png', origin=False):    
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, ax = plt.subplots(tight_layout=True)
    fig.suptitle(mzs.name)
    
    rbins = np.arange(0, MAX_SF_RADIUS, ZONE_WIDTH)
    rbin_centers = get_bin_centers(rbins)
    
    # Analytic model
    mw_disk = two_component_disk()
    total_disk = np.array([mw_disk(r) for r in rbin_centers])
    ax.plot(rbin_centers, total_disk, 'k-', label='Total disk')
    if components:
        thin_disk = np.array([mw_disk.thin_disk(r) for r in rbin_centers])
        ax.plot(rbin_centers, thin_disk, 'k--', label='Thin disk')
        thick_disk = np.array([mw_disk.thick_disk(r) for r in rbin_centers])
        ax.plot(rbin_centers, thick_disk, 'k-.', label='Thick disk')
    
    # Multi-zone output
    densities = stellar_density_gradient(mzs, rbins, origin=origin)
    ax.plot(rbin_centers, densities, color=color, linestyle='-', label=label)
    if components:
        onset = get_onset_time(mzs.name)
        thin_stars = mzs.filter({'formation_time': (onset, None)})
        densities = stellar_density_gradient(thin_stars, rbins, origin=origin)
        ax.plot(rbin_centers, densities, color=color, linestyle='--')
        thick_stars = mzs.filter({'formation_time': (0, onset)})
        densities = stellar_density_gradient(thick_stars, rbins, origin=origin)
        ax.plot(rbin_centers, densities, color=color, linestyle='-.')
    
    # Gas density
    sigma_gas, radii = gas_density_gradient(mzs.name)
    ax.plot(radii, sigma_gas, color=color, linestyle=':', label='Gas')
    
    ax.set_xlabel(r'$R_{\rm gal}$ [kpc]')
    # ax.set_ylabel(r'$\Delta\Sigma_\star/\Sigma_\star$')
    ax.set_ylabel(r'$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]')
    ax.set_yscale('log')
    ax.set_ylim((5e5, 6e9))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc='upper right', frameon=False)
    # ax.set_title('1 km/s radial gas flows')
    
    # Save
    fullpath = paths.extra / 'multizone' / mzs.name.replace('diskmodel', fname)
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    plt.savefig(fullpath, dpi=300)
    plt.close()


def get_onset_time(name, z=80):
    """
    Calculate the onset of the second gas infall epoch from a vice multioutput.

    Parameters
    ----------
    name : str
        Relative path to multizone output directory.
    z : int, optional
        Zone number to investigate. The default is 80.

    Returns
    -------
    float
        Onset time of the second gas infall epoch in Gyr.

    """
    multioutput = vice.output(str(paths.multizone / name))
    zone = multioutput.zones['zone%i' % z]
    ifr = zone.history['ifr']
    times = zone.history['time']
    ifr_change = [ifr[i+1] - ifr[i] for i in range(len(ifr)-1)]
    return times[ifr_change.index(max(ifr_change))+1]


def stellar_density_gradient(mzs, rbins, origin=False):
    """
    Calculate the stellar surface mass density gradient for the given VICE
    multi-zone model output.
    
    Parameters
    ----------
    mzs : MultizoneStars
    rbins : array
    origin : bool, optional
        If True, sort stars by birth radius instead of final. Default is False.
    
    Returns
    -------
    numpy.ndarray
        Stellar surface mass densities in each radius bin [Msun kpc^-2].
    
    """
    stars = mzs.stars.copy()
    if origin:
        rcol = 'galr_origin'
    else:
        rcol = 'galr_final'
    masses = stars.groupby(pd.cut(stars[rcol], rbins), 
                           observed=False)['mstar'].sum()
    areas = [m.pi * (rbins[i+1]**2 - rbins[i]**2) for i in range(len(rbins)-1)]
    return masses / np.array(areas)


def gas_density_gradient(name, zone_width=ZONE_WIDTH, index=-1):
    """
    Calculate the radial gas surface density gradient.
    
    Parameters
    ----------
    name : str
        Relative path to multizone output directory.
    rbins : array-like, optional
        Radial bins.
    index : int, optional
        Time-index to calculate the radial gradient at.
    
    Returns
    -------
    sigma_gas : numpy.ndarray
        Gas surface densities in each radius bin [Msun kpc^-2]
    radii : numpy.ndarray
        Inner radii of zones in kpc.
    
    """
    multioutput = vice.output(str(paths.multizone / name))
    gas_mass_gradient = radial_gradient(multioutput, 'mgas', 
                                        zone_width=zone_width, index=index)
    radii = [i * zone_width for i in range(len(gas_mass_gradient))]
    areas = [m.pi * ((r + zone_width)**2 - r**2) for r in radii]
    return np.array(gas_mass_gradient) / np.array(areas), np.array(radii)


def mstar_density_gradient(name, zone_width=ZONE_WIDTH):
    """
    Calculate the radial stellar surface density gradient from each zone.
    
    Parameters
    ----------
    name : str
        Relative path to multizone output directory.
    rbins : array-like
        Radial bins.
    
    Returns
    -------
    sigma_gas : numpy.ndarray
        Gas surface densities in each radius bin [Msun kpc^-2]
    radii : numpy.ndarray
        Inner radii of zones in kpc.
    
    """
    multioutput = vice.output(str(paths.multizone / name))
    mstar_mass_gradient = radial_gradient(multioutput, 'mstar', 
                                          zone_width=zone_width)
    radii = [i * zone_width for i in range(len(mstar_mass_gradient))]
    areas = [m.pi * ((r + zone_width)**2 - r**2) for r in radii]
    return np.array(mstar_mass_gradient) / np.array(areas), np.array(radii)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='density_gradient.py',
        description='Plot the stellar density gradient from a multizone run.'
    )
    parser.add_argument('output_name', metavar='NAME',
                        help='Name of VICE multizone output.')
    parser.add_argument('-c', '--components', action='store_true',
                        help='Plot thick & thin disk components.')
    parser.add_argument('-s', '--style', 
                        choices=['paper', 'poster'],
                        default='paper', 
                        help='Plot style to use (default: paper)')
    parser.add_argument('-o', '--origin', action='store_true',
                        help='Plot stellar density gradient at birth, ' + \
                            'rather than final radius.')
    args = parser.parse_args()
    main(**vars(args))
