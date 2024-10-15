"""
This script plots stellar density as a function of Galactocentric radius.
"""

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


def main(style='paper'):    
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, ax = plt.subplots(tight_layout=True)
    
    rbins = np.arange(0, MAX_SF_RADIUS, ZONE_WIDTH)
    rbin_centers = get_bin_centers(rbins)
    
    mw_disk = two_component_disk()
    total_disk = np.array([mw_disk(r) for r in rbin_centers])
    thin_disk = np.array([mw_disk.thin_disk(r) for r in rbin_centers])
    thick_disk = np.array([mw_disk.thick_disk(r) for r in rbin_centers])
    
    # Plot model baseline
    # ax.axhline(2, color='k', linestyle='-', label='Total disk')
    # ax.axhline(1, color='k', linestyle='--', label='Thin disk')
    # ax.axhline(0, color='k', linestyle=':', label='Thick disk')
    ax.plot(rbin_centers, total_disk, 'k-', label='Total disk')
    # ax.plot(rbin_centers, thin_disk, 'k--', label='Thin disk')
    # ax.plot(rbin_centers, thick_disk, 'k-.', label='Thick disk')
    
    # Static SFR mode
    name = 'nomigration/no_outflow/gasflow_in_1kms/J21/static/diskmodel'
    mzs = MultizoneStars.from_output(name)
    densities = stellar_density_gradient(mzs, rbins)
    ax.plot(rbin_centers, densities, 'r-', label='Static SFR (no outflow)')
    # Gas density
    sigma_gas, radii = gas_density_gradient(name)
    ax.plot(radii + ZONE_WIDTH, sigma_gas, 'r:', label='Gas')
    
    # Static IFR mode
    name = 'nomigration/no_outflow/gasflow_in_1kms/J21/static_infall/diskmodel'
    mzs = MultizoneStars.from_output(name)
    densities = stellar_density_gradient(mzs, rbins)
    ax.plot(rbin_centers, densities, 'b--', label='Static IFR (no outflow)')
    # Gas density
    sigma_gas, radii = gas_density_gradient(name)
    ax.plot(radii + ZONE_WIDTH, sigma_gas, 'b:', label='Gas')
    
    # Inside-out
    # insideout_name = 'nomigration/outflow/no_gasflow/J21/static_fine_dr/diskmodel'
    # insideout = MultizoneStars.from_output(insideout_name)
    # densities = surface_density_gradient(insideout, rbins)
    # ax.plot(rbin_centers, (densities - total_disk) / total_disk + 2, 'g-', label='dr=0.01')
    
    # Two-infall SFH with no migration
    # nomig_name = 'nomigration/outflow/no_gasflow/J21/twoinfall/diskmodel'
    # nomig = MultizoneStars.from_output(nomig_name)
    # densities = stellar_density_gradient(nomig, rbins)
    # ax.plot(rbin_centers, densities, 'g-', 
    #         label='Two-infall')
    
    # onset = get_onset_time(nomig_name)
    # nomig_thin = nomig.filter({'formation_time': (onset, None)})
    # densities = stellar_density_gradient(nomig_thin, rbins)
    # ax.plot(rbin_centers, densities, 'g--')
    
    # nomig_thick = nomig.filter({'formation_time': (0, onset)})
    # densities = stellar_density_gradient(nomig_thick, rbins)
    # ax.plot(rbin_centers, densities, 'g.-')
    
    ax.set_xlabel(r'$R_{\rm gal}$ [kpc]')
    # ax.set_ylabel(r'$\Delta\Sigma_\star/\Sigma_\star$')
    ax.set_ylabel(r'$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]')
    ax.set_yscale('log')
    ax.set_ylim((5e5, 6e9))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.xaxis.set_major_locator(MultipleLocator(4))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc='upper right', frameon=False)
    ax.set_title('1 km/s radial gas flows')
    
    plt.savefig(paths.figures / 'density_gradient')
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


def gas_density_gradient(name, zone_width=ZONE_WIDTH):
    """
    Calculate the radial gas surface density gradient.
    
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
    gas_mass_gradient = radial_gradient(multioutput, 'mgas', 
                                        zone_width=zone_width)
    radii = [i * zone_width for i in range(len(gas_mass_gradient))]
    areas = [m.pi * ((r + zone_width)**2 - r**2) for r in radii]
    return np.array(gas_mass_gradient) / np.array(areas), np.array(radii)


if __name__ == '__main__':
    main()
