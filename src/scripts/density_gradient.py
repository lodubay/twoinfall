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

from utils import get_bin_centers
from multizone.src.models.diskmodel import two_component_disk
import paths
import _globals


def main(style='paper'):    
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, ax = plt.subplots(tight_layout=True)
    
    dr = 0.1
    rbins = np.arange(0, _globals.MAX_SF_RADIUS + dr, dr)
    rbin_centers = get_bin_centers(rbins)
    
    # Plot expected gradients
    mw_disk = two_component_disk()
    ax.plot(rbin_centers, [mw_disk(r) for r in rbin_centers], 
            'k-', label='Two-component disk')
    
    # Individual thin and thick disk components
    ax.plot(rbin_centers, 
            [mw_disk.ratio * mw_disk.second(r) for r in rbin_centers], 
            'k--', label='Thin disk')
    ax.plot(rbin_centers, [mw_disk.first(r) for r in rbin_centers], 
            'k:', label='Thick disk')
    
    # Two-infall SFH with no migration
    nomig_name = 'nomigration/outflow/no_gasflow/J21/twoinfall/diskmodel'
    nomig = MultizoneStars.from_output(nomig_name)
    densities = surface_density_gradient(nomig, rbins)
    ax.plot(rbin_centers[:154], densities[:154], 'g-', label='No migration')
    
    onset = get_onset_time(nomig_name)
    nomig_thick = nomig.filter({'formation_time': (0, onset)})
    densities = surface_density_gradient(nomig_thick, rbins)
    ax.plot(rbin_centers[:154], densities[:154], 'g:')
    
    nomig_thin = nomig.filter({'formation_time': (onset, None)})
    densities = surface_density_gradient(nomig_thin, rbins)
    ax.plot(rbin_centers[:154], densities[:154], 'g--')
    
    # Two-infall SFH with Gaussian migration scheme
    gaussmig_name = 'gaussian/outflow/no_gasflow/J21/twoinfall/diskmodel'
    gaussmig = MultizoneStars.from_output(gaussmig_name)
    densities = surface_density_gradient(gaussmig, rbins)
    ax.plot(rbin_centers, densities, 'b-', label='Gaussian migration')
    
    # Two-infall components
    onset = get_onset_time(gaussmig_name)
    gaussmig_thick = gaussmig.filter({'formation_time': (0, onset)})
    densities = surface_density_gradient(gaussmig_thick, rbins)
    ax.plot(rbin_centers, densities, 'b:')
    gaussmig_thin = gaussmig.filter({'formation_time': (onset, None)})
    densities = surface_density_gradient(gaussmig_thin, rbins)
    ax.plot(rbin_centers, densities, 'b--')
    
    ax.set_xlabel(r'$R_{\rm gal}$ [kpc]')
    ax.set_ylabel(r'$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]')
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper right', frameon=False)
    
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


def surface_density_gradient(mzs, rbins, origin=False):
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
    areas = [np.pi * (rbins[i+1]**2 - rbins[i]**2) for i in range(len(rbins)-1)]
    return masses / np.array(areas)


if __name__ == '__main__':
    main()
