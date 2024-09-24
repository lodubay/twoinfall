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
    rbins = np.arange(0, _globals.MAX_SF_RADIUS, dr)
    rbin_centers = get_bin_centers(rbins)
    
    mw_disk = two_component_disk()
    total_disk = np.array([mw_disk(r) for r in rbins[:-1]])
    thin_disk = np.array([mw_disk.thin_disk(r) for r in rbins[:-1]])
    thick_disk = np.array([mw_disk.thick_disk(r) for r in rbins[:-1]])
    
    # Plot model baseline
    ax.axhline(2, color='k', linestyle='-', label='Total disk')
    ax.axhline(1, color='k', linestyle='--', label='Thin disk')
    ax.axhline(0, color='k', linestyle=':', label='Thick disk')
    
    # Two-infall SFH with no migration
    nomig_name = 'nomigration/outflow/no_gasflow/J21/twoinfall/diskmodel'
    nomig = MultizoneStars.from_output(nomig_name)
    densities = surface_density_gradient(nomig, rbins)
    ax.plot(rbins[:-1], (densities - total_disk) / total_disk + 2, 'g-', 
            label='dt=0.01')
    
    onset = get_onset_time(nomig_name)
    nomig_thin = nomig.filter({'formation_time': (onset, None)})
    densities = surface_density_gradient(nomig_thin, rbins)
    ax.plot(rbins[:-1], (densities - thin_disk) / thin_disk + 1, 'g--')
    
    nomig_thick = nomig.filter({'formation_time': (0, onset)})
    densities = surface_density_gradient(nomig_thick, rbins)
    ax.plot(rbins[:-1], (densities - thick_disk) / thick_disk, 'g:')
    
    # Two-infall SFH no migration (small integration dt)
    finedt_name = 'nomigration/outflow/no_gasflow/J21/twoinfall_fine_dt/diskmodel'
    finedt = MultizoneStars.from_output(finedt_name)
    densities = surface_density_gradient(finedt, rbins)
    ax.plot(rbins[:-1], (densities - total_disk) / total_disk + 2, 'b-', 
            label='dt=0.001')
    
    # Two-infall components
    onset = get_onset_time(finedt_name)
    finedt_thin = finedt.filter({'formation_time': (onset, None)})
    densities = surface_density_gradient(finedt_thin, rbins)
    ax.plot(rbins[:-1], (densities - thin_disk) / thin_disk + 1, 'b--')
    finedt_thick = finedt.filter({'formation_time': (0, onset)})
    densities = surface_density_gradient(finedt_thick, rbins)
    ax.plot(rbins[:-1], (densities - thick_disk) / thick_disk, 'b:')
    
    
    # Inside-out
    # insideout_name = 'nomigration/outflow/no_gasflow/J21/insideout/diskmodel'
    # insideout = MultizoneStars.from_output(insideout_name)
    # densities = surface_density_gradient(insideout, rbins)
    # ax.plot(rbin_centers, (densities - total_disk) / total_disk + 2, 'r-', label='Inside-out')
    
    # Two-infall SFH with Gaussian migration scheme
    # gaussmig_name = 'gaussian/outflow/no_gasflow/J21/twoinfall/diskmodel'
    # gaussmig = MultizoneStars.from_output(gaussmig_name)
    # densities = surface_density_gradient(gaussmig, rbins)
    # ax.plot(rbin_centers, densities, 'b-', label='Gaussian migration')
    
    # Two-infall components
    # onset = get_onset_time(gaussmig_name)
    # gaussmig_thick = gaussmig.filter({'formation_time': (0, onset)})
    # densities = surface_density_gradient(gaussmig_thick, rbins)
    # ax.plot(rbin_centers, densities, 'b:')
    # gaussmig_thin = gaussmig.filter({'formation_time': (onset, None)})
    # densities = surface_density_gradient(gaussmig_thin, rbins)
    # ax.plot(rbin_centers, densities, 'b--')
    
    ax.set_xlabel(r'$R_{\rm gal}$ [kpc]')
    # ax.set_ylabel(r'$\Sigma_{\rm\star, out} - \Sigma_{\rm \star, model}$ [M$_\odot$ kpc$^{-2}$]')
    ax.set_ylabel(r'$\Delta\Sigma_\star/\Sigma_\star$')
    # ax.set_yscale('log')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(1))
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
    areas = [m.pi * (rbins[i+1]**2 - rbins[i]**2) for i in range(len(rbins)-1)]
    return masses / np.array(areas)


if __name__ == '__main__':
    main()
