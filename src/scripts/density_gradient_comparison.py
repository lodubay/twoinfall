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

from density_gradient import stellar_density_gradient, gas_density_gradient, \
    mstar_density_gradient
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
    name = 'nomigration/no_outflow/no_gasflow/J21/static/diskmodel'
    mzs = MultizoneStars.from_output(name)
    densities = stellar_density_gradient(mzs, rbins)
    ax.plot(rbin_centers, densities, 'r-', label='Static SFR (no outflow)')
    # Gas density
    sigma_gas, radii = gas_density_gradient(name)
    ax.plot(radii, sigma_gas, 'r:', label='Gas')
    sigma_mstar, radii = mstar_density_gradient(name)
    print(sigma_mstar[:-1] / total_disk)
    
    # Static IFR mode
    name = 'nomigration/outflow/no_gasflow/J21/static_infall/diskmodel'
    mzs = MultizoneStars.from_output(name)
    densities = stellar_density_gradient(mzs, rbins)
    ax.plot(rbin_centers, densities, 'g--', label='Static IFR (outflows)')
    # Gas density
    sigma_gas, radii = gas_density_gradient(name)
    ax.plot(radii, sigma_gas, 'g:', label='Gas')
    sigma_mstar, radii = mstar_density_gradient(name)
    print(sigma_mstar[:-1] / total_disk)
    
    # Static IFR mode
    name = 'nomigration/no_outflow/no_gasflow/J21/static_infall/diskmodel'
    mzs = MultizoneStars.from_output(name)
    densities = stellar_density_gradient(mzs, rbins)
    ax.plot(rbin_centers, densities, 'b--', label='Static IFR (no outflow)')
    # Gas density
    sigma_gas, radii = gas_density_gradient(name)
    ax.plot(radii, sigma_gas, 'b:', label='Gas')
    sigma_mstar, radii = mstar_density_gradient(name)
    print(sigma_mstar[:-1]/ total_disk)
    
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
    # ax.set_title('1 km/s radial gas flows')
    
    plt.savefig(paths.figures / 'density_gradient')
    plt.close()


if __name__ == '__main__':
    main()
