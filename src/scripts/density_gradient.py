"""
This script plots stellar density as a function of Galactocentric radius.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from multizone_stars import MultizoneStars
from utils import get_bin_centers, Exponential
import paths
import _globals

def main(style='paper'):    
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, ax = plt.subplots(tight_layout=True)
    
    rbins = np.arange(0, 20., 0.1)
    rbin_centers = get_bin_centers(rbins)
    
    # Plot expected gradients
    mw_disk = TwoComponentDisk()
    ax.plot(rbin_centers, mw_disk(rbin_centers), 'k-', label='Two-component disk')
    
    # Individual thin and thick disk components
    ax.plot(rbin_centers, mw_disk.thin_disk(rbin_centers), 'k--', label='Thin disk')
    ax.plot(rbin_centers, mw_disk.thick_disk(rbin_centers), 'k:', label='Thick disk')
    
    # Two-infall SFH with no migration
    nomig = MultizoneStars.from_output('nomigration/twoinfall/plateau_width10/diskmodel')
    densities = surface_density_gradient(nomig, rbins)
    ax.plot(rbin_centers[:154], densities[:154], 'g-', label='No migration')
    
    nomig_thick = nomig.filter({'formation_time': (0, 4.)})
    densities = surface_density_gradient(nomig_thick, rbins)
    ax.plot(rbin_centers[:154], densities[:154], 'g:')
    
    # Two-infall SFH with Gaussian migration scheme
    twoinfall = MultizoneStars.from_output('gaussian/twoinfall/plateau_width10/diskmodel')
    densities = surface_density_gradient(twoinfall, rbins)
    ax.plot(rbin_centers, densities, 'r-', label='Gaussian migration')
    
    # Two-infall components
    twoinfall_thick = twoinfall.filter({'formation_time': (0, 4.)})
    densities = surface_density_gradient(twoinfall_thick, rbins)
    ax.plot(rbin_centers, densities, 'r:')
    twoinfall_thin = twoinfall.filter({'formation_time': (4., None)})
    densities = surface_density_gradient(twoinfall_thin, rbins)
    ax.plot(rbin_centers, densities, 'r--')
    
    # Inside-out SFH with analog migration scheme
    analog = MultizoneStars.from_output('diffusion/insideout/powerlaw_slope11/diskmodel')
    densities = surface_density_gradient(analog, rbins)
    ax.plot(rbin_centers, densities, 'b-', label='Analog migration')
    
    ax.set_xlabel(r'$R_{\rm gal}$ [kpc]')
    ax.set_ylabel(r'$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]')
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper right', frameon=False)
    
    plt.savefig(paths.figures / 'density_gradient')
    plt.close()


def surface_density_gradient(mzs, rbins):
    """
    Calculate the stellar surface mass density gradient for the given VICE
    multi-zone model output.
    
    Parameters
    ----------
    mzs : MultizoneStars
    rbins : array
    
    Returns
    -------
    numpy.ndarray
        Stellar surface mass densities in each radius bin [Msun kpc^-2].
    """
    stars = mzs.stars.copy()
    masses = stars.groupby(pd.cut(stars['galr_final'], rbins), 
                           observed=False)['mstar'].sum()
    areas = [np.pi * (rbins[i+1]**2 - rbins[i]**2) for i in range(len(rbins)-1)]
    return masses / np.array(areas)
    

class TwoComponentDisk:
    """
    A model for the Milky Way stellar disk consisting of a thick and a thin
    exponential disk.
    
    Attributes
    ----------
    thin_disk : exponential
        Thin exponential disk component.
    thick_disk : exponential
        Thick exponential disk component.
    
    Methods
    -------
    normalize(rmax=15.5, dr=0.1)
        Normalize the total disk mass to 1.
    """
    def __init__(self, ratio=_globals.THICK_TO_THIN_RATIO, 
                 mass=_globals.M_STAR_MW, 
                 rs_thin=_globals.THIN_DISK_SCALE_RADIUS,
                 rs_thick=_globals.THICK_DISK_SCALE_RADIUS):
        """
        Parameters
        ----------
        ratio : float, optional
            Ratio of thick disk to thin disk surface mass density at R=0. The
            default is 0.27.
        mass : float, optional
            Total mass of the disk in Msun. The default is 5.17e10.
        rs_thin : float, optional
            Thin disk scale radius in kpc. The default is 2.5.
        rs_thick : float, optional
            Thick disk scale radius in kpc. The default is 2.
        """
        self.thin_disk = Exponential(scale=-rs_thin)
        self.thick_disk = Exponential(scale=-rs_thick, coeff=ratio)
        norm = self.normalize()
        self.thin_disk.coeff *= mass * norm
        self.thick_disk.coeff *= mass * norm
    
    def __call__(self, radius):
        """
        Calculate the stellar surface density at the given radius.
        
        Parameters
        ----------
        radius : float or array-like
            Galactocentric radius or array of radii in kpc.
        """
        return self.thin_disk.__call__(radius) + self.thick_disk.__call__(radius)
    
    def normalize(self, rmax=_globals.MAX_SF_RADIUS, dr=0.1):
        """
        Normalize the total mass of the disk to 1.

        Parameters
        ----------
        rmax : float, optional
            Maximum radius of integration in kpc. The default is 15.5.
        dr : float, optional
            Integration step in kpc. The default is 0.1.

        Returns
        -------
        float
            Normalization coefficient [kpc^-2].

        """
        rvals = np.arange(0, rmax+dr, dr)
        integral = np.sum(self.__call__(rvals) * 2 * np.pi * rvals * dr)
        return 1 / integral
    
    @property
    def thin_disk(self):
        """
        Type: exponential
            The thin exponential disk component.
        """
        return self._thin_disk
    
    @thin_disk.setter
    def thin_disk(self, value):
        if isinstance(value, Exponential):
            self._thin_disk = value
        else:
            raise TypeError('Attribute `thin_disk` must be of type ' + 
                            '`exponential`. Got: %s.' % type(value))
                        
    @property
    def thick_disk(self):
        """
        Type: exponential
            The thick exponential disk component.
        """
        return self._thick_disk
    
    @thick_disk.setter
    def thick_disk(self, value):
        if isinstance(value, Exponential):
            self._thick_disk = value
        else:
            raise TypeError('Attribute `thick_disk` must be of type ' + 
                            '`exponential`. Got: %s.' % type(value))


if __name__ == '__main__':
    main()
