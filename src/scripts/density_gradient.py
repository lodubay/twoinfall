"""
This script plots stellar density as a function of Galactocentric radius.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from multizone_stars import MultizoneStars
from utils import get_bin_centers, Exponential
from multizone.src.models.normalize import twoinfall_ampratio
from multizone.src.models.gradient import gradient
from multizone.src.models.twoinfall import twoinfall
import paths
import _globals
import vice

def main(style='paper'):    
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, ax = plt.subplots(tight_layout=True)
    
    dr = 0.1
    rbins = np.arange(0, 20., dr)
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
    print(densities - mw_disk.thick_disk(rbins[:-1]))
    
    nomig_thin = nomig.filter({'formation_time': (4., None)})
    densities = surface_density_gradient(nomig_thin, rbins)
    ax.plot(rbin_centers[:154], densities[:154], 'g--')
    
    # Similar but using vice.history instead
    # nomig_out = vice.multioutput('../data/multizone/nomigration/twoinfall/plateau_width10/diskmodel')
    # thick_sigma = []
    # total_sigma = []
    # for i in range(154):
    #     zone = nomig_out.zones['zone%s' % i]
    #     mstar = zone.history['mstar']
    #     area = np.pi * (((i+1)*dr)**2 - (i*dr)**2)
    #     thick_sigma.append(mstar[399] / area)
    #     total_sigma.append(mstar[-1] / area)
    # ax.plot(rbin_centers[:154], total_sigma, 'r--', label='Zone-based')
    # ax.plot(rbin_centers[:154], thick_sigma, 'r:')
    
    # Plot BHG16 gradient
    # grad = np.array([gradient(r) for r in rbins])
    # integral = np.sum(grad * 2 * np.pi * rbins * 0.1)
    # grad *= _globals.M_STAR_MW / integral
    # ax.plot(rbins, grad, 'r:', label='BHG16')
    
    # Two-infall SFH with Gaussian migration scheme
    # twoinfall = MultizoneStars.from_output('gaussian/twoinfall/plateau_width10/diskmodel')
    # densities = surface_density_gradient(twoinfall, rbins)
    # ax.plot(rbin_centers, densities, 'r-', label='Gaussian migration')
    
    # Two-infall components
    # twoinfall_thick = twoinfall.filter({'formation_time': (0, 4.)})
    # densities = surface_density_gradient(twoinfall_thick, rbins)
    # ax.plot(rbin_centers, densities, 'r:')
    # twoinfall_thin = twoinfall.filter({'formation_time': (4., None)})
    # densities = surface_density_gradient(twoinfall_thin, rbins)
    # ax.plot(rbin_centers, densities, 'r--')
    
    # Inside-out SFH with analog migration scheme
    # analog = MultizoneStars.from_output('diffusion/insideout/powerlaw_slope11/diskmodel')
    # densities = surface_density_gradient(analog, rbins)
    # ax.plot(rbin_centers, densities, 'b-', label='Analog migration')
    
    ax.set_xlabel(r'$R_{\rm gal}$ [kpc]')
    ax.set_ylabel(r'$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]')
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper right', frameon=False)
    
    plt.savefig(paths.figures / 'density_gradient')
    plt.close()
    
    # Plot amplitude ratio, expected vs modeled
    fig, ax = plt.subplots(tight_layout=True)
    nomig_out = vice.multioutput('../data/multizone/nomigration/twoinfall/plateau_width10/diskmodel')
    ampratio = []
    for i in range(155):
        zone = nomig_out.zones['zone%s' % i]
        ifr = zone.history['ifr']
        ampratio.append(ifr[400] / ifr[0])
    ax.plot(rbin_centers[:155], ampratio, 'g-', label='Output')
    # Expected amplitude ratio
    ampratio = []
    for i, r in enumerate(rbins[:155]):
        ifr = twoinfall(r)
        ampratio.append(ifr.ratio)
    ax.plot(rbin_centers[:155], ampratio[:155], 'k-', label='Input')
    ax.set_xlabel('Rgal')
    ax.set_ylabel('Amplitude ratio')
    ax.legend()
    plt.savefig(paths.figures / 'ampratio')
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
