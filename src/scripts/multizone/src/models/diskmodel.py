r"""
This file contains analytical models for the stellar surface density of the
Milky Way disk at late times.

Contents
--------
two_component_disk : object
    A double exponential disk.
"""

import math as m
from .utils import double_exponential
from ..._globals import MAX_SF_RADIUS, M_STAR_MW, \
    THIN_DISK_SCALE_RADIUS, THICK_DISK_SCALE_RADIUS, THICK_TO_THIN_RATIO


class two_component_disk(double_exponential):
    r"""
    A two-component model for the Milky Way stellar disk.
    
    Parameters
    ----------
    ratio : float [default: 0.27]
        Ratio of thick disk to thin disk surface mass density at R=0.
    mass : float [default: 5.17e10]
        Total mass of the disk in Msun.
    rs_thin : float [default: 2.5]
        Thin disk scale radius in kpc.
    rs_thick : float [default: 2.0]
        Thick disk scale radius in kpc.
    rmax : float [default: 15.5]
        Maximum radius of star formation in kpc.
    
    Attributes
    ----------
    Inherits from ``utils.double_exponential``.
    
    Calling
    -------
    Returns the total surface density at the given Galactic radius in kpc.
    
    Methods
    -------
    normalize(rmax, dr=0.1)
        Normalize the total disk mass to 1.
    thick_disk(radius)
        The surface mass density of the thick disk at the given radius.
    thin_disk(radius)
        The surface mass density of the thick disk at the given radius.
    thick_to_thin(radius)
        The ratio of thick to thin disk surface mass density.

    """
    def __init__(self, ratio=THICK_TO_THIN_RATIO, 
                 mass=M_STAR_MW, 
                 rs_thin=THIN_DISK_SCALE_RADIUS,
                 rs_thick=THICK_DISK_SCALE_RADIUS,
                 rmax=MAX_SF_RADIUS):
        super().__init__(onset=0., ratio=1./ratio)
        self.first.timescale = rs_thick
        self.second.timescale = rs_thin
        norm = self.normalize(rmax)
        self.first.norm *= mass * norm
        self.second.norm *= mass * norm
        
    def normalize(self, rmax, dr=0.1):
        """
        Normalize the total mass of the disk to 1.

        Parameters
        ----------
        rmax : float
            Maximum radius of integration in kpc.
        dr : float [default: 0.1]
            Integration step in kpc.

        Returns
        -------
        float
            Normalization coefficient [kpc^-2].

        """
        integral = 0
        for i in range(int(rmax / dr)):
            integral += self.__call__(dr * (i + 0.5)) * m.pi * (
                (dr * (i + 1))**2 - (dr * i)**2
            )
        return 1 / integral
    
    def gradient(self, radius):
        r"""
        The unitless, un-normalized stellar surface density gradient.
        
        Parameters
        ----------
        radius : float
            Galactic radius in kpc.
        
        Returns
        -------
        float
            The value of $g(R_{\rm gal})$ as defined in Appendix B of
            Johnson et al. (2021).
        
        """
        return self.__call__(radius) / self.__call__(0)
    
    def thick_disk(self, radius):
        """
        The surface mass density of the thick disk at the given radius.
        
        Parameters
        ----------
        radius : float
            Galactic radius in kpc.
        
        Returns
        -------
        float
            Thick disk surface mass density.
        
        """
        return self.first(radius)
    
    def thin_disk(self, radius):
        """
        The surface mass density of the thin disk at the given radius.
        
        Parameters
        ----------
        radius : float
            Galactic radius in kpc.
        
        Returns
        -------
        float
            Thin disk surface mass density.
        
        """
        return self.ratio * self.second(radius)
        
    def thick_to_thin_ratio(self, radius):
        """
        Calculate the ratio of surface mass density between the components.
        
        Parameters
        ----------
        radius : float
            Galactic radius in kpc.
        
        Returns
        -------
        float
            The thick disk surface mass density divided by the thin disk.
        
        """
        return self.thick_disk(radius) / self.thin_disk(radius)


class BHG16(two_component_disk):
    """
    Subclass of ``two_component_disk`` which adopts the disk parameters from
    Bland-Hawthorn & Gerhard (2016).
    """
    def __init__(self):
        super().__init__(ratio=0.27, rs_thin=2.5, rs_thick=2.0)


class Palla20(two_component_disk):
    """
    Subclass of ``two_component_disk`` which adopts the disk parameters from
    the Palla et al. (2020) multizone GCE model.
    
    Parameters
    ----------
    sigma_thin : float [default: 54.0]
        Surface mass density of the thin disk in the Solar neighborhood in 
        Msun pc^-2.
    sigma_thick : float [default: 12.3]
        Surface mass density of the thick disk in the Solar neighborhood in 
        Msun pc^-2.
    rs_thin : float [default: 3.5]
        Thin disk scale radius in kpc.
    rs_thick : float [default: 2.3]
        Thick disk scale radius in kpc.
    R0 : float [default: 8.0]
        Galactocentric radius of the Sun in kpc.
    """
    def __init__(self, sigma_thin=54.0, sigma_thick=12.3, rs_thin=3.5, 
                 rs_thick=2.3, R0=8.0):
        # Stellar surface density ratio at the Solar neighborhood
        solar_ratio = sigma_thick / sigma_thin
        # Set the central thick-to-thin ratio
        central_ratio = solar_ratio * m.exp(-R0 * (1 / rs_thin - 1 / rs_thick))
        # Initialize with scale radii
        super().__init__(rs_thin=3.5, rs_thick=2.3, ratio=central_ratio)
        # Set the central surface mass density of the thin disk
        sigma_thin_center = sigma_thin * m.exp(R0 / rs_thin)
        self.second.norm = central_ratio * sigma_thin_center * 1e6 # Msun kpc^-2
        # Set the density in the Solar neighborhood of the thick disk
        self.first.norm *= sigma_thick * 1e6 / self.thick_disk(8)
