r"""
This file contains analytical models for the Milky Way disk.

Contents
--------
two_component_disk : object
    A double exponential disk.
"""

import math as m
from .utils import double_exponential
from ..._globals import MAX_SF_RADIUS, END_TIME, M_STAR_MW, \
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
        
    def thick_to_thin(self, radius):
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
