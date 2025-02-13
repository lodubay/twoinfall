"""
This file declares the time-dependence of the infall rate for a variant of
the two-infall model with a radially-dependent second infall timescale.
"""

import math as m
from .twoinfall import twoinfall

_SCALE_RADIUS_ = 7 # kpc
_SOLAR_TIMESCALE_ = 15 # Gyr

class twoinfall_expvar(twoinfall):
    """
    Variant of the two-infall SFH with a radially-dependent second infall 
    timescale.
    
    Parameters
    ----------
    radius : float
        The galactocentric radius in kpc of a given annulus in the model.
    
    Other parameters, arguments, and functionality are inherited from 
    ``twoinfall``.
    """
    def __init__(self, radius, **kwargs):
        super().__init__(
            radius, second_timescale=self.timescale(radius), **kwargs
        )
    
    @staticmethod
    def timescale(radius):
        """
        Timescale for the second infall which increases exponentially with radius.
        
        Parameters
        ----------
        radius : float
            Galactocentric radius in kpc.

        Returns
        -------
        float
            Timescale of the second infall in Gyr.
        """
        return _SOLAR_TIMESCALE_ * m.exp((radius - 8) / _SCALE_RADIUS_)
