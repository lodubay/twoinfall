"""
This file declares the time-dependence of the infall rate for a variant of
the two-infall model with a radially-dependent second infall timescale.
"""

from .insideout import insideout
from .twoinfall import twoinfall

class twoinfall_var(twoinfall):
    """
    Variant of the two-infall SFH with a radially-dependent second infall 
    timescale.
    
    Parameters
    ----------
    radius : float
        The galactocentric radius in kpc of a given annulus in the model.
    Re : float [default: 5]
        Effective radius of the Galaxy in kpc.
    
    Other parameters, arguments, and functionality are inherited from 
    ``twoinfall``.
    """
    def __init__(self, radius, **kwargs):
        super().__init__(
            radius, second_timescale=self.timescale(radius), **kwargs
        )
    
    @staticmethod
    def timescale(radius):
        return max(1.03 * radius - 1.27, 1.82)
