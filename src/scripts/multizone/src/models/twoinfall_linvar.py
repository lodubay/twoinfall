"""
This file declares the time-dependence of the infall rate for a variant of
the two-infall model with a radially-dependent second infall timescale.
"""

from .twoinfall import twoinfall

class twoinfall_linvar(twoinfall):
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
        """
        Timescale for the second infall which increases linearly with radius.
        
        Parameters
        ----------
        radius : float
            Galactocentric radius in kpc.

        Returns
        -------
        float
            Timescale of the second infall in Gyr (minimum of 1.83 Gyr).
        
        Notes
        ----
        Follows the prescription from Chiappinni et al. (2001).
        """
        return 1.033 * max(radius, 3) - 1.27
