"""
This file contains the class which implements a star formation law for the
two-infall model.
"""
import numbers
from .fiducial_sf_law import fiducial_sf_law

class twoinfall_sf_law(fiducial_sf_law):
    r"""
    The star formation law for the two-infall model.
    
    Parameters
    ----------
    area : real number
        The surface area in kpc^2 of the corresponding annulus in a 
        ``milkyway`` disk model.
    onset : real number [default : 4.0]
        The start time of the second gas infall epoch in Gyr.
    factor : real number [default : 0.5]
        The multiplicative factor on the star formation efficiency timescale
        during the first gas infall epoch.
    **kwargs : varying types
        Keyword arguments passed to ``fiducial_sf_law``.
    
    Attributes
    ----------
    onset : real number [default : 4.0]
        The start time of the second gas infall epoch in Gyr.
    factor : real number [default : 0.5]
        The multiplicative factor on the star formation efficiency timescale
        during the first gas infall epoch.
    
    Other attributes and functionality are inherited from ``fiducial_sf_law``.
    
    Calling
    -------
    Calling this object works similarly to ``J21_sf_law``, with the difference 
    that for ``time`` < ``onset`` (i.e., during the first infall), the value 
    of $\tau_\star$ is multiplied by ``factor`` [default: 0.5].
    
    Parameters:
        - time : real number
            Simulation time in Gyr.
        - mgas : real number
            Gas supply in M$_\odot$. Will be called by VICE directly.
    
    Returns:
        - tau_star : real number
            The star formation efficiency timescale in Gyr.
            
    Notes
    ----- 
    The default behavior of ``J21_sf_law`` is modified to produce a single
    power-law with a cutoff at high gas surface density. The time-dependent
    component is the molecular gas timescale, but during the first gas infall
    epoch the star formation efficiency timescale is multiplied by a factor
    [default: 0.5] as in, e.g., Nissen et al. (2020).
    
    """
    def __init__(self, area, onset=4.2, factor=0.5, **kwargs):
        super().__init__(area, mode="ifr", **kwargs)
        self.onset = onset
        self.factor = factor
    
    def __call__(self, time, mgas):
        if time < self.onset:
            prefactor = self.factor
        else:
            prefactor = 1.
        return prefactor * super().__call__(time, mgas)
        
    @property
    def onset(self):
        """
        float
            Start time of the second gas infall epoch in Gyr.
        """
        return self._onset
    
    @onset.setter
    def onset(self, value):
        if isinstance(value, numbers.Number):
            if value > 0:
                self._onset = value
            else:
                raise ValueError("Attribute ``onset`` must be positive.")
        else:
            raise TypeError("Attribute ``onset`` must be a number. Got:", 
                            type(value))
            
    @property
    def factor(self):
        """
        float
            Multiplicative factor on the star formation efficiency timescale
            during the first gas infall epoch.
        """
        return self._factor
    
    @factor.setter
    def factor(self, value):
        if isinstance(value, numbers.Number):
            if value > 0:
                self._factor = value
            else:
                raise ValueError("Attribute ``factor`` must be positive.")
        else:
            raise TypeError("Attribute ``factor`` must be a number. Got:", 
                            type(value))
