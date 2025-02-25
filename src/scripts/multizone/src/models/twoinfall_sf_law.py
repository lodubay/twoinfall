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
    sfe1 : real number [default: 2.]
        The multiplicative factor on the star formation efficiency in Gyr^-1
        during the first gas infall epoch. Note that a higher value leads to
        a shorter SFE timescale.
    sfe2 : real number [default: 1.]
        The multiplicative factor on the star formation efficiency in Gyr^-1
        during the second gas infall epoch. Note that a higher value leads to
        a shorter SFE timescale.
    **kwargs : varying types
        Keyword arguments passed to ``fiducial_sf_law``.
    
    Attributes
    ----------
    onset : real number [default : 4.0]
        The start time of the second gas infall epoch in Gyr.
    sfe1 : real number [default: 2.]
        The multiplicative factor on the star formation efficiency in Gyr^-1
        during the first gas infall epoch. Note that a higher value leads to
        a shorter SFE timescale.
    sfe2 : real number [default: 1.]
        The multiplicative factor on the star formation efficiency in Gyr^-1
        during the second gas infall epoch. Note that a higher value leads to
        a shorter SFE timescale.
    
    Other attributes and functionality are inherited from ``fiducial_sf_law``.
    
    Calling
    -------
    Calling this object works similarly to ``J21_sf_law``, with the difference 
    that for ``time`` < ``onset`` (i.e., during the first infall), the value 
    of $\tau_\star$ is divided by ``sfe1`` [default: 2.0], and for 
    ``time`` >= ``onset`` it is divided by ``sfe2`` [default: 1.0].
    
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
    def __init__(self, area, onset=4.2, sfe1=2., sfe2=1., **kwargs):
        super().__init__(area, mode="ifr", **kwargs)
        self.onset = onset
        self.sfe1 = sfe1
        self.sfe2 = sfe2
    
    def __call__(self, time, mgas):
        if time < self.onset:
            prefactor = 1. / self.sfe1
        else:
            prefactor = 1. / self.sfe2
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
    def sfe1(self):
        """
        float
            Multiplicative factor on the star formation efficiency
            during the first gas infall epoch.
        """
        return self._sfe1
    
    @sfe1.setter
    def sfe1(self, value):
        if isinstance(value, numbers.Number):
            if value > 0:
                self._sfe1 = value
            else:
                raise ValueError("Attribute ``sfe1`` must be positive.")
        else:
            raise TypeError("Attribute ``sfe1`` must be a number. Got:", 
                            type(value))
            
    @property
    def sfe2(self):
        """
        float
            Multiplicative factor on the star formation efficiency
            during the second gas infall epoch.
        """
        return self._sfe2
    
    @sfe2.setter
    def sfe2(self, value):
        if isinstance(value, numbers.Number):
            if value > 0:
                self._sfe2 = value
            else:
                raise ValueError("Attribute ``sfe2`` must be positive.")
        else:
            raise TypeError("Attribute ``sfe2`` must be a number. Got:", 
                            type(value))
