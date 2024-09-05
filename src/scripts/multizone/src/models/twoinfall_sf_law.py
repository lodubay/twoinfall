"""
This file contains the class which implements a star formation law for the
two-infall model.
"""
import numbers
from vice.toolkit import J21_sf_law

class twoinfall_sf_law(J21_sf_law):
    r"""
    The star formation law for the two-infall model.
    
    Inherits functionality from ``vice.toolkit.J21_sf_law``.
    
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
    index : real number [default : 1.5]
        The index of the power-law at gas surface densities below 
        ``Sigma_g_break``.
    Sigma_g_break : real number [default : 1.0e+08]
        The gas surface density at which there is a break in the
        Kennicutt-Schmidt relation. The star formation law is linear above this
        value. Assumes units of M$_\odot$ kpc$^{-2}$.
    **kwargs : varying types
        Keyword arguments passed to ``J21_sf_law``.
    
    Attributes
    ----------
    onset : real number [default : 4.0]
        The start time of the second gas infall epoch in Gyr.
    factor : real number [default : 0.5]
        The multiplicative factor on the star formation efficiency timescale
        during the first gas infall epoch.
    
    Other attributes are inherited from ``J21_sf_law``.
    
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
    def __init__(self, area, onset=3.5, factor=0.5, index=1.5, Sigma_g_break=1e8,
                 **kwargs):
        super().__init__(area, mode="ifr", index1=index, index2=index, 
                         Sigma_g2=Sigma_g_break, **kwargs)
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
