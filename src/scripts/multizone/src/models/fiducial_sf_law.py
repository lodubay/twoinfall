"""
This file contains the class which implements the fiducial star formation law.
"""

from vice.toolkit import J21_sf_law


class fiducial_sf_law(J21_sf_law):
    r"""
    The fiducial star formation law.
    
    Parameters
    ----------
    area : real number
        The surface area in kpc^2 of the corresponding annulus in a 
        ``milkyway`` disk model.
    mode : str [default: 'sfr']
        The star formation mode in which to run VICE, one of 'sfr', 'ifr', 
        or 'gas'.
    index : real number [default : 1.5]
        The index of the power-law at gas surface densities below 
        ``Sigma_g_break``.
    Sigma_g_break : real number [default : 1.0e+08]
        The gas surface density at which there is a break in the
        Kennicutt-Schmidt relation. The star formation law is linear above this
        value. Assumes units of M$_\odot$ kpc$^{-2}$.
    **kwargs : varying types
        Keyword arguments passed to ``J21_sf_law``.
        
    Attributes and functionality are inherited from ``vice.toolkit.J21_sf_law``.
            
    Notes
    ----- 
    The default behavior of ``J21_sf_law`` is modified to produce a single
    power-law with a cutoff at high gas surface density.
    
    """
    def __init__(self, area, mode='sfr', index=1.5, Sigma_g_break=1e8, **kwargs):
        # Set index1 to be different below an arbitrarily small Sigma_g1
        # to avoid divide-by-zero errors
        super().__init__(area, mode=mode, index1=1., Sigma_g1=1., index2=index, 
                         Sigma_g2=Sigma_g_break, **kwargs)
