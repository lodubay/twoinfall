"""
Utility functions for the custom yield presets.
"""

import vice

def ccsn_ratio(Fexp=0.75, Mmin=0.08, Mmax=120, Mthresh=8, dm=0.01, 
               imf=vice.imf.kroupa):
    r"""
    Calculate the core-collapse supernova ratio per total mass of stars.
    
    Parameters
    ----------
    Fexp : float, optional
        Fraction of massive stars which explode. The default is 0.75.
    Mmin : float, optional
        Minimum stellar mass for IMF integration. The default is 0.08.
    Mmax : float, optional
        Maximum stellar mass for IMF integration. The default is 120.
    Mthresh : float, optional
        Minimum mass of stars that explode as core-collapse supernovae.
        The default is 8.
    dm : float, optional
        Integration mass step size. The default is 0.1.
    imf : function, optional
        The initial mass function dN/dm. Must accept a single argument,
        which is stellar mass. The default is a Kroupa IMF.
        
    Returns
    -------
    float
        The core-collapse supernova ratio $R_{\rm cc}$.
    """
    # Integration masses
    masses = [m * dm + Mmin for m in range(int((Mmax + dm - Mmin) / dm))]
    N_massive_stars = sum([imf(m) * dm for m in masses if m >= Mthresh])
    total_mass = sum([m * imf(m) * dm for m in masses])
    return Fexp * N_massive_stars / total_mass
