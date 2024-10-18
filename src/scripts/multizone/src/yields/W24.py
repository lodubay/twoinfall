"""
This file implements the equilibrium-based yields of Weinberg et al. (2024).
"""

import vice

# Massive star explosion fraction
Fexp = 0.75
# Mean CCSN Fe yield (Msun)
mfecc = 0.058
# Mean SN Ia Fe yield (Msun)
mfeia = 0.7
# Fraction of solar Si produced by CC SNe
Fsicc = 0.8

# solar abundances by mass
# based on Magg et al. (2022) Table 5 + 0.04 dex to correct for diffusion
# vice.solar_z["o"] = 7.33e-3
# vice.solar_z["mg"] = 6.71e-4
# vice.solar_z["si"] = 8.51e-4
# vice.solar_z["fe"] = 1.37e-3

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

# IMF-averaged CCSN yields
# yield calibration is based on Weinberg++ 2023, eq. 10
afecc = {
    "o": 0.45,
    "mg": 0.45,
    "si": 0.35,
    "fe": 0.
}
Rcc = ccsn_ratio(Fexp=Fexp) # CCSNe per unit stellar mass
for el in ["o", "mg", "si", "fe"]:
    # yield mass per CCSN
    mcc = mfecc * 10 ** afecc[el] * vice.solar_z[el] / vice.solar_z["fe"]
    vice.yields.ccsne.settings[el] = Rcc * mcc

# population averaged SNIa Fe yield, integrated to t=infty
# for a constant SFR, will evolve to afeeq at late times
afeeq = -0.1
tau_sfh = 15
tau_Ia = 3
mu = tau_sfh / (tau_sfh - tau_Ia) # assuming tau_sfh >> minimum SN Ia delay time
Ria = (Rcc / mu) * (mfecc / mfeia) * (10 ** (afecc["o"] - afeeq) - 1.)
vice.yields.sneia.settings["fe"] = Ria * mfeia
vice.yields.sneia.settings["o"] = 0.
vice.yields.sneia.settings["mg"] = 0.
vice.yields.sneia.settings["si"] = (1 - Fsicc) / Fsicc * vice.yields.ccsne.settings["si"]
