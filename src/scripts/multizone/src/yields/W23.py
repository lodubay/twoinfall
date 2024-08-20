"""
This file implements the equilibrium-based yields of Weinberg et al. (2023).
"""

import vice
from .utils import ccsn_ratio

# Massive star explosion fraction
Fexp = 0.75
# Mean CCSN Fe yield (Msun)
mfecc = 0.058
# Mean SN Ia Fe yield (Msun)
mfeia = 0.7

# solar abundances by mass
# based on Magg et al. (2022) Table 5 + 0.04 dex to correct for diffusion
vice.solar_z["o"] = 7.33e-3
vice.solar_z["mg"] = 6.71e-4
vice.solar_z["si"] = 8.51e-4
vice.solar_z["fe"] = 1.37e-3

# IMF-averaged CCSN yields
# yield calibration is based on Weinberg++ 2023, eq. 10
afecc = {
    "o": 0.45,
    "mg": 0.45,
    "si": 0.36,
    "fe": 0.
}
Rcc = ccsn_ratio(Fexp=Fexp) # CCSNe per unit stellar mass
for el in ["o", "mg", "si", "fe"]:
    # yield mass per CCSN
    mcc = mfecc * 10 ** afecc[el] * vice.solar_z[el] / vice.solar_z["fe"]
    vice.yields.ccsne.settings[el] = Rcc * mcc

# population averaged SNIa Fe yield, integrated to t=infty
# for a constant SFR, will evolve to afeeq at late times
afeeq = 0.
tau_sfh = 15
tau_Ia = 1.5
mu = tau_sfh / (tau_sfh - tau_Ia) # assuming tau_sfh >> minimum SN Ia delay time
Ria = (Rcc / mu) * (mfecc / mfeia) * (10 ** (afecc["o"] - afeeq) - 1.)
vice.yields.sneia.settings["fe"] = Ria * mfeia
vice.yields.sneia.settings["o"] = 0.
vice.yields.sneia.settings["mg"] = 0.
vice.yields.sneia.settings["si"] = 0.
