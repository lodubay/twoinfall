"""
Created on Tue Oct 24 10:12:20 2023

@author: dubay.11
"""

import vice

# solar abundances by mass
# based on Magg et al. (2022) Table 5 + 0.04 dex to correct for diffusion
SolarO=7.33e-3
SolarMg=6.71e-4
SolarFe=1.37e-3

# IMF-averaged CCSN yields
# yield calibration is based on Weinberg++ 2023, eq. 11
afecc=0.45              # plateau value for [alpha/Fe]
mocc=0.973*SolarO*(0.00137/SolarFe)*(10**(afecc-0.45))  # CCSN oxygen
mfecc=mocc*(SolarFe/SolarO)*(10**(-afecc))              # CCSN iron
mmgcc=mocc*SolarMg/SolarO                               # CCSN magnesium

# population averaged SNIa Fe yield, integrated to t=infty
# for a constant SFR, will evolve to afeeq at late times
afeeq=0.05
mfeIa=mfecc*(10.**(afecc-afeeq) - 1.)

vice.yields.ccsne.settings["o"] = mocc
vice.yields.ccsne.settings["mg"] = mmgcc
vice.yields.ccsne.settings["fe"] = mfecc
vice.yields.sneia.settings["o"] = 0.
vice.yields.sneia.settings["mg"] = 0.
vice.yields.sneia.settings["fe"] = mfeIa
