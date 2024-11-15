from .W24 import ccsn_ratio
import vice

# Massive star explosion fraction
Fexp = 0.75
# Mean CCSN Fe yield (Msun)
mfecc = 0.058
# Mean SN Ia Fe yield (Msun)
mfeia = 0.7
# Rate of SNe Ia per solar mass of stars
# Ria = 1.3e-3 # Maoz & Graur (2017)
Ria = 1.6e-3

# IMF-averaged CCSN yields
# yield calibration is based on Weinberg++ 2023, eq. 10
afecc = 0.45
Rcc = ccsn_ratio(Fexp=Fexp) # CCSNe per unit stellar mass
mocc = mfecc * 10 ** afecc * vice.solar_z["o"] / vice.solar_z["fe"]
vice.yields.ccsne.settings["o"] = Rcc * mocc
vice.yields.ccsne.settings["fe"] = Rcc * mfecc

# population averaged SNIa Fe yield
vice.yields.sneia.settings["o"] = 0.
vice.yields.sneia.settings["fe"] = Ria * mfeia
