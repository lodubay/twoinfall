"""
This file sets the Solar-scaled nucleosynthetic yields according to y/Z=3.
"""
import vice

SOLAR_SCALE = 3.0 # y_O / Z_O,Sun
AFE_CC = 0.45 # CCSN [a/Fe] plateau
YIA_SCALE = 1.15 # arbitrary scaling of yIa to adjust chemical evolution endpoint

# IMF-averaged CCSN yields
vice.yields.ccsne.settings["o"] = SOLAR_SCALE * vice.solar_z["o"]
vice.yields.ccsne.settings["fe"] = SOLAR_SCALE * 10**-AFE_CC * vice.solar_z["fe"]

# population averaged SNIa Fe yield
vice.yields.sneia.settings["o"] = 0.
vice.yields.sneia.settings["fe"] = YIA_SCALE * SOLAR_SCALE * (1 - 10**-AFE_CC) * vice.solar_z["fe"]
