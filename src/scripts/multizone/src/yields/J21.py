"""
This file contains the fiducial yields from Johnson et al. (2021). They are
largely the same as VICE's "JW20" yields, but with a slightly higher SN Ia
Fe yield.
"""

import vice

vice.yields.ccsne.settings["o"] = 0.015
vice.yields.ccsne.settings["fe"] = 0.0012
vice.yields.sneia.settings["o"] = 0
vice.yields.sneia.settings["fe"] = 0.00214
