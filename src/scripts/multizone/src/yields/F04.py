"""
This file sets the yields according to those derived by 
Francois et al. (2004, A&A 421, 613)
"""

import warnings
from pathlib import Path
import numpy as np
import vice

# Silence warnings about missing elements
warnings.filterwarnings('ignore', category=vice.ScienceWarning)

# CC SNe
# O yield is unmodified from Woosley & Weaver (1995)
from vice.yields.ccsne import WW95
# WW95.set_params(m_upper=100)

# Read table of yields by progenitor mass
data = np.loadtxt(Path(__file__).parents[0] / 'francois2004_table1.dat')
masses = data[:,0]
imf = np.array([vice.imf.kroupa(m) for m in masses])
# Trapezoidal rule
dm = masses[1:] - masses[:-1]
weighted_yields = data[:,1:].T * imf
integral = np.sum(dm * (weighted_yields[:,:-1] + weighted_yields[:,1:]) / 2, axis=1)
integral /= np.sum([m * vice.imf.kroupa(m) * 0.1 for m in np.arange(0.1, 100.1, 0.1)])
# Assign yields
vice.yields.ccsne.settings['mg'] = integral[0]
vice.yields.ccsne.settings['si'] = integral[1]
vice.yields.ccsne.settings['fe'] = integral[2]

# SNe Ia
from vice.yields.sneia import iwamoto99
iwamoto99.set_params(model='W7')
vice.yields.sneia.settings['mg'] *= 5

# Re-allow science warnings
warnings.filterwarnings('default', category=vice.ScienceWarning)
