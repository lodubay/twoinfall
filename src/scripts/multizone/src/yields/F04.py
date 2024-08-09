"""
This file sets the yields according to those derived by 
Francois et al. (2004, A&A 421, 613)
"""

import warnings
# import numpy as np
import vice

# CC SNe
# Silence warnings about missing elements
warnings.filterwarnings('ignore', category=vice.ScienceWarning)
# O and Fe yields are unmodified from WW95
from vice.yields.ccsne import WW95
# WW95.set_params(m_upper=100)
# Scale Mg yields by mass range
# dm = 0.1
# masses = np.arange(8, 100, dm)
# imf = np.array([vice.imf.kroupa(m) for m in masses])
# mg_scaling = np.where(masses < 20, 7., 0.5)
# print(np.sum(dm * mg_scaling * imf) / np.sum(dm * imf))


# Read table of yields by progenitor mass
# data = np.loadtxt('francois2004_table1.dat')
# masses = data[:,0]
# imf = np.array([vice.imf.kroupa(m) for m in masses])
# Trapezoidal rule
# dm = masses[1:] - masses[:-1]
# weighted_yields = data[:,1:].T * imf
# integral = np.sum(dm * (weighted_yields[:,:-1] + weighted_yields[:,1:]) / 2, axis=1)
# integral /= np.sum([m * vice.imf.kroupa(m) * 0.1 for m in np.arange(0.1, 100.1, 0.1)])


# SNe Ia
from vice.yields.sneia import iwamoto99
iwamoto99.set_params(model='W7')
# vice.yields.sneia.settings['mg'] *= 5

# Re-allow science warnings
warnings.filterwarnings('default', category=vice.ScienceWarning)
