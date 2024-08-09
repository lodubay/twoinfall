"""
Minimal example calculating gross SN yields from a yield table.
"""

import warnings
import vice
import numpy as np

# Get WW95 Fe yield table, convert to arrays
yield_table = vice.yields.ccsne.table('fe', study='WW95')
masses = np.array(yield_table.keys())
yields = np.array(list(yield_table.todict().values()))
imf = np.array([vice.imf.kroupa(m) for m in masses])
# Trapezoidal rule
dm = masses[1:] - masses[:-1]
trap_mean = (yields[:-1] * imf[:-1] + yields[1:] * imf[1:])/2
all_masses = np.arange(0.1, 100, 0.1)
integral = np.sum(trap_mean * dm) / np.sum([m * vice.imf.kroupa(m) * 0.1 for m in all_masses])
print('The estimated gross yield is ', integral)

# Check the actual gross yield
warnings.filterwarnings('ignore', category=vice.ScienceWarning)
from vice.yields.ccsne import WW95
print('The VICE gross yield is ', vice.yields.ccsne.settings['fe'])
