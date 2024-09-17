r"""
This file declares the time-dependence of the star formation history at a
given radius under the two-infall model.
"""

from ..._globals import END_TIME
from .utils import double_exponential
from .normalize import normalize_ifrmode, twoinfall_ampratio
from .gradient import gradient
from .insideout import insideout
import math as m

FIRST_TIMESCALE = 1. # Gyr
SECOND_TIMESCALE = 10. # Gyr
SECOND_ONSET = 4.2 # Gyr

class twoinfall(double_exponential):

    def __init__(self, radius, onset=SECOND_ONSET, 
                 first_timescale=FIRST_TIMESCALE, 
                 second_timescale=SECOND_TIMESCALE,
                 dt = 0.01, dr = 0.1, outflows='default'):
        super().__init__(onset=onset, ratio=1.)
        self.first.timescale = first_timescale 
        self.second.timescale = second_timescale 
        self.ratio = twoinfall_ampratio(self, radius, onset=self.onset, 
                                        dr = dr, dt = dt, outflows=outflows)
        prefactor = normalize_ifrmode(self, gradient, radius, dt = dt,
            dr = dr, which_tau_star = 'twoinfall', outflows=outflows)
        self.first.norm *= prefactor
        self.second.norm *= prefactor

class twoinfall_var(double_exponential):

    def __init__(self, radius, onset=SECOND_ONSET, 
                 first_timescale=FIRST_TIMESCALE,
                 dt = 0.01, dr = 0.1):
        super().__init__() # dummy initial parameters
        self.onset = onset
        self.first.timescale = first_timescale 
        self.second.timescale = insideout.timescale(radius) 
        self.ratio = twoinfall_ampratio(self, radius, onset=self.onset, 
                                        dr = dr, dt = dt)
        prefactor = normalize_ifrmode(self, gradient, radius, dt = dt,
            dr = dr, which_tau_star = 'twoinfall')
        self.first.norm *= prefactor
        self.second.norm *= prefactor
