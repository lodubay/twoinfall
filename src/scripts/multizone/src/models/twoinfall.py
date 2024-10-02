r"""
This file declares the time-dependence of the star formation history at a
given radius under the two-infall model.
"""

from ..._globals import END_TIME
from .utils import double_exponential
from .normalize import normalize_ifrmode, twoinfall_ampratio
from .gradient import gradient, thick_to_thin_ratio
from .insideout import insideout
import math as m

FIRST_TIMESCALE = 1. # Gyr
SECOND_TIMESCALE = 10. # Gyr
SECOND_ONSET = 4.2 # Gyr

class twoinfall(double_exponential):
    r"""
    The infall history of the two-infall model.

    Parameters
    ----------
    radius : float
        The galactocentric radius in kpc of a given annulus in the model.
    onset : float [default: 4.2]
        The onset time of the second exponential infall in Gyr.
    first_timescale : float [default: 1.0]
        The timescale of the first exponential infall in Gyr.
    second_timescale : float [default: 10.0]
        The timescale of the second exponential infall in Gyr.
    dt : float [default : 0.01]
        The timestep size of the model in Gyr.
    dr : float [default : 0.1]
        The width of the annulus in kpc.
    
    Attributes
    ----------
    Inherits from ``utils.double_exponential``.

    """
    def __init__(self, radius, onset=SECOND_ONSET, 
                 first_timescale=FIRST_TIMESCALE, 
                 second_timescale=SECOND_TIMESCALE,
                 dt = 0.01, dr = 0.1, outflows="default"):
        super().__init__(onset=onset, ratio=1.)
        self.first.timescale = first_timescale 
        self.second.timescale = second_timescale 
        # for i in range(3):
        self.ratio = twoinfall_ampratio(self, thick_to_thin_ratio, radius,
                                        onset=self.onset, 
                                        dr = dr, dt = dt, outflows=outflows)
        prefactor = normalize_ifrmode(self, gradient, radius, dt = dt,
            dr = dr, which_tau_star = "twoinfall", outflows=outflows)
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
            dr = dr, which_tau_star = "twoinfall")
        self.first.norm *= prefactor
        self.second.norm *= prefactor
