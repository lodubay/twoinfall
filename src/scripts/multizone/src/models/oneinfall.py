r"""
This file declares the time-dependence of the infall rate at a
given radius in the early-burst model.
"""

from .utils import exponential
from .normalize import normalize_ifrmode
from .fiducial_sf_law import fiducial_sf_law
from .gradient import gradient
from .insideout import insideout
import vice
import math as m

class oneinfall(exponential):
    r"""
    The early-burst IFR model.

    Parameters
    ----------
    radius : float
        The galactocentric radius in kpc of a given annulus in the model.
    vgas : float [default: 0.0]
        Radial gas velocity in kpc/Gyr. Positive for outward flow.
    dt : float [default : 0.01]
        The timestep size of the model in Gyr.
    dr : float [default : 0.1]
        The width of the annulus in kpc.

    Functions
    ---------
    - timescale [staticmethod]

    Other atributes and functionality are inherited from
    ``modified_exponential`` declared in ``src/simulations/models/utils.py``.
    """
    def __init__(self, radius, vgas = 0., dt = 0.01, dr = 0.1,
                 mass_loading = vice.milkyway.default_mass_loading):
        super().__init__(timescale = insideout.timescale(radius))
        area = m.pi * ((radius + dr/2.)**2 - (radius - dr/2.)**2)
        tau_star = fiducial_sf_law(area, onset=self.onset)
        eta = mass_loading(radius)
        self.norm *= normalize_ifrmode(self, gradient, tau_star, radius, 
                                       eta=eta, vgas = vgas, dt = dt, dr = dr)
