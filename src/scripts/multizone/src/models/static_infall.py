r"""
This file declares the time-dependence of the star formation history at a
given radius in the constant SFH model from Johnson et al. (2021).
"""

from .utils import constant
from .normalize import normalize_ifrmode
from .diskmodel import BHG16
from .fiducial_sf_law import fiducial_sf_law
import math as m
import vice


class static_infall(constant):

    r"""
    The constant SFH model from Johnson et al. (2021).

    Parameters
    ----------
    radius : float
        The galactocentric radius in kpc of a given annulus in the model.
    mass_loading : <function> [defualt: ``vice.milkyway.default_mass_loading``]
        The dimensionless mass-loading factor as a function of radius.
    diskmodel : object [default: ``.diskmodel.BHG16()``]
        Instance of a disk model object which includes a ``gradient`` function 
        that takes an argument for radius.
    vgas : float [default: 0.0]
        Radial gas velocity in kpc/Gyr. Positive for outward flow.
    dt : float [default : 0.01]
        The timestep size of the model in Gyr.
    dr : float [default : 0.1]
        The width of the annulus in kpc.

    All attributes and functionality are inherited from ``constant`` declared
    in ``src/simulations/models/utils.py``.
    """

    def __init__(self, radius, dt = 0.01, dr = 0.1, vgas = 0.,
                 mass_loading=vice.milkyway.default_mass_loading,
                 diskmodel = BHG16()):
        super().__init__()
        area = m.pi * ((radius + dr/2.)**2 - (radius - dr/2.)**2)
        tau_star = fiducial_sf_law(area)
        eta = mass_loading(radius)
        self.amplitude *= normalize_ifrmode(
            self, 
            diskmodel.gradient, 
            tau_star, 
            radius, 
            eta = eta, 
            vgas = vgas, 
            dt = dt, 
            dr = dr
        )
