r"""
This file declares the time-dependence of the infall rate at a
given radius in the early-burst model.
"""

from .utils import exponential, get_bin_number, interpolate
from .normalize import normalize_ifrmode
from .gradient import gradient
from .insideout import _read_sanchez_data

class earlyburst_ifr(exponential):
    r"""
    The early-burst IFR model.

    Parameters
    ----------
    radius : float
        The galactocentric radius in kpc of a given annulus in the model.
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
    def __init__(self, radius, dt = 0.01, dr = 0.1):
        super().__init__(timescale = self.timescale(radius))
        self.norm *= normalize_ifrmode(self, gradient, radius, dt = dt, dr = dr,
                                       which_tau_star = "earlyburst")

    @staticmethod
    def timescale(radius, Re = 5):
        r"""
        Determine the timescale of star formation at a given radius reported
        by Sanchez (2020) [1]_.

        Parameters
        ----------
        radius : real number
            Galactocentric radius in kpc.
        Re : real number [default : 5]
            The effective radius (i.e. half-mass radius) of the galaxy in kpc.

        Returns
        -------
        tau_sfh : real number
            The e-folding timescale of the star formation history at that
            radius. The detailed time-dependence on the star formation history
            has the following form:

            .. math:: \dot{M}_\star \sim
                (1 - e^{-t / \tau_\text{rise}})e^{-t / \tau_\text{sfh}}

            where :math:`\tau_\text{rise}` = 2 Gyr and :math:`\tau_\text{sfh}`
            is the value returned by this function.

        .. [1] Sanchez (2020), ARA&A, 58, 99
        """
        radius /= Re # convert to units of Re
        radii, timescales = _read_sanchez_data()
        idx = get_bin_number(radii, radius)
        if idx != -1:
            return interpolate(radii[idx], timescales[idx], radii[idx + 1],
                timescales[idx + 1], radius)
        else:
            return interpolate(radii[-2], timescales[-2], radii[-1],
                timescales[-1], radius)
