r"""
This file declares the time-dependence of the star formation history at a
given radius under the two-infall model.
"""

from ..._globals import END_TIME
from .utils import double_exponential
from .normalize import normalize_ifrmode, integrate_infall
from .gradient import gradient, thick_to_thin_ratio
from .twoinfall_sf_law import twoinfall_sf_law
import vice
import math as m

FIRST_TIMESCALE = 1. # Gyr
SECOND_TIMESCALE = 15. # Gyr
SECOND_ONSET = 4.2 # Gyr

class twoinfall(double_exponential):
    r"""
    The infall history of the two-infall model.

    Parameters
    ----------
    radius : float
        The galactocentric radius in kpc of a given annulus in the model.
    mass_loading : <function> [defualt: ``vice.milkyway.default_mass_loading``]
        The dimensionless mass-loading factor as a function of radius.
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

    Methods
    -------
    ampratio(radius) : float
        Calculate the ratio of the second infall amplitude to the first.
    normalize(radius) : float
        Normalize the infall rate according to the desired surface density.
    """
    def __init__(self, radius, mass_loading=vice.milkyway.default_mass_loading, 
                 onset=SECOND_ONSET, 
                 first_timescale=FIRST_TIMESCALE, 
                 second_timescale=SECOND_TIMESCALE,
                 dt = 0.01, dr = 0.1):
        super().__init__(onset=onset, ratio=1.)
        self.first.timescale = first_timescale 
        self.second.timescale = second_timescale
        # Calculate amplitude ratio
        self.ratio = self.ampratio(radius, mass_loading = mass_loading, 
                                   dr = dr, dt = dt)
        # Normalize infall rate
        prefactor = self.normalize(radius, mass_loading = mass_loading, 
                                   dt = dt,  dr = dr)
        self.first.norm *= prefactor
        self.second.norm *= prefactor


    def ampratio(self, radius, mass_loading = vice.milkyway.default_mass_loading, 
                 dt = 0.01, dr = 0.1, recycling = 0.4):
        r"""
        Calculate the ratio of the second infall amplitude to the first.
    
        Parameters
        ----------
        radius : float
            The galactocentric radius at which to evaluate the amplitude ratio.
        mass_loading : <function> [default: ``vice.milkyway.default_mass_loading``]
            The dimensionless mass-loading factor as a function of radius.
        dt : real number [default : 0.01]
            The timestep size in Gyr.
        dr : real number [default : 0.1]
            The width of each annulus in kpc.
        recycling : real number [default : 0.4]
            The instantaneous recycling mass fraction for a single stellar
            population. The default is calculated for the Kroupa IMF.
    
        Returns
        -------
        float
            The amplitude ratio between the second and first infalls.
        """
        area = m.pi * ((radius + dr/2.)**2 - (radius - dr/2.)**2)
        tau_star = twoinfall_sf_law(area, onset=self.onset)
        eta = mass_loading(radius)
        times, sfh = integrate_infall(self, tau_star, eta, 
                                      recycling=recycling, dt=dt)
        mstar_final = calculate_mstar(sfh, END_TIME, dt=dt, recycling=recycling)
        mstar_onset = calculate_mstar(sfh, self.onset, dt=dt, recycling=recycling)
        ratio = thick_to_thin_ratio(radius)
        return ratio**-1 * mstar_onset / (mstar_final - mstar_onset)
    
    
    def normalize(self, radius, mass_loading = vice.milkyway.default_mass_loading, 
                  dt = 0.01, dr = 0.1, recycling = 0.4):
        r"""
        Normalize the infall rate according to the desired surface density.
        
        Parameters
        ----------
        radius : float
            The galactocentric radius at which to normalize the gas infall.
        mass_loading : <function> [default: ``vice.milkyway.default_mass_loading``]
            The dimensionless mass-loading factor as a function of radius.
        dt : real number [default : 0.01]
            The timestep size in Gyr.
        dr : real number [default : 0.1]
            The width of each annulus in kpc.
        recycling : real number [default : 0.4]
            The instantaneous recycling mass fraction for a single stellar
            population. The default is calculated for the Kroupa IMF.
        """
        area = m.pi * ((radius + dr/2.)**2 - (radius - dr/2.)**2)
        tau_star = twoinfall_sf_law(area, onset=self.onset)
        eta = mass_loading(radius)
        return normalize_ifrmode(self, gradient, tau_star, eta = eta,
                                 dt = dt, dr = dr, recycling = recycling)


def calculate_mstar(sfh, time, dt=0.01, recycling=0.4):
    r"""
    Calculate the stellar mass at the given time from the star formation history.

    Parameters
    ----------
    sfh : <function>
        The star formation history in Msun/yr as a function of time in Gyr.
    time : float
        The time in Gyr at which to calculate the total stellar mass.
    dt : float [default: 0.01]
        The timestep in Gyr.
    recycling : float [default: 0.4]
        The dimensionless recycling parameter.

    Returns
    -------
    lloat
        Stellar mass at the given time.

    """
    mstar = 0
    for i in range(int(time / dt)):
        mstar += sfh(i * dt) * dt * 1e9 * (1 - recycling)
    return mstar
