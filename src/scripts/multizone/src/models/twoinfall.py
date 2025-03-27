r"""
This file declares the time-dependence of the star formation history at a
given radius under the two-infall model.

Contents
-------
twoinfall : object
    The time-dependence of the infall rate for the two-infall model.
calculate_mstar : <function>
    Calculate the stellar mass at a given time from the star formation history.
"""

from ..._globals import END_TIME
from .utils import double_exponential
from .normalize import normalize_ifrmode, integrate_infall
from .twoinfall_sf_law import twoinfall_sf_law
from .diskmodel import BHG16
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
    diskmodel : object [default: ``.diskmodel.BHG16()``]
        Instance of a disk model object which includes ``gradient`` and 
        ``thick_to_thin_ratio`` functions that take an argument for radius.
    mass_loading : <function> [defualt: ``vice.milkyway.default_mass_loading``]
        The dimensionless mass-loading factor as a function of radius.
    onset : float [default: 4.2]
        The onset time of the second exponential infall in Gyr.
    first_timescale : float [default: 1.0]
        The timescale of the first exponential infall in Gyr.
    second_timescale : float [default: 10.0]
        The timescale of the second exponential infall in Gyr.
    vgas : float [default: 0.0]
        Radial gas velocity in kpc/Gyr. Positive for outward flow.
    dt : float [default : 0.01]
        The timestep size of the model in Gyr.
    dr : float [default : 0.1]
        The width of the annulus in kpc.
    sfe1 : float [default: 2.0]
        The pre-factor on the star formation efficiency during the 
        first infall epoch (higher value means longer SFE timescale).
    sfe2 : float [default: 1.0]
        The pre-factor on the star formation efficiency during the 
        second infall epoch (higher value means longer SFE timescale).
    
    Attributes
    ----------
    Inherits from ``utils.double_exponential``.

    Methods
    -------
    ampratio(radius) : float
        Calculate the ratio of the second infall amplitude to the first.
    """
    def __init__(
            self, 
            radius, 
            diskmodel = BHG16(),
            mass_loading = vice.milkyway.default_mass_loading,
            onset = SECOND_ONSET, 
            first_timescale = FIRST_TIMESCALE, 
            second_timescale = SECOND_TIMESCALE,
            vgas = 0.,
            dt = 0.01, 
            dr = 0.1,
            sfe1 = 2.,
            sfe2 = 1.,
            niter = 5,
    ):
        super().__init__(onset=onset, ratio=1.)
        self.first.timescale = first_timescale 
        self.second.timescale = second_timescale
        # Initialize the star formation law
        area = m.pi * ((radius + dr/2.)**2 - (radius - dr/2.)**2)
        self.tau_star = twoinfall_sf_law(
            area, onset=self.onset, sfe1=sfe1, sfe2=sfe2,
        )
        eta = mass_loading(radius)
        # Run several times to converge
        for i in range(niter):
            # Calculate amplitude ratio
            self.ratio *= self.ampratio(
                radius, diskmodel.thick_to_thin_ratio, 
                eta = eta, vgas = vgas, dt = dt
            )
            # Normalize infall rate
            prefactor = normalize_ifrmode(
                self, diskmodel.gradient, self.tau_star, radius, 
                eta = eta, vgas = vgas, dt = dt, dr = dr, recycling = 0.4
            )
            self.first.norm *= prefactor
            self.second.norm *= prefactor


    def ampratio(self, radius, disk_ratio, eta=0., 
                 vgas = 0., dt = 0.01, recycling = 0.4):
        r"""
        Calculate the ratio of the second infall amplitude to the first.
    
        Parameters
        ----------
        radius : float
            The galactocentric radius at which to evaluate the amplitude ratio.
        disk_ratio : <function>
            The ratio of the thick disk to thin disk surface density as a 
            function of radius in kpc.
        eta : float [default: 0.0]
            The outflow mass-loading factor at the given radius.
        vgas : float [default: 0.0]
            Radial gas velocity in kpc/Gyr. Positive for outward flow.
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
        times, sfh = integrate_infall(self, self.tau_star, radius, eta=eta, 
                                      vgas=vgas, recycling=recycling, dt=dt)
        mstar_final = calculate_mstar(sfh, END_TIME, dt=dt, recycling=recycling)
        mstar_onset = calculate_mstar(sfh, self.onset, dt=dt, recycling=recycling)
        ratio = disk_ratio(radius)
        return ratio**-1 * mstar_onset / (mstar_final - mstar_onset)


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
