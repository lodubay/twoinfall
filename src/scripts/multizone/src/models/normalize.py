r"""
This file implements the normalization calculation in Appendix B of
Johnson et al. (2021).
"""

from .earlyburst_tau_star import earlyburst_tau_star
from .twoinfall_sf_law import twoinfall_sf_law
from .fiducial_sf_law import fiducial_sf_law
from .mass_loading import equilibrium_mass_loading
from ..._globals import MAX_SF_RADIUS, END_TIME, M_STAR_MW, \
    THIN_DISK_SCALE_RADIUS, THICK_DISK_SCALE_RADIUS, THICK_TO_THIN_RATIO
import vice
from vice.toolkit import J21_sf_law
import math as m
import numbers


def normalize(time_dependence, radial_gradient, dt = 0.01, dr = 0.1,
              recycling = 0.4):
    r"""
    Determine the prefactor on the surface density of star formation as a
    function of time as described in Appendix A of Johnson et al. (2021).

    Parameters
    ----------
    time_dependence : <function>
        A function accepting time in Gyr specifying the time-dependence of the 
        star formation history. Return value assumed to be unitless and
        unnormalized.
    radial_gradient : <function>
        A function accepting galactocentric radius in kpc specifying the
        desired stellar radial surface density gradient at the present day.
        Return value assumed to be unitless and unnormalized.
    dt : real number [default : 0.01]
        The timestep size in Gyr.
    dr : real number [default : 0.1]
        The width of each annulus in kpc.
    recycling : real number [default : 0.4]
        The instantaneous recycling mass fraction for a single stellar
        population. Default is calculated for the Kroupa IMF [1]_.

    Returns
    -------
    A : real number
        The prefactor on the surface density of star formation at that radius
        such that when used in simulation, the correct total stellar mass with
        the specified radial gradient is produced.

    Notes
    -----
    This function automatically adopts the desired maximum radius of star
    formation, end time of the model, and total stellar mass declared in
    ``_globals.py``.

    .. [1] Kroupa (2001), MNRAS, 322, 231
    """

    time_integral = 0
    sfh = []
    for i in range(int(END_TIME / dt)):
        sfh.append(time_dependence(i * dt))
        time_integral += time_dependence(i * dt) * dt * 1.e9 # yr to Gyr
        if recycling == "continuous":
            time_integral -= continuous_recycling(sfh, dt=dt)

    radial_integral = 0
    for i in range(int(MAX_SF_RADIUS / dr)):
        radial_integral += radial_gradient(dr * (i + 0.5)) * m.pi * (
            (dr * (i + 1))**2 - (dr * i)**2
        )

    if recycling == "continuous":
        recycling = 0
    
    return M_STAR_MW / ((1 - recycling) * radial_integral * time_integral)


def normalize_ifrmode(time_dependence, radial_gradient, radius, dt = 0.01,
                      dr = 0.1, recycling = 0.4, which_tau_star='default',
                      outflows = 'default'):
    r"""
    Performs essentially the same thing as ``normalize`` but for models ran in
    infall mode.
    """
    area = m.pi * ((radius + dr/2.)**2 - (radius - dr/2.)**2)
    tau_star = {
        'fiducial': fiducial_sf_law,
        'earlyburst': earlyburst_tau_star,
        'twoinfall': twoinfall_sf_law,
    }[which_tau_star.lower()](area)
    eta = {
        'default': vice.milkyway.default_mass_loading(radius),
        'equilibrium': equilibrium_mass_loading()(radius),
        'none': 0
    }[outflows]
    times, sfh = integrate_infall(time_dependence, tau_star, eta, 
                                  recycling=recycling, dt=dt)
    sfh = vice.toolkit.interpolation.interp_scheme_1d(times, sfh)
    return normalize(sfh, radial_gradient, dt = dt, dr = dr, 
                     recycling = recycling)


def twoinfall_ampratio(time_dependence, thick_to_thin_ratio, radius, 
                       onset = 4, outflows='default',
                       dt = 0.01, dr = 0.1, recycling = 0.4):
    area = m.pi * ((radius + dr/2.)**2 - (radius - dr/2.)**2)
    tau_star = twoinfall_sf_law(area, onset=onset)
    if outflows not in ['default', 'equilibrium', 'none']:
        raise ValueError('Parameter ``outflows`` must be one of "default", \
"equilibrium", or "none".')
    eta = {
        'default': vice.milkyway.default_mass_loading(radius),
        'equilibrium': equilibrium_mass_loading()(radius),
        'none': 0.
    }[outflows]

    times, sfh = integrate_infall(time_dependence, tau_star, eta, 
                                  recycling=recycling, dt=dt)
    mstar = calculate_mstar(sfh, dt=dt, recycling=recycling)
    mstar_final = mstar[-1]
    mstar_onset = mstar[int(onset/dt)-1]
    ratio = thick_to_thin_ratio(radius)
    return ratio**-1 * mstar_onset / (mstar_final - mstar_onset)


def integrate_infall(time_dependence, tau_star, eta, recycling=0.4, dt=0.01):
    r"""
    Calculate the star formation history from a prescribed infall rate history.
    
    Parameters
    ----------
    time_dependence : <function>
        Time-dependence of the infall rate. Accepts one parameter: time in Gyr.
    tau_star : <function>
        Star formation efficiency timescale. Accepts two parameters: 
        time in Gyr, gas mass [Msun] or surface density [Msun kpc^-2].
    eta : float
        Dimensionless mass-loading factor.
    recycling : float [default: 0.4]
        Dimensionless recycling parameter.
    dt : float [default: 0.01]
        Integration timestep in Gyr.

    Returns
    -------
    times : list
        Integration times in Gyr.
    sfh : list
        Star formation rate in Msun yr^-1
        
    """
    if isinstance(recycling, str) and recycling.lower() == "continuous":
        r = 0
        continuous = True
    elif isinstance(recycling, numbers.Number):
        r = recycling
        continuous = False
    else:
        raise TypeError("Parameter 'recycling' must be a float or 'continuous'.")
    
    mgas = 0
    time = 0
    sfh = []
    times = []
    while time < END_TIME:
        sfr = mgas / tau_star(time, mgas) # Msun / Gyr
        sfh.append(1.e-9 * sfr)
        mgas += time_dependence(time) * dt * 1.e9 # yr-Gyr conversion
        mgas -= sfr * dt * (1 + eta - r)
        if continuous:
            mgas += continuous_recycling(sfh, dt=dt) * dt * 1e9
        times.append(time)
        time += dt
    return times, sfh


def calculate_mstar(sfh, dt=0.01, recycling=0.4):
    r"""
    Calculate the stellar mass at each timestep from the star formation history.

    Parameters
    ----------
    sfh : list of floats
        The star formation history in Msun/yr.
    dt : float [default: 0.01]
        The timestep in Gyr.
    recycling : float or str [default: 0.4]
        The recycling parameter. If "continuous", implements continuous
        recycling calculations; otherwise, assumes instantaneous recycling.

    Returns
    -------
    list of floats
        Stellar mass at each timestep.

    """
    if isinstance(recycling, str) and recycling.lower() == "continuous":
        r = 0
        continuous = True
    elif isinstance(recycling, numbers.Number):
        r = recycling
        continuous = False
    else:
        raise TypeError("Parameter 'recycling' must be a float or 'continuous'.")
    
    mstar = [0]
    for i in range(1, len(sfh)):
        if continuous:
            dm = (sfh[i] - continuous_recycling(sfh[:i], dt=dt)) * dt * 1e9
        else:
            dm = sfh[i] * dt * 1e9 * (1 - r)
        mstar.append(mstar[i-1] + dm)
    return mstar


def continuous_recycling(sfh, dt=0.01):
    r"""
    Numerically approximate the mass recycling rate.
    
    Parameters
    ----------
    sfh : list of floats
        The star formation history in Msun/yr.
    dt : float [default: 0.01]
        The timestep in Gyr.
    
    Returns
    -------
    float
        Recycling rate in Msun/yr.
    
    """
    recycling_rate = 0.
    crf = vice.cumulative_return_fraction # alias for readability
    for i in range(len(sfh)):
        # Increasing lookback time
        recycling_rate += sfh[-(i+1)] * (crf((i+1) * dt) - crf(i * dt))
    return recycling_rate
    