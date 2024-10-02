r"""
This file implements the normalization calculation in Appendix B of
Johnson et al. (2021).
"""

from ..._globals import MAX_SF_RADIUS, END_TIME, M_STAR_MW
import vice
import math as m


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
    for i in range(int(END_TIME / dt)):
        time_integral += time_dependence(i * dt) * dt * 1.e9 # yr to Gyr

    radial_integral = 0
    for i in range(int(MAX_SF_RADIUS / dr)):
        radial_integral += radial_gradient(dr * (i + 0.5)) * m.pi * (
            (dr * (i + 1))**2 - (dr * i)**2
        )
    
    return M_STAR_MW / ((1 - recycling) * radial_integral * time_integral)


def normalize_ifrmode(time_dependence, radial_gradient, tau_star, eta=0.,
                      dt = 0.01, dr = 0.1, recycling = 0.4):
    r"""
    Wrapper for ``normalize`` for models in infall mode.
    
    Parameters
    ----------
    time_dependence : <function>
        A function accepting time in Gyr specifying the time-dependence of the 
        gas infall history. Return value assumed to be unitless and 
        unnormalized.
    radial_gradient : <function>
        A function accepting galactocentric radius in kpc specifying the
        desired stellar radial surface density gradient at the present day.
        Return value assumed to be unitless and unnormalized.
    tau_star : <function>
        The star formation efficiency timescale. Accepts two parameters: 
        time in Gyr, and gas mass [Msun] or surface density [Msun kpc^-2].
        Returns a value with units of Gyr.
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
        The prefactor on the surface density of gas infall at that radius
        such that when used in simulation, the correct total stellar mass with
        the specified radial gradient is produced.
        
    """
    times, sfh = integrate_infall(time_dependence, tau_star, eta, 
                                  recycling=recycling, dt=dt)
    return normalize(sfh, radial_gradient, dt = dt, dr = dr, 
                     recycling = recycling)


def integrate_infall(time_dependence, tau_star, eta=0., recycling=0.4, dt=0.01):
    r"""
    Calculate the star formation history from a prescribed infall rate history.
    
    Parameters
    ----------
    time_dependence : <function>
        Time-dependence of the infall rate. Accepts one parameter: time in Gyr.
    tau_star : <function>
        The star formation efficiency timescale. Accepts two parameters: 
        time in Gyr, and gas mass [Msun] or surface density [Msun kpc^-2].
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
    mgas = 0
    time = 0
    sfh = []
    times = []
    while time < END_TIME:
        sfr = mgas / tau_star(time, mgas) # Msun / Gyr
        sfh.append(1.e-9 * sfr)
        mgas += time_dependence(time) * dt * 1.e9 # yr-Gyr conversion
        mgas -= sfr * dt * (1 + eta - recycling)
        times.append(time)
        time += dt
    # Interpolate the resulting list
    sfh = vice.toolkit.interpolation.interp_scheme_1d(times, sfh)
    return times, sfh
    