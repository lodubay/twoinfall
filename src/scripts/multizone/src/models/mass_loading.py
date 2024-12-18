import math as m
import vice

class equilibrium_mass_loading:
    """
    An exponential outflow mass-loading parameter.
    
    Tuned to produce the given equilibrium abundance at the Solar radius
    assuming an exponential star formation history
    
    Parameters
    ----------
    equilibrium : float [default: 0.0]
        The equilibrium [O/H] abundance at the Solar radius.
    recycling : float [default: 0.4]
        The instantaneous recycling parameter.
    tau_star : float [default: 3.0]
        The star formation efficiency timescale in Gyr.
    tau_sfh : float [default: 15.0]
        The star formation history timescale in Gyr.
    gradient : float [default: -0.08]
        The radial metallicity gradient in dex/kpc.
    
    Attributes
    ----------
    _eta_sun : float
        The mass-loading factor at the Solar radius.
    _scale_radius : float
        The exponential scale radius for the mass-loading factor in kpc.
    
    Calling
    -------
    Returns the value of the mass-loading factor at the given radius.
    
    Parameters:
        - radius : float
            Galactocentric radius in kpc.
    
    Returns:
        - eta : float
            The mass-loading factor at the given radius.
    """
    def __init__(self, equilibrium=0., recycling=0.4, tau_star=3., tau_sfh=15.,
                 gradient=-0.08):
        # Calculate eta for exponential SFH to reach desired equilibrium [O/H]
        Z_alpha_eq = vice.solar_z["o"] * 10 ** equilibrium
        yield_ratio = vice.yields.ccsne.settings["o"] / Z_alpha_eq
        self._eta_sun = yield_ratio - 1 + recycling + tau_star / tau_sfh
        self._scale_radius = -1 / (gradient * m.log(10))
        
    def __call__(self, radius):
        return self._eta_sun * m.exp((radius - 8) / self._scale_radius)


def no_outflows(radius):
    """A dummy function returning 0 for all inputs."""
    return 0.
