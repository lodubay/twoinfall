import math as m
import vice
from ..._globals import ETA_SCALE_RADIUS


class exponential_mass_loading:
    r"""
    An exponential outflow mass-loading parameter $\eta$.

    Parameters
    ----------
    See Attributes.
    
    Attributes
    ----------
    solar_value : float
        Value of the mass-loading factor for the Solar annulus.
    scale_radius : float
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
    def __init__(self, solar_value, scale_radius):
        self.solar_value = solar_value
        self.scale_radius = scale_radius

    def __call__(self, radius):
        return self.solar_value * m.exp((radius - 8.0) / self.scale_radius)


class yZ1(exponential_mass_loading):
    """Subclass of ``exponential_mass_loading`` tuned to the yZ1 yields."""
    def __init__(self):
        super().__init__(0.2, 4.0)


class yZ2(exponential_mass_loading):
    """Subclass of ``exponential_mass_loading`` tuned to the yZ2 yields."""
    def __init__(self):
        super().__init__(1.4, 5.0)


class equilibrium_mass_loading(exponential_mass_loading):
    """
    An exponential outflow mass-loading parameter.
    
    Tuned to produce the given equilibrium abundance at the Solar radius
    assuming an exponential star formation history.
    
    Parameters
    ----------
    equilibrium : float [default: 0.0]
        The equilibrium [O/H] abundance at the Solar radius.
    recycling : float [default: 0.4]
        The instantaneous recycling parameter.
    tau_star : float [default: 0.0]
        The star formation efficiency timescale in Gyr. If zero, this negates
        the star formation correction factor.
    tau_sfh : float [default: 15.0]
        The star formation history timescale in Gyr.
    gradient : float [default: -0.06]
        The radial metallicity gradient in dex/kpc.
    
    Notes
    -----
    Attributes and functionality are inherited from ``exponential_mass_loading``.
    """
    def __init__(self, equilibrium=0., recycling=0.4, tau_star=0., tau_sfh=15.,
                 gradient=-0.06):
        # Calculate eta for exponential SFH to reach desired equilibrium [O/H]
        Z_alpha_eq = vice.solar_z["o"] * 10 ** equilibrium
        yield_ratio = vice.yields.ccsne.settings["o"] / Z_alpha_eq
        eta_sun = yield_ratio - 1 + recycling + tau_star / tau_sfh
        scale_radius = -1 / (gradient * m.log(10))
        super().__init__(eta_sun, scale_radius)


def no_outflows(radius):
    """A dummy function returning 0 for all inputs."""
    return 0.
