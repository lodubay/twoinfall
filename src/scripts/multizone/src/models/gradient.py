r"""
This file declares the radial dependence of the stellar surface density at
late times in the Johnson et al. (2021) models.
"""

import math as m
from ..._globals import THIN_DISK_SCALE_RADIUS, THICK_DISK_SCALE_RADIUS, \
    CENTRAL_DISK_RATIO


def gradient(radius):
    r"""
    The gradient in stellar surface density defined in Bland-Hawthorn &
    Gerhard (2016) [1]_.

    Parameters
    ----------
    radius : real number
        Galactocentric radius in kpc.

    Returns
    -------
    sigma : real number
        The radial surface density at that radius defined by the following
        double-exponential profile:

        .. math:: \Sigma_\star(r) = e^{-r/r_t} + Ae^{-r/r_T}

        where :math:`r_t` = 2.5 kpc is the scale radius of the thin disk,
        :math:`r_T` = 2.0 kpc is the scale radius of the thick disk, and
        :math:`A = \Sigma_T / \Sigma_t \approx` 0.27 is the ratio of thick to
        thin disks at :math:`r = 0`.

        .. note:: This gradient is un-normalized.

    .. [1] Bland-Hawthorn & Gerhard (2016), ARA&A, 54, 529
    """
    return (
        m.exp(-radius / THIN_DISK_SCALE_RADIUS) +
        CENTRAL_DISK_RATIO * m.exp(-radius / THICK_DISK_SCALE_RADIUS)
    )


def thick_to_thin_ratio(radius):
    r"""
    The stellar surface density ratio between the thick and thin disks
    as defined in Bland-Hawthorn & Gerhard (2016).
    
    Parameters
    ----------
    radius : real number
        Galactocentric radius in kpc.
    
    Returns
    -------
    float
        The ratio between the thick and thin disk stellar surface density
        at the given radius.
    """
    return CENTRAL_DISK_RATIO * m.exp(
        radius * (1 / THIN_DISK_SCALE_RADIUS - 1 / THICK_DISK_SCALE_RADIUS)
    )

