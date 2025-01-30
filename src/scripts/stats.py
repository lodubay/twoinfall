r"""
Generic statistical routines for this project (thanks James!).
"""

from scipy.optimize import curve_fit
from scipy.stats import skewnorm
import numpy as np
import pandas as pd
import random
import vice


def skewnormal(x, a, mean, std):
    r"""
    A generic skew-normal distribution. See scipy.stats.skewnorm.pdf.
    """
    return 1 / std * skewnorm.pdf((x - mean) / std, a)


def skewnormal_estimate_mode(a, mean, std):
    r"""
    A numerical estimate of the mode of a skewnormal distribution.
    """
    delta = a / np.sqrt(1 + a**2)
    term1 = (4 - np.pi) / 2 * delta**3 / (np.pi - 2 * delta**2)
    sgn = int(a > 0) - int(a < 0)
    factor = np.sqrt(2 / np.pi) * (delta - term1) - sgn / 2 * np.exp(
        -2 * np.pi / abs(a))
    return mean + std * factor


def skewnormal_mode_sample(sample, bins = np.linspace(-3, 2, 1001), **kwargs):
    """
    Fit a skewnormal distribution to the sample and estimate the mode.
    """
    centers = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]
    dist, _ = np.histogram(sample, bins = bins, density = True, **kwargs)
    opt, cov = curve_fit(skewnormal, centers, dist, p0 = [1, 0, 1])
    return skewnormal_estimate_mode(opt[0], opt[1], opt[2])


def jackknife_summary_statistic(sample, fcn, n_resamples = 10, seed = None,
    **kwargs):
    r"""
    Estimate the uncertainty on a given summary statistic for a particular
    sample via jackknife resampling.

    kwargs are passed on to `fcn`.
    """
    if isinstance(sample, np.ndarray) or isinstance(sample, pd.Series):
        sample = sample.to_list()
    assert isinstance(sample, list)
    assert callable(fcn)
    assert isinstance(n_resamples, int)
    random.seed(a = seed)
    jackknife_subsample = []
    for i in range(len(sample)):
        jackknife_subsample.append(int(n_resamples * random.random()))
    data = vice.dataframe({
        'sample': sample,
        'jackknife_subsample': jackknife_subsample
    })
    resampled_values = []
    for _ in range(n_resamples):
        sub = data.filter('jackknife_subsample', '!=', _)
        resampled_values.append(fcn(sub['sample'], **kwargs))
    mean = np.mean(resampled_values)
    var = 0
    for value in resampled_values: var += (value - mean)**2
    return np.sqrt((n_resamples - 1) / n_resamples * var)
