r"""
Generic statistical routines for this project (thanks James!).
"""

import random
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
import numpy as np
import pandas as pd
import vice
from _globals import RANDOM_SEED

def cross_entropy(pk, qk):
    """
    Calculate the cross entropy between two distributions.
    
    The cross entropy is defined as CE = -sum(pk * log(qk)). This function will 
    normalize pk and qk to 1 if needed.
    
    Parameters
    ----------
    pk : numpy.ndarray
        The discrete probability distribution.
    qk : numpy.ndarray
        The probability distribution against which to compute the cross entropy.
        
    Returns
    -------
    CE : float
        The cross entropy of the input distributions
    """
    if pk.shape != qk.shape:
        raise ValueError('Arrays logp and logq must have the same shape.')
    # Normalize distributions
    pk /= np.sum(pk)
    qk /= np.sum(qk)
    # Mask array 0s with smallest non-zero value
    qk[qk == 0] = np.min(qk[qk > 0])
    return -np.sum(pk * np.log(qk))


def kl_divergence(pk, qk, dx):
    r"""
    Calculate the Kullback-Leibler (KL) divergence between two distributions.
    
    For a continuous random variable, KL divergence is defined to be
    $D_{\rm{KL}}(P\parallel Q) = \int_{-\infty}^{\infty} p(x)\log(p(x)/q(x))dx$
    
    Parameters
    ----------
    pk : numpy.ndarray
        Probability density of the observed (true) distribution.
    qk : numpy.ndarray
        Probability density of the model distribution.
    dx : float
        Integration step of the observed variable.
    
    Returns
    -------
    kl : float
        The KL divergence between the two distributions, which is 0 if they
        are identical and positive otherwise.
    """
    # mask zeroes with smallest non-zero value
    pk_nz = np.where(pk != 0, pk, np.min(pk[pk > 0]))
    qk_nz = np.where(qk != 0, qk, np.min(qk[qk > 0]))
    return np.sum(np.where(pk != 0, pk * np.log(pk_nz / qk_nz) * dx, 0))


def kl_div_2D(x, y):
    """
    Compute the Kullback-Leibler divergence between two multivariate samples.
    
    Parameters
    ----------
    x : 2D array (n,d)
        Samples from distribution P, which typically represents the true
        distribution.
    y : 2D array (m,d)
        Samples from distribution Q, which typically represents the approximate
        distribution.
        
    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
        
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
        continuous distributions IEEE International Symposium on Information
        Theory, 2008.
    Source: https://mail.python.org/pipermail/scipy-user/2011-May/029521.html
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    return np.log(s/r).sum() * d / n + np.log(m / (n - 1.))


def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE).

    Parameters
    ----------
    x : array-like
    y : array-like
    bandwidth : float
    xbins : complex, optional [default: 100j]
    ybins : complex, optional [default: 100j]

    Other keyword arguments are passed to sklearn.neighbors.KernelDensity

    Returns
    -------
    xx : MxN numpy array
        Density grid x-coordinates (M=xbins, N=ybins)
    yy : MxN numpy array
        Density grid y-coordinates
    logz : MxN numpy array
        Grid of log-likelihood density estimates
    """
    from sklearn.neighbors import KernelDensity
    # Error handling for xbins and ybins
    if type(xbins) == np.ndarray and type(ybins) == np.ndarray:
        if xbins.shape == ybins.shape:
            if len(xbins.shape) == 2 and len(ybins.shape) == 2:
                xx = xbins
                yy = ybins
            else:
                raise ValueError('Input xbins and ybins must have dimension 2.')
        else:
            raise ValueError('Got xbins and ybins of different shape.')
    elif type(xbins) == complex and type(ybins) == complex:
        # create grid of sample locations (default: 100x100)
        xx, yy = np.mgrid[x.min():x.max():xbins,
                          y.min():y.max():ybins]
    else:
        raise TypeError('Input xbins and ybins must have type complex ' + \
                        '(e.g. 100j) or numpy.ndarray.')

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(kernel='gaussian', bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    logz = kde_skl.score_samples(xy_sample)
    return xx, yy, np.reshape(logz, xx.shape)


def median_standard_error(x, B=1000, seed=RANDOM_SEED):
    """
    Use bootstrapping to calculate the standard error of the median.
    
    Parameters
    ----------
    x : array-like
        Data array.
    B : int, optional
        Number of bootstrap samples. The default is 1000.
    
    Returns
    -------
    float
        Standard error of the median.
    """
    rng = np.random.default_rng(seed)
    # Randomly sample input array *with* replacement, all at once
    samples = rng.choice(x, size=len(x) * B, replace=True).reshape((B, len(x)))
    medians = np.median(samples, axis=1)
    # The standard error is the standard deviation of the medians
    return np.std(medians)


def weighted_quantile(df, val, weight, quantile=0.5):
    """
    Calculate the quantile of a pandas column weighted by another column.
    
    Parameters
    ----------
    df : pandas.DataFrame
    val : str
        Name of values column.
    weight : str
        Name of weights column.
    quantile : float, optional
        The quantile to calculate. Must be in [0,1]. The default is 0.5.
    
    Returns
    -------
    wq : float
        The weighted quantile of the dataframe column.
    """
    if quantile >= 0 and quantile <= 1:
        if df.shape[0] == 0:
            return np.nan
        else:
            df_sorted = df.sort_values(val)
            cumsum = df_sorted[weight].cumsum()
            cutoff = df_sorted[weight].sum() * quantile
            wq = df_sorted[cumsum >= cutoff][val].iloc[0]
            return wq
    else:
        raise ValueError("Quantile must be in range [0,1].")


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
