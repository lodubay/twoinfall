"""
Functions used by many plotting scripts.
"""

from numbers import Number
import math as m

import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm, LogNorm
from matplotlib.cm import ScalarMappable
from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coords
import vice

from multizone.src.models import twoinfall, insideout, earlyburst_ifr
from multizone.src.models.gradient import gradient
from _globals import RANDOM_SEED, MAX_SF_RADIUS, ZONE_WIDTH


# =============================================================================
# VICE FUNCTIONALITY EXTENSIONS
# =============================================================================


class twoinfall_onezone(twoinfall):
    """A sub-class of the twoinfall SFH which incorporates the value of the
    stellar surface density gradient, and returns an infall gas mass rather
    than surface density when called."""
    def __init__(self, radius, dr=0.1, **kwargs):
        super().__init__(radius, dr=dr, **kwargs)
        area = m.pi * ((radius + dr/2)**2 - (radius - dr/2)**2)
        self.first.norm *= area * gradient(radius)
        self.second.norm *= area * gradient(radius)


class insideout_onezone(insideout):
    """A sub-class of the inside-out SFH which incorporates the value of the
    stellar surface density gradient, and returns an infall gas mass rather
    than surface density when called."""
    def __init__(self, radius, dr=0.1, **kwargs):
        super().__init__(radius, dr=dr, **kwargs)
        area = m.pi * ((radius + dr/2)**2 - (radius - dr/2)**2)
        self.norm *= area * gradient(radius)


def vice_to_apogee_col(col):
    """
    Convert VICE multizone output to APOGEE column labels.
    
    Parameters
    ----------
    col : str
        Name of column in VICE multizone stars output.
    
    Returns
    -------
    str
        Column label in APOGEE data.
        
    """
    return col[1:-1].replace('/', '_').upper()


def capitalize_abundance(abund):
    """
    Convert abundance label from VICE (lowercase) to plot label style.
    
    Parameters
    ----------
    abund : str
        Abundance label from VICE.

    Returns
    -------
    str
        Abundance label with proper capitalization
    
    """
    assert abund[0] == '[' and abund[-1] == ']'
    elements = [e.capitalize() for e in abund[1:-1].split('/')]
    assert len(elements) == 2
    return '[%s/%s]' % tuple(elements)


def radial_gradient(multioutput, parameter, index=-1, 
                    Rmax=MAX_SF_RADIUS, zone_width=ZONE_WIDTH):
    """
    Return the value of the given model parameter at all zones.
    
    Parameters
    ----------
    multioutput : vice.multioutput
        VICE multi-zone output instance for the desired model.
    parameter : str
        Name of parameter in vice.history dataframe.
    index : int, optional
        Index to select for each zone. The default is -1, which corresponds
        to the last simulation timestep or the present day.
    Rmax : float, optional
        Maximum radius in kpc. The default is 15.5.
    zone_width : float, optional
        Annular zone width in kpc. The default is 0.1.
        
    Returns
    -------
    list
        Parameter values at each zone at the given time index.
    """
    return [multioutput.zones['zone%i' % z].history[index][parameter] 
            for z in range(int(Rmax/zone_width))]


# =============================================================================
# GENERIC CLASSES
# =============================================================================
    
class Exponential:
    """
    A numpy-friendly generic class for exponential functions.
    
    Attributes
    ----------
    scale : float
        The exponential scale factor.
    coeff : float
        The pre-factor for the exponential function.
    """
    def __init__(self, scale=1, coeff=1):
        """
        Parameters
        ----------
        scale : float, optional
            The exponential scale factor. If positive, produces exponential 
            growth; if negative, produces exponential decay. The default is 1.
        coeff : float, optional
            The pre-factor for the exponential function. The default is 1.
        """
        self.scale = scale
        self.coeff = coeff
        
    def __call__(self, x):
        """
        Evaluate the exponential at the given value(s).

        Parameters
        ----------
        x : float or array-like

        Returns
        -------
        float or numpy.ndarray
        """
        return self.coeff * np.exp(x / self.scale)
                    
    @property
    def scale(self):
        """
        Type: float
            The exponential scale factor.
        """
        return self._scale
    
    @scale.setter
    def scale(self, value):
        if isinstance(value, Number):
            self._scale = float(value)
        else:
            raise TypeError('Attribute `scale` must be a number. Got: %s.' % 
                            type(value))
                        
    @property
    def coeff(self):
        """
        Type: float
            Pre-factor for the exponential function.
        """
        return self._coeff
    
    @coeff.setter
    def coeff(self, value):
        if isinstance(value, Number):
            self._coeff = float(value)
        else:
            raise TypeError('Attribute `coeff` must be a number. Got: %s.' % 
                            type(value))

# =============================================================================
# DATA UTILITY FUNCTIONS
# =============================================================================

def fits_to_pandas(path, **kwargs):
    """
    Import a table in the form of a FITS file and convert it to a pandas
    DataFrame.

    Parameters
    ----------
    path : Path or str
        Path to fits file
    Other keyword arguments are passed to astropy.table.Table

    Returns
    -------
    df : pandas DataFrame
    """
    # Read FITS file into astropy table
    table = Table.read(path, format='fits', **kwargs)
    # Filter out multidimensional columns
    cols = [name for name in table.colnames if len(table[name].shape) <= 1]
    # Convert byte-strings to ordinary strings and convert to pandas
    df = decode(table[cols].to_pandas())
    return df


def split_multicol(multicol, names=[]):
    """
    Split a multi-dimensional column from an Astropy table into a DataFrame.
    
    Parameters
    ----------
    multicol : astropy Table multidimensional column
    names : list, optional
        List of column names.
    
    Returns
    -------
    pandas.DataFrame
    """
    # Make sure it's a multidimensional column
    if len(multicol.shape) != 2:
        raise ValueError('Input column has wrong dimensionality.')
    # Generate column names if needed
    if names == []:
        names = [multicol.name + str(i) for i in range(multicol.shape[1])]
    if len(names) != multicol.shape[1]:
        raise ValueError('Length of names does not match number of columns.')
    # Convert to DataFrame
    # This is kind of dumb but I needed to avoid an "endianness" problem
    data = np.array(multicol.data)
    data_dict = dict()
    for i in range(multicol.shape[1]):
        data_dict[names[i]] = list(data[:,i])
    return pd.DataFrame(data_dict)


def decode(df):
    """
    Decode DataFrame with byte strings into ordinary strings.

    Parameters
    ----------
    df : pandas DataFrame
    """
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df


def quad_add(arr1, arr2):
    """
    Add two input arrays in quadrature.
    """
    return np.sqrt(arr1**2 + arr2**2)
        

def group_by_bins(df, bin_col, bins=10):
    """
    Bin a DataFrame column and group data by those bins.
    
    Parameters
    ----------
    df : pandas.DataFrame
    bin_col : str
        Column name by which to bin the data.
    bins : int or array-like
        If an int is provided, the number of bins between the min and max
        data values. If an array, the bin edges to use. The default is 10.
    
    Returns
    -------
    grouped : pandas.DataFrameGroupBy
        A groupby object that contains information about the groups.
    """
    df = df.copy()
    # Handle different types for "bins" parameter
    if isinstance(bins, int):
        bin_edges = np.linspace(df[bin_col].min(), df[bin_col].max(), bins)
    elif isinstance(bins, (list, np.ndarray, pd.Series)):
        bin_edges = np.array(bins)
    else:
        raise ValueError('Parameter "bins" must be int or array-like.')
    # segment and sort data into bins
    bin_centers = get_bin_centers(bin_edges)
    df.insert(len(df.columns), 
              'bin', 
              pd.cut(df[bin_col], bin_edges, labels=bin_centers))
    # group data by bins
    return df.groupby('bin', observed=False)


def get_bin_centers(bin_edges):
    """
    Calculate the centers of bins defined by the given bin edges.
    
    Parameters
    ----------
    bin_edges : array-like of length N
        Edges of bins, including the left-most and right-most bounds.
     
    Returns
    -------
    bin_centers : numpy.ndarray of length N-1
        Centers of bins
    """
    bin_edges = np.array(bin_edges, dtype=float)
    if len(bin_edges) > 1:
        return 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        raise ValueError('The length of bin_edges must be at least 2.')


def sample_rows(df, n, weights=None, reset=True, seed=RANDOM_SEED):
    """
    Randomly sample n unique rows from a pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
    n : int
        Number of random samples to draw
    weights : array, optional
        Probability weights of the given DataFrame
    reset : bool, optional
        If True, reset sample DataFrame index

    Returns
    -------
    pandas DataFrame
        Re-indexed DataFrame of n sampled rows
    """
    if isinstance(df, pd.DataFrame):
        # Number of samples can't exceed length of DataFrame
        n = min(n, df.shape[0])
        # Initialize default numpy random number generator
        rng = default_rng(seed)
        # Randomly sample without replacement
        rand_indices = rng.choice(df.index, size=n, replace=False, p=weights)
        sample = df.loc[rand_indices]
        if reset:
            sample.reset_index(inplace=True, drop=True)
        return sample
    else:
        raise TypeError('Expected pandas DataFrame.')


def model_uncertainty(x, err, how='linear', seed=RANDOM_SEED):
    """
    Apply Gaussian uncertainty to the given data array.
    
    Parameters
    ----------
    x : array-like
        Input (clean) data.
    err : float
        Standard deviation of the Gaussian.
    how : str, optional
        How the uncertainty should be applied to the data. Options are 'linear',
        'logarithmic' or 'log', and 'fractional' or 'frac'. The default is 
        'linear'.
    
    Returns
    -------
    y : array-like
        Noisy data, with same dimensions as x.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=err, size=x.shape[0])
    if how.lower() == 'linear':
        y = x + noise
    elif how.lower() in ['logarithmic', 'log']:
        y = x * 10 ** noise
    elif how.lower() in ['fractional', 'frac']:
        y = x * (1 + noise)
    else:
        raise ValueError('Parameter "how" must be one of ("linear", ' + 
                         '"logarithmic", "log", "fractional", "frac")')
    return y


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def axes_grid(rows, cols, width=8, xlim=None, ylim=None):
    """
    Set up a blank grid of axes plus a colorbar axis.

    Parameters
    ----------
    rows : int
        Number of rows of axes
    cols : int
        Number of columns of axes
    width : float, optional
        Width of the figure in inches. The default is 8 in.
    xlim : tuple or None, optional
        Limits of x-axis for all axes
    ylim : tuple or None, optional
        Limits of y-axis for all axes

    Returns
    -------
    fig : matplotlib figure
    axs : list of axes
    cax : axis object for colorbar
    """
    fig, axs = plt.subplots(rows, cols, figsize=(width, (width/cols)*rows),
                            sharex=True, sharey=True)
    # Configure plot dimensions
    plt.subplots_adjust(right=0.98, left=0.05, bottom=0.09, top=0.94,
                        wspace=0.05, hspace=0.05)
    # Configure axis limits and ticks (will be applied to all axes)
    axs[0,0].set_xlim(xlim)
    axs[0,0].set_ylim(ylim)
    # axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    # axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # axs[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    # axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    return fig, axs


def scatter_hist(ax, x, y, xlim=None, ylim=None, log_norm=True, cmap='gray',
                 cmin=10, vmin=None, vmax=None, nbins=50, color='k',
                 rasterized=True, scatter_size=0.5):
    """
    Generate a scatter plot and overlayed 2D histogram for dense data.

    Parameters
    ----------
    ax : matplotlib.axis.Axes
        Axes object on which to plot the data.
    x : array-like
        Horizontal coordinates of the data points.
    y : array-like
        Vertical coordinates of the data points.
    xlim : float, optional
        Bounds for x-axis. The default is None.
    ylim : float, optional
        Bounds for y-axis. The default is None.
    log_norm : bool, optional
        Shade the 2D histogram on a logarithmic scale. The default is True.
    cmap : str, optional
        Colormap for 2D histogram. The default is'gray'.
    cmin : int, optional
        Minimum counts per bin; any number below this will show individual points.
        The default is 10.
    vmin : float or None, optional
        Value to map to minimum of histogram normalization. The default is None.
    vmax : float or None, optional
        Value to map to maximum of histogram normalization. The default is None.
    nbins : int or tuple of ints, optional
        Number of histogram bins. If a tuple, presumed to be (xbins, ybins).
        The default is 50.
    color : str, optional
        Color of individual points. The default is 'k'.
    rasterized : bool, optional [default: True]
        Whether to rasterize the scattered points
    scatter_size : float, optional
        Size of scattered points. The default is 0.5.
    """
    # Set automatic plot bounds
    if not xlim:
        xlim = (np.min(x), np.max(x))
    if not ylim:
        ylim = (np.min(y), np.max(y))
    # Set bin edges
    if type(nbins) == 'tuple':
        xbins, ybins = nbins
    else:
        xbins = ybins = nbins
    xbins = np.linspace(xlim[0], xlim[1], num=xbins, endpoint=True)
    ybins = np.linspace(ylim[0], ylim[1], num=ybins, endpoint=True)
    # Histogram normalization
    if log_norm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    # Plot
    ax.scatter(x, y, c=color, s=scatter_size, rasterized=rasterized, 
               edgecolor='none')
    return ax.hist2d(x, y, bins=[xbins, ybins], cmap=cmap, norm=norm, cmin=cmin)


def get_color_list(cmap, bins):
    """
    Split a discrete colormap into a list of colors based on bin edges.
    
    Parameters
    ----------
    cmap : matplotlib colormap
    bins : array-like
        Bin edges, including left- and right-most edges
    
    Returns
    -------
    list
        List of colors of length len(bins) - 1
    """
    rmin, rmax = bins[0], bins[-2]
    colors = cmap([(r-rmin)/(rmax-rmin) for r in bins[:-1]])
    return colors


def discrete_colormap(cmap_name, bounds):
    """
    Convert a continuous colormap into a discrete one.
    
    Parameters
    ----------
    cmap_name : str
        Name of matplotlib colormap
    bounds : array-like
        Bounds of discrete colormap
    
    Returns
    -------
    cmap : matplotlib colormap
    norm : colormap normalization
    """
    cmap = plt.get_cmap(cmap_name)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def setup_discrete_colorbar(fig, cmap, norm, label='', width=0.6):
    """
    Adds a colorbar for a discrete colormap to the bottom of the figure.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
    cmap : matplotlib colormap
    norm : matplotlib colormap normalization
    label : str, optional
        Colorbar label. The default is ''.
    width : float, optional
        Width of colorbar as fraction of whole figure. The default is 0.6.
    
    Returns
    -------
    cax : matplotlib.axes.Axes
        Colorbar axes
    """
    fig.subplots_adjust(bottom=0.2)
    cax = plt.axes([0.5 - (width / 2), 0.09, width, 0.02])
    # Add colorbar
    cbar = fig.colorbar(ScalarMappable(norm, cmap), cax,
                        orientation='horizontal')
    cbar.set_label(label)
    return cax


def highlight_panels(fig, axs, idx, color='#cccccc'):
    """
    Add a colored box behind subplots to make them stand out.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Main plot figure.
    axs : list of matplotlib.axes.Axes
        All subplots of the figure.
    idx : tuple or list of tuples
        Index of the subplot(s) to highlight.
    color : str, optional
        Color to highlight the panel. The default is '#cccccc'.
    """
    # Get spacing between subplots (assuming all are identical)
    # Note: bbox coordinates converted from display to figure
    bbox00 = axs[0,0].get_window_extent().transformed(fig.transFigure.inverted())
    bbox01 = axs[0,1].get_window_extent().transformed(fig.transFigure.inverted())
    bbox10 = axs[1,0].get_window_extent().transformed(fig.transFigure.inverted())
    pad_h = bbox01.x0 - bbox00.x0 - bbox00.width
    pad_v = bbox00.y0 - bbox10.y0 - bbox10.height
    if not isinstance(idx, list):
        idx = [idx]
    for i in idx:
        bbox = axs[i].get_tightbbox().transformed(fig.transFigure.inverted())
        fig.patches.extend([plt.Rectangle((bbox.x0 - pad_h/2, bbox.y0 - pad_v/4),
                                          bbox.x1 - bbox.x0 + pad_h, # width
                                          bbox.y1 - bbox.y0 + pad_v/2, # height
                                          fill=True, color=color, zorder=-1,
                                          transform=fig.transFigure, figure=fig)])


def contour_levels_2D(arr2d, enclosed=[0.8, 0.3]):
    """
    Calculate the contour levels which contain the given enclosed probabilities.
    
    Parameters
    ----------
    arr2d : np.ndarray
        2-dimensional array of densities.
    enclosed : list, optional
        List of enclosed probabilities of the contour levels. The default is
        [0.8, 0.3].
    """
    levels = []
    l = 0.
    i = 0
    while l < 1 and i < len(enclosed):
        frac_enclosed = np.sum(arr2d[arr2d > l]) / np.sum(arr2d)
        if frac_enclosed <= enclosed[i] + 0.01:
            levels.append(l)
            i += 1
        l += 0.01
    return levels


def box_smooth(hist, bins, width):
    """
    Box-car smoothing function for a pre-generated histogram.

    Parameters
    ----------
    bins : array-like
        Bins dividing the histogram, including the end. Length must be 1 more
        than the length of hist, and bins must be evenly spaced.
    hist : array-like
        Histogram of data
    width : float
        Width of the box-car smoothing function in data units
    """
    bin_width = bins[1] - bins[0]
    box_width = int(width / bin_width)
    box = np.ones(box_width) / box_width
    hist_smooth = np.convolve(hist, box, mode='same')
    return hist_smooth


def gaussian_smooth(hist, bins, width):
    """
    Box-car smoothing function for a pre-generated histogram.

    Parameters
    ----------
    bins : array-like
        Bins dividing the histogram, including the end. Length must be 1 more
        than the length of hist, and bins must be evenly spaced.
    hist : array-like
        Histogram of data
    width : float
        Standard deviation of the Gaussian in data units
    """
    from scipy.stats import norm
    bin_width = bins[1] - bins[0]
    sigma = int(width / bin_width)
    gaussian = norm.pdf(np.arange(-5*sigma, 5*sigma), loc=0, scale=sigma)
    hist_smooth = np.convolve(hist, gaussian, mode='same')
    return hist_smooth

# =============================================================================
# SCIENCE FUNCTIONS
# =============================================================================    

def galactic_to_galactocentric(l, b, distance):
    r"""
    Use astropy's SkyCoord to convert Galactic (l, b, distance) coordinates
    to galactocentric (r, phi, z) coordinates.

    Parameters
    ----------
    l : array-like
        Galactic longitude in degrees
    b : array-like
        Galactic latitude in degrees
    distance : array-like
        Distance (from Sun) in kpc

    Returns
    -------
    galr : numpy array
        Galactocentric radius in kpc
    galphi : numpy array
        Galactocentric phi-coordinates in degrees
    galz : numpy arraay
        Galactocentric z-height in kpc
    """
    l = np.array(l)
    b = np.array(b)
    d = np.array(distance)
    if l.shape == b.shape == d.shape:
        if not isinstance(l, u.quantity.Quantity):
            l *= u.deg
        if not isinstance(b, u.quantity.Quantity):
            b *= u.deg
        if not isinstance(d, u.quantity.Quantity):
            d *= u.kpc
        galactic = coords.SkyCoord(l=l, b=b, distance=d, frame=coords.Galactic())
        galactocentric = galactic.transform_to(frame=coords.Galactocentric())
        galactocentric.representation_type = 'cylindrical'
        galr = galactocentric.rho.to(u.kpc).value
        galphi = galactocentric.phi.to(u.deg).value
        galz = galactocentric.z.to(u.kpc).value
        return galr, galphi, galz
    else:
        raise ValueError('Arrays must be of same length.')
