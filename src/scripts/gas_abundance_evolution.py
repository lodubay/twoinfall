"""
This script plots the evolution of gas abundance in the Solar ring for 
multi-zone models with different yields and mass-loading factors.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import vice

import CheapTools
from apogee_sample import APOGEESample
from multizone_stars import MultizoneStars
from utils import get_bin_centers, truncate_colormap
from _globals import ONE_COLUMN_WIDTH
import paths
from colormaps import paultol

OH_LIM = (-0.8, 0.6)
FEH_LIM = (-0.8, 0.6)
OFE_LIM = (-0.2, 0.5)
AGE_LIM = (-1, 14)
GALR_LIM = (7, 9)
ABSZ_LIM = (0, 0.5)
SMOOTH_WIDTH = 0.05
GRIDSIZE = 30

OUTPUT_NAMES = [
    'yZ1-fiducial/diskmodel',
    'yZ2-fiducial/diskmodel',
    'yZ2-earlyonset/diskmodel',
]
LABELS = [
    r'$y/Z_\odot=1$ (fiducial)',
    r'$y/Z_\odot=2$ (fiducial)',
    r'$y/Z_\odot=2$ ($t_{\rm max}=2.2$ Gyr)',
]


def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', paultol.bright.colors)

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH, 2*ONE_COLUMN_WIDTH))
    gs = fig.add_gridspec(3, 2, width_ratios=(1, 4), wspace=0., hspace=0.)
    ax0 = fig.add_subplot(gs[0,0])
    ax0.tick_params(axis='x', labelbottom=False)
    ax1 = fig.add_subplot(gs[0,1], sharey=ax0)
    ax1.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax2 = fig.add_subplot(gs[1,0], sharex=ax0)
    ax2.tick_params(axis='x', labelbottom=False)
    ax3 = fig.add_subplot(gs[1,1], sharex=ax1, sharey=ax2)
    ax3.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax4 = fig.add_subplot(gs[2,0], sharex=ax2)
    ax5 = fig.add_subplot(gs[2,1], sharex=ax3, sharey=ax4)
    ax5.tick_params(axis='y', labelleft=False)
    axs = np.array([[ax0, ax1], [ax2, ax3], [ax4, ax5]])

    # Plot APOGEE abundances + Leung et al. (2023) ages
    apogee_sample = APOGEESample.load()
    local_sample = apogee_sample.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
    age_bin_width = 1. # Gyr
    age_bins = np.arange(0, 13 + age_bin_width, age_bin_width)
    age_bin_centers = get_bin_centers(age_bins)
    age_col = 'L23_AGE'
    data_color = '0.6'
    data_size = 1
    mode_color = 'k'
    # Rolling median
    sorted_ages = local_sample.data.sort_values(age_col)[
        [age_col, 'O_H', 'FE_H', 'O_FE']
    ]
    rolling_medians = sorted_ages.rolling(
        1000, min_periods=100, step=100, on=age_col, center=True
    ).median()
    # Median age errors as a function of time
    big_age_bins = np.arange(0, 15, 4)
    median_age_errors = local_sample.binned_intervals(
        '%s_ERR' % age_col, age_col, big_age_bins, quantiles=[0.5]
    )
    xval_err = get_bin_centers(big_age_bins)
    yval_err = [-0.7, -0.7, -0.15]
    abund_range = [OH_LIM, FEH_LIM, OFE_LIM]
    # Normalize colormap
    norm = Normalize(vmin=0, vmax=250)
    cmap = truncate_colormap(plt.get_cmap('binary'), minval=0., maxval=0.8)
    for i, abund in enumerate(['O_H', 'FE_H', 'O_FE']):
        # Hexbin of APOGEE stars
        pcm = axs[i,1].hexbin(
            local_sample(age_col), local_sample(abund),
            gridsize=GRIDSIZE, cmap=cmap, norm=norm, linewidths=0.2, mincnt=1,
            extent=[AGE_LIM[0], AGE_LIM[1], abund_range[i][0], abund_range[i][1]]
        )
        # Plot rolling median
        axs[i,1].plot(
            rolling_medians[age_col], rolling_medians[abund], 'w-',
            zorder=2, linewidth=2
        )
        axs[i,1].plot(
            rolling_medians[age_col], rolling_medians[abund], 'k-',
            zorder=2, label='APOGEE median'
        )
        # Plot abundance modes in bins of stellar age
        abund_bins = local_sample.binned_modes(abund, age_col, age_bins)
        axs[i,1].errorbar(age_bin_centers, abund_bins['mode'], 
                    xerr=age_bin_width/2, yerr=abund_bins['error'],
                    linestyle='none', c=mode_color, capsize=1, marker='.',
                    zorder=10, label='APOGEE mode')
        # Median errors at different age bins
        median_abund_errors = local_sample.binned_intervals(
            '%s_ERR' % abund, age_col, big_age_bins, quantiles=[0.5]
        )
        axs[i,1].errorbar(
            xval_err, yval_err[i] * np.ones(xval_err.shape), 
            xerr=median_age_errors[0.5], yerr=median_abund_errors[0.5],
            c=data_color, marker='.', ms=0, zorder=3, 
            linestyle='none', elinewidth=1, capsize=0,
        )
        # Plot APOGEE abundance distributions in marginal panels
        abund_df, bin_edges = local_sample.mdf(
            col=abund, range=abund_range[i], smoothing=SMOOTH_WIDTH
        )
        axs[i,0].plot(abund_df / max(abund_df), get_bin_centers(bin_edges),
                color=data_color, linestyle='-', linewidth=2, marker=None)
    # Colorbar for APOGEE hexbin
    cax = axs[2,1].inset_axes([0.06, 0.88, 0.68, 0.06])
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), 
        cax=cax, 
        orientation='horizontal',
        label='# APOGEE Stars'
    )

    # Plot multizone gas abundance
    for i, output_name in enumerate(OUTPUT_NAMES):
        mzs = MultizoneStars.from_output(output_name)
        mzs.model_uncertainty(apogee_sample.data, inplace=True)
        mzs_local = mzs.region(galr_lim=GALR_LIM, absz_lim=ABSZ_LIM)
        plot_abundance_history(
            axs[0], mzs_local, '[o/h]', range=OH_LIM, smoothing=SMOOTH_WIDTH,
            zorder=5,
            label=LABELS[i]
        )
        plot_abundance_history(
            axs[1], mzs_local, '[fe/h]', range=FEH_LIM, smoothing=SMOOTH_WIDTH,
            zorder=5,
        )
        plot_abundance_history(
            axs[2], mzs_local, '[o/fe]', range=OFE_LIM, smoothing=SMOOTH_WIDTH,
            zorder=5,
        )
    
    # Plot Palicio et al. (2023) model for comparison
    # Model parameters
    chemdict = dict()
    chemdict["omega"] = 0.8
    chemdict["R"] = 0.285
    chemdict["nuL"] = 0.75
    # Infall parameters
    chemdict["tauj"] = np.array([0.4, 7.])
    chemdict["tj"] = [0., 3.]
    chemdict["sigma_gas_0"] = 1E-8 # Sigma_gas_0 should be very close to zero but not zero
    chemdict["Aj"] =  [35.128, 10.207] # Makes 47 Msun/pc**2 today (McKee et al. 2015)
    time, feh, ofe = analytic_model(chemdict)
    zorder = 4
    color = 'gray'
    axs[0,1].plot(time[::-1], ofe + feh, 'w-', linewidth=2, zorder=zorder)
    axs[0,1].plot(time[::-1], ofe + feh, linestyle='--', c=color, zorder=zorder, 
                  label='Palicio et al. (2023)')
    axs[1,1].plot(time[::-1], feh, 'w-', linewidth=2, zorder=zorder)
    axs[1,1].plot(time[::-1], feh, linestyle='--', c=color, zorder=zorder)
    axs[2,1].plot(time[::-1], ofe, 'w-', linewidth=2, zorder=zorder)
    axs[2,1].plot(time[::-1], ofe, linestyle='--', c=color, zorder=zorder)

    # Format axes
    ax0.set_ylabel('[O/H]')
    ax0.set_xlim((1.2, 0))
    ax0.set_ylim(OH_LIM)
    ax0.yaxis.set_major_locator(MultipleLocator(0.5))
    ax0.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.set_xlim(AGE_LIM)
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    
    ax2.set_ylabel('[Fe/H]')
    ax2.set_ylim(FEH_LIM)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax4.set_ylabel('[O/Fe]')
    ax4.set_xlabel('P([X/H])', size='small')
    ax4.set_ylim(OFE_LIM)
    ax4.yaxis.set_major_locator(MultipleLocator(0.2))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax5.set_xlabel('Lookback Time [Gyr]')

    # Legend for data
    handles, labels = ax1.get_legend_handles_labels()
    ax0.legend(
        [handles[0], handles[-1], handles[-2]], 
        [labels[0], labels[-1], labels[-2]],
        loc='lower left', 
        bbox_to_anchor=[-0.5, 1], 
        borderaxespad=0.
        # title='APOGEE (NN ages)'
    )
    # Legend for models
    ax1.legend(
        handles[1:4], 
        labels[1:4], 
        loc='lower right', 
        bbox_to_anchor=[1, 1],
        borderaxespad=0.,
    )

    fig.savefig(paths.figures / 'gas_abundance_evolution')
    plt.close()


def plot_abundance_history(axs, mzs, col, label='', c=None, ls='-', range=None, smoothing=0., **kwargs):
    # Plot gas abundance evolution
    zone = int(0.5 * (mzs.galr_lim[0] + mzs.galr_lim[1]) / mzs.zone_width)
    zone_path = str(mzs.fullpath / ('zone%d' % zone))
    hist = vice.history(zone_path)
    axs[1].plot(hist['lookback'], hist[col], color='w', ls='-', linewidth=2, **kwargs)
    axs[1].plot(hist['lookback'], hist[col], label=label, color=c, ls=ls, linewidth=1, **kwargs)
    # Plot MDFs
    mdf, mdf_bins = mzs.mdf(col, range=range, smoothing=smoothing, bins=100)
    axs[0].plot(mdf / mdf.max(), get_bin_centers(mdf_bins), color='w', ls='-', linewidth=2, **kwargs)
    axs[0].plot(mdf / mdf.max(), get_bin_centers(mdf_bins), color=c, ls=ls, linewidth=1, **kwargs)


def analytic_model(
        chemdict,
        TypeIa_SNe_ratio = 0.54/100.*1E9,
        Area = np.pi*(20.**2-3.**2)*1E6,
        today = 13.8,
        Solar_values = {"FeH":-2.752, "OFe":0.646, "SiFe":-0.291}
    ):
    """Run analytic model from Palicio et al. (2023)."""
    # Integration time
    t_gyr = np.arange(0.01, today, 0.00125)# Gyr
    t_gyr = t_gyr[t_gyr<today]

    # DTD parameters:
    chemdict = CheapTools.Load_MR01_dict( chemdict )# For example, let's use the MR01 DTD
    # Using "chemdict" as input, the output will have all the key-values of the input

    # Now we have to provide a value for CIa, but instead of setting CIa directly we make
    # use of the present-day Type Ia ratio:
    chemdict["CIa"] = CheapTools.Get_CIa(TypeIa_SNe_ratio, Area, chemdict, present_day_time=today)# This computes CIa

    # Now add the parameters associated with the iron element, with zero initial density.
    # using the add_element() function for the values considered in Palicio et al. (submitted).
    chemdict_Fe = CheapTools.add_element(chemdict, "Fe", 0.0)# For the moment, this function only works for Fe, O and Si,
    chemdict_O = CheapTools.add_element(chemdict, "O", 0.0)

    # Solve the Chemical Evolution Model equation:
    Sigma_MR01_Fe = CheapTools.SolveChemEvolModel( t_gyr, chemdict_Fe)# The output is the density of iron as a function of time
    Sigma_MR01_O = CheapTools.SolveChemEvolModel( t_gyr, chemdict_O)

    # It is more intuitive to work with [Fe/H] rather than sigma_Fe:
    Abund_MR01_Fe = CheapTools.FromSigmaToAbundance(t_gyr, Sigma_MR01_Fe, chemdict_Fe)# From sigma (surface density) to abundance:
    Abund_MR01_O = CheapTools.FromSigmaToAbundance(t_gyr, Sigma_MR01_O, chemdict_O)

    # We have to implement the correction due to the solar values for the iron:
    FeH_MR01 = Abund_MR01_Fe -Solar_values["FeH"] + 0.125 # The factor 0.125 comes from log10(0.75), since 3/4 of the gas is made by Hydrogen
    OFe_MR01 = Abund_MR01_O-Abund_MR01_Fe-Solar_values["OFe"]

    return t_gyr, FeH_MR01, OFe_MR01


if __name__ == '__main__':
    main()
