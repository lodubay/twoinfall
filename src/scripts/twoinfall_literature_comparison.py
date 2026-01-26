import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import vice

from CheapTools import *
from utils import twoinfall_onezone
from multizone.src import models, outflows
from _globals import END_TIME, ONEZONE_DEFAULTS, TWO_COLUMN_WIDTH
import paths


def main():
    plt.style.use(paths.styles / 'paper.mplstyle')

    # Represents the [O/Fe] and [Si/Fe] for the DTDs used in our work
    TypeIa_SNe_ratio = 0.54/100.*1E9;# +/-0.12 events/cent
    Area = np.pi*(20.**2-3.**2)*1E6;# In pc**2
    # Area = np.pi*(8.1**2-8.**2)*1E6;# In pc**2
    today = 13.8; # Gyr
    # today = 13.2; # Gyr
    Solar_values = {'FeH':-2.752, 'OFe':0.646,'SiFe':-0.291}# Fe/H, O/Fe, Si/Fe. Related to the ratio of solar Fe, O, and Si densities.

    # Integration time
    t_gyr = np.arange(0.01, today, 0.00125)# Gyr
    t_gyr = t_gyr[t_gyr<today]

    # Model parameters
    chemdict = dict()
    chemdict['omega'] = 0.8
    chemdict['R'] = 0.285
    chemdict['nuL'] = 0.75
    # chemdict['omega'] = 0.2
    # chemdict['R'] = 0.4
    # chemdict['nuL'] = 0.25

    # Infall parameters:
    chemdict['tauj'] = np.array([0.4, 7.])
    chemdict['tj'] = [0., 3.]
    chemdict['sigma_gas_0'] = 1E-8 # Sigma_gas_0 should be very close to zero but not zero
    chemdict['Aj'] =  [35.128, 10.207] # Makes 54 Msun/pc**2 today (Vincenzo et al. 2017)
    # chemdict['tauj'] = np.array([1, 15])
    # chemdict['tj'] = [0., 4.2]
    # chemdict['sigma_gas_0'] = 1E-8 # Sigma_gas_0 should be very close to zero but not zero
    # chemdict['Aj'] =  [35, 50] # Makes 54 Msun/pc**2 today (Vincenzo et al. 2017)

    # DTD parameters:
    chemdict = Load_MR01_dict( chemdict )# For example, let's use the MR01 DTD
    # Using 'chemdict' as input, the output will have all the key-values of the input

    # Now we have to provide a value for CIa, but instead of setting CIa directly we make
    # use of the present-day Type Ia ratio:
    chemdict['CIa'] = Get_CIa(TypeIa_SNe_ratio, Area, chemdict, present_day_time=today)# This computes CIa
    # chemdict['CIa'] = 1.55e-3

    # Now add the parameters associated with the iron element, with zero initial density.
    #    We can it both by:
    # chemdict_Fe = chemdict.copy()
    # chemdict_Fe['yx'] = 4.58e-4 * 1/(1-chemdict['R'])# preferred yield here
    # chemdict_Fe['mx1a'] = 0.7 # preferred yield here
    # chemdict_Fe['sigmaX_0'] = 0 # assume no initial content of iron
    # chemdict_O = chemdict.copy()
    # chemdict_O['yx'] = 5.72e-3 * 1/(1-chemdict['R'])# preferred yield here
    # chemdict_O['mx1a'] = 0 # preferred yield here
    # chemdict_O['sigmaX_0'] = 0 # assume no initial content of iron
    # chemdict_Si = chemdict.copy()
    # chemdict_Si['yx'] = 1e-3# preferred yield here
    # chemdict_Si['mx1a'] = 0.1 # preferred yield here
    # chemdict_Si['sigmaX_0'] = 0 # assume no initial content of iron
    #    ... or using the add_element() function for the values considered in Palicio et al. (submitted).
    chemdict_Fe = add_element(chemdict, 'Fe', 0.0)# For the moment, this function only works for Fe, O and Si,
    chemdict_O = add_element(chemdict, 'O', 0.0)
    chemdict_Si = add_element(chemdict, 'Si', 0.0)

    # Solve the Chemical Evolution Model equation:
    Sigma_MR01_Fe = SolveChemEvolModel( t_gyr, chemdict_Fe)# The output is the density of iron as a function of time
    Sigma_MR01_O = SolveChemEvolModel( t_gyr, chemdict_O)
    Sigma_MR01_Si = SolveChemEvolModel( t_gyr, chemdict_Si)

    # It is more intuitive to work with [Fe/H] rather than sigma_Fe:
    Abund_MR01_Fe = FromSigmaToAbundance(t_gyr, Sigma_MR01_Fe, chemdict_Fe)# From sigma (surface density) to abundance:
    Abund_MR01_O = FromSigmaToAbundance(t_gyr, Sigma_MR01_O, chemdict_O)
    Abund_MR01_Si = FromSigmaToAbundance(t_gyr, Sigma_MR01_Si, chemdict_Si)

    # We have to implement the correction due to the solar values for the iron:
    FeH_MR01 = Abund_MR01_Fe -Solar_values['FeH'] + 0.125 # The factor 0.125 comes from log10(0.75), since 3/4 of the gas is made by Hydrogen
    OFe_MR01 = Abund_MR01_O-Abund_MR01_Fe-Solar_values['OFe']
    SiFe_MR01 = Abund_MR01_Si-Abund_MR01_Fe-Solar_values['SiFe']
    #-------------------------------------------------------------------

    # Attempt to match Palicio with a one-zone model
    from multizone.src.yields import yZ1
    radius = 8
    eta = outflows.yZ1(radius)
    # eta = 0.4
    zone_width = 0.1
    local_disk_ratio = 0.12
    dt = ONEZONE_DEFAULTS['dt']
    simtime = np.arange(0, END_TIME + dt, dt)
    area = np.pi * ((radius + zone_width)**2 - radius**2)
    # Prescription for disk surface density as a function of radius
    diskmodel = models.diskmodel.two_component_disk.from_local_ratio(
        local_ratio = local_disk_ratio
    )
    # Run one-zone model
    name = paths.data/'onezone'/'litmatch'/'yZ1'
    if not name.parent.is_dir():
        name.parent.mkdir(parents=True)
    name = str(name)
    ifr = twoinfall_onezone(
        radius, 
        diskmodel=diskmodel,
        mass_loading=eta, 
        dt=dt, 
        dr=zone_width, 
        first_timescale=0.3,
        second_timescale=15,
        onset=3.2,
    )
    sz = vice.singlezone(
        name=name,
        func=ifr, 
        mode='ifr',
        **ONEZONE_DEFAULTS
    )
    sz.tau_star = models.twoinfall_sf_law(area, onset=ifr.onset)
    sz.eta = eta
    sz.run(simtime, overwrite=True)
    onezone_yZ1_hist = vice.history(name)

    from multizone.src.yields import yZ2
    eta = outflows.yZ2(radius)
    # eta = 0.4
    local_disk_ratio = 0.12
    # Prescription for disk surface density as a function of radius
    diskmodel = models.diskmodel.two_component_disk.from_local_ratio(
        local_ratio = local_disk_ratio
    )
    # Run one-zone model
    name = paths.data/'onezone'/'litmatch'/'yZ2'
    if not name.parent.is_dir():
        name.parent.mkdir(parents=True)
    name = str(name)
    ifr = twoinfall_onezone(
        radius, 
        diskmodel=diskmodel,
        mass_loading=eta, 
        dt=dt, 
        dr=zone_width, 
        first_timescale=0.3,
        second_timescale=15,
        onset=2.2,
    )
    sz = vice.singlezone(
        name=name,
        func=ifr, 
        mode='ifr',
        **ONEZONE_DEFAULTS
    )
    sz.tau_star = models.twoinfall_sf_law(area, onset=ifr.onset)
    sz.eta = eta
    sz.run(simtime, overwrite=True)
    onezone_yZ2_hist = vice.history(name)

    # Plotting the results
    # -------------------------------------------------------------
    # Figure
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.5 * TWO_COLUMN_WIDTH))
    gs = GridSpec(2, 2, figure=fig)

    # APOGEE for comparison
    data = pd.read_csv(paths.data / 'APOGEE' / 'sample.csv')
    local_sample = data[(data['GALR'] >= 7) & (data['GALZ'] < 9)]

    # Multizone models
    zone = 80
    yZ1_hist = vice.history(str(paths.data/f'multizone/yZ1-fiducial/diskmodel.vice/zone{zone}'))
    yZ2_hist = vice.history(str(paths.data/f'multizone/yZ2-earlyonset/diskmodel.vice/zone{zone}'))

    # In the left panel, plot [O/Fe] vs [Fe/H]
    ax0 = fig.add_subplot(gs[:,0])
    hexargs = {'cmap': 'binary_r', 'gridsize': 50, 'linewidths': 0.2, 'mincnt': 1, 'norm': LogNorm()}
    ax0.hexbin(local_sample['FE_H'], local_sample['O_FE'], **hexargs)
    ax0.plot(FeH_MR01, OFe_MR01, color='b', label='Palicio et al. (2023)')
    ax0.plot(yZ1_hist['[fe/h]'], yZ1_hist['[o/fe]'], c='orange', label='new yZ1-fiducial')
    ax0.plot(yZ2_hist['[fe/h]'], yZ2_hist['[o/fe]'], c='r', label='new yZ2-earlyonset')
    # ax0.plot(onezone_yZ1_hist['[fe/h]'], onezone_yZ1_hist['[o/fe]'], c='g', label='best match y/Z=1 model')
    # ax0.plot(onezone_yZ2_hist['[fe/h]'], onezone_yZ2_hist['[o/fe]'], c='yellow', label='best match y/Z=2 model')
    ax0.set_xlabel('[Fe/H]')
    ax0.set_ylabel('[O/Fe]')
    ax0.set_xlim((-1.5, 0.6))
    ax0.set_ylim((-0.2, 0.6))
    ax0.legend(frameon=False)

    # In the right panels, plot abundance evolution
    ax1 = fig.add_subplot(gs[0,1])
    ax1.hexbin(local_sample['L23_AGE'], local_sample['FE_H'], **hexargs)
    ages = t_gyr[::-1]
    ax1.plot(ages, FeH_MR01, color='b')
    ax1.plot(yZ1_hist['lookback'], yZ1_hist['[fe/h]'], c='orange')
    ax1.plot(yZ2_hist['lookback'], yZ2_hist['[fe/h]'], c='r')
    # ax1.plot(onezone_yZ1_hist['lookback'], onezone_yZ1_hist['[fe/h]'], c='g')
    # ax1.plot(onezone_yZ2_hist['lookback'], onezone_yZ2_hist['[fe/h]'], c='yellow')
    ax1.set_ylabel('[Fe/H]')
    ax1.set_ylim((-1.5, 0.6))

    ax2 = fig.add_subplot(gs[1,1], sharex=ax1)
    ax2.hexbin(local_sample['L23_AGE'], local_sample['O_FE'], **hexargs)
    ax2.plot(ages, OFe_MR01, color='b')
    ax2.plot(yZ1_hist['lookback'], yZ1_hist['[o/fe]'], c='orange')
    ax2.plot(yZ2_hist['lookback'], yZ2_hist['[o/fe]'], c='r')
    # ax2.plot(onezone_yZ1_hist['lookback'], onezone_yZ1_hist['[o/fe]'], c='g')
    # ax2.plot(onezone_yZ2_hist['lookback'], onezone_yZ2_hist['[o/fe]'], c='yellow')
    ax2.set_xlabel('Age [Gyr]')
    ax2.set_ylabel('[O/Fe]')
    ax2.set_ylim((-0.2, 0.6))

    gs.tight_layout(fig)
    fig.savefig(paths.extra / 'twoinfall_literature_comparison.png')
    plt.close()


if __name__ == '__main__':
    main()
