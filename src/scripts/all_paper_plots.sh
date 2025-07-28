#!/bin/bash

# Generate APOGEE sample data table
python apogee_sample.py

# Tables and other output
python uncertainties.py
python yields.py

# Plots
python smooth_vs_twoinfall.py
python star_formation_history.py
python infall_parameters.py
python gas_abundance_evolution.py
python stellar_abundance_evolution.py
python mdf_evolution.py
python ofe_feh_density.py
python ofe_distributions.py
python ofe_feh_best.py
python lmr_ages.py
python sfe_hiatus.py
