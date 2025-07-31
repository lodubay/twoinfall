#!/bin/bash

# Generate APOGEE sample data table
python apogee_sample.py

# Tables and other output
echo "Generating Table 2..."
python uncertainties.py
echo "Generating Table 4..."
python yields.py

# Plots
echo "Plotting Figure 1..."
python smooth_vs_twoinfall.py
echo "Plotting Figure 2..."
python star_formation_history.py
echo "Plotting Figure 3..."
python infall_parameters.py
echo "Plotting Figure 4..."
python gas_abundance_evolution.py
echo "Plotting Figure 5..."
python stellar_abundance_evolution.py
echo "Plotting Figure 6..."
python mdf_evolution.py
echo "Plotting Figure 7..."
python ofe_feh_density.py
echo "Plotting Figure 8..."
python ofe_distributions.py
echo "Plotting Figure 9..."
python ofe_feh_best.py
echo "Plotting Figure 10..."
python lmr_ages.py
echo "Plotting Figure 11..."
python sfe_hiatus.py
echo "Done!"
