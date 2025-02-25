#!/bin/bash

# Tables and other output
python apogee_regions_table.py
python yields.py

# Plots
python yield_outflow.py
python onezone_params.py
python spitoni_comparison.py
python sfe_prefactor.py
python star_formation_history.py
python ofe_df_comparison.py
python ofe_feh_density.py
python abundance_evolution.py
python mdf_evolution.py
python onezone_sfe_hiatus.py
