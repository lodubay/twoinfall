# Challenges to the Two-Infall Scenario

Welcome to the repository for **Dubay et al. (2026), "Challenges
to the Two-Infall Scenario by Large Stellar Age Catalogues"** (accepted
for publication in The Astrophysical Journal).
Read the paper at [arXiv:2508.00988](https://arxiv.org/abs/2508.00988).

To re-build the article yourself, first ensure you have the right
packages by creating a Conda environment from the `environment.yml` file:
```
$ conda env create -f environment.yml
$ conda activate twoinfall
```

Next, download and extract the model outputs from the Zenodo archive at 
[10.5281/zenodo.16649938](https://doi.org/10.5281/zenodo.16649938).
Place the extracted `multizone` directory within the `src/data` folder.

Alternatively, all 11 multi-zone models can be run from scratch:
```
$ ulimit -n 30000
$ bash run_all_models.sh
```
Please allow approximately 6 hours for all the models to run.
**Note:** This will replace everything in the `src/data/multizone` directory, including
the output files from Zenodo.

To re-create all the plots and tables in the paper, run the following:
```
$ bash all_paper_plots.sh
```

## Repository Structure

```
.
├── src
│   ├── data                # Catalog files and model outputs (ignored by git)
│   ├── extra               # Non-paper figures and other outputs (ignored by git)
│   ├── scripts             # Python scripts for figures and models
│   ├── ├── multizone       # Source code for multi-zone chemical evolution models
│   ├── tex                 # LaTeX files
│   ├── ├── figures         # Programatically-generated figures
│   ├── ├── output          # Programatically-generated tables and other outputs
├── all_paper_plots.sh      # Script to produce all paper figures
├── run_all_models.sh       # Script to run all multi-zone models
├── environment.yml         # Package dependencies
├── LICENSE
└── README.md
```

Note that some scripts, such as [multizone_plots.py](src/scripts/multizone_plots.py), generate
additional figures for multi-zone model outputs that not included in the manuscript.
The output of these scripts is saved in [src/extra](src/extra).

Source code for the models is located within the [multizone](/src/scripts/multizone/) directory.
To run a single multi-zone model with custom parameters, run the following:
```
$ cd src/scripts
$ python -m multizone [OPTIONS...]
```

## Software Dependencies

This code is built on the [Versatile Integrator for Chemical Evolution (VICE)](https://github.com/giganano/vice)
package [(Johnson & Weinberg 2020)](https://doi.org/10.1093/mnras/staa2431). VICE is necessary for the code
to run, and should be installed automatically using `environment.yml`.

We also include a copy of the [Chemical Evolution Analysis Package (ChEAP)](https://bitbucket.org/pedroap/cheap/)
developed by [Palicio et al. (2023)](https://doi.org/10.1051/0004-6361/202346567). This is included at
[src/scripts/CheapTools.py](src/scripts/CheapTools.py) for convenience, so it does not need to be
installed separately.
