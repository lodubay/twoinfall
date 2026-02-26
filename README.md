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

Source code for the models is located within the [src/scripts/multizone/](/src/scripts/multizone/) directory.
To run a single multi-zone model with custom parameters, run the following:
```
$ cd src/scripts
$ python -m multizone [OPTIONS...]
```
