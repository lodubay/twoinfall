# Challenges to the Two-Infall Scenario

Welcome to the repository for **Dubay et al. (2025, submitted), "Challenges
to the Two-Infall Scenario by Large Stellar Age Catalogues"**.

To re-build the article yourself, first download and extract the model outputs
from the Zenodo archive at [10.5281/zenodo.16649938](https://doi.org/10.5281/zenodo.16649938).
Place the extracted `multizone` directory within the `src/data` folder.

Alternatively, all 11 multi-zone models can be run by first navigating to the
`src/scripts` directory and running:
```
$ bash run_all_models.sh
```
Please allow approximately 6 hours for all the models to run.
**Note:** This will replace everything in the `src/data/multizone` directory, including
the output files from Zenodo.

To re-create all the plots and tables in the paper, run the following
within the `src/scripts` directory:
```
$ bash all_paper_plots.sh
```

Source code for the models is located within the [src/scripts/multizone/](/src/scripts/multizone/) directory.
To run a single multi-zone model with custom parameters, run the following:
```
$ cd src/scripts
$ python -m multizone [OPTIONS...]
```
