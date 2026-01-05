# Cheminformatics characterization of Pseudo-Natural Products inspired by aryloctahydroindole alkaloids

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository contains the code and raw data to reproduce the cheminformatics characterization of Pseudo-Natural Products inspired by aryloctahydroindole alkaloids as described in our manuscript **Design, Synthesis and Biological Evaluation of Pseudo-Natural Products Inspired by Aryloctahydroindole Alkaloids** ([Waldmann *et al.*, ChemRxiv **2025**](https://chemrxiv.org/engage/chemrxiv/article-details/692d9031a10c9f5ca1998cef)).

## Requirements

For reproducibility purposes, it is highly recommended to run the code in a dedicated conda environment. Only common Python libraries were used for our cheminformatics analyses as indicated in the `environment.yml` file. To set up the environment, first install a conda distribution (e.g. miniconda; follow instructions in the [official Anaconda website](https://www.anaconda.com/docs/main)). Afterward, clone the repository and create the environment as usual:

```bash
$ git clone https://gitlab.gwdg.de/mpi-dortmund/comas/2025_greiner_pseudo-np.git
$ cd 2025_greiner_pseudo-np
$ conda env create -f environment.yml
```

## Contents

The code used for our analyses is presented as Python scripts ([scripts](./scripts/) folder). A short description of each script is presented below.

- `01_retrieve_nps_chembl.py`: this script presents a programatic access to the **ChEMBL 35** database using `SQLite`. It retrieves all compounds flagged as natural products.
- `02_prepare_datasets.py`: it helps preparing the different datasets used in the manuscript, including ChEMBL natural products, compounds reported in the **DrugBank**, as well as a 50k sample from the *Enamine Advanced Screening Collection*. Preparation of alkaloid families including monoterpene alkaloids and Amaryllidaceae alkaloids reported in ChEMBL as well as Hasubanan alkaloids reported in the **COCONUT** database is alse included. The prepare datasets are stored as CSV files with standardized SMILES.
- `03_feature_calculation.py`: it allows calculating a set of 17 descriptors (molecular, atomic and drug-like features) for all the compounds sets. Results are stored to file.
- `04_pca.py`: used to run Principal Component Analysis on the descriptors obtained by `03_feature_calculation.py`. Both PCA loadings and explained variance are saved to disk.
- `05_get_scores.py`: it calculates different drug-like scores, including the NP likeness, Quantitative Estimate of Drug-likeness (QED), Böttcher score, and the normalized Spacial Score (SPS).
- `06_get_pmis.py`: the last script perform molecular embedding, a simple force field energy minimization, and subsequent calculation of the Principal Moments of Inertia (PMIs) as defined in the `RDKit`.

The data obtained from running those scripts is found in the [reports](./reports/) folder.

## Citation

If you use this code or parts of the content from this repository, please cite it as:

```
@article{Waldmann_2025,
  doi = {10.26434/chemrxiv-2025-7s9lp},
  url = {https://chemrxiv.org/engage/chemrxiv/article-details/692d9031a10c9f5ca1998cef},
  year = {2025},
  month = {december},
  publisher = {Cambridge University Press},
  volume = {},
  pages = {},
  author = {Waldmann, Herbert; Greiner, Luca C.; Bernal, Freddy A.; Thavam, Sasikala; Brachtshäuser, Maite; Sievers, Sonja; Ziegler, Slava},
  title = {Design, Synthesis and Biological Evaluation of Pseudo-Natural Products Inspired by Aryloctahydroindole Alkaloids},
  journal = {ChemRxiv}
}
```
