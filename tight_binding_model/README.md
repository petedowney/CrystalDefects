# Tight Binding Model

This folder iteratively solves for the tight binding coefficients from the density of states.

## Setup

1. Create a folder and put all data into `all_dos_data_full`.
2. Run `find_spin_neutral.py` and `filter_spin_neutral.py` to get a valid set of paths to run on. *(Note: These two Python files need to be combined into a single one)*.

## Local Execution

Run all cells in `create_hamiltonians.ipynb`. I would recommend running locally with a subset of the data to generate an initial set of onsite energies and hopping parameters.

## Cluster Execution

To run on a cluster, the setup is the same. Then run `run_cluster.sh` with the associated `sbatch` parameters.

> **Note:** Some of the save/load paths will likely need to be changed to successfully run.
