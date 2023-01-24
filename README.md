# MLCSHE: Machine Learning Component Systemic Hazard Envelope

ML Component Systemic Hazard Envelope project, or *MLCSHE* (pronounced /'mɪlʃ/), is a cooperative coevolutionary search algorithm that automatically identifies the *hazard boundary* of a ML component in an ML-enabled Autonomous System (MLAS), given a system safety requirement.

## Publications

__Identifying the Hazard Boundary of ML-enabled Autonomous Systems__ by Sepehr Sharifi, Donghwan Shin, and Lionel C. Briand, arXiv pre-prints, January 2023, DOI:XXXX

## Installation

The details on setting up MLCSHE are provided in the [installation guide](INSTALL.md).

## Usage

- To run MLCSHE, you can use the following command:

```bash
python run_mlcshe.py
```

*NOTE 1*: the parameters of the search can be updated in `~/pylot/search_config.py`. The parameters of the simulation can be updated in `~/simulation_config.py`.

*NOTE 2*: the simulations are run in `2` parallel jobs. If you want to run more jobs, you can update the `num_jobs` variable in `~/simulation_config.py`. However, you need to make sure that your machine has enough resources (especially RAM and GPU memory) to run the jobs in parallel.

- To run Random Search, you can use the following command:

```bash
python run_random_search.py <SIM_BUDGET>
```

where `<SIM_BUDGET>` is the number of simulations to run.

- To run Genetic Algorithm Search, you can use the following command:

```bash
python run_ga_search.py <SIM_BUDGET> <POP_SIZE> <MAX_NUM_GEN>
```

where `<SIM_BUDGET>`, `<POP_SIZE>` and `<MAX_NUM_GEN>` are the number of simulations to run, the population size and the maximum number of generations, respectively.


This work is done at [Nanda Lab](https://www.nanda-lab.ca/), EECS Department, University of Ottawa.
