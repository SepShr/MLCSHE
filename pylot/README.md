# Pylot

We use Pylot to evaluate the effectiveness and efficiency of MLCSHE. The encodings of Scenarios and ML Component Outputs are defined in [`~/pylot/ENCODINGS.md`](ENCODINGS.md). The postprocess script is provided in `~/pylot/postprocess.ipynb`. The evaluation results are stored in `~/pylot/results.zip`.

# Reproducibility Instructions

To run the experiments, you need to have set up the environment as described in the [installation guide](INSTALL.md).

To run MLCSHE, random search, or genetic algorithm, refer to the [usage section](#usage).

# Evaluation Results

The results of the evaluations including the simulation logs, raw data, processed data (as `pandas.DataFrame`), and figures are stored in `~/pylot/results.zip`.
The folder contains the postprocess results (i.e., dataframes and figures) in the `/postprocess` folder and the results of the experiments (i.e., raw data and simulation logs) in the `~/results/pylot_<EXPERIMENT_NAME>_Results.zip` folders.

To reproduce the postprocess results, unzip the `~/results.zip` file to the `~/pylot` folder. Then use the `postprocess.ipynb` notebook to reproduce the results.

The dataframes that contain the answers to *RQ1* and *RQ2* are recorded as pickled files in the `~/pylot/results/postprocess` folder. The figures that visualize the answers to *RQ1* and *RQ2* are recorded as `png` and `pdf` files in the `~/pylot/results/postprocess/diagrams` folder.