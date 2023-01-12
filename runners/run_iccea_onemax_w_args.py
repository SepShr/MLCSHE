"""
MLCSHE Runner.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
sys.path.append(os.path.dirname(__file__))  # nopep8

from datetime import datetime
import pathlib
from benchmark.onemax import problem
from src.main.MLCSHE import MLCSHE
from src.utils.PairwiseDistance import PairwiseDistance
from src.utils.utility import setup_logger
import sys
import importlib

# NOTE: ICCEA is an algorithm, which is independent of a problem structure


def main():
    # Instantiate simulator instance.
    simulator = None

    # Instantiate pairwise distance instance.
    pairwise_distance_cs = PairwiseDistance(
        cs_list=[],
        numeric_ranges=cfg.numeric_ranges,
        categorical_indices=cfg.categorical_indices
    )

    pairwise_distance_scen = PairwiseDistance(
        cs_list=[],
        numeric_ranges=cfg.numeric_ranges_scen,
        categorical_indices=[]
    )

    pairwise_distance_mlco = PairwiseDistance(
        cs_list=[],
        numeric_ranges=cfg.numeric_ranges_mlco,
        categorical_indices=[]
    )

    try:
        update_archive_strategy = cfg.update_archive_strategy
    except:
        update_archive_strategy = 'best random'

    config_values = {
        'min_distance': cfg.min_distance,
        'tournament_selection': cfg.tournament_selection,
        'crossover_probability': cfg.crossover_probability,
        'guassian_mutation_mean': cfg.guassian_mutation_mean,
        'guassian_mutation_std': cfg.guassian_mutation_std,
        'guassian_mutation_probability': cfg.guassian_mutation_probability,
        'integer_mutation_probability': cfg.integer_mutation_probability,
        'bitflip_mutation_probability': cfg.bitflip_mutation_probability,
        'population_archive_size': cfg.population_archive_size,
        'scenario_population_size': cfg.scenario_population_size,
        'mlco_population_size': cfg.mlco_population_size,
        'enumLimits': cfg.enumLimits
    }

    # Pass the config values to problem.py to setup the search funcitons.
    creator, toolbox = problem.setup_problem(config_values)

    solver = MLCSHE(
        creator=creator,
        toolbox=toolbox,
        simulator=simulator,
        pairwise_distance_cs=pairwise_distance_cs,
        # more parameters can be added to better define the problem
        pairwise_distance_p1=pairwise_distance_scen,
        pairwise_distance_p2=pairwise_distance_mlco,
        first_population_enumLimits=cfg.enumLimits,
        second_population_enumLimits=cfg.enumLimits,
        update_archive_strategy=update_archive_strategy
    )

    hyperparameters = [
        cfg.min_distance,
        cfg.tournament_selection,
        cfg.crossover_probability,
        cfg.guassian_mutation_mean,
        cfg.guassian_mutation_std,
        cfg.guassian_mutation_probability,
        cfg.integer_mutation_probability,
        cfg.bitflip_mutation_probability,
        cfg.population_archive_size
    ]

    # Get current timestamp to use as a unique ID.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path('results').joinpath(
        timestamp + '_' + cfg.output_dir_name)
    # Create the output folder.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger.
    setup_logger(
        file_name='CCEA_ONEMAX',
        output_directory=output_dir,
        file_log_level='DEBUG',
        stream_log_level='INFO'
    )

    # User does not need to modify anything but `problem.py`
    solution = solver.solve(
        max_gen=cfg.number_of_generations,
        hyperparameters=hyperparameters,
        max_num_evals=cfg.max_num_evals,
        radius=cfg.region_radius,
        output_dir=output_dir,
        seed=cfg.random_seed
    )


if __name__ == "__main__":
    # Import the config file given its path.
    module = sys.argv[1].rstrip('\r')
    cfg = importlib.import_module(module)
    print(f'{module} imported as cfg.')
    main()
