"""
MLCSHE Runner.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
sys.path.append(os.path.dirname(__file__))  # nopep8

from src.main.MLCSHE import MLCSHE
from src.utils.utility import setup_logger
from benchmark.griegwangk import problem

import benchmark.griegwangk.search_config as cfg


# NOTE: MLCSHE is an algorithm, which is independent of a problem structure

def main():
    solver = MLCSHE(
        creator=problem.creator,
        toolbox=problem.toolbox,
        simulator=None,
        # more parameters can be added to better define the problem
        first_population_enumLimits=cfg.enumLimits,
        second_population_enumLimits=cfg.enumLimits
    )

    hyperparameters = [
        cfg.min_distance,
        cfg.tournament_selection,
        cfg.crossover_probability,
        cfg.guassian_mutation_mean,
        cfg.guassian_mutation_std,
        cfg.guassian_mutation_probability,
        cfg.integer_mutation_probability,
        cfg.bitflip_mutation_probability
    ]

    # Setup logger.
    setup_logger(file_name='CCEA_GRWGNK', file_log_level='DEBUG',
                 stream_log_level='INFO')

    # User does not need to modify anything but `problem.py`
    solution = solver.solve(
        max_gen=cfg.number_of_generations, hyperparameters=hyperparameters, max_num_evals=cfg.max_num_evals, seed=cfg.random_seed)


if __name__ == "__main__":
    main()
