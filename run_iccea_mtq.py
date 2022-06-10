"""
iCCEA Runner.
"""
from src.main.ICCEA import ICCEA
from src.utils.PairwiseDistance import PairwiseDistance
from src.utils.utility import setup_logger
from benchmark.mtq import problem
from simulation_runner import Simulator

import benchmark.mtq.search_config as cfg


# NOTE: ICCEA is an algorithm, which is independent of a problem structure

def main():
    # Instantiate simulator instance.
    simulator = Simulator()

    # Instantiate pairwise distance instance.
    pairwise_distance = PairwiseDistance(
        cs_list=[],
        numeric_ranges=cfg.numeric_ranges,
        categorical_indices=cfg.categorical_indices
    )

    solver = ICCEA(
        creator=problem.creator,
        toolbox=problem.toolbox,
        simulator=simulator,
        pairwise_distance_cs=pairwise_distance,
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
    setup_logger(file_name='CCEA_MTQ', file_log_level='DEBUG',
                 stream_log_level='INFO')

    # User does not need to modify anything but `problem.py`
    solution = solver.solve(
        max_gen=cfg.number_of_generations,
        hyperparameters=hyperparameters,
        max_num_evals=cfg.max_num_evals,
        radius=cfg.region_radius,
        seed=cfg.random_seed
    )


if __name__ == "__main__":
    main()
