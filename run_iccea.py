"""
iCCEA Runner.
"""

from datetime import datetime
import pathlib
import problem
import search_config as cfg
# from Simulator import Simulator
from simulation_manager_cluster import ContainerSimManager
from src.main.ICCEA import ICCEA
from src.utils.PairwiseDistance import PairwiseDistance
from src.utils.utility import setup_logger

# NOTE: ICCEA is an algorithm, which is independent of a problem structure


def main():
    # Instantiate simulator instance.
    # simulator = Simulator()
    sim_manager = ContainerSimManager(
        cfg.input_directory, cfg.output_directory)

    # Instantiate pairwise distance instance.
    pairwise_distance = PairwiseDistance(
        cs_list=[],
        numeric_ranges=cfg.numeric_ranges,
        categorical_indices=cfg.categorical_indices
    )

    pairwise_distance_scen = PairwiseDistance(
        cs_list=[],
        numeric_ranges=cfg.scenario_numeric_ranges,
        categorical_indices=cfg.scenario_catgorical_indices
    )

    pairwise_distance_mlco = PairwiseDistance(
        cs_list=[],
        numeric_ranges=cfg.mlco_numeric_ranges,
        categorical_indices=cfg.mlco_categorical_indices
    )

    solver = ICCEA(
        creator=problem.creator,
        toolbox=problem.toolbox,
        # simulator=simulator,
        simulator=sim_manager,
        pairwise_distance_cs=pairwise_distance,
        # more parameters can be added to better define the problem
        pairwise_distance_p1=pairwise_distance_scen,
        pairwise_distance_p2=pairwise_distance_mlco,
        first_population_enumLimits=cfg.scenario_enumLimits,
        update_archive_strategy=cfg.update_archive_strategy,
        fitness_function_target_probability=cfg.fitness_function_target_probability
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
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = pathlib.Path('results').joinpath(
    #     timestamp + '_' + cfg.output_dir_name)
    output_dir = pathlib.Path('results').joinpath(cfg.output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output folder.

    # Setup logger.
    setup_logger(file_name='CCEA', output_directory=output_dir, file_log_level='DEBUG',
                 stream_log_level='INFO')

    # User does not need to modify anything but `problem.py`
    solution = solver.solve(
        max_gen=cfg.number_of_generations,
        hyperparameters=hyperparameters,
        max_num_evals=cfg.max_num_evals,
        radius=cfg.region_radius,
        output_dir=output_dir,
        seed=cfg.random_seed)


if __name__ == "__main__":
    main()
