import logging
import os
import pathlib
import pickle
import time
from datetime import datetime

import numpy as np
from deap import base, creator, tools

import search_config as cfg
from problem_utils import initialize_mlco, problem_joint_fitness
from simulation_runner import Simulator
from src.utils.utility import (create_complete_solution,
                               initialize_hetero_vector, setup_logbook_file,
                               setup_logger)

TIME_BUDGET = 86_400  # Search time budget in seconds (currently == 24 hr).

scen_pop_size = cfg.scenario_population_size  # Size of the scenario population
mlco_pop_size = cfg.mlco_population_size  # Size of the MLC output population
min_distance = cfg.min_distance  # Minimum distance between members of an archive

# The list of lower and upper limits for enumerationed types in sceanrio.
scen_enumLimits = cfg.scenario_enumLimits


def evaluate_joint_fitness(simulator, toolbox, c):
    """Evaluates the joint fitness of a complete solution.

    It takes the complete solution as input and returns its joint
    fitness as output.
    """
    # For benchmarking problems.
    # x = c[0][0]
    # y = c[1][0]

    x = c[0]
    y = c[1]

    joint_fitness_value = toolbox.problem_jfit(simulator, x, y)

    return (joint_fitness_value,)


def setup_file(file_name: str):
    """
    Sets up a formatted file name in the results folder with `file_name`.
    """
    # Create the results folder if it does not exist.
    pathlib.Path('results/').mkdir(parents=True, exist_ok=True)

    # Get current timestamp to use as a unique ID.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_id = str(timestamp) + file_name + '.log'
    file = os.path.join('results', file_id)

    return file


def main():
    # Setup logger.
    logger = logging.getLogger(__name__)

    logger.info("Random search started.")
    logger.info('time_budget = {}'.format(TIME_BUDGET))

    # Define problem and individuals.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    creator.create("Scenario", creator.Individual)
    creator.create("OutputMLC", creator.Individual)

    # Instantiate toolbox.
    toolbox = base.Toolbox()

    # Define functions and register them in toolbox.
    toolbox.register(
        "scenario", initialize_hetero_vector,
        class_=creator.Scenario, limits=scen_enumLimits
    )

    toolbox.register(
        "mlco", initialize_mlco,
        creator.OutputMLC
    )

    toolbox.register("problem_jfit", problem_joint_fitness)

    # Adding multiple statistics.
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Instantiating logbook that records search-specific statistics.
    logbook = tools.Logbook()

    # Format logbook.
    logbook.header = "gen", "min", "avg", "max", "std"

    # Initialize simulator.
    simulator = Simulator()

    # Initialize lists.
    scenarios = []
    mlcos = []
    complete_solutions = []

    counter = 1  # Initialize counter.

    # Initialize start and end time.
    start_time = time.time()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Search the scenario and mlco parameter space randomly until the time is up.
    while elapsed_time < TIME_BUDGET:

        # Create a random scenario.
        scenario = initialize_hetero_vector(scen_enumLimits, creator.Scenario)
        # logger.debug('scenario={}'.format(scenario))
        scenarios.append(scenario)

        # Create a random mlco.
        mlco = initialize_mlco(creator.OutputMLC)
        # logger.debug("mlco={}".format(mlco))
        mlcos.append(mlco)

        # Create complete solution.
        complete_solution = creator.Individual(
            create_complete_solution(scenario, mlco, creator.Scenario))

        # Evaluate complete solution.
        complete_solution.fitness.values = evaluate_joint_fitness(
            simulator, toolbox, complete_solution)
        logger.info("joint fitness evaluation complete")
        logger.info('complete_solution.fitness.values={}'.format(
            complete_solution.fitness.values))

        complete_solutions.append(complete_solution)

        # Report the best complete solution.
        best_solution = sorted(
            complete_solutions, key=lambda x: x.fitness.values[0])[0]
        logger.info(
            'best_complete_solution={} | fitness={}'.format(best_solution, best_solution.fitness.values[0]))

        # Compiling statistics on the populations and completeSolSet.
        record_complete_solutions = stats.compile(complete_solutions)

        logbook.record(gen=counter, **record_complete_solutions)
        print(logbook.stream)

        end_time = time.time()

        counter += 1

        logger.info('complete_solutions_size={}'.format(
            len(complete_solutions)))

        elapsed_time = end_time - start_time
        logger.info('elapsed_time={}'.format(elapsed_time))
        logger.info('remaining_time={}'.format(TIME_BUDGET - elapsed_time))

    # Record the list of found sceanrios.
    scenarios_file = setup_file('_scenarios')
    with open(scenarios_file, 'wb') as s_file:
        pickle.dump(scenarios, s_file)

    # Record the list of found complete_solutions.
    mlcos_file = setup_file('_mlcos')
    with open(mlcos_file, 'wb') as m_file:
        pickle.dump(mlcos, m_file)

    # Record the list of found complete_solutions.
    complete_solutions_file = setup_file('_complete_solutions')
    with open(complete_solutions_file, 'wb') as cs_file:
        pickle.dump(complete_solutions, cs_file)

    # Record the complete logbook.
    logbook_file = setup_logbook_file()
    with open(logbook_file, 'wb') as lb_file:
        pickle.dump(logbook, lb_file)

    logger.info("End of random search.")


if __name__ == "__main__":
    setup_logger(file_name='RS')
    main()
