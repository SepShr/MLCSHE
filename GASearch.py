"""
Genetic Algorithm Search
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
sys.path.append(os.path.dirname(__file__))  # nopep8

from deap import base, creator, tools
import numpy as np
import logging
import time
import pathlib
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import repeat
from multiprocessing import cpu_count
from pathlib import Path

import search_config as cfg
from src.main.fitness_function import calculate_fitness
from problem_utils import initialize_mlco, mutate_mlco, mutate_scenario
from simulation_manager_cluster import (ContainerSimManager,
                                        prepare_for_computation,
                                        start_computation)
from src.utils.PairwiseDistance import PairwiseDistance
from src.utils.utility import (create_complete_solution, flatten_list,
                               initialize_hetero_vector, log_and_pickle, setup_file, setup_logbook_file, setup_logger)

JOBS_QUEUE_SIZE = 10


class GASearch:
    def __init__(self, scen_enumLimits, radius,
                 sim_input_dir, sim_output_dir):
        self.scen_enumLimits = scen_enumLimits
        self.radius = radius
        self.sim_input_dir = sim_input_dir
        self.output_directory = sim_output_dir

        self.simulator = ContainerSimManager(
            self.sim_input_dir, self.output_directory)

        # Initialize lists.
        self.scenario_list = []
        self.mlco_list = []
        self.cs_archive = []
        self.pop = []

        self.sim_index = 1

        self._logger = logging.getLogger('GASearch')
        self._logbook_file = setup_logbook_file(
            output_dir=self.output_directory)
        self.cs_list_gen_file = setup_file(
            '_cs_list_gen', self.output_directory)

        self.pairwise_distance = PairwiseDistance(
            cs_list=[],
            numeric_ranges=cfg.numeric_ranges,
            categorical_indices=cfg.categorical_indices
        )
        self.creator = creator

        self.sim_counter = 0
        self.jobs_size = JOBS_QUEUE_SIZE

        self.ff_target_prob = cfg.finess_function_target_probability

    def define_problem(self, pop_size, tourn_size=3, cxpb=0.85, mutpb=0.01, mutgmu=0.0, mutgsig=40):
        """Define the problem.
        """
        # Log the hyperparameters.
        self._logger.info('tourn_size={}'.format(tourn_size))
        self._logger.info('cxpb={}'.format(cxpb))
        self._logger.info('mutpb={}'.format(mutpb))
        self._logger.info('mutgmu={}'.format(mutgmu))
        self._logger.info('mutgsig={}'.format(mutgsig))

        # Define problem and individuals.
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list,
                            fitness=self.creator.FitnessMin, safety_req_value=float)
        self.creator.create("Scenario", self.creator.Individual)
        self.creator.create("OutputMLC", self.creator.Individual)

        # Adding multiple statistics.
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        # Instantiating logbook that records search-specific statistics.
        self.logbook = tools.Logbook()

        # Format logbook.
        self.logbook.header = "gen", "min", "avg", "max", "std"

        counter = 1  # Initialize counter.

        # Randomly create complete solutions.
        while counter <= pop_size:

            # Create a random scenario.
            scenario = initialize_hetero_vector(
                self.scen_enumLimits, creator.Scenario)
            # self.scenario_list.append(scenario)

            # Create a random mlco.
            mlco = initialize_mlco(creator.OutputMLC)
            # self.mlco_list.append(mlco)

            # Create a complete solution.
            complete_solution = creator.Individual(
                create_complete_solution(scenario, mlco, creator.Scenario))

            self.pop.append(complete_solution)

            counter += 1

            self.toolbox = base.Toolbox()

            self.toolbox.register(
                "select", tools.selTournament, tournsize=tourn_size, fit_attr='fitness')
            self.toolbox.register("mate", tools.cxUniform, indpb=cxpb)
            self.toolbox.register("mutate", self.mutate_cs, mutgmu=mutgmu,
                                  mutgsig=mutgsig, mutgpb=mutpb, mutipb=mutpb, mutbpb=mutpb, scenario_enumLimits=self.scen_enumLimits)

    def run(self, pop_size, max_gen, max_evals):
        """Runs the genetic algorithm search.
        """
        # Setup logger.
        self._logger.info('pop_size={}'.format(pop_size))
        self._logger.info('max_gen={}'.format(max_gen))
        self._logger.info('max_evals={}'.format(max_evals))

        start_time = time.time()

        for num_gen in range(1, max_gen + 1):
            if num_gen == 1:
                # Initialize the first generation.
                self.define_problem(pop_size)
                # self._logger.info('Generation {} initialized.'.format(num_gen))
            else:
                # Evolve the next generation.
                self.breed_population()

            self._logger.info('current_generation={}'.format(num_gen))
            self._logger.info('population={}'.format(self.pop))

            # Evaluate the entire population.
            self.evaluate_population(self.pop)

            self.record_results(num_gen=num_gen)

            # Record the list of complete solutions per generation.
            with open(self.cs_list_gen_file, 'at') as csgf:
                csgf.write(
                    '--------------- GENERATION_NUMBER {} ---------------\n'.format(num_gen))
                for cs in self.cs_archive:
                    csgf.write('cs={} | jfit_value={} | safety_req_value={}\n'.format(
                        cs, cs.fitness.values[0], cs.safety_req_value))

            self._logger.info("Results recorded.")

            if self.sim_counter >= max_evals:
                self._logger.info("Maximum number of evaluations reached.")
                break

            current_time = time.time()
            elapsed_time = current_time - start_time
            self._logger.info('elapsed_time={}'.format(elapsed_time))

        end_time = time.time()
        total_time = end_time - start_time
        self._logger.info('total_search_time={}'.format(total_time))
        self._logger.info("End of GA search.")

    def breed_population(self):
        """Breed the population.
        """
        # Select the next generation individuals.
        selected_offspring = self.toolbox.select(self.pop, len(self.pop))
        self._logger.info(f'offspring_after_selection={selected_offspring}')
        # Clone the selected individuals.
        offspring = list(map(self.toolbox.clone, selected_offspring))
        # Apply crossover and mutation on the offspring.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            self.toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        print(f'offspring_after_mate={offspring}')
        mutated_offspring = []
        for mutant in offspring:
            mutant = self.toolbox.mutate(cs=mutant)
            del mutant.fitness.values
            mutated_offspring.append(mutant)

        # The population is entirely replaced by the offspring.
        self.pop[:] = mutated_offspring

    def evaluate_population(self, pop=None):
        """Evaluates the joint fitness of a complete solution.

        It takes the complete solution as input and returns its joint
        fitness as output.
        """
        jobs = [i for i in pop if not self.individual_in_list(
            i, self.cs_archive)]

        if len(jobs) > 0:
            self.sim_index, results = self.compute_safety_value_cs_list(
                self.simulator, jobs, self.sim_index)

            for c, result in zip(jobs, results):
                DfC_min, DfV_min, DfP_min, DfM_min, DT_min, traffic_lights_max = result
                c.safety_req_value = DfV_min
                self._logger.info('safety_req_value={}'.format(DfV_min))
                self.sim_counter += 1
                self._logger.info(
                    f'{self.sim_counter} simulations completed.')

            # Calculate the pairwise distance between all simulated complete solutions.
            self.pairwise_distance.update_dist_matrix(jobs)
            self._logger.info('Pairwise distance matrix updated.')

            # Add items of jobs to self.cs_archive.
            for job in jobs:
                self.cs_archive.append(job)

        else:
            self._logger.info('No new complete solutions to evaluate.')

        self.calculate_fitness(self.cs_archive)
        for i in self.pop:
            for c in self.cs_archive:
                if self.individual_is_equal(i, c):
                    i.fitness.values = c.fitness.values

    def compute_safety_value_cs_list(self, simulator, cs_list, sim_index):
        updated_sim_index = prepare_for_computation(
            cs_list, simulator, sim_index)
        results = start_computation(simulator)
        return updated_sim_index, results

    def calculate_fitness(self, pop=None):
        # Calculate fitness values for all complete solutions in parallel.
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            for cs, result in zip(pop, executor.map(calculate_fitness, pop, repeat(self.pairwise_distance.cs_list), repeat(self.pairwise_distance.dist_matrix_sq), repeat(self.radius), repeat(self.ff_target_prob))):
                cs.fitness.values = (result,)
        self._logger.info('Fitness values calculated.')

    def record_results(self, num_gen=None):
        # Record best solution.
        best_solution = sorted(
            self.pop, key=lambda x: x.fitness.values[0])[0]
        self._logger.info(
            'best_complete_solution={} | fitness={}'.format(best_solution, best_solution.fitness.values[0]))

        log_and_pickle(object=self.pop, file_name='_GA_pop',
                       output_dir=self.output_directory)

        log_and_pickle(object=self.cs_archive, file_name='_GA_cs',
                       output_dir=self.output_directory)

        # Compiling statistics on the populations and completeSolSet.
        record_complete_solutions = self.stats.compile(self.pop)

        self.logbook.record(gen=num_gen, **record_complete_solutions)
        print(self.logbook.stream)
        log_and_pickle(object=self.logbook, file_name='_GA_logbook',
                       output_dir=self.output_directory)

    def mutate_cs(self, cs, mutgmu, mutgsig, mutgpb, mutipb, mutbpb, scenario_enumLimits):
        """Mutates a complete solution.

        It takes the complete solution as input and returns the mutated
        complete solution as output.
        """
        cs_class = type(cs)
        scen, mlco = cs

        # Mutate the scenario.
        scen = mutate_scenario(scen, scenario_enumLimits,
                               mutbpb, mutgmu, mutgsig, mutgpb, mutipb)
        # Mutate the mlco.
        mlco = mutate_mlco(mlco, mutgmu, mutgsig, mutgpb, mutipb)

        return cs_class([scen, mlco])

    def individual_in_list(self, individual, individuals_list):
        for cs in individuals_list:
            if self.individual_is_equal(individual, cs):
                return True
        return False

    def individual_is_equal(self, base_individual, target_individual):
        base = flatten_list(base_individual)
        target = flatten_list(target_individual)

        if base == target:
            return True
        else:
            return False


def main(sim_budget, pop_size, max_gen):
    # Setup directories
    # Get current timestamp to use as a unique ID.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = str(timestamp) + '_GA_Pylot'
    input_directory = Path.cwd().joinpath('temp').joinpath(output_dir_name)
    output_directory = Path.cwd().joinpath('results').joinpath(output_dir_name)

    output_dir = pathlib.Path('results').joinpath(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output folder.

    # Setup logger.
    setup_logger(file_name='GA', output_directory=output_dir, file_log_level='DEBUG',
                 stream_log_level='INFO')

    logger = logging.getLogger('GA')
    logger.info("Starting GA search.")
    logger.info(f'sim_budget={sim_budget}')
    logger.info(f'pop_size={pop_size}')
    logger.info(f'max_gen={max_gen}')

    # Instantiate the random search object.
    ga_search = GASearch(scen_enumLimits=cfg.scenario_enumLimits, radius=cfg.region_radius,
                         sim_input_dir=input_directory, sim_output_dir=output_directory)

    # Run the random search.
    ga_search.run(pop_size=pop_size, max_gen=max_gen, max_evals=sim_budget)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('sim_budget', nargs='?', type=int,
                           default=cfg.max_num_evals, help="simulation budget")
    argparser.add_argument('pop_size', nargs='?', type=int,
                           default=20, help="popuation size")
    argparser.add_argument('max_gen', nargs='?', type=int,
                           default=100, help="max number of generations")
    args = argparser.parse_args()
    main(sim_budget=args.sim_budget, pop_size=args.pop_size, max_gen=args.max_gen)
