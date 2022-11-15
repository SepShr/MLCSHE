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
from copy import deepcopy
import pathlib
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import repeat
from multiprocessing import cpu_count
from pathlib import Path

import search_config as cfg
from fitness_function import fitness_function
from problem_utils import initialize_mlco
from simulation_manager_cluster import (ContainerSimManager,
                                        prepare_for_computation,
                                        start_computation)
from src.utils.PairwiseDistance import PairwiseDistance
from src.utils.utility import (create_complete_solution,
                               initialize_hetero_vector, log_and_pickle, setup_logbook_file, setup_logger)


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

        self.sim_index = 1

        self._logger = logging.getLogger(__name__)
        self._logbook_file = setup_logbook_file(
            output_dir=self.output_directory)

        self.pairwise_distance = PairwiseDistance(
            cs_list=[],
            numeric_ranges=cfg.numeric_ranges,
            categorical_indices=cfg.categorical_indices
        )
        self.creator = creator

        self.sim_counter = 0
        self.jobs_size = cfg.jobs_queue_size

    def define_problem(self, pop_size):
        """Define the problem.
        """
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

            self.toolbox.register("select", tools.selTournament, tournsize=3)
            self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
            self.toolbox.register("mutate", self.mutate_cs)

    def run(self, pop_size, cxpb, mutpb, max_gen, max_evals, ncores, seed):
        """Runs the genetic algorithm search.
        """
        # Setup logger.
        self._logger.info("Starting GA search.")
        self._logger.info('pop_size={}'.format(pop_size))
        self._logger.info('max_gen={}'.format(max_gen))
        self._logger.info('max_evals={}'.format(max_evals))
        self._logger.info('cxpb={}'.format(cxpb))
        self._logger.info('mutpb={}'.format(mutpb))

        start_time = time.time()

        for num_gen in range(1, max_gen + 1):
            if num_gen == 1:
                # Initialize the first generation.
                self.define_problem(pop_size)
                self._logger.info('Generation {} initialized.'.format(num_gen))
                self._logger.info('population={}'.format(self.pop))
            else:
                # Select the next generation individuals.
                offspring = self.toolbox.select(self.pop, len(self.pop))
                # Clone the selected individuals.
                offspring = list(map(self.toolbox.clone, offspring))

                # Apply crossover and mutation on the offspring.
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if np.random.random() < cxpb:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if np.random.random() < mutpb:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # The population is entirely replaced by the offspring.
                self.pop[:] = offspring

            self._logger.info('current_generation={}'.format(num_gen))
            # Evaluate the entire population.
            self.evaluate_population()

            self.record_results()
            self._logger.info("Results recorded.")

            if self.sim_counter >= max_evals:
                break

            current_time = time.time()
            elapsed_time = current_time - start_time
            self._logger.info('elapsed_time={}'.format(elapsed_time))

        end_time = time.time()
        total_time = end_time - start_time
        self._logger.info('total_search_time={}'.format(total_time))
        self._logger.info("End of GA search.")

    def evaluate_population(self, jobs=None):
        """Evaluates the joint fitness of a complete solution.

        It takes the complete solution as input and returns its joint
        fitness as output.
        """
        self.sim_index, results = self.compute_safety_value_cs_list(
            self.simulator, jobs, self.sim_index)

        for c, result in zip(jobs, results):
            DfC_min, DfV_min, DfP_min, DfM_min, DT_min, traffic_lights_max = result
            c.safety_req_value = DfV_min
            self._logger.info('safety_req_value={}'.format(DfV_min))
            self.sim_counter += 1
            self._logger.info(
                f'{self.sim_counter} simulations completed out of {len(self.cs_archive)}.')

        # Calculate the pairwise distance between all simulated complete solutions.
        self.pairwise_distance.update_dist_matrix(jobs)
        self._logger.info('Pairwise distance matrix updated.')

        # Add items of jobs to self.completed_jobs.
        for job in jobs:
            self.completed_jobs.append(job)

        self.calculate_fitness()

    def compute_safety_value_cs_list(self, simulator, cs_list, sim_index):
        updated_sim_index = prepare_for_computation(
            cs_list, simulator, sim_index)
        results = start_computation(simulator)
        return updated_sim_index, results

    def calculate_fitness(self):
        # Calculate fitness values for all complete solutions in parallel.
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            for cs, result in zip(self.completed_jobs, executor.map(fitness_function, self.pop, repeat(self.pairwise_distance.cs_list), repeat(self.pairwise_distance.dist_matrix_sq), repeat(self.radius), repeat(self.ff_target_prob))):
                cs.fitness.values = (result,)
        self._logger.info('Fitness values calculated.')

    def record_results(self):
        # Record best solution.
        best_solution = sorted(
            self.pop, key=lambda x: x.fitness.values[0])[0]
        self._logger.info(
            'best_complete_solution={} | fitness={}'.format(best_solution, best_solution.fitness.values[0]))

        log_and_pickle(object=self.pop, file_name='_GA_pop',
                       output_dir=self.output_directory)

        # Compiling statistics on the populations and completeSolSet.
        record_complete_solutions = self.stats.compile(self.pop)

        self.logbook.record(gen=1, **record_complete_solutions)
        print(self.logbook.stream)
        log_and_pickle(object=self.logbook, file_name='_GA_logbook',
                       output_dir=self.output_directory)

    def mutate_cs(self, cs):
        """Mutates a complete solution.

        It takes the complete solution as input and returns the mutated
        complete solution as output.
        """
        cs_class = type(cs)
        scen, mlco = cs

        # Mutate the scenario.
        scen = self.mutate_scenario(scen)
        
        # Mutate the mlco.
        mlco = self.mutate_mlco(mlco)

        return cs_class(scen, mlco)

