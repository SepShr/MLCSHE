"""
Random Search
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import repeat
from multiprocessing import cpu_count

import numpy as np
from deap import base, creator, tools

import search_config as cfg
from problem_utils import initialize_mlco
from simulation_manager_cluster import (ContainerSimManager,
                                        prepare_for_computation,
                                        start_computation)
from src.main.fitness_function import calculate_fitness
from src.utils.PairwiseDistance import PairwiseDistance
from src.utils.utility import (create_complete_solution,
                               initialize_hetero_vector, log_and_pickle,
                               setup_logbook_file)

JOBS_QUEUE_SIZE = 10


class RandomSearch:
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

        self.sim_index = 1  # Initialize simulation index.

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
        self.jobs_size = JOBS_QUEUE_SIZE

        self.ff_target_prob = cfg.finess_function_target_probability

    def define_problem(self, cs_archive_size):
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

        # Search the scenario and mlco parameter space randomly until the time is up.

        # Randomly create complete solutions.
        while counter <= cs_archive_size:

            # Create a random scenario.
            scenario = initialize_hetero_vector(
                self.scen_enumLimits, creator.Scenario)
            self.scenario_list.append(scenario)

            # Create a random mlco.
            mlco = initialize_mlco(creator.OutputMLC)
            self.mlco_list.append(mlco)

            # Create a complete solution.
            complete_solution = creator.Individual(
                create_complete_solution(scenario, mlco, creator.Scenario))

            self.cs_archive.append(complete_solution)

            counter += 1

        self.job_list = deepcopy(self.cs_archive)
        self.completed_jobs = []

        self._logger.info(f'{cs_archive_size} complete solutions created.')
        self._logger.info('cs_archive={}'.format(self.cs_archive))

    def run(self, sim_budget):
        self._logger.info('Starting random search.')
        self._logger.info('sim_budget={}'.format(sim_budget))
        start_time = time.time()

        self.define_problem(sim_budget)

        while len(self.job_list) > 0:
            # Add self.jobs_size number of items from self.job_list to jobs.
            if len(self.job_list) > self.jobs_size:
                jobs = self.job_list[:self.jobs_size]
                self.job_list = self.job_list[self.jobs_size:]
            else:
                jobs = self.job_list
                self.job_list = []

            self.evaluate_joint_fitness(jobs=jobs)
            self.record_results()
            self._logger.info("Results recorded.")

            current_time = time.time()
            elapsed_time = current_time - start_time
            self._logger.info('elapsed_time={}'.format(elapsed_time))

        end_time = time.time()
        total_time = end_time - start_time
        self._logger.info('total_search_time={}'.format(total_time))
        self._logger.info("End of random search.")

    def record_results(self):
        # Record best solution.
        best_solution = sorted(
            self.completed_jobs, key=lambda x: x.fitness.values[0])[0]
        self._logger.info(
            'best_complete_solution={} | fitness={}'.format(best_solution, best_solution.fitness.values[0]))

        log_and_pickle(object=self.scenario_list, file_name='_RS_scen',
                       output_dir=self.output_directory)

        log_and_pickle(object=self.mlco_list, file_name='_RS_mlco',
                       output_dir=self.output_directory)

        log_and_pickle(object=self.completed_jobs, file_name='_RS_cs',
                       output_dir=self.output_directory)

        # Compiling statistics on the populations and completeSolSet.
        record_complete_solutions = self.stats.compile(self.completed_jobs)

        self.logbook.record(gen=1, **record_complete_solutions)
        print(self.logbook.stream)
        log_and_pickle(object=self.logbook, file_name='_RS_logbook',
                       output_dir=self.output_directory)

    def evaluate_joint_fitness(self, jobs=None):
        """Evaluates the joint fitness of a complete solution.

        Parameters:
        - jobs (list): list of jobs to evaluate the joint fitness on

        Returns:
        - None
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
            for cs, result in zip(self.completed_jobs, executor.map(calculate_fitness, self.cs_archive, repeat(self.pairwise_distance.cs_list), repeat(self.pairwise_distance.dist_matrix_sq), repeat(self.radius), repeat(self.ff_target_prob))):
                cs.fitness.values = (result,)
        self._logger.info('Fitness values calculated.')
