from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import repeat
import pathlib
import pickle
import random
import logging
from time import time

import numpy as np
from deap import tools
from fitness_function import fitness_function
from src.utils.utility import (collaborate,
                               create_complete_solution, evaluate_individual,
                               find_individual_collaborator,
                               find_max_fv_individual, flatten_list,
                               identify_nominal_indices,
                               index_in_complete_solution,
                               max_rank_change_fitness, measure_heom_distance,
                               rank_change, setup_file, setup_logbook_file)


class ICCEA:
    def __init__(self, creator, toolbox, simulator, pairwise_distance_cs, pairwise_distance_p1, pairwise_distance_p2, first_population_enumLimits=None, second_population_enumLimits=None, update_archive_strategy='bestRandom'):
        self.toolbox = toolbox
        self.creator = creator
        self.p1_enumLimits = first_population_enumLimits
        self.p2_enumLimits = second_population_enumLimits

        self.pairwise_distance_p1 = pairwise_distance_p1
        self.pairwise_distance_p2 = pairwise_distance_p2

        self.simulator = simulator
        self.num_sim = 1
        self.pairwise_distance = pairwise_distance_cs

        self.update_archive_strategy = update_archive_strategy

        # Setup logger and logbook.
        self._logger = logging.getLogger(__name__)

    def solve(self, max_gen, hyperparameters, max_num_evals, radius, output_dir, seed=None):
        self.radius = radius
        self._output_directory = output_dir

        # Setup logbook.
        self._logbook_file = setup_logbook_file(
            output_dir=self._output_directory)

        self._logger.info("CCEA search started.")
        self._logger.info(
            'max_number_of_generations={}'.format(max_gen))
        self._logger.info('max_number_of_evaluations={}'.format(max_num_evals))

        # Set the random module seed.
        random.seed(seed)
        self._logger.info('random_seed={}'.format(seed))

        # Instantiate individuals and populations
        popScen = self.toolbox.popScen()
        self._logger.info('scenario_population_size={}'.format(len(popScen)))
        popMLCO = self.toolbox.popMLCO()
        self._logger.info('mlco_population_size={}'.format(len(popMLCO)))
        arcScen = self.toolbox.clone(popScen)
        arcMLCO = self.toolbox.clone(popMLCO)

        # Initialize the pairwise distance matrix for the initial populations.
        self.pairwise_distance_p1.update_dist_matrix(popScen)
        self.pairwise_distance_p2.update_dist_matrix(popMLCO)

        # complete_solution_archive = []
        self.cs_archive = []
        self.solution_archive = []

        # Setup files to log populations and archives.
        cs_archive_file = setup_file('_cs_archive', self._output_directory)
        cs_list_gen_file = setup_file('_cs_list_gen', self._output_directory)
        pop_scen_file = setup_file('_pop_scen', self._output_directory)
        pop_mlco_file = setup_file('_pop_mlco', self._output_directory)
        arc_scen_file = setup_file('_arc_scen', self._output_directory)
        arc_mlco_file = setup_file('_arc_mlco', self._output_directory)

        # Adding multiple statistics.
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        # stats_size = tools.Statistics(key=len)
        # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats = tools.MultiStatistics(fitness=stats_fit)
        mstats.register("avg", np.mean, axis=0)
        mstats.register("std", np.std, axis=0)
        mstats.register("min", np.min, axis=0)
        mstats.register("max", np.max, axis=0)

        # Instantiating logbook that records search-specific statistics.
        logbook = tools.Logbook()

        # FIXME: Maybe add the number of evaluations
        # logbook.header = "gen", "type", "fitness", "size"
        logbook.header = "gen", "type", "fitness"
        logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        # logbook.chapters["size"].header = "min", "avg", "max"

        # Set the hyperparameter values.
        min_dist,\
            ts, \
            cxpb, \
            mut_guass_mu, \
            mut_guass_sig, \
            mut_guass_pb, \
            mut_int_pb, \
            mut_bit_pb, \
            pop_arc_size = hyperparameters

        self._logger.info(
            'minimum_distance_threshold={}'.format(min_dist))
        self._logger.info('tournament_selection_size={}'.format(ts))
        self._logger.info('crossover_probability={}'.format(cxpb))
        self._logger.info('gaussian_mutation_mean={}'.format(mut_guass_mu))
        self._logger.info(
            'gaussian_mutation_standard_deviation={}'.format(mut_guass_sig))
        self._logger.info(
            'gaussian_mutation_probability={}'.format(mut_guass_pb))
        self._logger.info('integer_mutation_probability={}'.format(mut_int_pb))
        self._logger.info('bitflip_mutation_probability={}'.format(mut_bit_pb))

        start_time = time()

        # Cooperative Coevolutionary Search
        for num_gen in range(1, max_gen+1):
            self._logger.info('current_generation={}'.format(num_gen))
            self._logger.debug('scenario_population={}'.format(popScen))
            self._logger.debug('mlco_population={}'.format(popMLCO))
            self._logger.debug('scenario_archive={}'.format(arcScen))
            self._logger.debug('mlco_archive={}'.format(arcMLCO))
            self._logger.debug('pop_arc_size={}'.format(pop_arc_size))

            # self.cs_archive = complete_solution_archive

            # Create complete solutions and evaluate individuals
            # completeSolSet, popScen, popMLCO = self.evaluate(
            #     popScen, arcScen, popMLCO, arcMLCO, self.creator.Individual, 1)
            cs_archive_gen, popScen, popMLCO = self.evaluate(
                popScen, arcScen, popMLCO, arcMLCO, self.creator.Individual, 1)

            # for cs in completeSolSet:
            #     complete_solution_archive.append(cs)

            # Record the complete solutions archive.
            with open(cs_archive_file, 'wt') as csaf:
                for cs in self.solution_archive:
                    csaf.write('cs={}, jfit_value={}, safety_req_value={}\n'.format(
                        cs, cs.fitness.values[0], cs.safety_req_value))

            # Record the list of complete solutions per generation.
            with open(cs_list_gen_file, 'at') as csgf:
                csgf.write(
                    '--------------- GENERATION_NUMBER {} ---------------\n'.format(num_gen))
                for cs in cs_archive_gen:
                    csgf.write('cs={}, jfit_value={}, safety_req_value={}\n'.format(
                        cs, cs.fitness.values[0], cs.safety_req_value))

            # Record scenarios in popScen per generation.
            with open(pop_scen_file, 'at') as psf:
                psf.write(
                    '--------------- GENERATION_NUMBER {} ---------------\n'.format(num_gen))
                for scen in popScen:
                    psf.write('scen={}, fv={}\n'.format(
                        scen, scen.fitness.values[0]))

            # Record mlcos in popMLCO per generation.
            with open(pop_mlco_file, 'at') as pmf:
                pmf.write(
                    '--------------- GENERATION_NUMBER {} ---------------\n'.format(num_gen))
                for mlco in popMLCO:
                    pmf.write('mlco={}, fv={}\n'.format(
                        mlco, mlco.fitness.values[0]))

            # Record scenarios in arcScen per generation.
            with open(arc_scen_file, 'at') as asf:
                asf.write(
                    '--------------- GENERATION_NUMBER {} ---------------\n'.format(num_gen))
                for scen in arcScen:
                    asf.write('scen={}, fv={}\n'.format(
                        scen, scen.fitness.values))

            # Record mlcos in arcMLCO per generation.
            with open(arc_mlco_file, 'at') as amf:
                amf.write(
                    '--------------- GENERATION_NUMBER {} ---------------\n'.format(num_gen))
                for mlco in arcMLCO:
                    amf.write('mlco={}, fv={}\n'.format(
                        mlco, mlco.fitness.values))

            # Compiling statistics on the populations and completeSolSet.
            record_scenario = mstats.compile(popScen)
            record_mlco = mstats.compile(popMLCO)
            record_complete_solution = mstats.compile(cs_archive_gen)
            # record_complete_solution = mstats.compile(self.solution_archive)

            logbook.record(gen=num_gen, type='scen',
                           **record_scenario)
            logbook.record(gen=num_gen, type='mlco', **record_mlco)
            logbook.record(gen=num_gen, type='cs',
                           **record_complete_solution)

            logbook_stream = logbook.stream
            print(logbook_stream)

            with open(self._logbook_file, 'wt') as lb_file:
                print(logbook, file=lb_file)

            # Get the number of evaluated complete solutions.
            csa_len = len(self.cs_archive)
            self._logger.info(
                'complete_solution_archive_len={}'.format(csa_len))
            self._logger.info('num_simulations={}'.format(self.num_sim))

            # best_solution_overall = sorted(
            #     self.solution_archive, key=lambda x: x.fitness.values[0])[-1]
            # For testing the new fitness function
            best_solution_overall = sorted(
                self.solution_archive, key=lambda x: x.fitness.values[0])[0]
            # self._logger.info('at generation={}, complete_solution_archive_size={}'.format(
            #     num_gen, len(complete_solution_archive)))
            self._logger.info(
                'best_complete_solution_overall={} | fitness={}'.format(best_solution_overall, best_solution_overall.fitness.values[0]))

            # best_solution_gen = sorted(
            #     cs_archive_gen, key=lambda x: x.fitness.values[0])[-1]
            # For testing the new fitness function
            best_solution_gen = sorted(
                cs_archive_gen, key=lambda x: x.fitness.values[0])[0]
            # self._logger.debug('complete_solutions={}'.format(
            #     completeSolSet))
            self._logger.info(
                'best_complete_solution_gen={} | fitness={}'.format(best_solution_gen, best_solution_gen.fitness.values[0]))

            if (csa_len >= max_num_evals):
                self._logger.info(
                    'Maximum number of evaluations (max_num_evals={}) reached! Ending the search ...'.format(max_num_evals))
                self._logger.info(
                    f'execution_duration={time() - start_time} seconds')
                break

            if (num_gen >= max_gen):
                self._logger.info(
                    'Maximum number of generations (max_gen={}) reached! Ending the search ...'.format(max_gen))
                self._logger.info(
                    f'execution_duration={time() - start_time} seconds')
                break

            self._logger.info("Updating archives...")
            # Evolve archives and populations for the next generation
            # arcScen = self.update_archive(
            #     popScen, popMLCO, completeSolSet, min_dist
            # )
            if self.update_archive_strategy == 'best':
                arcScen = self.update_archive_diverse_elitist(
                    popScen, pop_arc_size, min_dist, self.pairwise_distance_p1)
            elif self.update_archive_strategy == 'bestRandom':
                arcScen = self.update_archive_diverse_best_random(
                    popScen, pop_arc_size, min_dist, self.pairwise_distance_p1)
            elif self.update_archive_strategy == 'random':
                arcScen = self.update_archive_diverse_random(
                    popScen, pop_arc_size, min_dist, self.pairwise_distance_p1)
            else:
                raise ValueError(
                    f'update_archive strategy can only be: best, best random, or random. It cannot be {self.update_archive_strategy}')

            # arcMLCO = self.update_archive(
            #     popMLCO, popScen, completeSolSet, min_dist
            # )
            if self.update_archive_strategy == 'best':
                arcMLCO = self.update_archive_diverse_elitist(
                    popMLCO, pop_arc_size, min_dist, self.pairwise_distance_p2)
            elif self.update_archive_strategy == 'bestRandom':
                arcMLCO = self.update_archive_diverse_best_random(
                    popMLCO, pop_arc_size, min_dist, self.pairwise_distance_p2)
            elif self.update_archive_strategy == 'random':
                arcMLCO = self.update_archive_diverse_random(
                    popMLCO, pop_arc_size, min_dist, self.pairwise_distance_p2)
            else:
                raise ValueError(
                    f'update_archive strategy can only be: best, bestRandom, or random. It cannot be {self.update_archive_strategy}')

            # Select, mate (crossover) and mutate individuals that are not in archives.
            # Breed the next generation of populations.
            self._logger.info("Breeding the populations...")
            popScen = self.breed(
                popScen, arcScen, self.p1_enumLimits, ts, cxpb, mut_bit_pb,
                mut_guass_mu, mut_guass_sig, mut_guass_pb, mut_int_pb)

            # Handle benchmark problems that have similar populations.
            if self.p2_enumLimits:
                popMLCO = self.breed(
                    popMLCO, arcMLCO, self.p2_enumLimits, ts, cxpb, mut_bit_pb,
                    mut_guass_mu, mut_guass_sig, mut_guass_pb, mut_int_pb)
            else:
                popMLCO = self.breed_mlco(
                    popMLCO, arcMLCO, ts, cxpb,
                    mut_guass_mu, mut_guass_sig, mut_guass_pb, mut_int_pb)

            popScen += arcScen
            popMLCO += arcMLCO
            # popScen.append(x for x in arcScen)
            # popMLCO.append(x for x in arcMLCO)

        # Record the complete logbook.
        with open(pathlib.Path((self._logbook_file)+'.pkl'), 'wb') as lb_file:
            pickle.dump(logbook, lb_file)

        return self.solution_archive

    def evaluate_joint_fitness(self, cs_list):
        """Evaluates the joint fitness of a complete solution.

        It takes the complete solution as input and returns its joint
        fitness as output.
        """
        # For benchmarking problems.
        # x = c[0][0]
        # y = c[1][0]

        # Ensure that no repetitive simulation would run.
        # print(f'cs_list is: {cs_list}')
        new_cs_list = [
            c for c in cs_list if not self.individual_in_list(c, self.cs_archive)]
        # print(f'new_cs_list is: {new_cs_list}')

        # Handle benchmarking problems.
        if self.p2_enumLimits:
            for c in new_cs_list:
                c.safety_req_value = self.get_safety_req_value(c)
                # Add cs that have been simulated to cs_archive.
                self.cs_archive.append(c)
        else:
            # self.num_sim = prepare_for_computation(
            #     new_cs_list, self.simulator, self.num_sim)
            # start_computation(self.simulator)
            self.num_sim, results = self.toolbox.compute_safety_cs_list(
                self.simulator, new_cs_list, self.num_sim)
            for c, result in zip(new_cs_list, results):
                # FIXME: Results assignment should be done in the problem.
                DfC_min, DfV_min, DfP_min, DfM_min, DT_min, traffic_lights_max = result
                # print(f'c={c}, c.safety_req_value={DfC_min}')
                # c.safety_req_value = DfC_min
                # self._logger.info('safety_req_value={}'.format(DfC_min))
                c.safety_req_value = DfV_min
                self._logger.info('safety_req_value={}'.format(DfV_min))
                self.cs_archive.append(c)

        # print(self.cs_archive)

        # Calculate the pairwise distance between all simulated complete solutions.
        self.pairwise_distance.update_dist_matrix(new_cs_list)

        self.solution_archive.clear()

        self.solution_archive = [deepcopy(c) for c in self.cs_archive]
        # print(self.solution_archive)
        # for c in cs_list:
        # for c in self.solution_archive:
        #     c.fitness.values = (fitness_function(
        #         cs=c,
        #         cs_list=self.pairwise_distance.cs_list,
        #         dist_matrix=self.pairwise_distance.dist_matrix_sq,
        #         max_dist=self.radius
        #     ),)

        # Calculate fitness values for all complete solutions in parallel.
        # FIXME: Do we need to reevaluate solution_archive every generation?
        with ProcessPoolExecutor() as executor:
            # results = [executor.submit(fitness_function, c, self.pairwise_distance.cs_list,
            #                            self.pairwise_distance.dist_matrix_sq, self.radius) for c in self.solution_archive]
            # for cs, result in zip(self.solution_archive, results):
            #     cs.fitness.values = (result.result(),)

            for cs, result in zip(self.solution_archive, executor.map(fitness_function, self.solution_archive, repeat(self.pairwise_distance.cs_list), repeat(self.pairwise_distance.dist_matrix_sq), repeat(self.radius))):
                cs.fitness.values = (result,)

        # Return a cs_list with fv per generation.
        return [c for c in self.solution_archive if self.individual_in_list(c, cs_list)]

    def get_safety_req_value(self, c):
        x = c[0]
        y = c[1]

        return self.toolbox.compute_safety_req_value(self.simulator, x, y)

    def evaluate(
            self, first_population, first_archive,
            second_population, second_archive, joint_class, min_num_evals):
        """Forms complete solutions, evaluates their joint fitness and
        evaluates the individual fitness values.

        :param first_population: the population (list) of scenarios.
        :param first_archive: the archive (list) of scenarios.
        :param second_population: the population of MLC outputs.
        :param second_archive: the archive of MLC outputs.
        :param joint_class: type into which each complete solution will
                            be typecasted.
        :param min_num_evals: the minimum number of collaborations and
                            thus, joint fitness evaluations per
                            individual.
        :returns: set of complete solutions with their fitness values,
                set of scenarios with their individual fitness values,
                and the set of MLC outputs with their individual
                fitness values.
        """
        # Exception handling must be added.

        first_component_class = type(first_population[0])
        complete_solutions_set = collaborate(
            first_archive,
            first_population,
            second_archive,
            second_population,
            joint_class,
            first_component_class,
            min_num_evals)

        # FIXME: Enable parallel processing.

        cs_list_per_gen = self.evaluate_joint_fitness(complete_solutions_set)

        # Evaluate individual fitness values.
        # for individual in first_population:
        #     individual.fitness.values = evaluate_individual(
        #         individual, self.solution_archive, 0)
        for individual in first_population:
            individual.fitness.values = evaluate_individual(
                individual, cs_list_per_gen, 0)

        # for individual in second_population:
        #     individual.fitness.values = evaluate_individual(
        #         individual, self.solution_archive, 1)
        for individual in second_population:
            individual.fitness.values = evaluate_individual(
                individual, cs_list_per_gen, 1)

        # return complete_solutions_set, first_population, second_population
        return cs_list_per_gen, first_population, second_population

    def breed(
            self, population, archive, enumLimits, tournSize, cxpb,
            mutbpb, mutgmu, mutgsig, mutgpb, mutipb):
        """Breeds, i.e., performs selection, crossover (exploitation) and
        mutation (exploration) on individuals of the Scenarios. It takes an old
        generation of scenarios as input and returns an evolved generation.

        :param popScen: the population of scenarios.
        :param arcScen: the list of all memebrs of the archive.
        :param enumLimits: a 2D list that contains a lower and upper
                           limits for the mutation of elements in a
                           scenario of type int.
        :param tournSize: the size of the tournament to be used by the
                          tournament selection algorithm.
        :param cxpb: the probability that a crossover happens between
                     two individuals.
        :param mutbpb: the probability that a binary element might be
                       mutated by the `tools.mutFlipBit()` function.
        :param mutgmu: the normal distribution mean used in
                       `tools.mutGaussian()`.
        :param mutgsig: the normal distribution standard deviation used
                        in `tools.mutGaussian()`.
        :param mutgpb: the probability that a real element might be
                       mutated by the `tools.mutGaussian()` function.
        :param mutipb: the probability that a integer element might be
                       mutated by the `mutUniformInt()` function.
        :returns: a list of bred scenario individuals (that will be
                  appended to the archive of scenarios to form the next
                  generation of the population).
        """
        # Registering evolutionary operators in the toolbox.
        self.toolbox.register(
            "select", tools.selTournament,
            tournsize=tournSize, fit_attr='fitness'
        )
        self.toolbox.register("crossover", tools.cxUniform, indpb=cxpb)

        # # Find the complement (population minus the archive).
        # breeding_population = [ele for ele in popScen if ele not in archive]
        breeding_population = population

        # Select 2 parents, cx and mut them until satisfied.
        offspring_list = []
        size = len(population) - len(archive)
        while size > 0:
            # Select 2 parents from the breeding_population via the select
            # fucntion.
            parents = self.toolbox.select(breeding_population, k=2)
            # Perform crossover.
            offspring_pair = self.toolbox.crossover(parents[0], parents[1])
            # # Choose a random offspring and typecast it into list.
            # offspring = list(offspring_pair[random.getrandbits(1)])
            offspring = offspring_pair[random.getrandbits(1)]
            # Mutate the offspring.
            offspring = self.mutate_flat_hetero_individual(
                offspring, enumLimits, mutbpb, mutgmu,
                mutgsig, mutgpb, mutipb
            )
            offspring_list.append(offspring)
            size = size - 1

        return offspring_list

    def breed_mlco(
            self, population, archive, tournSize, cxpb,
            mutgmu, mutgsig, mutgpb, mutipb):
        """Breeds, i.e., performs selection, crossover (exploitation) and
        mutation (exploration) on individuals of the Scenarios. It takes an old
        generation of scenarios as input and returns an evolved generation.

        :param popScen: the population of scenarios.
        :param arcScen: the list of all memebrs of the archive.
        :param enumLimits: a 2D list that contains a lower and upper
                           limits for the mutation of elements in a
                           scenario of type int.
        :param tournSize: the size of the tournament to be used by the
                          tournament selection algorithm.
        :param cxpb: the probability that a crossover happens between
                     two individuals.
        :param mutbpb: the probability that a binary element might be
                       mutated by the `tools.mutFlipBit()` function.
        :param mutgmu: the normal distribution mean used in
                       `tools.mutGaussian()`.
        :param mutgsig: the normal distribution standard deviation used
                        in `tools.mutGaussian()`.
        :param mutgpb: the probability that a real element might be
                       mutated by the `tools.mutGaussian()` function.
        :param mutipb: the probability that a integer element might be
                       mutated by the `mutUniformInt()` function.
        :returns: a list of bred scenario individuals (that will be
                  appended to the archive of scenarios to form the next
                  generation of the population).
        """
        # # Find the complement (population minus the archive).
        # breeding_population = [ele for ele in popScen if ele not in archive]
        breeding_population = population

        # Select 2 parents, cx and mut them until satisfied.
        offspring_list = []
        size = len(population) - len(archive)
        while size > 0:
            # Select 2 parents from the breeding_population via the select
            # fucntion.
            parents = self.toolbox.select(
                breeding_population, k=2, tournsize=tournSize)
            # Perform crossover.
            offspring_pair = self.toolbox.crossover(
                parents[0], parents[1], indpb=cxpb)
            # # Choose a random offspring and typecast it into list.
            # offspring = list(offspring_pair[random.getrandbits(1)])
            offspring = offspring_pair[random.getrandbits(1)]
            # Mutate the offspring.
            offspring = self.toolbox.mutate_mlco(
                offspring, mutgmu,
                mutgsig, mutgpb, mutipb
            )

            offspring_list.append(offspring)
            size = size - 1

        return offspring_list

    def mutate_flat_hetero_individual(
            self, individual, intLimits, mutbpb, mutgmu,
            mutgsig, mutgpb, mutipb):
        """Mutates a flat list of heterogeneous types individual. Input is
        an unmutated scenario, while the output is a mutated individual.
        The function applies one of the 3 mutators to the elements depending
        on their type, i.e., `mutGaussian()` (Guass distr) to Floats,
        `mutFlipBit()` (bitflip) to Booleans and `mutUniformInt()` 
        (integer-randomization) to Integers.

        :param scenario: a scenario type individual to be mutated by the
                         function.
        :param intLimits: a 2D list that contains a lower and upper
                          limits for the mutation of elements in a
                          scenario that are of type int.
        :param mutbpb: the probability that a binary element might be
                       mutated by the `tools.mutFlipBit()` function.
        :param mutgmu: the normal distribution mean used in
                       `tools.mutGaussian()`.
        :param mutgsig: the normal distribution standard deviation used
                        by `tools.mutGaussian()`.
        :param mutgpb: the probability that a real element might be
                       mutated by the `tools.mutGaussian()` function.
        :param mutipb: the probability that a integer element might be
                       mutated by the `mutUniformInt()` function.
        """
        self.toolbox.register("mutateScenBool", tools.mutFlipBit, indpb=mutbpb)
        self.toolbox.register("mutateScenFlt", tools.mutGaussian, mu=mutgmu,
                              sigma=mutgsig, indpb=mutgpb)

        # LIMITATION: assumes a specific format for intLimits.

        cls = type(individual)
        mutatedScen = []

        for i in range(len(individual)):
            buffer = [individual[i]]

            if type(buffer[0]) is int:
                buffer = tools.mutUniformInt(
                    buffer, low=intLimits[i][0],
                    up=intLimits[i][1], indpb=mutipb
                )
                buffer = list(buffer[0])

            if type(buffer[0]) is bool:
                buffer = self.toolbox.mutateScenBool(buffer)
                buffer = list(buffer[0])

            if type(buffer[0]) is float:
                buffer = self.toolbox.mutateScenFlt(buffer)
                buffer = list(buffer[0])

            mutatedScen += buffer

        return cls(mutatedScen)

    def flatten(self, list_of_lists):
        """Flattens a list of lists. It returns the `flattened_list`. Note that
        this function is recursive.

        :param list_of_lists: a list of lists. It can be an irregular
                              nested list.
        :returns: flattened list.
        """
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list) or \
                isinstance(list_of_lists[0], self.creator.Individual) or \
                isinstance(list_of_lists[0], self.creator.Scenario) or \
                isinstance(list_of_lists[0], self.creator.OutputMLC):
            return self.flatten(list_of_lists[0]) + self.flatten(list_of_lists[1:])
        return list_of_lists[:1] + self.flatten(list_of_lists[1:])

    def prepare_for_distance_evaluation(self, irregular_nested_list):
        """Prepares an irregular nested list for distance evaluation. It
        returns the flattened list and the list of indices for nominal values.

        :param irregular_nested_list: an irregular nested list, i.e., a
                                      list that may have lists or other
                                      types such as `int`, `str`, or
                                      `flt` as elements.
        :returns: a flattened list and the list of indices that
                  correspond to the nominal values.
        """
        flattened_list = self.flatten(irregular_nested_list)

        nominal_values_indices = identify_nominal_indices(flattened_list)

        return flattened_list, nominal_values_indices

    # FIXME: Needs to be refactored!!

    def is_similar(
            self, candidate, collaborator, archive,
            archive_members_and_collaborators_dictionary,
            min_distance, first_item_class):
        """The algorithm evaluates if a `candidate` and its `collaborator` are
        similar to the memebrs of an `archive` and their collaborators
        (recorded in `archive_members_and_collaborators_dictionary`).

        Similarity uses the criteria `min_distance` to decide.
        """
        # cand = deepcopy(candidate)
        # collab = deepcopy(collaborator)
        flat_complete_solutions_list = []
        ficls = first_item_class
        archive_dict = archive_members_and_collaborators_dictionary
        # Create the complete solution of cand and collab
        main_complete_solution = create_complete_solution(
            candidate, collaborator, ficls)

        # # Determine the nan equivalent value
        # nan_eqv = np.Nan

        # Prepare main_complete_solution for similarity assessment
        main_complete_solution_flat, nominal_values_indices = \
            self.prepare_for_distance_evaluation(main_complete_solution)

        # Add main_complete_solution_flat to the list of flat complete solutions
        flat_complete_solutions_list.append(main_complete_solution_flat)

        # Create the list of complete solutions that are to be used for distance
        # evaluations.
        for i in range(len(archive)):
            arc_nom_indices = []
            archive_complete_solution = create_complete_solution(
                archive[i], archive_dict[str(archive[i])], ficls
            )
            archive_complete_solution_flat, arc_nom_indices = \
                self.prepare_for_distance_evaluation(archive_complete_solution)
            if arc_nom_indices != nominal_values_indices:
                self._logger.error(
                    'The nominal values between ' +
                    str(archive_complete_solution) +
                    ' and ' + str(main_complete_solution) +
                    ' do not match!')
            flat_complete_solutions_list.append(archive_complete_solution_flat)

        distance_values = measure_heom_distance(
            flat_complete_solutions_list, nominal_values_indices)
        distance_values.pop(0)

        # Assess similarity between the main_complete_solution and the rest.
        similarity_list = []
        for i in range(len(distance_values)):
            if distance_values[i] <= min_distance:
                similarity_list.append(1)
            else:
                similarity_list.append(0)

        if sum(similarity_list) == len(distance_values):
            return True
        else:
            return False

    def individual_is_equal(self, base_individual, target_individual):
        # base = deepcopy(base_individual)
        # target = deepcopy(target_individual)

        # base = self.flatten(base_individual)
        # target = self.flatten(target_individual)
        base = flatten_list(base_individual)
        target = flatten_list(target_individual)

        if base == target:
            return True
        else:
            return False

    def individual_in_list(self, individual, individuals_list):
        # evaluation_list = []
        for cs in individuals_list:
            if self.individual_is_equal(individual, cs):
                # evaluation_list.append(1)
                return True
            # else:
            #     evaluation_list.append(0)
        # return sum(evaluation_list) > 0
        return False

    def calculate_fit_given_archive(self, archive, population, complete_solutions_set):
        # Initialization of variables
        fit_1 = []
        joint_class = type(complete_solutions_set[0])
        first_item_class = type(complete_solutions_set[0][0])

        for x in population:
            x_index_in_cs = index_in_complete_solution(
                x, complete_solutions_set[0])
            comp_sol_set_archive = []
            for ind in archive:
                c = create_complete_solution(ind, x, first_item_class)
                c = joint_class(c)
                # FIXME: Needs to be optimized!
                if not self.individual_in_list(c, complete_solutions_set):
                    # c.fitness.values = self.evaluate_joint_fitness([c])
                    self.evaluate_joint_fitness([c])
                    comp_sol_set_archive.append(c)
                    complete_solutions_set.append(c)
                else:
                    for cs in range(len(complete_solutions_set)):
                        if self.individual_is_equal(c, complete_solutions_set[cs]):
                            comp_sol_set_archive.append(
                                complete_solutions_set[cs])
                            break

                # print('x is: ' + str(x))
                # print('set considered for evaluateIndividual is: %s' % comp_sol_set_archive)
                fit_1.append(evaluate_individual(
                    x, comp_sol_set_archive, x_index_in_cs)[0])

        return fit_1

    # NO TESTCASE, NO DOCSTRING

    def calculate_fit_given_archive_and_i(
            self, individual, population, complete_solutions_set, fit_given_archive):
        # Initialization of variables
        fitness_list = []
        joint_class = type(complete_solutions_set[0])
        first_item_class = type(complete_solutions_set[0][0])

        for x in population:
            c = joint_class(create_complete_solution(
                individual, x, first_item_class))
            if not self.individual_in_list(c, complete_solutions_set):
                self.evaluate_joint_fitness([c])
                # c.fitness.values = self.evaluate_joint_fitness(c)
                complete_solutions_set.append(c)
                fitness_value = max(
                    fit_given_archive[population.index(x)], c.fitness.values[0])
                fitness_list.append(fitness_value)
            else:
                for cs in complete_solutions_set:
                    if self.individual_is_equal(c, cs):
                        fitness_value = max(
                            fit_given_archive[population.index(x)], cs.fitness.values[0])
                        fitness_list.append(fitness_value)
                        break
        return fitness_list

    def update_archive(self, population, other_population,
                       complete_solutions_set, min_distance):
        """Updates the archive according to iCCEA updateArchive algorithm.

        It starts with an empty archive for p, i.e., `archive_p` and
        adds informative members to it. The initial member is the
        individual with the highest fitness value. Individuals that:
        1. change the fitness ranking of the other population,
        2. are not similar to existing members of `archive_p`, and
        3. have the highest fitness ranking will be added to the
        `archive_p` for the next generation of the coevolutionary search.
        """
        # ???: Don't we have a specific type for each input?
        # for example, when `test_update_archive.py` is executed
        # population is `1D-list` and other_population is `2D-list`.
        # Is this always the case? or are they flexible?

        # Initialization of variables.
        pop = self.toolbox.clone(population)
        pop_prime = self.toolbox.clone(other_population)
        complete_solutions_set_internal = self.toolbox.clone(
            complete_solutions_set)
        archive_p = []
        ineligible_p = []

        first_item_class = type(complete_solutions_set[0][0])

        # archive_p starts with an individual with the maximum fitness value
        max_fitness_value_individual = find_max_fv_individual(pop)
        # max(
        # pop, key=lambda ind: ind.fitness.values[0])
        archive_p.append(max_fitness_value_individual)

        # INITIALIZE THE DICT OF ARCHIVE MEMBERS AND THE HIGHEST COLLABORATOR
        dict_archive_members_and_collaborators = {
            str(max_fitness_value_individual): find_individual_collaborator(
                max_fitness_value_individual,
                complete_solutions_set_internal
            )
        }

        exit_condition = False
        while not exit_condition:
            fit_incl_i = []  # A list of individual fitness values incl i
            dict_fit_incl_i = {}  # A dict of fit_incl_i with str(i) as keys
            dict_fit_rank_change_incl_i = {}  #

            pop_minus_archive = [i_c for i_c in pop if i_c not in archive_p]

            pop_minus_archive_and_ineligible = [i_c for i_c in pop_minus_archive
                                                if i_c not in ineligible_p]

            # Check if all individuals of pop have been considered archive.
            if not pop_minus_archive_and_ineligible:
                # self._logger.info(
                #     "No individual left to be considered for archive membership.")
                break

            # Calculate the individual fitness of pop_prime individuals,
            # given the collaborations made only with the members of archive_p
            fit_excl_i = self.calculate_fit_given_archive(
                archive_p, pop_prime, complete_solutions_set_internal)

            for i in pop_minus_archive:
                # Calculate fitness of pop_prime individuals given archive and i
                fit_incl_i_for_x = self.calculate_fit_given_archive_and_i(
                    i, pop_prime, complete_solutions_set_internal, fit_excl_i)

                fit_incl_i.append(fit_incl_i_for_x)
                dict_fit_incl_i[str(i)] = fit_incl_i_for_x

                # Create a dictionary that records rank change fitness values
                # for each i
                index_i = pop_minus_archive.index(i)
                dict_fit_rank_change_incl_i[str(i)] = rank_change(
                    pop_prime, fit_excl_i, fit_incl_i, index_i)

            # Calculate the max rank change fitness for each i
            max_fit_i = max_rank_change_fitness(
                pop_minus_archive_and_ineligible, dict_fit_rank_change_incl_i)

            # Find the maximum of all max_fit_i values and its corresponding i
            max_fit = max(max_fit_i)
            if max_fit != -1 * np.inf:
                # Find a, i.e., the candidate member to be added to archive_p
                index_a = max_fit_i.index(max_fit)
                a = pop_minus_archive_and_ineligible[index_a]

                # Find a's collaborator that has maximum fitness value
                max_fit_rank_change_a = dict_fit_rank_change_incl_i[str(a)]
                for x in range(len(pop_prime)):
                    if max_fit in max_fit_rank_change_a[x]:
                        x_a = pop_prime[max_fit_rank_change_a[x].index(
                            max_fit)]

                # Check the distance between a and other members of archive_p
                if self.is_similar(a, x_a, archive_p,
                                   dict_archive_members_and_collaborators,
                                   min_distance, first_item_class):
                    ineligible_p.append(a)
                else:
                    archive_p.append(a)
                    dict_archive_members_and_collaborators[str(a)] = x_a
            else:
                exit_condition = True

        return archive_p

    def update_archive_elitist(self, population, archive_size):
        """
        Updates and archive by selecting only a number of best
        individuals.
        """
        pop = self.toolbox.clone(population)

        archive_p = []

        pop_sorted = sorted(
            pop, key=lambda x: x.fitness.values[0])

        for i in range(archive_size):
            archive_p.append(pop_sorted.pop(-1))

        return archive_p

    def update_archive_diverse_elitist(self, population,
                                       max_archive_size, min_distance, pairwise_distance):
        """
        Updates and archive by selecting only a number of best
        individuals that are distinct enought.
        """
        pop = self.toolbox.clone(population)

        archive_p = []

        pop_sorted = sorted(
            pop, key=lambda x: x.fitness.values[0])

        # # Add the first member of the archive
        # archive_p.append(pop_sorted.pop(-1))

        # Add the other members of the archive while
        for i in range(max_archive_size):
            candidate = pop_sorted.pop(-1)
            if len(archive_p) > 0:
                if not self.is_similar_individual(
                        candidate, archive_p, min_distance, pairwise_distance):
                    archive_p.append(candidate)
                # if self.is_different(
                #         candidate, archive_p, min_distance):
                #     archive_p.append(candidate)
            else:
                # Add the first member of the archive
                archive_p.append(candidate)

        return archive_p

    def update_archive_random(self, population, archive_size):
        """
        Creates an archive of size `archive_size` by randomly 
        selecting from the members of the population.
        """
        population_copy = self.toolbox.clone(population)

        archive_p = []

        for i in range(archive_size):
            candidate = population_copy.pop(
                random.randint(0, len(population_copy)-1))
            archive_p.append(candidate)

        return archive_p

    def update_archive_best_random(self, population, archive_size):
        """
        Updates and archive by selecting the best individual and
        `archive_size - 1` random individuals.
        """
        population_copy = self.toolbox.clone(population)

        population_copy_sorted = sorted(
            population_copy, key=lambda x: x.fitness.values[0])

        archive_p = []

        for i in range(archive_size):
            if len(archive_p) > 0:
                candidate = population_copy_sorted.pop(
                    random.randint(0, len(population_copy_sorted)-1))

            else:
                # Select the best indiviudal as the first candidate
                candidate = population_copy_sorted.pop(-1)

            archive_p.append(candidate)

        return archive_p

    def update_archive_diverse_random(self, population, archive_size, min_distance, pairwise_distance):
        """
        Updates and archive by selecting diverse random individuals.
        """
        population_copy = self.toolbox.clone(population)

        archive_p = []

        for i in range(archive_size):
            candidate = population_copy.pop(
                random.randint(0, len(population_copy)-1))
            if len(archive_p) > 0:
                if not self.is_similar_individual(
                        candidate, archive_p, min_distance, pairwise_distance):
                    archive_p.append(candidate)
            else:
                archive_p.append(candidate)

        return archive_p

    def update_archive_diverse_best_random(self, population, max_archive_size, min_distance, pairwise_distance):
        """
        Updates and archive by selecting diverse best and random individuals.
        """
        population_copy = self.toolbox.clone(population)

        population_copy_sorted = sorted(
            population_copy, key=lambda x: x.fitness.values[0])

        archive_p = []

        for i in range(max_archive_size):
            if len(archive_p) > 0:
                candidate = population_copy.pop(
                    random.randint(0, len(population_copy)-1))
                if not self.is_similar_individual(
                        candidate, archive_p, min_distance, pairwise_distance):
                    archive_p.append(candidate)
            else:
                # candidate = population_copy_sorted.pop(-1)
                # For testing the new fitness function
                candidate = population_copy_sorted.pop(0)
                archive_p.append(candidate)

        return archive_p

    def is_similar_individual(self, candidate, archive, min_distance, pairwise_distance):
        """The algorithm evaluates if an individual is
        similar to the memebrs of an `archive`.

        Similarity uses the criteria `min_distance` to decide.
        """
        distance_values = []
        for ind in archive:
            distance_values.append(
                pairwise_distance.get_distance(candidate, ind))

        # Assess similarity between the main_complete_solution and the rest.
        for i in range(len(distance_values)):
            if distance_values[i] <= min_distance:
                return True

        return False

    def is_different(self, candidate, archive, dist_threshold):
        assert candidate is not None, "candidate cannot be None."
        assert candidate != [], "candidate cannot be an empty list."
        assert archive is not None, "archive cannot be None."

        if len(archive) == 0:
            return True

        for member in archive:
            if self.pairwise_distance.get_distance(candidate, member) >= dist_threshold:
                return True
        else:
            return False
