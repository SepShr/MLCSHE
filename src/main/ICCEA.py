import random
from copy import deepcopy

import numpy as np

from deap import tools

from src.utils.utility import create_complete_solution, evaluate, \
    evaluate_joint_fitness, evaluate_individual, breed_mlco, \
    identify_nominal_indices, measure_heom_distance, \
    index_in_complete_solution, find_individual_collaborator, rank_change, \
    max_rank_change_fitness, find_max_fv_individual, violate_safety_requirement


class ICCEA:
    def __init__(self, creator, toolbox, enumLimits):
        self.toolbox = toolbox
        self.creator = creator
        self.enumLimits = enumLimits

    def solve(self, max_gen):
        # Instantiate individuals and populations
        popScen = self.toolbox.popScen()
        popMLCO = self.toolbox.popMLCO()
        arcScen = self.toolbox.clone(popScen)
        arcMLCO = self.toolbox.clone(popMLCO)
        solutionArchive = []

        print('popScen is: ' + str(popScen))
        print('PopMLCO is: ' + str(popMLCO))

        # Cooperative Coevolutionary Search
        for num_gen in range(max_gen):
            print('the current generation is: ' + str(num_gen))
            # Create complete solutions and evaluate individuals
            completeSolSet, popScen, popMLCO = evaluate(
                popScen, arcScen, popMLCO, arcMLCO, self.creator.Individual, 1)

            # Record the complete solutions that violate the requirement r
            # solutionArchive.append(
            #     cs for cs in completeSolSet if violate_safety_requirement(cs))
            for cs in completeSolSet:
                # if violate_safety_requirement(cs):  # DEBUG: to add all complete solutions
                solutionArchive.append(cs)

            # Some probes
            fitness_scen_list = [ind.fitness.values[0] for ind in popScen]
            avg_fitness_scen = sum(fitness_scen_list) / len(popScen)
            print('the avg for popScen fitness is: ' + str(avg_fitness_scen))

            fitness_mlco_list = [ind.fitness.values[0] for ind in popMLCO]
            avg_fitness_mlco = sum(fitness_mlco_list) / len(popMLCO)
            print('the avg for popMLCO fitness is: ' + str(avg_fitness_mlco))

            # print best complete solution found
            best_solution = sorted(solutionArchive, key=lambda x: x.fitness.values[0])[-1]
            print(f'len(solutionArchive): {len(solutionArchive)}')
            print(f'the best complete solution in solutionArchive: {best_solution} (fitness: {best_solution.fitness.values[0]})')

            # Evolve archives and populations for the next generation
            min_distance = 1.3
            arcScen = self.update_archive(
                popScen, popMLCO, completeSolSet, min_distance
            )
            arcMLCO = self.update_archive(
                popMLCO, popScen, completeSolSet, min_distance
            )

            # Select, mate (crossover) and mutate individuals that are not in archives.
            ts = 2
            cxpb = 0.5
            mut_bit_pb = 1
            mut_guass_mu = 0.5
            mut_guass_sig = 0.125
            mut_guass_pb = 0.5
            mut_int_pb = 0.5
            popScen = self.breed_scenario(
                popScen, arcScen, self.enumLimits, ts, cxpb, mut_bit_pb,
                mut_guass_mu, mut_guass_sig, mut_guass_pb, mut_int_pb)
            popMLCO = self.breed_scenario(
                popMLCO, arcMLCO, self.enumLimits, ts, cxpb, mut_bit_pb,
                mut_guass_mu, mut_guass_sig, mut_guass_pb, mut_int_pb)

            popScen += arcScen
            popMLCO += arcMLCO
            # popScen.append(x for x in arcScen)
            # popMLCO.append(x for x in arcMLCO)

            print('popScen is: ' + str(popScen))
            print('PopMLCO is: ' + str(popMLCO))
        return solutionArchive

    def breed_scenario(
            self, popScen, arcScen, enumLimits, tournSize, cxpb,
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

        # Find the complement (population minus the archive).
        breeding_population = [ele for ele in popScen if ele not in arcScen]

        # Select 2 parents, cx and mut them until satisfied.
        offspring_list = []
        size = len(popScen) - len(arcScen)
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
            offspring = self.mutate_scenario(
                offspring, enumLimits, mutbpb, mutgmu,
                mutgsig, mutgpb, mutipb
            )
            offspring_list.append(offspring)
            size = size - 1

        return offspring_list

    def mutate_scenario(
            self, scenario, intLimits, mutbpb, mutgmu,
            mutgsig, mutgpb, mutipb):
        """Mutates a scenario individual. Input is an unmutated scenario, while
        the output is a mutated scenario. The function applies one of the 3
        mutators to the elements depending on their type, i.e., `mutGaussian()`
        (Guass distr) to Floats, `mutFlipBit()` (bitflip) to Booleans and
        `mutUniformInt()` (integer-randomization) to Integers.

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

        cls = type(scenario)
        mutatedScen = []

        for i in range(len(scenario)):
            buffer = [scenario[i]]

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

    def is_similar(
            self, candidate, collaborator, archive,
            archive_members_and_collaborators_dictionary,
            min_distance, first_item_class):
        """The algorithm evaluates if a `candidate` and its `collaborator` are
        similar to the memebrs of an `archive` and their collaborators
        (recorded in `archive_members_and_collaborators_dictionary`).

        Similarity uses the criteria `min_distance` to decide.
        """
        cand = deepcopy(candidate)
        collab = deepcopy(collaborator)
        flat_complete_solutions_list = []
        ficls = first_item_class
        archive_dict = archive_members_and_collaborators_dictionary
        # Create the complete solution of cand and collab
        main_complete_solution = create_complete_solution(
            cand, collab, ficls)

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
                print(
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
        base = deepcopy(base_individual)
        target = deepcopy(target_individual)

        base = self.flatten(base)
        target = self.flatten(target)

        if base == target:
            return True
        else:
            return False

    def individual_in_list(self, individual, individuals_list):
        evaluation_list = []
        for cs in individuals_list:
            if self.individual_is_equal(individual, cs):
                evaluation_list.append(1)
            else:
                evaluation_list.append(0)
        return sum(evaluation_list) > 0

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
                if not self.individual_in_list(c, complete_solutions_set):
                    c.fitness.values = evaluate_joint_fitness(c)
                    # counter_jfe += 1
                    # print(counter_jfe)
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
                c.fitness.values = evaluate_joint_fitness(c)
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
        pop = deepcopy(population)
        pop_prime = deepcopy(other_population)
        complete_solutions_set_internal = deepcopy(complete_solutions_set)
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
                print('No individual left to be considered for archive membership.')
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
