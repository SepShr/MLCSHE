import random
from copy import deepcopy

import numpy as np

from deap import creator
from deap import base
from deap import tools

# Sample initalization function for Scenarios


def initialize_scenario(class_, limits):
    """Initializes a heterogeneous vector of type `class_` based on the values
    in `limits`.

    :param class_: the class into which the final list will be
                   typecasted into.
    :param limits: a list that determines whether an element of the
                   individual is a bool, int or float. It also
                   provides lower and upper limits for the int and
                   float elements.
    :returns: a heterogeneous vector of type `class_`.

    Furthermore, it assumes that `limits` is a list and it elements
    have the folowing format:
    `['float', min_flt, max_flt]` for float values;
    `['int', min_int, max_int]` for int values; and
    `'bool'` for boolean values.
    """

    x = []
    for i in range(0, len(limits)):
        if limits[i] == 'bool':
            x += [bool(random.getrandbits(1))]
        else:
            if type(limits[i][0]) == float:
                x += [random.uniform(limits[i][0], limits[i][1])]
            if type(limits[i][0]) == int:
                x += [random.randint(limits[i][0], limits[i][1])]

    return class_(x)

    # Uncomment the following line while commenting the rest to have a
    # minimally executable code skeleton.
    # return print("initializeScenario() returned.\n")

# Sample initialization function for MLC Outputs


def initializeMLCO(class_, size):
    """Initializes an individual of the MLCO population."""
    # return b
    return print("initializeMLCO() returned.\n")


def create_complete_solution(element, other_element, first_component_class):
    """Creates a complete solution from two elements such that the one with the
    type `first_component_class` is the first component of the complete
    solution."""
    if type(element) == first_component_class:
        c = [element, other_element]
    else:
        c = [other_element, element]
    return c


def collaborate_archive(archive, population, joint_class, ficls):
    """Create collaborations between individuals of archive and population. The
    complete solutions (collaborations) are of class icls. Output is the list
    of complete solutions `complete_solution_set`.

    :param archive: the archive (set) of individuals with which
                    every memeber of population should collaborate.
    :param population: the set of individuals that should collaborate
                       with the memebers of the archive.
    :param joint_class: the name of the class into which a complete
                        solution or `c` would be typecasted into.
    :param ficls: the name of the first individual's class to be
                  included in the complete solution. Defines the
                  format of a complete solution including 2
                  individuals of different types.
    """
    # Ensure that the input is not None. Exception handling should be added.
    assert archive and population and joint_class and ficls,\
        "Input to collaborate_archive cannot be None."

    # Deepcopy the lists.
    arc = deepcopy(archive)
    pop = deepcopy(population)

    complete_solution_set = []

    for i in arc:
        for j in pop:
            c = create_complete_solution(i, j, ficls)
            complete_solution_set = complete_solution_set + joint_class([c])

    return complete_solution_set

    # Uncomment the following line while commenting the rest to have a
    # minimally executable code skeleton.
    # return print("collabArc() returned.\n")


def collaborate_complement(
        first_population, first_archive, second_population,
        min_num_evals, joint_class, first_component_class):
    """Create collaborations between the members of `(first_population -
    first_archive)` and `second_population`. It returns a set of complete
    solutions `complete_solution_set` which has collaborations between the
    members of `second_population` and.

    `(first_population - first_archive)` or `pAComplement`.

    :param first_population: population A which is a list of
                             individuals.
    :param first_archive: an archive (set) of individuals, also a
                          subset of `first_population`.
    :param second_population: the set of individuals that should
                              collaborate with the memebers of the
                              `first_population - first_archive`.
    :param min_num_evals: the number of collaborations that each
                          individual in `second_population` should
                          participate in. Note that they have already
                          participated `len(first_archive)` times.
    :param joint_class: the name of the class into which a complete
                        solution or `c` would be typecasted into.
    :param first_component_class: the name of the first individual's
                                  class to be included in the
                                  complete solution. Defines the
                                  format of a complete solution
                                  including 2 individuals of
                                  different types.
    """
    if min_num_evals <= len(first_archive):
        return []

    pA = deepcopy(first_population)
    pB = deepcopy(second_population)
    aA = deepcopy(first_archive)

    # Ensure that first_archive is a subset of first_population
    assert all(x in pA for x in aA), \
        "first_archive is not a subset of first_population"

    complete_solution_set = []

    # Find {pA - aA}
    pAComplement = [ele for ele in pA if ele not in aA]

    # Create complete solution between all members of pB and
    # (min_num_evals - len(aA)) members of pAComplement
    while min_num_evals - len(aA) > 0:
        random_individual = \
            pAComplement[random.randint(0, len(pAComplement) - 1)]
        for i in pB:
            c = create_complete_solution(
                random_individual, i, first_component_class)
            complete_solution_set = \
                complete_solution_set + joint_class([c])
        min_num_evals = min_num_evals - 1

    return complete_solution_set

    # Uncomment the following line while commenting the rest of the method to
    # have a minimally executable code skeleton.
    # return print("collbaComp() returned.\n")


def collaborate(
        first_archive, first_population,
        second_archive, second_population,
        joint_class, first_component_class, min_num_evals):
    """Creates a complete solution from two sets of individuals. It takes two
    sets (arc and pop) and the type of individuals as input. It returns a set
    of unique complete solutions (collaborations).

    :param first_archive: an archive (list) of individuals. It is a subset
                          of first_population.
    :param first_population: a list of individuals that have to collaborate
                             with members of second_archive and possibly some
                             members of `(second_population - second_archive)`.
    :param second_archive: an archive (list) of individuals. It is a subset of
                        second_population.
    :param second_population: a list of individuals that have to collaborate
                              with members of first_archive and possibly some
                              members of `(first_population - first_archive)`.
    :param joint_class: the name of the class into which a complete solution
                        would be typecasted into.
    :param first_component_class: the name of the first individual's class to
                                  be included in the complete solution. Defines
                                  the format of a complete solution including
                                  2 individuals of different types.
    :param min_num_evals: the number of collaborations that each individual in
                          a population should participate in. Note that they
                          have already participated with every member of the
                          archive of other population, `len(archive)` times.
    """
    # Exeption handling needs to be implemented.
    assert first_archive or second_archive or first_population \
        or second_population is not None, \
        "Populations or archives cannot be None."
    assert first_archive or second_archive or first_population \
        or second_population != [], \
        "Populations or archives cannot be empty."

    # Ensure that arc_A is a subset of pop_A
    assert all(x in first_population for x in first_archive), \
        "first_archive is not a subset of first_population"
    assert all(x in second_population for x in second_archive), \
        "second_archive is not a subset of pop2"

    # Deepcopy archives and populations.
    a1 = deepcopy(first_archive)
    p1 = deepcopy(first_population)
    a2 = deepcopy(second_archive)
    p2 = deepcopy(second_population)

    # Create complete solutions from collaborations with the archives.
    complete_solutions_set = \
        collaborate_archive(a1, p2, joint_class, first_component_class) \
        + collaborate_archive(a2, p1, joint_class, first_component_class) \
        + collaborate_complement(
            p1, a1, p2, min_num_evals,
            joint_class, first_component_class) \
        + collaborate_complement(
            p2, a2, p1, min_num_evals,
            joint_class, first_component_class)

    # Remove repetitive complete solutions.
    complete_solutions_set_unique = []
    for x in complete_solutions_set:
        if x not in complete_solutions_set_unique:
            complete_solutions_set_unique.append(x)

    # Typcast every complete solution into cscls
    complete_solutions_set_unique_typecasted = []
    for c in complete_solutions_set_unique:
        c = joint_class(c)
        complete_solutions_set_unique_typecasted.append(c)

    return complete_solutions_set_unique_typecasted

    # Uncomment the following line while commenting the rest of the method to
    # have a minimally executable code skeleton.
    # collaborate_archive(arc1, pop1, cscls)
    # collaborateComplement(pop1, arc2, pop2, k, cscls)
    # return print("collaborate() returned.\n")


def evaluate(
        first_population, first_archive,
        second_population, second_archive, joint_class, min_num_evals):
    """Forms complete solutions, evaluates their joint fitness and evaluates
    the individual fitness values.

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

    # Deep copy the inputs
    population_one = deepcopy(first_population)
    population_two = deepcopy(second_population)
    archive_one = deepcopy(first_archive)
    archive_two = deepcopy(second_archive)

    first_component_class = type(population_one[0])
    complete_solutions_set = collaborate(
        archive_one,
        population_one,
        archive_two,
        population_two,
        joint_class,
        first_component_class,
        min_num_evals)

    # Evaluate joint fitness and record its value.
    for c in complete_solutions_set:
        c.fitness.values = evaluate_joint_fitness(c)

    # Evaluate individual fitness values.
    for individual in population_one:
        individual.fitness.values = evaluate_individual(
            individual, complete_solutions_set, 0)

    for individual in population_two:
        individual.fitness.values = evaluate_individual(
            individual, complete_solutions_set, 1)

    return complete_solutions_set, population_one, population_two

    # Uncomment the following line while commenting the rest of the method to
    # have a minimally executable code skeleton.
    # collaborate(popScen, arcScen, popMLCO, arcMLCO, cscls, fcls, k)
    # return print("evaluate() returned.\n")

# Evaluate the joint fitness of a complete solution.


def evaluate_joint_fitness(c):
    """Evaluates the joint fitness of a complete solution.

    It takes the complete solution as input and returns its joint
    fitness as output.
    """
    # Returns a random value for now.
    return (random.uniform(-5.0, 5.0),)

    # Uncomment the following line while commenting the rest of the method to
    # have a minimally executable code skeleton.
    # return print("evaluateJFit() returned.\n")


def evaluate_individual(individual, complete_solution_set, index):
    """Aggregates joint fitness values that an individual has been invovled in.

    It takes an individual, a list of all complete solutions,
    `completeSolSet`, that include the `individual` at the `index` of
    the complete solution; it returns the aggregate fitness value for
    `individual` as a real value.
    """
    weights_joint_fitness_involved = []
    values_joint_fitness_involved = []

    # Add the joint fitness values of complete solutions in which
    # individual has been a part of.
    for cs in complete_solution_set:
        if cs[index] == individual:
            weights_joint_fitness_involved += [cs.fitness.values]

    for i in weights_joint_fitness_involved:
        values_joint_fitness_involved += list([i[0]])

    # Aggregate the joint fitness values. For now, maximum values is used.
    individual_fitness_value = max(values_joint_fitness_involved)

    return (individual_fitness_value,)

# Breed scenarios.


def breed_scenario(
        popScen, arcScen, enumLimits, tournSize, cxpb,
        mutbpb, mutgmu, mutgsig, mutgpb, mutipb):
    """Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the Scenarios. It takes an old generation
    of scenarios as input and returns an evolved generation.

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
    toolbox.register(
        "select", tools.selTournament,
        tournsize=tournSize, fit_attr='fitness'
    )
    toolbox.register("crossover", tools.cxUniform, indpb=cxpb)

    # Find the complement (population minus the archive).
    breeding_population = [ele for ele in popScen if ele not in arcScen]

    # Select 2 parents, cx and mut them until satisfied.
    offspring_list = []
    size = len(popScen) - len(arcScen)
    while size > 0:
        # Select 2 parents from the breeding_population via the select
        # fucntion.
        parents = toolbox.select(breeding_population, k=2)
        # Perform crossover.
        offspring_pair = toolbox.crossover(parents[0], parents[1])
        # Choose a random offspring and typecast it into list.
        offspring = list(offspring_pair[random.getrandbits(1)])
        # Mutate the offspring.
        offspring = mutate_scenario(
            offspring, enumLimits, mutbpb, mutgmu,
            mutgsig, mutgpb, mutipb
        )
        offspring_list += [offspring]
        size = size - 1

    return offspring_list

    # Uncomment the following line while commenting the rest of the method to
    # have a minimally executable code skeleton.
    # mutate_scenario(popScen, enumLimits)
    # return print("breedScen() returned.\n")

# Breed MLC outputs.


def breed_mlco(outputMLC):
    """Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the MLC output population.

    It takes an old generation of MLC ouptputs as input and return an
    evovled generation.
    """
    # return outputMLC
    return print("breedMLCO() returned.\n")

# Mutate scenarios


def mutate_scenario(
        scenario, intLimits, mutbpb, mutgmu,
        mutgsig, mutgpb, mutipb):
    """Mutates a scenario individual. Input is an unmutated scenario, while the
    output is a mutated scenario. The function applies one of the 3 mutators to
    the elements depending on their type, i.e., `mutGaussian()` (Guass distr)
    to Floats, `mutFlipBit()` (bitflip) to Booleans and `mutUniformInt()`
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
    toolbox.register("mutateScenBool", tools.mutFlipBit, indpb=mutbpb)
    toolbox.register("mutateScenFlt", tools.mutGaussian, mu=mutgmu,
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
            buffer = toolbox.mutateScenBool(buffer)
            buffer = list(buffer[0])

        if type(buffer[0]) is float:
            buffer = toolbox.mutateScenFlt(buffer)
            buffer = list(buffer[0])

        mutatedScen += buffer

    return cls(mutatedScen)

    # Uncomment the following line while commenting the rest to have a
    # minimally executable code skeleton.
    # return print("mutate_MLCO() returned.\n")


def flatten(list_of_lists):
    """Flattens a list of lists. It returns the `flattened_list`. Note that
    this function is recursive.

    :param list_of_lists: a list of lists. It can be an irregular
                          nested list.
    :returns: flattened list.
    """
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list) or \
            isinstance(list_of_lists[0], creator.Individual) or \
            isinstance(list_of_lists[0], creator.Scenario) or \
            isinstance(list_of_lists[0], creator.OutputMLC):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def identify_nominal_indices(flat_list):
    """Identifies nominal values and returns their index in a list.

    :param flat_list: a flat list that contains elements of type
                      `int`, `str`, `bool`, or `float`. The first
                      3 types are considered as nominal values.
    :returns: a list of the nominal values indices,
              `nominal_values_indices`.
    """
    nominal_values_indices = []

    for i in range(len(flat_list)):
        if isinstance(flat_list[i], int):
            nominal_values_indices.append(i)
        else:
            if isinstance(flat_list[i], str):
                nominal_values_indices.append(i)
            if isinstance(flat_list[i], bool):
                if flat_list[i]:
                    element = 1
                    nominal_values_indices.append(i)
                else:
                    element = 0
                    nominal_values_indices.append(i)

    return nominal_values_indices


def prepare_for_distance_evaluation(irregular_nested_list):
    """Prepares an irregular nested list for distance evaluation. It returns
    the flattened list and the list of indices for nominal values.

    :param irregular_nested_list: an irregular nested list, i.e., a
                                  list that may have lists or other
                                  types such as `int`, `str`, or
                                  `flt` as elements.
    :returns: a flattened list and the list of indices that
              correspond to the nominal values.
    """
    flattened_list = flatten(irregular_nested_list)

    nominal_values_indices = identify_nominal_indices(flattened_list)

    return flattened_list, nominal_values_indices


def gather_values_in_np_array(two_d_list, numeric_value_index):
    """Gathers all numeric values from a 2D list, located at a specific column.

    :param two_d_list: a 2D list.
    :param numeric_value_index: the index of a numeric value, i.e.,
                                a coloumn in the 2D array.

    :returns: a numpy array.
    """
    # DOES NOT HANDLE NOMINAL VALUE INPUTS.
    # DOES NOT HANDLE CASES WERE THE NUMERIC VALUE INDEX IS OUT OF RANGE.
    numeric_values_array = np.zeros(len(two_d_list))

    for i in range(len(two_d_list)):
        numeric_values_array[i] = two_d_list[i][numeric_value_index]

    return numeric_values_array


def gather_values_in_list(two_d_list, numeric_value_index):
    """Gathers all the numeric values from a 2D list, located at a specific
    column.

    :param two_d_list: a 2D list.
    :param numeric_value_index: the index of a numeric value, i.e.,
                                a coloumn in the 2D array.

    :returns: a list.
    """
    # DOES NOT HANDLE NOMINAL VALUE INPUTS.
    # DOES NOT HANDLE CASES WERE THE NUMERIC VALUE INDEX IS OUT OF RANGE.
    numeric_values_list = []

    for i in range(len(two_d_list)):
        numeric_values_list.append(two_d_list[i][numeric_value_index])

    return numeric_values_list

# NO TESTCASE


def calculate_std(two_d_list, numeric_value_index):
    """Calculates the standard deviation for the numeric values whose index is
    provided.

    The values are in a 2D list.
    """
    X = gather_values_in_np_array(two_d_list, numeric_value_index)

    return np.std(X)

# NO TESTCASE


def calculate_max(two_d_list, numeric_value_index):
    """Calculates the maximum value along a column of a 2D list."""
    X = gather_values_in_list(two_d_list, numeric_value_index)

    return max(X)

# NO TESTCASE


def calculate_min(two_d_list, numeric_value_index):
    """Calculates the minimum value along a column of a 2D list."""
    X = gather_values_in_list(two_d_list, numeric_value_index)

    return min(X)

# ASSUMPTIONS: X HAS NO MISSING VALUES. IMPLEMENTATION SHOULD BE IMPROVED.


def measure_heom_distance(
        X: list,
        cat_ix: list,
        nan_equivalents: list = [np.nan, 0],
        normalised: str = "normal"
) -> list:
    """Calculate the Heterogeneous Euclidean-Overlap Metric (HEOM)- difference
    between a list located at X[0] and the rest of the lists of similar size.
    (TODO: what do you mean by the lists of 'similar' size?)

    :param X: X is a 2D list of flattened heterogeneuous lists.
    :param cat_ix: is a list of indices of the categorical values.
    :param nan_equivalents: list of values that are considered as
                            missing values.
    :param normalised: normalization method, can be "normal" or "std".
    :return: a list of normalized distance (TODO: is this correct?)
    """

    assert len(X) > 1  # measure distance between at least two lists
    for col in range(1, len(X)):
        # the length of each list must be the same
        assert len(X[col-1]) == len(X[col])

    nan_eqvs = nan_equivalents  # FIXME: `nan_eqvs` is never used later
    cat_ix = cat_ix
    row_x = len(X)
    col_x = len(X[0])

    # Initialize numeric_range list.
    numeric_range = []
    for i in range(len(X[0])):
        numeric_range.append(1)

    # Initialize the results array
    results_array = np.zeros((row_x, col_x))

    # Get indices for missing values, if any

    # Calculate the distance for missing values elements
    # Hint: the distance for missing values is equal to one!

    # Get categorical indices without missing values elements

    # Calculate the distance for categorical elements
    for index in cat_ix:
        for row in range(1, row_x):
            if X[0][index] != X[row][index]:
                results_array[row][index] = 1

    # Get numerical indices without missing values elements
    num_ix = [i for i in range(col_x) if i not in cat_ix]

    # Calculate range for numeric values.
    # TODO: check issue #8
    for i in range(len(X[0])):
        if i in num_ix:
            if normalised == "std":
                numeric_range[i] = 4 * calculate_std(X, i)  # ???: why multiply by 4?
            else:
                numeric_range[i] = calculate_max(X, i) - calculate_min(X, i)
                if numeric_range[i] == 0.0:
                    numeric_range[i] = 0.0001
                    # To avoid division by zero in case of similar values.

    # Calculate the distance for numerical elements
    for index in num_ix:
        for row in range(1, row_x):
            # FIXME: this is strange; check the formula again
            # FIXME: for example, np.sqrt(np.square(column_difference)) == abs(column_difference)
            # FIXME: the corresponding test case must be updated too
            column_difference = X[0][index] - X[row][index]
            results_array[row, index] = \
                np.sqrt(np.square(column_difference)) / numeric_range[index]
            # USE THE ABSOLUTE VALUE FOR DIFFERENCE

    heom_distance_values = \
        list(np.sqrt(np.sum(np.square(results_array), axis=1)))
    return heom_distance_values

# DOCSTRING INCOMPLETE


def is_similar(
        candidate, collaborator, archive,
        archive_members_and_collaborators_dictionary,
        min_distance, first_item_class):
    """The algorithm evaluates if a `candidate` and its `collaborator` are
    similar to the memebrs of an `archive` and their collaborators (recorded in
    `archive_members_and_collaborators_dictionary`).

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
        prepare_for_distance_evaluation(main_complete_solution)

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
            prepare_for_distance_evaluation(archive_complete_solution)
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

# NO TESTCASE, NO DOCSTRING


def individual_is_equal(base_individual, target_individual):
    base = deepcopy(base_individual)
    target = deepcopy(target_individual)

    base = flatten(base)
    target = flatten(target)

    if base == target:
        return True
    else:
        return False

# NO TESTCASE, NO DOCSTRING


def individual_in_list(individual, individuals_list):
    evaluation_list = []
    for cs in individuals_list:
        if individual_is_equal(individual, cs):
            evaluation_list.append(1)
        else:
            evaluation_list.append(0)
    return sum(evaluation_list) > 0

# NO TESTCASE, INCOMPLETE DOCSTRING
# POPULATION HAS ONLY ONE TYPE OF INDIVIDUAL


def find_max_fv_individual(population):
    """Finds the indiviudal with the maximum indiviudal fitness value."""
    values = []

    for ind in population:
        individual_fitness_value = ind.fitness.values[0]
        values.append(individual_fitness_value)

    max_fitness_value = max(values)
    index_max_fitness_value = values.index(max_fitness_value)

    return population[index_max_fitness_value]

# NO TESTCASE, NO DOCSTRING


def index_in_complete_solution(individual, complete_solution):
    """Returns the index of an individual in a complete solution based on its
    type."""
    for i in range(len(complete_solution)):
        if type(complete_solution[i]) == type(individual):
            return i
        # else:
        #     return None

# NO TESTCASE, NO DOCSTRING


def find_individual_collaborator(individual, complete_solutions_set):
    index_in_cs = index_in_complete_solution(
        individual, complete_solutions_set[0]
    )
    for cs in complete_solutions_set:
        if cs[index_in_cs] == individual:
            if cs.fitness.values == individual.fitness.values:
                cs_deepcopy = deepcopy(cs)
                cs_deepcopy.pop(index_in_cs)
                return cs_deepcopy[0]

# NO TESTCASE, NO DOCSTRING


def rank_change(population, fit_no_i, fit_with_i, index_i):
    # Create a dictionary that records fit_3 values for each i
    row, col = len(population), len(population)

    # fit_3_i: the fitness of the populations individual in case of
    # fitness rank change, when i is also evaluated.
    fit_3_i = [[0 for _ in range(col)] for _ in range(row)]

    for x in population:
        for y in population:
            index_x = population.index(x)
            index_y = population.index(y)
            if ((fit_no_i[index_x] <= fit_no_i[index_y]) and (
                    fit_with_i[index_i][index_x] > fit_with_i[index_i][index_y])):
                fit_3_i[index_x][index_y] = fit_with_i[index_i][index_x]
            else:
                fit_3_i[index_x][index_y] = (-1 * np.inf)
    return fit_3_i

# NO TESTCASE, NO DOCSTRING


def calculate_fit_given_archive(archive, population, complete_solutions_set):
    # Initialization of variables
    fit_1 = []
    dict_fit_1 = {}
    joint_class = type(complete_solutions_set[0])
    first_item_class = type(complete_solutions_set[0][0])

    for x in population:
        x_index_in_cs = index_in_complete_solution(
            x, complete_solutions_set[0])
        comp_sol_set_archive = []
        for ind in archive:
            c = create_complete_solution(ind, x, first_item_class)
            c = joint_class(c)
            if not individual_in_list(c, complete_solutions_set):
                c.fitness.values = evaluate_joint_fitness(c)
                # counter_jfe += 1
                print('Joint fitness evaluation performed!')
                # print(counter_jfe)
                comp_sol_set_archive.append(c)
                complete_solutions_set.append(c)
            else:
                for cs in range(len(complete_solutions_set)):
                    if individual_is_equal(c, complete_solutions_set[cs]):
                        comp_sol_set_archive.append(complete_solutions_set[cs])
                        break

            # print('x is: ' + str(x))
            # print('set considered for evaluateIndividual is: %s' % comp_sol_set_archive)
            fit_1.append(evaluate_individual(
                x, comp_sol_set_archive, x_index_in_cs)[0])
        # print('fit_1 is: %s' % fit_1)
        dict_fit_1[str(x)] = evaluate_individual(
            x, comp_sol_set_archive, x_index_in_cs)[0]
        # print(str(dict_fit_1))
    return fit_1, dict_fit_1

# NO TESTCASE, NO DOCSTRING


def calculate_fit_given_archive_and_i(
        individual, population, complete_solutions_set, fit_given_archive):
    # Initialization of variables
    fitness_list = []
    joint_class = type(complete_solutions_set[0])
    first_item_class = type(complete_solutions_set[0][0])

    for x in population:
        c = joint_class(create_complete_solution(
            individual, x, first_item_class))
        if not individual_in_list(c, complete_solutions_set):
            c.fitness.values = evaluate_joint_fitness(c)
            complete_solutions_set.append(c)
            fitness_value = max(
                fit_given_archive[population.index(x)], c.fitness.values[0])
            fitness_list.append(fitness_value)
        else:
            for cs in complete_solutions_set:
                if individual_is_equal(c, cs):
                    fitness_value = max(
                        fit_given_archive[population.index(x)], cs.fitness.values[0])
                    fitness_list.append(fitness_value)
                    break
    return fitness_list

# NO TESTCASE, INCOMPLETE DOCSTRING


def max_rank_change_fitness(population, fitness_dict):
    """
    Find the max value for a dict with 2D lists as key values. Returns the max
    values in a list.
    """
    max_fitness_list = []
    # Find maximum of fitness_matrix for each i
    for i in population:
        max_fit_i_x = []
        fitness_matrix = fitness_dict[str(i)]
        for j in range(len(fitness_matrix)):
            fitness_matrix_row = fitness_matrix[j]
            max_fit_i_x.append(max(fitness_matrix_row))
        max_fitness_list.append(max(max_fit_i_x))
    return max_fitness_list

# NO TESTCASE


def update_archive(population, other_population,
                   complete_solutions_set, joint_class, min_distance):
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

    max_fitness_value_individual = find_max_fv_individual(pop)
    archive_p.append(max_fitness_value_individual)

    # INITIALIZE THE DICT OF ARCHIVE MEMEBERS AND THE HIGHEST COLLABORATOR
    dict_archive_memebers_and_collaborators = {}
    dict_archive_memebers_and_collaborators[str(max_fitness_value_individual)] = \
        find_individual_collaborator(
            max_fitness_value_individual,
            complete_solutions_set_internal
    )

    exit_condition = False
    while not exit_condition:
        fit_2_i = []
        dict_fit_2_i = {}
        dict_fit_3_xy_i = {}
        max_fit_i = []  # FIXME: never used

        pop_minus_archive = [ele for ele in pop if ele not in archive_p]

        pop_minus_archive_and_ineligible = [ele for ele in pop_minus_archive
                                            if ele not in ineligible_p]

        # Check if all individuals of pop have been considered archive.
        if not pop_minus_archive_and_ineligible:
            print('No individual left to be considered for archive memebership.')
            break

        # Line 6 - 12 of psuedo code

        # Calculate the individual fitness of pop_prime individuals,
        # given the collaborations made only with the members of archive_p
        fit_1, dict_fit_1 = calculate_fit_given_archive(
            archive_p, pop_prime, complete_solutions_set_internal)

        for i in pop_minus_archive:
            # Calculate fitness of pop_prime individuals given archive and i
            fit_2_i_x = calculate_fit_given_archive_and_i(
                i, pop_prime, complete_solutions_set_internal, fit_1)

            fit_2_i.append(fit_2_i_x)
            dict_fit_2_i[str(i)] = fit_2_i_x

            # Create a dictionary that records fit_3 values for each i
            index_i = pop_minus_archive.index(i)
            dict_fit_3_xy_i[str(i)] = rank_change(
                pop_prime, fit_1, fit_2_i, index_i)

        max_fit_i = max_rank_change_fitness(
            pop_minus_archive_and_ineligible, dict_fit_3_xy_i)

        # Find the maximum of all max_fit_i values and its corresponding i
        max_fit = max(max_fit_i)
        if max_fit != -1 * np.inf:
            # Find a, i.e., the candidate member to be added to archive_p
            index_a = max_fit_i.index(max_fit)
            a = pop_minus_archive_and_ineligible[index_a]

            # Find a's collaborator that has maximum fitness value
            max_fit_3_a = dict_fit_3_xy_i[str(a)]
            for x in range(len(pop_prime)):
                if max_fit in max_fit_3_a[x]:
                    x_a = pop_prime[max_fit_3_a[x].index(max_fit)]

            # Check the distance between a and other members of archive_p
            if is_similar(a, x_a, archive_p,
                          dict_archive_memebers_and_collaborators,
                          min_distance, first_item_class):
                ineligible_p.append(a)
            else:
                archive_p.append(a)
                dict_archive_memebers_and_collaborators[str(a)] = x_a
        else:
            exit_condition = True

    return archive_p


def violate_safety_requirement(complete_solution):
    """Checks whether a complete_solution violates a safety requirement. In
    case of violation, the function returns `True`, otherwise it returns
    `False`. Since the definition of fitness function is based on the safety
    requirement and it is defined in a way that its positive sign indicates a
    violation and vice versa.

    :param complete_solution: a complete solution that has a fitness
                              value.
    """
    assert complete_solution is not None, \
        "complete_solution cannot be None."

    assert complete_solution.fitness.values is not None, \
        "complete_solution must have a real value."

    if complete_solution.fitness.values[0] > 0:
        return True
    else:
        return False


# Create fitness and individual datatypes.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("Scenario", creator.Individual)
creator.create("OutputMLC", creator.Individual)

SCEN_IND_SIZE = 1  # Size of a scenario individual
MLCO_IND_SIZE = 2  # Size of an MLC output individual
SCEN_POP_SIZE = 1  # Size of the scenario population
MLCO_POP_SIZE = 1  # Size of the MLC output population
MIN_DISTANCE = 1  # Minimum distance between members of an archive

# The list of lower and upper limits for enumerationed types in sceanrio.
enumLimits = [np.nan, np.nan, (1, 6)]

toolbox = base.Toolbox()

# Define functions and register them in toolbox.
toolbox.register(
    "scenario", initialize_scenario,
    creator.Individual, SCEN_IND_SIZE
)
toolbox.register(
    "outputMLC", initializeMLCO,
    creator.Individual, MLCO_IND_SIZE
)
toolbox.register(
    "popScen", tools.initRepeat, list,
    toolbox.scenario, n=SCEN_POP_SIZE
)
toolbox.register(
    "popMLCO", tools.initRepeat, list,
    toolbox.outputMLC, n=MLCO_POP_SIZE
)

# ----------------------


def main():
    # Instantiate individuals and populations
    popScen = toolbox.popScen()
    popMLCO = toolbox.popMLCO()
    arcScen = toolbox.clone(popScen)
    arcMLCO = toolbox.clone(popMLCO)
    solutionArchive = []

    # Create complete solutions and evaluate individuals
    completeSolSet = evaluate(
        popScen, arcScen, popMLCO, arcMLCO, creator.Individual, 1)

    # # Record the complete solutions that violate the requirement r
    # for c in completeSolSet:
    #     if violateSafetyReq(c) is True:
    #         solutionArchive = solutionArchive + c

    # Evolve archives and populations for the next generation
    min_distance = 1
    arcScen = update_archive(
        popScen, popMLCO, completeSolSet,
        creator.Individual, min_distance
    )
    arcMLCO = update_archive(
        popMLCO, popScen, completeSolSet,
        creator.Individual, min_distance
    )

    # Select, mate (crossover) and mutate individuals that are not in archives.
    popScen = breed_scenario(popScen, enumLimits)
    popMLCO = breed_mlco(popMLCO)


if __name__ == "__main__":
    main()
