"""
Set of utility functions which are independent from the `problem` structure.
"""

from deap import tools, creator
from datetime import datetime
import logging
import os
from pathlib import Path
import random
from copy import deepcopy

import numpy as np


def initialize_hetero_vector(limits, class_=None):
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

    if class_:
        return class_(x)
    else:
        return x


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
    # arc = deepcopy(archive)
    # pop = deepcopy(population)

    complete_solution_set = []

    for i in archive:
        for j in population:
            c = create_complete_solution(i, j, ficls)
            complete_solution_set = complete_solution_set + joint_class([c])

    return complete_solution_set


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

    # pA = deepcopy(first_population)
    # pB = deepcopy(second_population)
    # aA = deepcopy(first_archive)

    # Ensure that first_archive is a subset of first_population
    assert all(x in first_population for x in first_archive), \
        "first_archive is not a subset of first_population"

    complete_solution_set = []

    # Find {pA - aA}
    pAComplement = [
        ele for ele in first_population if ele not in first_archive]

    # Create complete solution between all members of pB and
    # (min_num_evals - len(aA)) members of pAComplement
    while min_num_evals - len(first_archive) > 0:
        random_individual = \
            pAComplement[random.randint(0, len(pAComplement) - 1)]
        for i in second_population:
            c = create_complete_solution(
                random_individual, i, first_component_class)
            complete_solution_set = \
                complete_solution_set + joint_class([c])
        min_num_evals = min_num_evals - 1

    return complete_solution_set


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
    # a1 = deepcopy(first_archive)
    # p1 = deepcopy(first_population)
    # a2 = deepcopy(second_archive)
    # p2 = deepcopy(second_population)

    # Create complete solutions from collaborations with the archives.
    complete_solutions_set = \
        collaborate_archive(
            first_archive, second_population,
            joint_class, first_component_class) \
        + collaborate_archive(
            second_archive, first_population,
            joint_class, first_component_class) \
        + collaborate_complement(
            first_population, first_archive,
            second_population, min_num_evals,
            joint_class, first_component_class) \
        + collaborate_complement(
            second_population, second_archive,
            first_population, min_num_evals,
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


def mutate_flat_hetero_individual(
        individual, intLimits, mutbpb, mutgmu,
        mutgsig, mutgpb, mutipb):
    """Mutates a flat list of heterogeneous types individual. Input is
    an unmutated flat_hetero_individual, while the output is a mutated 
    individual.

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

    # LIMITATION: assumes a specific format for intLimits.
    #  FIXME: Write an assertion to check intLimits format.

    cls = type(individual)
    mutated_individual = []

    for i in range(len(individual)):
        buffer = [individual[i]]

        if type(buffer[0]) is int:
            buffer = tools.mutUniformInt(
                buffer, low=intLimits[i][0],
                up=intLimits[i][1], indpb=mutipb
            )
            buffer = list(buffer[0])

        if type(buffer[0]) is bool:
            buffer = tools.mutFlipBit(buffer, indpb=mutbpb)
            buffer = list(buffer[0])

        if type(buffer[0]) is float:
            buffer = tools.mutGaussian(buffer, mu=mutgmu,
                                       sigma=mutgsig, indpb=mutgpb)
            buffer = list(buffer[0])

        mutated_individual += buffer

    return cls(mutated_individual)


def evaluate_individual(individual, complete_solution_set, index):
    """Aggregates joint fitness values that an individual has been invovled in.

    It takes an individual, a list of all complete solutions,
    `completeSolSet`, that include the `individual` at the `index` of
    the complete solution; it returns the aggregate fitness value for
    `individual` as a real value.
    """
    values_joint_fitness_involved = []

    # Add the joint fitness values of complete solutions in which
    # individual has been a part of.
    for cs in complete_solution_set:
        if cs[index] == individual:
            values_joint_fitness_involved.append(cs.fitness.values[0])

    # Aggregate the joint fitness values. For now, maximum values is used.
    # individual_fitness_value = max(values_joint_fitness_involved)
    # For testing the new fitness function!
    individual_fitness_value = min(values_joint_fitness_involved)

    return (individual_fitness_value,)


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


def calculate_std(two_d_list, numeric_value_index):
    """Calculates the standard deviation for the numeric values whose index is
    provided.

    The values are in a 2D list.
    """
    X = gather_values_in_np_array(two_d_list, numeric_value_index)

    return np.std(X)


def calculate_max(two_d_list, numeric_value_index):
    """Calculates the maximum value along a column of a 2D list."""
    X = gather_values_in_list(two_d_list, numeric_value_index)

    return max(X)


def calculate_min(two_d_list, numeric_value_index):
    """Calculates the minimum value along a column of a 2D list."""
    X = gather_values_in_list(two_d_list, numeric_value_index)

    return min(X)


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
                # ???: why multiply by 4?
                numeric_range[i] = 4 * calculate_std(X, i)
            else:
                numeric_range[i] = calculate_max(X, i) - calculate_min(X, i)
                if numeric_range[i] == 0.0:
                    numeric_range[i] = 0.0001
                    # To avoid division by zero in case of similar values.

    # Calculate the distance for numerical elements
    for index in num_ix:
        for row in range(1, row_x):
            column_difference = X[0][index] - X[row][index]
            results_array[row, index] = \
                np.abs(column_difference) / \
                numeric_range[index]

            # USE THE ABSOLUTE VALUE FOR DIFFERENCE

    heom_distance_values = \
        list(np.sqrt(np.sum(np.square(results_array)/col_x, axis=1)))
    return heom_distance_values


def index_in_complete_solution(individual, complete_solution):
    """Returns the index of an individual in a complete solution based on its
    type."""
    for i in range(len(complete_solution)):
        if type(complete_solution[i]) == type(individual):
            return i
        # else:
        #     return None


def find_individual_collaborator(individual, complete_solutions_set):
    # ???: not clear what does this function do.
    # Is it okay to return only one of all collaborators?

    index_in_cs = index_in_complete_solution(
        individual, complete_solutions_set[0]
    )
    for cs in complete_solutions_set:
        if cs[index_in_cs] == individual:
            if cs.fitness.values == individual.fitness.values:
                cs_deepcopy = deepcopy(cs)
                cs_deepcopy.pop(index_in_cs)
                return cs_deepcopy[0]


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


def find_max_fv_individual(population):
    """
    Finds the indiviudal with the maximum indiviudal fitness value.
    """
    values = []
    for ind in population:
        individual_fitness_value = ind.fitness.values[0]
        values.append(individual_fitness_value)
    max_fitness_value = max(values)
    index_max_fitness_value = values.index(max_fitness_value)
    return population[index_max_fitness_value]


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


def setup_logger(file_name: str, output_directory: Path = 'results', file_log_level='DEBUG', stream_log_level='INFO'):
    """Initilizes and formats the root logger. It also sets the log
    levels for the log file and stream handler.
    """
    # Create the results folder if it does not exist.
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    # Setup logger.
    logger = logging.getLogger()

    # Get current timestamp to use as a unique ID.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # # Create a log folder for a run.
    # log_folder = os.path.join('results', str(timestamp))
    # pathlib.Path(log_folder).mkdir(parents=True, exist_ok=True)

    # parser = configparser.ConfigParser()
    # parser.set()

    log_id = str(timestamp) + '_' + file_name + '.log'
    log_file = os.path.join(output_directory, log_id)
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')

    # Set logger's logging level.
    if file_log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif file_log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif file_log_level == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif file_log_level == 'ERROR':
        logger.setLevel(logging.ERROR)
    else:
        raise ValueError(
            "file_log_level can only be DEBUG, INFO, WARNING or ERROR.")

    # Initialize and format the stream_handler.
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    stream_handler.setFormatter(formatter)

    # Set stream_handler logging level.
    if stream_log_level == 'DEBUG':
        stream_handler.setLevel(logging.DEBUG)
    elif stream_log_level == 'INFO':
        stream_handler.setLevel(logging.INFO)
    elif stream_log_level == 'WARNING':
        stream_handler.setLevel(logging.WARNING)
    elif stream_log_level == 'ERROR':
        stream_handler.setLevel(logging.ERROR)
    else:
        raise ValueError(
            "stream_log_level can only be DEBUG, INFO, WARNING or ERROR.")

    logger.addHandler(stream_handler)


def setup_file(file_name: str, output_directory: Path = 'results', file_extension: str = '.log'):
    # Create the results folder if it does not exist.
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Get current timestamp to use as a unique ID.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_id = str(timestamp) + file_name + '.log'

    # file = output_directoty.joinpath(file_id)
    file = os.path.join(output_directory, file_id)

    return file


def setup_logbook_file(output_dir: Path = 'results'):
    return setup_file(file_name='_logbook', output_directory=output_dir)


def flatten_list(nested_list):
    """Flattens a nested list into a 1D list.
    """
    # return sum(map(flatten_list, nested_list), []) \
    #     if (isinstance(nested_list, list) or
    #         isinstance(nested_list, creator.Individual) or
    #         isinstance(nested_list, creator.Scenario) or
    #         isinstance(nested_list, creator.OutputMLC)) else [nested_list]
    return sum(map(flatten_list, nested_list), []) \
        if isinstance(nested_list, list) else [nested_list]
