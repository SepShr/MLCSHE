import random
from copy import deepcopy

import numpy as np

from deap import creator
from deap import base
from deap import tools
from numpy.core.fromnumeric import argmax

# Sample initalization function for Scenarios
def initialize_scenario(class_, limits):
    """
    Initializes a heterogeneous vector of type `class_` based on the values
    in `limits`.

    :param class_: the class into which the final list will be typecasted into.
    :param limits: a list that determines whether an element of the individual
                   is a bool, int or float. It also provides lower and upper 
                   limits for the int and float elements.
    :returns: a heterogeneous vector of type `class_`.

    Furthermore, it assumes that `limits` is a list and it elements have 
    the folowing format:
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
    """
    Initializes an individual of the MLCO population.
    """
    # return b
    return print("initializeMLCO() returned.\n")

def create_complete_solution(element, other_element, first_component_class):
    """
    Creates a complete solution from two elements such that the one with the 
    type `first_component_class` is the first component of the complete solution.
    """
    if type(element) == first_component_class:
        c = [element, other_element]
    else:
        c = [other_element, element]
    return c

def collaborate_archive(archive, population, joint_class, ficls):
    """
    Create collaborations between individuals of archive and population.
    The complete solutions (collaborations) are of class icls. Output
    is the list of complete solutions `complete_solution_set`. 

    :param archive: the archive (set) of individuals with which every 
                    memeber of population should collaborate.
    :param population: the set of individuals that should collaborate
                       with the memebers of the archive.
    :param joint_class: the name of the class into which a complete solution
                        or `c` would be typecasted into.
    :param ficls: the name of the first individual's class to be included 
                  in the complete solution. Defines the format of a complete
                  solution including 2 individuals of different types.
    """
    # Ensure that the input is not None. Exception handling should be added.
    assert archive and population and joint_class and ficls ,\
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

def collaborate_complement(first_population, first_archive, second_population,
     min_num_evals, joint_class, first_component_class):
    """
    Create collaborations between the members of `(first_population - 
    first_archive)` and `second_population`. It returns a set of complete 
    solutions `complete_solution_set` which has collaborations between the 
    members of `second_population` and `(first_population - first_archive)`
    or `pAComplement`.

    :param first_population: population A which is a list of individuals.
    :param first_archive: an archive (set) of individuals, also a subset of
                          `first_population`.
    :param second_population: the set of individuals that should collaborate
                              with the memebers of the `first_population - 
                              first_archive`.
    :param min_num_evals: the number of collaborations that each individual
                          in `second_population` should participate in. Note
                          that they have already participated 
                          `len(first_archive)` times.
    :param joint_class: the name of the class into which a complete solution
                        or `c` would be typecasted into.
    :param first_component_class: the name of the first individual's class to
                                  be included in the complete solution. Defines
                                  the format of a complete solution including 
                                  2 individuals of different types.
    """
    if min_num_evals <= len(first_archive):
        return []

    pA = deepcopy(first_population)
    pB = deepcopy(second_population)
    aA = deepcopy(first_archive)

    # Ensure that first_archive is a subset of first_population
    assert all(x in pA for x in aA), "first_archive is not a subset of \
        first_population"

    complete_solution_set = []
    
    # Find {pA - aA}
    pAComplement = [ele for ele in pA]
    for _ in aA:
        if _ in pAComplement:
            pAComplement.remove(_)

    # Create complete solution between all members of pB and 
    # (min_num_evals - len(aA)) members of pAComplement
    while min_num_evals - len(aA) > 0:
        random_individual = pAComplement[random.randint(0, len(pAComplement)-1)]
        for i in pB:
            c = create_complete_solution(random_individual, i, first_component_class)
            complete_solution_set = complete_solution_set + joint_class([c])
        min_num_evals = min_num_evals - 1

    return complete_solution_set

    # Uncomment the following line while commenting the rest of the method to 
    # have a minimally executable code skeleton.
    # return print("collbaComp() returned.\n")

def collaborate(first_archive, first_population, second_archive, \
    second_population, joint_class, first_component_class, min_num_evals):
    """
    Creates a complete solution from two sets of individuals. It takes 
    two sets (arc and pop) and the type of individuals as input. It
    returns a set of unique complete solutions (collaborations).

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
            + collaborate_complement(p1, a1, p2, min_num_evals, joint_class, \
                first_component_class) \
                + collaborate_complement(p2, a2, p1, min_num_evals, \
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

def evaluate(first_population, first_archive, second_population, second_archive,\
     joint_class, min_num_evals):
    """
    Forms complete solutions, evaluates their joint fitness and evaluates the
    individual fitness values.

    :param first_population: the population (list) of scenarios.
    :param first_archive: the archive (list) of scenarios.
    :param second_population: the population of MLC outputs.
    :param second_archive: the archive of MLC outputs.
    :param joint_class: type into which each complete solution will be typecasted.
    :param min_num_evals: the minimum number of collaborations and thus, 
                          joint fitness evaluations per individual.
    :returns: set of complete solutions with their fitness values, set of 
              scenarios with their individual fitness values, and the set of
              MLC outputs with their individual fitness values.
    """
    # Exception handling must be added.
    
    # Deep copy the inputs
    population_one = deepcopy(first_population)
    population_two = deepcopy(second_population)
    archive_one = deepcopy(first_archive)
    archive_two = deepcopy(second_archive)

    first_component_class = type(population_one[0])
    complete_solutions_set = collaborate(archive_one, population_one,\
         archive_two, population_two, joint_class, first_component_class, \
             min_num_evals)

    # Evaluate joint fitness and record its value.
    for c in complete_solutions_set:
        c.fitness.values = evaluate_joint_fitness(c)
  
    # Evaluate individual fitness values.
    for individual in population_one:
        individual.fitness.values = evaluateIndividual(individual,\
            complete_solutions_set, 0)
  
    for individual in population_two:
        individual.fitness.values = evaluateIndividual(individual,\
            complete_solutions_set, 1)
        
    return complete_solutions_set, population_one, population_two
    
    # Uncomment the following line while commenting the rest of the method to
    # have a minimally executable code skeleton.
    # collaborate(popScen, arcScen, popMLCO, arcMLCO, cscls, fcls, k)
    # return print("evaluate() returned.\n")

# Evaluate the joint fitness of a complete solution.
def evaluate_joint_fitness(c):
    """
    Evaluates the joint fitness of a complete solution. It takes the complete
    solution as input and returns its joint fitness as output.
    """
    # Returns a random value for now.
    return (random.uniform(-5.0, 5.0),)

    # Uncomment the following line while commenting the rest of the method to
    # have a minimally executable code skeleton.
    # return print("evaluateJFit() returned.\n")

def evaluateIndividual(individual, complete_solution_set, index):
    """
    Aggregates joint fitness values that an individual has been invovled in.
    It takes an individual, a list of all complete solutions, `completeSolSet`,
    that include the `individual` at the `index` of the complete solution; it 
    returns the aggregate fitness value for `individual` as a real value.
    """
    weights_joint_fitness_involved = []
    values_joint_fitness_involved = []

    # Add the joint fitness values of complete solutions in which individual
    # has been a part of.
    for cs in complete_solution_set:
        if cs[index] == individual:
            weights_joint_fitness_involved += [cs.fitness.values]
    
    for i in weights_joint_fitness_involved:
        values_joint_fitness_involved += list([i[0]])

    
    # Aggregate the joint fitness values. For now, maximum values is used.
    individual_fitness_value = max(values_joint_fitness_involved)

    return (individual_fitness_value,)

# Breed scenarios.
def breedScenario(popScen, arcScen, enumLimits, tournSize, cxpb,  mutbpb,\
    mutgmu, mutgsig, mutgpb, mutipb):
    
    """
    Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the Scenarios population. It takes an old 
    generation of scenarios as input and returns an evolved generation.
    
    :param popScen: the population of scenarios.
    :param arcScen: the list of all memebrs of the archive.
    :param enumLimits: a 2D list that contains a lower and upper limits for 
                      the mutation of elements in a scenario of type int.
    :param tournSize: the size of the tournament to be used by the tournament 
                      selection algorithm.
    :param cxpb: the probability that a crossover happens between two individuals.
    :param mutbpb: the probability that a binary element might be mutated by the 
                   tools.mutFlipBit() function.
    :param mutgmu: the normal distribution mean used in tools.mutGaussian().
    :param mutgsig: the normal distribution standard deviation used in 
                    tools.mutGaussian().
    :param mutgpb: the probability that a real element might be mutated by the
                   tools.mutGaussian() function.
    :param mutipb: the probability that a integer element might be mutated by
                   the mutUniformInt() function.
    :returns: a list of bred scenario individuals (that will be appended to the
              archie of scenarios to form the next generation of the population).
    """
    # Registering evolutionary operators in the toolbox.
    toolbox.register("select", tools.selTournament, tournsize=tournSize,\
        fit_attr='fitness')
    toolbox.register("crossover", tools.cxUniform, indpb=cxpb)

    # Find the complement (population minus the archive).
    breeding_population = [ele for ele in popScen]
    for _ in arcScen:
        if _ in breeding_population:
            breeding_population.remove(_)

    # Select 2 parents, cx and mut them until satisfied.
    offspring_list = []
    size = len(popScen) - len(arcScen)
    while size > 0:
        # Select 2 parents from the breeding_population via the select fucntion.
        parents = toolbox.select(breeding_population, k=2)
        # Perform crossover.
        offspring_pair = toolbox.crossover(parents[0], parents[1])
        # Choose a random offspring and typecast it into list.
        offspring = list(offspring_pair[random.getrandbits(1)])
        # Mutate the offspring.
        offspring = mutateScenario(offspring, enumLimits,  mutbpb, mutgmu, \
            mutgsig, mutgpb, mutipb)
        offspring_list += [offspring]
        size = size - 1

    return offspring_list

    # Uncomment the following line while commenting the rest of the method to
    # have a minimally executable code skeleton.
    # mutateScenario(popScen, enumLimits)
    # return print("breedScen() returned.\n")

# Breed MLC outputs.
def breedMLCO(outputMLC):
    """
    Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the MLC output population. It takes an old
    generation of MLC ouptputs as input and return an evovled generation.
    """
    # return outputMLC
    return print("breedMLCO() returned.\n")

# Mutate scenarios
def mutateScenario(scenario, intLimits, mutbpb, mutgmu, mutgsig, mutgpb, mutipb):
    """
    Mutates a scenario individual. Input is an unmutated scenario, while the
    output is a mutated scenario. The function applies one of the 3 mutators
    to the elements depending on their type, i.e., `mutGaussian()` (Guass distr)
    to Floats, `mutFlipBit()` (bitflip) to Booleans and `mutUniformInt()` 
    (integer-randomization) to Integers.

    :param scenario: a scenario type individual to be mutated by the function.
    :param intLimits: a 2D list that contains a lower and upper limits for 
                      the mutatio of elements in a scenario that are of type int. 
    :param mutbpb: the probability that a binary element might be mutated by the 
                   tools.mutFlipBit() function.
    :param mutgmu: the normal distribution mean used in tools.mutGaussian().
    :param mutgsig: the normal distribution standard deviation used in 
                    tools.mutGaussian().
    :param mutgpb: the probability that a real element might be mutated by the
                   tools.mutGaussian() function.
    :param mutipb: the probability that a integer element might be mutated by
                   the mutUniformInt() function.
    """
    toolbox.register("mutateScenBool", tools.mutFlipBit, indpb=mutbpb)
    toolbox.register("mutateScenFlt", tools.mutGaussian, mu=mutgmu,\
        sigma=mutgsig, indpb=mutgpb)
    
    # LIMITATION: assumes a specific format for intLimits.

    cls = type(scenario)
    mutatedScen = []

    for i in range(len(scenario)):
        buffer = [scenario[i]]

        if type(buffer[0]) is int:
            buffer = tools.mutUniformInt(buffer, low= intLimits[i][0],\
                up=intLimits[i][1], indpb=mutipb)
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

def updateArc_Scen(archive, pop):
    """
    Updates the archive of scenarios for the next generation.
    """
    # return archive
    return print("updateArc_Scen returned.\n")

def updateArc_MLCO(archive, pop):
    """
    Update the archive of MLC outputs for the next generation.
    """
    # return archive
    return print("updateArc_MLCO() returned.\n")

def flatten(list_of_lists):
    """
    Flattens a list of lists. It returns the `flattened_list`. Note that this
    function is recursive.

    :param list_of_lists: a list of lists. It can be an irregular nested list.
    :returns: flattened list.
    """
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list) or isinstance(list_of_lists[0], creator.Individual) or \
                    isinstance(list_of_lists[0], creator.Scenario) or isinstance(list_of_lists[0], creator.OutputMLC):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def identify_nominal_indices(flat_list):
    """
    Identifies nominal values and returns their index in a list.

    :param flat_list: a flat list that contains elements of type `int`, `str`,
                      `bool`, or `float`. The first 3 types are considered as
                      nominal values.
    :returns: a list of the nominal values indices, `nominal_values_indices`.
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
    """
    Prepares an irregular nested list for distance evaluation. It returns the
    flattened list and the list of indices for nominal values.

    :param irregular_nested_list: an irregular nested list, i.e., a list that
                                  that may have lists or other types such as 
                                  `int`, `str`, or `flt` as elements.
    :returns: a flattened list and the list of indices that correspond to the
              nominal values.
    """
    flattened_list = flatten(irregular_nested_list)

    nominal_values_indices = identify_nominal_indices(flattened_list)

    return flattened_list, nominal_values_indices

def gather_values_in_np_array(two_d_list, numeric_value_index):
    """
    Gathers all numeric values from a 2D list, located at a specific column.

    :param two_d_list: a 2D list. 
    :param numeric_value_index: the index of a numeric value, i.e., a coloumn 
                                in the 2D array.

    :returns: a numpy array.
    """
    ## DOES NOT HANDLE NOMINAL VALUE INPUTS.
    ## DOES NOT HANDLE CASES WERE THE NUMERIC VALUE INDEX IS OUT OF RANGE.
    numeric_values_array = np.zeros(len(two_d_list))

    for i in range(len(two_d_list)):
        numeric_values_array[i] = two_d_list[i][numeric_value_index]
    
    return numeric_values_array

def gather_values_in_list(two_d_list, numeric_value_index):
    """
    Gathers all the numeric values from a 2D list, located at a specific column.

    :param two_d_list: a 2D list. 
    :param numeric_value_index: the index of a numeric value, i.e., a coloumn 
                                in the 2D array.

    :returns: a list.
    """
    ## DOES NOT HANDLE NOMINAL VALUE INPUTS.
    ## DOES NOT HANDLE CASES WERE THE NUMERIC VALUE INDEX IS OUT OF RANGE.
    numeric_values_list = []

    for i in range(len(two_d_list)):
        numeric_values_list.append(two_d_list[i][numeric_value_index])
    
    return numeric_values_list

## NO TESTCASE
def calculate_std(two_d_list, numeric_value_index):
    """
    Calculates the standard deviation for the numeric values whose index is
    provided. The values are in a 2D list.
    """
    X = gather_values_in_np_array(two_d_list, numeric_value_index)

    return np.std(X)

## NO TESTCASE
def calculate_max(two_d_list, numeric_value_index):
    """
    Calculates the maximum value along a column of a 2D list.
    """
    X = gather_values_in_list(two_d_list, numeric_value_index)
    
    return max(X)

## NO TESTCASE
def calculate_min(two_d_list, numeric_value_index):
    """
    Calculates the minimum value along a column of a 2D list.
    """
    X = gather_values_in_list(two_d_list, numeric_value_index)
    
    return min(X)

def measure_heom_distance(X, cat_ix, nan_equivalents=[np.nan, 0], normalised="normal"):
    """
    Calculate the HEOM difference between a list located at X[0] and the rest
    of the lists of similar size.

    :param X: X is a 2D list of flattened heterogeneuous lists.
    :param cat_ix: is a list of indices of the categorical values.
    :param nan_equivalents: list of values that are considered as missing values.
    :param normalised: normalization method, can be "normal" or "std".
    """
    ## ASSUMPTIONS: X HAS NO MISSING VALUES. IMPLEMENTATION SHOULD BE IMPROVED.
    nan_eqvs = nan_equivalents
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
    num_ix = []
    for i in range(col_x):  # The assumption is that there are no missing values
        if i not in cat_ix:
            num_ix.append(i)

    # Calculate range for numeric values.
    for i in range(len(X[0])):
        if i in num_ix:
            if normalised == "std":
                numeric_range[i] = 4 * calculate_std(X, i)
            else:
                numeric_range[i] = calculate_max(X, i) - calculate_min(X, i)
                if numeric_range[i] == 0 or numeric_range[i] == 0.0:
                    numeric_range[i] = 0.0001  ## To avoid divide by zero in case of similar values.

    # Calculate the distance for numerical elements
    for index in num_ix:
        for row in range(1, row_x):  ## DOUBLE-CHECK THE RANGE VALUES
            column_difference = X[0][index] - X[row][index]
            results_array[row, index] = np.sqrt(np.square(column_difference)) / numeric_range[index]  ## USE THE ABSOLUTE VALUE FOR DIFFERENCE

    heom_distance_values = list(np.sqrt(np.sum(np.square(results_array), axis = 1)))
    return heom_distance_values

## DOCSTRING INCOMPLETE
def is_similar(candidate, collaborator, archive, \
            archive_members_and_collaborators_dictionary, \
               min_distance, first_item_class):
    """
    The algorithm evaluates if a `candidate` and its `collaborator` are similar to 
    the memebrs of an `archive` and their collaborators (recorded in 
    `archive_members_and_collaborators_dictionary`). Similarity uses the criteria
    `min_distance` to decide.
    """
    cand = deepcopy(candidate)
    collab = deepcopy(collaborator)
    flat_complete_solutions_list = []
    ficls = first_item_class
    archive_dict = archive_members_and_collaborators_dictionary
    # Create the complete solution of cand and collab
    main_complete_solution = create_complete_solution(cand, collab, ficls)

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
        archive_complete_solution = create_complete_solution(archive[i], archive_dict[str(archive[i])], ficls)
        archive_complete_solution_flat, arc_nom_indices = \
            prepare_for_distance_evaluation(archive_complete_solution)
        if arc_nom_indices != nominal_values_indices:
            print('The nominal values between ' + str(archive_complete_solution) \
                 + ' and ' + str(main_complete_solution) + ' do not match!')
        flat_complete_solutions_list.append(archive_complete_solution_flat)

    distance_values = measure_heom_distance(flat_complete_solutions_list, \
                                            nominal_values_indices)
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

## NO TESTCASE
def update_archive(population, other_population, \
    complete_solutions_set, joint_class, min_distance):
    """
    Updates the archive according to iCCEA updateArchive algorithm. It starts
    with an empty archive for p, i.e., `archive_p` and adds informative members
    to it. The initial member is the individual with the highest fitness value.
    Individuals that: 1. change the fitness ranking of the other population, 
    2. are not similar to existing members of `archive_p`, and 3. have the
    highest fitness ranking will be added to the `archive_p` for the next
    generation of the coevolutionary search.
    """
    # VALUES FOR TESTING
    # scen1 = creator.Scenario([1, False, 5.0])
    # scen1.fitness.values = (random.randint(-10, 10),)
    # print('the fitness value of ' +  str(scen1) + ' is: ' + str(scen1.fitness.values))
    # mlco1 = creator.OutputMLC([[8, 'a'], [2, 'b']])
    # mlco1.fitness.values = (random.randint(-10, 10),)
    # print('the fitness value of ' +  str(mlco1) + ' is: ' + str(mlco1.fitness.values))
    # scen2 = creator.Scenario([4, True, -7.8])
    # scen2.fitness.values = (random.randint(-10, 10),)
    # print('the fitness value of ' +  str(scen2) + ' is: ' + str(scen2.fitness.values))
    # scen3 = creator.Scenario([-2, False, 4.87])
    # scen3.fitness.values = (random.randint(-10,10),)
    # print('the fitness value of ' +  str(scen3) + ' is: ' + str(scen3.fitness.values))
    # mlco2 = creator.OutputMLC([[1, 'a'], [21, 'd']])
    # mlco2.fitness.values = (random.randint(-10, 10),)
    # print('the fitness value of ' +  str(mlco2) + ' is: ' + str(mlco2.fitness.values))
    # mlco3 = creator.OutputMLC([[-2, 'e'], [10, 'f']])
    # mlco3.fitness.values = (random.randint(-10,10),)
    # print('the fitness value of ' +  str(mlco3) + ' is: ' + str(mlco3.fitness.values))
    # scen4 = creator.Scenario([2, True, 0.24])
    # scen4.fitness.values = (random.randint(-10,10),)
    # print('the fitness value of ' +  str(scen4) + ' is: ' + str(scen4.fitness.values))
    # mlco4 = creator.OutputMLC([[4, 'g'], [-1, 'h']])
    # mlco4.fitness.values = (random.randint(-10,10),)
    # print('the fitness value of ' +  str(mlco4) + ' is: ' + str(mlco4.fitness.values))
    # pScen = [scen1, scen2, scen3, scen4]
    # pMLCO = [mlco1, mlco2, mlco3, mlco4]
    # cls = creator.Individual
    # cs1 = creator.Individual([scen1, mlco1])
    # cs1.fitness.values = (random.randint(-10, 10),)
    # print('the fitness value of ' +  str(cs1) + ' is: ' + str(cs1.fitness.values))
    # cs2 = creator.Individual([scen1, mlco2])
    # cs2.fitness.values = (random.randint(-10, 10),)
    # print('the fitness value of ' +  str(cs2) + ' is: ' + str(cs2.fitness.values))
    # cs3 = creator.Individual([scen2, mlco1])
    # cs3.fitness.values = (random.randint(-10, 10),)
    # print('the fitness value of ' +  str(cs3) + ' is: ' + str(cs3.fitness.values))
    # cs4 = creator.Individual([scen2, mlco2])
    # cs4.fitness.values = (random.randint(-10, 10),)
    # print('the fitness value of ' +  str(cs4) + ' is: ' + str(cs4.fitness.values))
    # css = [cs1, cs2, cs3, cs4]

    pop = deepcopy(population)
    pop_prime = deepcopy(other_population)
    complete_solutions_set_internal = deepcopy(complete_solutions_set)

    archive_p =[]
    ineligible_p =[]

    # Add the first individual to the archive_p, which has the highest
    # fitness value
    values = []
    for ind in pop:
        individual_fitness_value = ind.fitness.values[0]
        values.append(individual_fitness_value)
    max_fitness_value = max(values)
    index_max_fitness_value = values.index(max_fitness_value)
    max_fitness_value_individual = pop[index_max_fitness_value]
    archive_p.append(max_fitness_value_individual)
    print('the first individual that is added to the archive_p is: %s' % \
          max_fitness_value_individual)

    ## INITIALIZE THE DICT OF ARCHIVE MEMEBERS AND THE HIGHEST COLLABORATOR
    dict_archive_memebers_and_collaborators = {}

    exit_condition = False
    # counter = 0
    # while counter < 3:
    while exit_condition == False:
        # print('This is loop number: %s' % counter)  # counter loop
        dict_fit_1 = {}
        dict_fit_2_i = {}
        dict_fit_3_xy_i = {}
        max_fit_i = []
        dict_max_fit_i = {}
        
        pop_minus_archive = pop
        for _ in archive_p:
          print('in for 0')
          if _ in pop_minus_archive:
            pop_minus_archive.remove(_)
        print('pop - archive_p is: %s' % pop_minus_archive)
        
        pop_minus_archive_and_ineligible = pop_minus_archive
        for _ in ineligible_p:
          if _ in pop_minus_archive_and_ineligible:
            pop_minus_archive_and_ineligible.remove(_)
        print('pop - archive_p - ineligible_p is: %s' % pop_minus_archive_and_ineligible)

        # Line 6 - 12
        fit_1 = []
        fit_2_i = []

        for x in pop_prime:
            print('Calculating fit_1')
            print('in for 1')
            comp_sol_set_archive = []
            for ind in archive_p:
                c = create_complete_solution(ind, x, type(complete_solutions_set[0][0]))
                print('the created complete solution is: %s' % c)
                print('in for 3')
                if c not in complete_solutions_set:
                    c = joint_class(c)
                    c.fitness.values = evaluate_joint_fitness(c)
                    comp_sol_set_archive.append(c)
                    print('type of the first element of comp_sol_set_archive is: %s' % \
                        type(comp_sol_set_archive[0]) )
                    print('the fitness value for c is: %s' % \
                        c.fitness.values[0])
                    print(str(comp_sol_set_archive))
                    complete_solutions_set_internal.append(c)
                else:
                    for cs in complete_solutions_set_internal:
                        print('in for 6')
                        if c == cs:
                            comp_sol_set_archive.append(cs)
                            print('the complete solution already exists')
                            break
                
            print('set considered for evaluateIndividual is: %s' % comp_sol_set_archive)
            fit_1 += [evaluateIndividual(x, comp_sol_set_archive, 1)[0]]
            print('fit_1 is: %s' % fit_1)
            string_x = deepcopy(x)
            string_x = str(string_x)
            dict_fit_1[string_x] = evaluateIndividual(x, comp_sol_set_archive, 1)[0]
            print(str(dict_fit_1))

        for i in pop_minus_archive:
            print('in for 2')
            fit_2_i_x =[]
            # FIT1 EVALUATION WAS HERE
            
            archive_p_incl_i = []
            archive_p_incl_i = deepcopy(archive_p)
            archive_p_incl_i.append(i)
            print(str(archive_p_incl_i))
            print('the i under consideration is: %s' % i)

            for x in pop_prime:
                print('Calculating fit_2')
                c = create_complete_solution(i, x, type(complete_solutions_set[0][0]))
                print('the created complete solution is: %s' % c)

                if c not in complete_solutions_set_internal:
                        c = joint_class(c)
                        c.fitness.values = evaluate_joint_fitness(c)
                        print('the fitness value for c is: %s' % c.fitness.values[0])
                        complete_solutions_set_internal.append(c)
                        fit_2_i_x_value = max(fit_1[pop_prime.index(x)], c.fitness.values[0])
                        fit_2_i_x.append(fit_2_i_x_value)
                else:
                    for cs in complete_solutions_set_internal:
                        print('in for 6')
                        if c == cs:
                            fit_2_i_x_value = max(fit_1[pop_prime.index(x)], cs.fitness.values[0])
                            fit_2_i_x.append(fit_2_i_x_value)
                            print('the complete solution already existed')
                            break

            fit_2_i.append(fit_2_i_x)
            print('fit_2_i is: %s' % fit_2_i)
            dict_fit_2_i[str(i)] = fit_2_i_x
            print(str(dict_fit_2_i))

            # Create a dictionary that records fit_3 values for each i
            row, col = len(pop_prime), len(pop_prime)
            fit_3_i = [[0 for _ in range(col)] for _ in range(row)]
            index_i = pop_minus_archive.index(i)
            for x in pop_prime:
                for y in pop_prime:
                    index_x = pop_prime.index(x)
                    index_y = pop_prime.index(y)
                    if ((fit_1[index_x] <= fit_1[index_y]) and \
                        (fit_2_i[index_i][index_x] > fit_2_i[index_i][index_y])):
                        fit_3_i[index_x][index_y] = fit_2_i[index_i][index_x]
                    else:
                        fit_3_i[index_x][index_y] = (-1 * np.inf)
            
            print('fit_3_i is: %s' % fit_3_i)
            dict_fit_3_xy_i[str(i)] = fit_3_i

        print('dict_fit_3_xy is: %s' % dict_fit_3_xy_i)
        
        # Find maximum of fit_3_xy for each i
        for i in pop_minus_archive_and_ineligible:
            max_fit_i_x = []
            fit_3_xy = dict_fit_3_xy_i[str(i)]
            print('fit_3_xy is: %s' % fit_3_xy)
            for j in range(len(fit_3_xy)):
                fit_3_x = fit_3_xy[j]
                print('fit_3_x is: %s' % fit_3_x)
                max_fit_i_x.append(max(fit_3_x))
            max_fit_i.append(max(max_fit_i_x))
            print('max_fit_i is: %s' % max_fit_i)
        
        # Find the maximum of all max_fit_i values and its corresponding i
        max_fit = max(max_fit_i)
        print('max_fit is: %s' % max_fit)
        if max_fit != -1 * np.inf:
            # Find a, i.e., the candidate member to be added to archive_p
            index_a = max_fit_i.index(max_fit)
            a = pop_minus_archive_and_ineligible[index_a]
            print('a is: %s' % a)

            # Find a's collaborator that has maximum fitness value
            max_fit_3_a = dict_fit_3_xy_i[str(a)]
            for x in range(row):
                if max_fit in max_fit_3_a[x]:
                    x_a = pop_prime[max_fit_3_a[x].index(max_fit)]
            print('x_a is: %s' % x_a)      

            # Check the distance between a and other members of archive_p
            if is_similar(a, x_a, archive_p, \
                            dict_archive_memebers_and_collaborators, min_distance):
                ineligible_p.append(a)
            else:
                archive_p.append(a)
                dict_archive_memebers_and_collaborators[str(a)] = x_a
        else:
            exit_condition = True

        print('archive_p is: %s' % archive_p)
        print('ineligible_p is: %s' % ineligible_p)

    return archive_p, ineligible_p
        
def violate_safety_requirement(complete_solution):
    """
    Checks whether a complete_solution violates a safety requirement. In case
    of violation, the function returns `True`, otherwise it returns `False`.
    Since the definition of fitness function is based on the safety requirement
    and it is defined in a way that its positive sign indicates a violation and
    vice versa.

    :param complete_solution: a complete solution that has a fitness value. 
    """
    assert complete_solution is not None, "complete_solution cannot be None."

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

# The list of lower and upper limits for enumerationed types in sceanrio.
enumLimits = [np.nan, np.nan, (1,6)]

toolbox = base.Toolbox()

# Define functions and register them in toolbox.
toolbox.register("scenario", initialize_scenario, creator.Individual, SCEN_IND_SIZE)
toolbox.register("outputMLC", initializeMLCO, creator.Individual, MLCO_IND_SIZE)
toolbox.register("popScen", tools.initRepeat, list, toolbox.scenario, n=SCEN_POP_SIZE)
toolbox.register("popMLCO", tools.initRepeat, list, toolbox.outputMLC, n=MLCO_POP_SIZE)

 

# ----------------------

def main():
    # Instantiate individuals and populations
    popScen = toolbox.popScen()
    popMLCO = toolbox.popMLCO()
    arcScen = toolbox.clone(popScen)
    arcMLCO = toolbox.clone(popMLCO)
    solutionArchive = []

    # Create complete solutions and evaluate individuals
    completeSolSet = evaluate(popScen, arcScen, popMLCO, arcMLCO, creator.Individual, 1)

    # # Record the complete solutions that violate the requirement r
    # for c in completeSolSet:
    #     if violateSafetyReq(c) is True:
    #         solutionArchive = solutionArchive + c

    # Evolve archives and populations for the next generation
    arcScen = updateArc_Scen(arcScen, popScen)
    arcMLCO = updateArc_MLCO(arcMLCO, popMLCO)

    # Select, mate (crossover) and mutate individuals that are not in archives.
    popScen = breedScenario(popScen, enumLimits)
    popMLCO = breedMLCO(popMLCO)


if __name__ == "__main__":
    main()