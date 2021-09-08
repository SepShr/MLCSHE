import random
from copy import deepcopy

import numpy as np

from deap import creator
from deap import base
from deap import tools

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
            # This line makes the function casestudy-dependant.
            if type(i) == ficls:
                c = [i, j]
            else:
                c = [j, i]
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
            # This line makes the function casestudy-dependant.
            if type(i) == first_component_class:
                c = [i, random_individual]
            else:
                c = [random_individual, i]
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