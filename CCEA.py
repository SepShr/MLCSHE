import random
from copy import deepcopy

import numpy as np

from deap import creator
from deap import base
from deap import tools

    
# Sample initalization function for Scenarios
def initializeScenario(scls, size):
    """
    Initializes an individual of the Scenario population.
    """
    # x = scls()
    # return x
    return print("initializeScenario() returned.\n")

# Sample initialization function for MLC Outputs
def initializeMLCO(bcls, size):
    """
    Initializes an individual of the MLCO population.
    """
    # return b
    return print("initializeMLCO() returned.\n")

def collaborateArchive(archive, population, icls, ficls):
    """
    Create collaborations between individuals of archive and population.
    The complete solutions (collaborations) are of class icls. Output
    is the list of complete solutions `compSolSet`. 

    :param archive: the archive (set) of individuals with which every 
                    memeber of population should collaborate.
    :param population: the set of individuals that should collaborate
                       with the memebers of the archive.
    :param icls: the name of the class into which a complete solution
                 or `c` would be typecasted into.
    :param ficls: the name of the first individual's class to be included 
                  in the complete solution. Defines the format of a complete
                  solution including 2 individuals of different types.
    """
    # Ensure that the input is not None. Exception handling should be added.
    assert archive and population and icls, "Input to collaborateArchive \
        cannot be None."

    # Deepcopy the lists.
    arc = deepcopy(archive)
    pop = deepcopy(population)

    compSolSet = []

    for i in arc:
        for j in pop:
            # This line makes the function casestudy-dependant.
            if type(i) == ficls:
                c = [i, j]
            else:
                c = [j, i]
            compSolSet = compSolSet + icls([c])
    
    return compSolSet

    # Uncomment the following line while commenting the rest to have a
    # minimally executable code skeleton.
    # return print("collabArc() returned.\n")

def collaborateComplement(pop_A, arc_A, pop_B, numTest, icls, ficls):
    """
    Create collaborations between the members of (pop_A - arc_A) and 
    pop_B. It returns a complete solutions set `compSolSet` which has
    collaborations between the members of pop_B and (pop_A - arc_A) or
    `pAComplement`.

    :param pop_A: population A which is a list of individuals.
    :param arc_A: an archive (set) of individuals, also a subset of
                  pop_A.
    :param pop_B: the set of individuals that should collaborate
                       with the memebers of the pop_A - arc_A.
    :param numTest: the number of collaborations that each individual
                    in pop_B should participate in. Note that they have
                    already participated `len(arc_A)` times.
    :param icls: the name of the class into which a complete solution
                 or `c` would be typecasted into.
    :param ficls: the name of the first individual's class to be included 
                  in the complete solution. Defines the format of a complete
                  solution including 2 individuals of different types.
    """
    if numTest <= len(arc_A):
        return []

    pA = deepcopy(pop_A)
    pB = deepcopy(pop_B)
    aA = deepcopy(arc_A)

    # Ensure that arc_A is a subset of pop_A
    assert all(x in pA for x in aA), "arc_A is not a subset of pop_A"

    compSolSet = []
    
    # Find {pA - aA}
    pAComplement = [ele for ele in pA]
    for _ in aA:
        if _ in pAComplement:
            pAComplement.remove(_)

    # Create complete solution between all members of pB and 
    # (numTest - len(aA)) members of pAComplement
    while numTest - len(aA) > 0:
        random_individual = pAComplement[random.randint(0, len(pAComplement)-1)]
        for i in pB:
            # This line makes the function casestudy-dependant.
            if type(i) == ficls:
                c = [i, random_individual]
            else:
                c = [random_individual, i]
            compSolSet = compSolSet + icls([c])
        numTest = numTest - 1

    return compSolSet

    # Uncomment the following line while commenting the rest to have a
    # minimally executable code skeleton.
    # return print("collbaComp() returned.\n")

def collaborate(arc1, pop1, arc2, pop2, cscls, fcls, k):
    """
    Creates a complete solution from two sets of individuals. It takes 
    two sets (arc and pop) and the type of individuals as input. It
    returns a set of unique complete solutions (collaborations).

    :param arc1: an archive (list) of individuals. It is a subset of pop1.
    :param pop1: a list of individuals that have to collaborate with members
                 of arc2 and possibly some members of `(pop2 - arc2)`.
    :param arc2: an archive (list) of individuals. It is a subset of pop2.
    :param pop2: a list of individuals that have to collaborate with members
                 of arc1 and possibly some members of `(pop1 - arc1)`.
    :param cscls: the name of the class into which a complete solution
                  would be typecasted into.
    :param k: the number of collaborations that each individual
                    in a population should participate in. Note that they have
                    already participated with every member of the archive of 
                    the other population (`len(arc)` times).
    :param fcls: the name of the first individual's class to be included 
                  in the complete solution. Defines the format of a complete
                  solution including 2 individuals of different types.
    """
    # Exeption handling needs to be implemented.
    assert arc1 or arc2 or pop1 or pop2 is not None, \
        "Populations or archives cannot be None."
    assert arc1 or arc2 or pop1 or pop2 != [], \
        "Populations or archives cannot be empty."
    
    # Ensure that arc_A is a subset of pop_A
    assert all(x in pop1 for x in arc1), "arc1 is not a subset of pop1"
    assert all(x in pop2 for x in arc2), "arc1 is not a subset of pop1"

    # Deepcopy archives and populations.
    a1 = deepcopy(arc1)
    p1 = deepcopy(pop1)
    a2 = deepcopy(arc2)
    p2 = deepcopy(pop2)

    # REMOVE REDUNDANT COMPLETE SOLUTIONS FROM THE COMPLETE SOLUTIONS SET.
    # set --> list

    # Create complete solutions from collaborations with the archives.
    complete_solutions_set = collaborateArchive(a1, p2, cscls, fcls) \
        + collaborateArchive(a2, p1, cscls, fcls) \
            + collaborateComplement(p1, a1, p2, k, cscls, fcls) \
                + collaborateComplement(p2, a2, p1, k, cscls, fcls)
    
    # Remove repetitive complete solutions.
    complete_solutions_set_unique = []
    for x in complete_solutions_set:
        if x not in complete_solutions_set_unique:
            complete_solutions_set_unique.append(x)

    return complete_solutions_set_unique

    # Uncomment the following line while commenting the rest to have a
    # minimally executable code skeleton.
    # collaborateArchive(arc1, pop1, cscls)
    # collaborateComplement(pop1, arc2, pop2, k, cscls)
    # return print("collaborate() returned.\n")

def evaluate(popScen, arcScen, popMLCO, arcMLCO, cscls, k):
    """
    Forms complete solutions, evaluates their joint fitness and evaluates the
    individual fitness values.
    """
    # # # Check for null inputs.
    # # if popScen or popMLCO is None:
    # #     raise TypeError
    # # if arcScen or arcMLCO is None:
    # #     raise TypeError
    
    # # # Check whether the type of the individuals in the sets is the same.
    # # if type(popScen[0]) != type(popMLCO[0]):
    # #     raise TypeError

    # # Deep copy the inputs
    # pScen = deepcopy(popScen)
    # pMLCO = deepcopy(popMLCO)
    # aScen = deepcopy(arcScen)
    # aMLCO = deepcopy(arcMLCO)

    # completeSol = collaborate(aScen, pScen, aMLCO, pMLCO, cscls, k)

    # for c in completeSol:
    #     c.fitness.values = evaluateJFit(c)
    
    # return completeSol

    collaborate(cscls, popScen, arcScen, popMLCO, arcMLCO, k)
    return print("evaluate() returned.\n")

# Evaluate the joint fitness of a complete solution.
def evaluateJFit(c):
    """
    Evaluates the joint fitness of a complete solution. It takes the complete
    solution as input and returns its joint fitness as output.
    """
    # return c
    return print("evaluateJFit() returned.\n")

# Evaluate scenario individuals.
def evaluateScen(compSolSet, scenario):
    """
    Aggreagates the joint fitness values that a scenario has been invovled in.
    It takes a list of all the complete solutions that include scenario x with
    their jFit values and returns the fitness value of x.
    """
    # csSet = copy.deepcopy(compSolSet)
    # scen = copy.deepcopy(scenario)

    # jFitInvolved = []

    # for i in csSet:
    #     if i[1] == scen:
    #         jFitInvolved.append(i.Fitness.values)
    
    # scenarioFitValue = max(jFitInvolved)

    # return scenarioFitValue
    return print("evaluateScen() returned.\n")

# Evaluate MLC output individuals.
def evaluateMLCO(csSet, outputMLC):
    """
    Aggreagates the joint fitness values that an MLC output sequence has been 
    invovled in. It takes a list of all the complete solutions that include 
    MLC output b with their jFit values and returns the fitness value of x.
    """
    # return outputMLC.fitness.values
    return print("evaluateMLCO() returned.\n")

# Breed scenarios.
def breedScen(popScen, enumLimits):
    
    """
    Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the Scenarios population. It takes an old 
    generation of scenarios as input and returns an evolved generation.
    """
    # pScen = toolbox.clone(popScen)
    # pScen=sorted(pScen, key=(_.fitness for _ in pScen), reverse=True)
    
    # return pScen
    mutateScenario(popScen, enumLimits)
    return print("breedScen() returned.\n")

# Breed MLC outputs.
def breedMLCO(outputMLC):
    """
    Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the MLC output population. It takes an old
    generation of MLC ouptputs as input and return an evovled generation.
    """
    # return outputMLC
    return print("breedMLCO() returned.\n")

#%%
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
    toolbox.register("mutateScenFlt", tools.mutGaussian, mu=mutgmu, sigma=mutgsig, indpb=mutgpb)
    
    # Limitation: assumes a special format for intLimits.

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
#%%
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

def violateSafetyReq(completeSolution):
    """
    Checks whether a completeSolution violate a safety requirement.
    """
    # if completeSolution.fitness.values > 0:
    #     return True
    # else:
    #     return False

    return True

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
toolbox.register("scenario", initializeScenario, creator.Individual, SCEN_IND_SIZE)
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
    popScen = breedScen(popScen, enumLimits)
    popMLCO = breedMLCO(popMLCO)


if __name__ == "__main__":
    main()