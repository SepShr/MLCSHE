import random
from copy import deepcopy

import numpy as np

from deap import creator
from deap import base
from deap import tools

    
# Sample initalization function for Scenarios
def initScen(scls, size):
    """
    Initializes an individual of the Scenario population.
    """
    return x

# Sample initialization function for MLC Outputs
def initMLCO(bcls, size):
    """
    Initializes an individual of the MLCO population.
    """
    return b

def collabArc(archive, population, icls):
    """
    
    """
    arc = toolbox.clone(archive)
    pop = toolbox.clone(population)
    
    # icls = super(type(arc[0]), arc[0])
    c = icls()
    compSolSet = []


    for i in arc:
        for j in pop:
            if icls == creator.Scenario:
                c.setValues([(i.getValues())+(j.getValues())])
            else:
                c.setValues([(j.getValues())+(i.getValues())])
            compSolSet = compSolSet.append(c)
    
    return compSolSet

def collabComp(pop_A, arc_A, pop_B, numTest, icls):
    """
    Create collaborations between the members of {pop_A - arc_A} and pop_B.
    """
    if numTest <= len(arc_A):
        return []

    pA = toolbox.clone(pop_A)
    pB = toolbox.clone(pop_B)
    aA = toolbox.clone(arc_A)

    # icls = super(type(pA[0]), pA[0])
    c = icls()
    compSolSet = []
    
    # Find {pA - aA}
    pAComplement = [ele for ele in pA]
    for _ in aA:
        if _ in pAComplement:
            pAComplement.remove(_)

    # Create complete solution between all members of pB and 
    # (numTest - len(aA)) members of pAComplement
    while numTest - len(aA) > 0:
        randInd = pAComplement[random.randint(0, len(pAComplement))]
        for i in pB:
            if icls == creator.Scenario:
                c.setValues([(randInd.getValues())+(i.getValues())])
            else:
                c.setValues([(i.getValues())+(randInd.getValues())])
            compSolSet = compSolSet.append(c)

    return compSolSet

def collaborate(cscls, arc1, pop1, arc2, pop2, k):
    """
    Creates a complete solution from two sets of individuals. It takes 
    two sets (arc and pop) and the type of individuals as input. It
    returns a set of complete solutions (collaborations).
    """
    # if arc or pop is None:
    #     raise TypeError
    
    # if icls is None:
    #     raise TypeError
    
    # for ind in range(arc):
    #     if type(ind) is not icls:
    #         raise TypeError
    
    # for ind in range(pop):
    #     if type(ind) is not icls:
    #         raise TypeError

    a1 = toolbox.clone(arc1)
    p1 = toolbox.clone(pop1)
    a2 = toolbox.clone(arc2)
    p2 = toolbox.clone(pop2)

    # Create complete solutions from collaborations with the archives.
    completeSolutionsSet = collabArc(a1, p2) + collabArc(a2, p1) \
        + collabComp(p1, a1, p2, k, cscls) + collabComp(p2, a2, p1, k, cscls)
        
    return completeSolutionsSet

def evaluate(cscls, popScen, arcScen, popMLCO, arcMLCO, k):
    """
    Forms complete solutions, evaluates their joint fitness and evaluates the
    individual fitness values.
    """
    # # Check for null inputs.
    # if popScen or popMLCO is None:
    #     raise TypeError
    # if arcScen or arcMLCO is None:
    #     raise TypeError
    
    # # Check whether the type of the individuals in the sets is the same.
    # if type(popScen[0]) != type(popMLCO[0]):
    #     raise TypeError

    # Deep copy the inputs
    pScen = deepcopy(popScen)
    pMLCO = deepcopy(popMLCO)
    aScen = deepcopy(arcScen)
    aMLCO = deepcopy(arcMLCO)

    completeSol = collaborate(cscls, aScen, pScen, aMLCO, pMLCO, k)

    for c in completeSol:
        c.fitness.values = evaluateJFit(c)
    
    return completeSol

# Evaluate the joint fitness of a complete solution.
def evaluateJFit(c):
    """
    Evaluates the joint fitness of a complete solution. It takes the complete
    solution as input and returns its joint fitness as output.
    """
    return c

# Evaluate scenario individuals.
def evaluateScen(c, scenario):
    """
    Aggreagates the joint fitness values that a scenario has been invovled in.
    It takes a list of all the complete solutions that include scenario x with
    their jFit values and returns the fitness value of x.
    """
    return scenario.Fitness.values

# Evaluate MLC output individuals.
def evaluateMLCO(c, outputMLC):
    """
    Aggreagates the joint fitness values that an MLC output sequence has been 
    invovled in. It takes a list of all the complete solutions that include 
    MLC output b with their jFit values and returns the fitness value of x.
    """
    return outputMLC.Fitness.values

# Breed scenarios.
def breedScen(popScen):
    
    """
    Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the Scenarios population. It takes an old 
    generation of scenarios as input and returns an evolved generation.
    """
    pScen = toolbox.clone(popScen)
    pScen=sorted(pScen, key=(_.fitness for _ in pScen), reverse=True)
    
    return pScen

# Breed MLC outputs.
def breedMLCO(outputMLC):
    """
    Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the MLC output population. It takes an old
    generation of MLC ouptputs as input and return an evovled generation.
    """
    return outputMLC

# Mutate scenarios
def mutateScen(scenario, intLimits):
    """
    Mutates a scenario individual. Input is an unmutated scenario, while the
    output is a mutated scenario. The function applies one of the 3 mutators
    to the elements depending on their type, i.e., Guassian to Floats, bitflip
    to Booleans and integer-randomization to Integers.
    """
    toolbox.register("mutateScenBool", tools.mutFlipBit, indpb=0.05)
    toolbox.register("mutateScenFlt", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
    # toolbox.register("mutateScenInt", tools.mutUniformInt, low=lowerLim, up=UpperLim, indpb=0.05)
    
    
    mutatedScen = []
    limits = deepcopy(intLimits)

    # Check every element and apply the appropriate mutator to it. 
    for i in range(len(scenario)):
        buffer = scenario.getValues()[i]
        

        if type(buffer) is int:
            buffer = tools.mutUniformInt(list(buffer), low= limits[i][0], up=limits[i][1], indpb=0.05)
        
        if type(buffer) is bool:
            buffer = toolbox.mutateScenBool(list(buffer))
        
        if type(buffer) is float:
            buffer = toolbox.mutateScenFlt(list(buffer))
        
        mutatedScen.append(buffer[0])

    return mutatedScen

def updateArc_Scen(archive, pop):
    """
    Updates the archive of scenarios for the next generation.
    """
    return archive

def updateArc_MLCO(archive, pop):
    """
    Update the archive of MLC outputs for the next generation.
    """
    return archive

# Create fitness and individual datatypes.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("Scenario", creator.Individual)
creator.create("OutputMLC", creator.Individual)

SCEN_IND_SIZE = 10  # Size of a scenario individual
MLCO_IND_SIZE = 30  # Size of an MLC output individual
SCEN_POP_SIZE = 10  # Size of the scenario population
MLCO_POP_SIZE = 10  # Size of the MLC output population

enumLimits = [np.nan, np.nan, (1,6)]

toolbox = base.Toolbox()

# Define functions and register them in toolbox.
toolbox.register("scenario", tools.initScen, creator.Individual, n=SCEN_IND_SIZE)
toolbox.register("outputMLC", tools.initMLCO, creator.Individual, n=MLCO_IND_SIZE)
toolbox.register("popScen", tools.initRepeat, list, toolbox.scenario, n=SCEN_POP_SIZE)
toolbox.register("popMLCO", tools.initRepeat, list, toolbox.outputMLC, n=MLCO_POP_SIZE)

# ----------------------

def main():
    # Instantiate individuals and populations
    popScen = toolbox.popScen()
    popMLCO = toolbox.popMLCO()
    arcScen = toolbox.clone(popScen)
    arcMLCO = toolbox.clone(popMLCO)

    # Create complete solutions and evaluate individuals
    completeSol = evaluate(popScen, arcScen, popMLCO, arcMLCO)

    # Create complete solutions and evaluate individuals
    completeSol = evaluate(popScen, arcScen, popMLCO, arcMLCO)

    # Record the complete solutions that violate the requirement r


    # Evolve archives and populations for the next generation
    arcScen = updateArc_Scen(arcScen, popScen)
    arcMLCO = updateArc_MLCO(arcMLCO, popMLCO)

    # Select, mate (crossover) and mutate individuals that are not in archives.


if __name__ == "__main__":
    main()