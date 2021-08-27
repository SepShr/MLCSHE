import copy
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

def collabArc(archive, population, icls):
    """
    Create collaborations between individuals of archive and population.
    The complete solutions (collaborations) are of class icls.
    """
    # arc = toolbox.clone(archive)
    # pop = toolbox.clone(population)

    # c = icls()
    # compSolSet = []

    # for i in arc:
    #     for j in pop:
    #         if icls == creator.Scenario:
    #             c.setValues([(i.getValues())+(j.getValues())])
    #         else:
    #             c.setValues([(j.getValues())+(i.getValues())])
    #         compSolSet = compSolSet.append(c)
    
    # return compSolSet
    return print("collabArc() returned.\n")

def collabComp(pop_A, arc_A, pop_B, numTest, icls):
    """
    Create collaborations between the members of {pop_A - arc_A} and pop_B.
    """
    # if numTest <= len(arc_A):
    #     return []

    # pA = toolbox.clone(pop_A)
    # pB = toolbox.clone(pop_B)
    # aA = toolbox.clone(arc_A)

    # # icls = super(type(pA[0]), pA[0])
    # c = icls()
    # compSolSet = []
    
    # # Find {pA - aA}
    # pAComplement = [ele for ele in pA]
    # for _ in aA:
    #     if _ in pAComplement:
    #         pAComplement.remove(_)

    # # Create complete solution between all members of pB and 
    # # (numTest - len(aA)) members of pAComplement
    # while numTest - len(aA) > 0:
    #     randInd = pAComplement[random.randint(0, len(pAComplement))]
    #     for i in pB:
    #         if icls == creator.Scenario:
    #             # c.setValues([(randInd.getValues())+(i.getValues())])
    #             c.setValues([randInd,i])
    #         else:
    #             # c.setValues([(i.getValues())+(randInd.getValues())])
    #             c.setValues([i, randInd])
    #         compSolSet = compSolSet.append(c)

    # return compSolSet
    return print("collbaComp() returned.\n")

def collaborate(cscls, arc1, pop1, arc2, pop2, k):
    """
    Creates a complete solution from two sets of individuals. It takes 
    two sets (arc and pop) and the type of individuals as input. It
    returns a set of complete solutions (collaborations).
    """
    # # Exeption handling needs to be implemented.
    # assert arc1 or arc2 or pop1 or pop2 is not None, \
    #     "Populations or archives cannot be None."
    # assert arc1 or arc2 or pop1 or pop2 != [], \
    #     "Populations or archives cannot be empty."

    # # Deepcopy archives and populations.
    # a1 = toolbox.clone(arc1)
    # p1 = toolbox.clone(pop1)
    # a2 = toolbox.clone(arc2)
    # p2 = toolbox.clone(pop2)

    # # Create complete solutions from collaborations with the archives.
    # completeSolutionsSet = collabArc(a1, p2) + collabArc(a2, p1) \
    #     + collabComp(p1, a1, p2, k, cscls) + collabComp(p2, a2, p1, k, cscls)
        
    # return completeSolutionsSet

    collabArc(arc1, pop1, cscls)
    collabComp(pop1, arc2, pop2, k, cscls)
    return print("collaborate() returned.\n")

def evaluate(cscls, popScen, arcScen, popMLCO, arcMLCO, k):
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

    # completeSol = collaborate(cscls, aScen, pScen, aMLCO, pMLCO, k)

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
    mutateScen(popScen, enumLimits)
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

# Mutate scenarios
def mutateScen(scenario, intLimits):
    """
    Mutates a scenario individual. Input is an unmutated scenario, while the
    output is a mutated scenario. The function applies one of the 3 mutators
    to the elements depending on their type, i.e., Guassian to Floats, bitflip
    to Booleans and integer-randomization to Integers.
    """
    # toolbox.register("mutateScenBool", tools.mutFlipBit, indpb=0.05)
    # toolbox.register("mutateScenFlt", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
    # # toolbox.register("mutateScenInt", tools.mutUniformInt, low=lowerLim, up=UpperLim, indpb=0.05)
    
    # mutatedScen = []
    # limits = deepcopy(intLimits)

    # # Check every element and apply the appropriate mutator to it. 
    # for i in range(len(scenario)):
    #     buffer = scenario.getValues()[i]
        

    #     if type(buffer) is int:
    #         buffer = tools.mutUniformInt(list(buffer), low= limits[i][0], up=limits[i][1], indpb=0.05)
        
    #     if type(buffer) is bool:
    #         buffer = toolbox.mutateScenBool(list(buffer))
        
    #     if type(buffer) is float:
    #         buffer = toolbox.mutateScenFlt(list(buffer))
        
    #     mutatedScen.append(buffer[0])

    # return mutatedScen
    return print("mutate_MLCO() returned.\n")

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
    completeSolSet = evaluate(creator.Individual, popScen, arcScen, popMLCO, arcMLCO, 1)

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