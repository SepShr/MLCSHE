import random

from deap import creator
from deap import base
from deap import tools

    
# Sample initalization function for Scenarios
def initScen(scls, size):
    '''
    Initializes an individual of the Scenario population.
    '''
    return x

# Sample initialization function for MLC Outputs
def initMLCO(bcls, size):
    '''
    Initializes an individual of the MLCO population.
    '''
    return b

# Evaluate the joint fitness of a complete solution.
def evaluateJFit(c):
    '''
    Evaluates the joint fitness of a complete solution. It takes the complete
    solution as input and returns its joint fitness as output.
    '''
    return c
        
def evaluateScen(c, scenario):
    '''
    Aggreagates the joint fitness values that a scenario has been invovled in.
    It takes a list of all the complete solutions that include scenario x with
    their jFit values and returns the fitness value of x.
    '''
    return scenario.Fitness.values
    
def evaluateMLCO(c, outputMLC):
    '''
    Aggreagates the joint fitness values that an MLC output sequence has been 
    invovled in. It takes a list of all the complete solutions that include 
    MLC output b with their jFit values and returns the fitness value of x.
    '''
    return outputMLC.Fitness.values


def breedScen(scenario):
    '''
    Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the Scenarios population. It takes an old 
    generation of scenarios as input and returns an evolved generation.
    '''
    scenario=sorted(scenario, reverse=True)
    
    return scenario

def breedoutputMLC(outputMLC):
    '''
    Breeds, i.e., performs selection, crossover (exploitation) and mutation
    (exploration) on individuals of the MLC output population. It takes an old
    generation of MLC ouptputs as input and return an evovled generation.
    '''
    return outputMLC

# Create datatypes.
creator.create("FitnessMax", base.fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

SCEN_IND_SIZE = 10  # Size of a scenario individual
MLCO_IND_SIZE = 30  # Size of an MLC output individual
SPECIES_SIZE = 10  # Size of the species (population)

toolbox = base.Toolbox()

# Define functions and register them in toolbox.
toolbox.register("scenario", tools.initScen, creator.Individual, SCEN_IND_SIZE)
toolbox.register("outputMLC", tools.initMLCO, creator.Individual, MLCO_IND_SIZE)