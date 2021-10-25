"""
Set of variables that define the `problem` structure, especially `creator`
and `toolbox`.

This is the only file that a user has to modify for a given problem to
solve.
"""

from deap import creator, base, tools
from src.utils.utility import initialize_hetero_vector

SCEN_IND_SIZE = 1  # Size of a scenario individual
MLCO_IND_SIZE = 2  # Size of an MLC output individual
SCEN_POP_SIZE = 2  # Size of the scenario population
MLCO_POP_SIZE = 2  # Size of the MLC output population
MIN_DISTANCE = 1  # Minimum distance between members of an archive

# The list of lower and upper limits for enumerationed types in sceanrio.
enumLimits = ['bool', [1, 5], 'bool', [1.35, 276.87]]
# [np.nan, np.nan, (1, 6)]

# Create fitness and individual datatypes.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("Scenario", creator.Individual)
creator.create("OutputMLC", creator.Individual)

toolbox = base.Toolbox()

# Define functions and register them in toolbox.
toolbox.register(
    "scenario", initialize_hetero_vector,
    creator.Scenario, enumLimits
)

# enum_limits is different for the two types of individuals
toolbox.register(
    "mlco", initialize_hetero_vector,
    creator.OutputMLC, enumLimits
)
toolbox.register(
    "popScen", tools.initRepeat, list,
    toolbox.scenario, n=SCEN_POP_SIZE
)
toolbox.register(
    "popMLCO", tools.initRepeat, list,
    toolbox.mlco, n=MLCO_POP_SIZE
)
