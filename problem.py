"""
Set of variables that define the `problem` structure, especially `creator`
and `toolbox`.

This is the only file that a user has to modify for a given problem to
solve.
"""
from deap import base, creator, tools

import search_config as cfg

from problem_utils import initialize_mlco, problem_joint_fitness

# FIXME: This should be imported from problem_utils.py
from src.utils.utility import initialize_hetero_vector

scen_pop_size = cfg.scenario_population_size  # Size of the scenario population
mlco_pop_size = cfg.mlco_population_size  # Size of the MLC output population
min_distance = cfg.min_distance  # Minimum distance between members of an archive

# The list of lower and upper limits for enumerationed types in sceanrio.
scen_enumLimits = cfg.scenario_enumLimits

TOTAL_MLCO_MESSAGES = 10
TOTAL_OBSTACLES_PER_MESSAGE = 3


# Create fitness and individual datatypes.
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))   # Original formulation of the problem.
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("Scenario", creator.Individual)
creator.create("OutputMLC", creator.Individual)

toolbox = base.Toolbox()

# Define functions and register them in toolbox.
toolbox.register(
    "scenario", initialize_hetero_vector,
    class_=creator.Scenario, limits=scen_enumLimits
)

# FIXME: Add variable to initialize_mlco function such as BBsize.
toolbox.register(
    "mlco", initialize_mlco,
    creator.OutputMLC
)

toolbox.register(
    "popScen", tools.initRepeat, list,
    toolbox.scenario, n=scen_pop_size
)
toolbox.register(
    "popMLCO", tools.initRepeat, list,
    toolbox.mlco, n=mlco_pop_size
)

toolbox.register("problem_jfit", problem_joint_fitness)
