"""
Set of variables that define the `problem` structure, especially `creator`
and `toolbox`.

This is the only file that a user has to modify for a given problem to
solve.
"""
from math import cos, sqrt

import benchmark.griegwangk.search_config as cfg
from deap import base, creator, tools
from src.utils.utility import (initialize_hetero_vector,
                               mutate_flat_hetero_individual)

scen_pop_size = cfg.scenario_population_size  # Size of the scenario population
mlco_pop_size = cfg.mlco_population_size  # Size of the MLC output population
min_distance = cfg.min_distance  # Minimum distance between members of an archive

# The list of lower and upper limits.
enumLimits = cfg.enumLimits


def joint_fitness_griegwangk(simulator, x, y):
    """This is the problem-specific joint fitness evaluation.
    """
    # The Griewangk domain.

    cf = 5.0  # Correction factor that controls the granularity of x and y.

    x_bar = 10.24 * (x[0] / cf) - 5.12
    y_bar = 10.24 * (y[0] / cf) - 5.12

    f = -1.0 - (pow(x_bar, 2) / 4000) - (pow(y_bar, 2) / 4000) + \
        cos(x_bar) * cos(y_bar/(sqrt(2)))

    return f


# Create fitness and individual datatypes.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))   # Maximization
creator.create("Individual", list, fitness=creator.FitnessMax)  # Maximization
creator.create("Scenario", creator.Individual)
creator.create("OutputMLC", creator.Individual)

toolbox = base.Toolbox()

# Define functions and register them in toolbox.
toolbox.register(
    "scenario", initialize_hetero_vector,
    class_=creator.Scenario, limits=enumLimits
)

toolbox.register(
    "mlco", initialize_hetero_vector,
    class_=creator.OutputMLC, limits=enumLimits
)

toolbox.register(
    "popScen", tools.initRepeat, list,
    toolbox.scenario, n=scen_pop_size
)
toolbox.register(
    "popMLCO", tools.initRepeat, list,
    toolbox.mlco, n=mlco_pop_size
)

toolbox.register("problem_jfit", joint_fitness_griegwangk)

toolbox.register(
    "select", tools.selTournament,
    tournsize=cfg.tournament_selection, fit_attr='fitness'
)

toolbox.register("crossover", tools.cxUniform)

toolbox.register("mutate_mlco", mutate_flat_hetero_individual)
toolbox.register("mutate_scenario", mutate_flat_hetero_individual)
