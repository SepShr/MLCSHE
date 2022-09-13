"""
Set of variables that define the `problem` structure, especially `creator`
and `toolbox`.

This is the only file that a user has to modify for a given problem to
solve.
"""
from math import floor
import random
import benchmark.onemax.search_config as cfg
from deap import base, creator, tools
from src.utils.utility import (initialize_hetero_vector,
                               mutate_flat_hetero_individual)

scen_pop_size = cfg.scenario_population_size  # Size of the scenario population
mlco_pop_size = cfg.mlco_population_size  # Size of the MLC output population
min_distance = cfg.min_distance  # Minimum distance between members of an archive

# The list of lower and upper limits.
enumLimits = cfg.enumLimits


def compute_safety_req_value(simulator, x, y):
    """This is the problem-specific joint fitness evaluation.
    """
    # The OneMax problem with a boundary constraint.

    return sum(x) + sum(y) - len(x)


# Create fitness and individual datatypes.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Structure initializers
creator.create("Individual", list, fitness=creator.FitnessMin,
               safety_req_value=float)  # Minimization
creator.create("Scenario", creator.Individual)
creator.create("OutputMLC", creator.Individual)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
# Define functions and register them in toolbox.
toolbox.register(
    "scenario", tools.initRepeat,
    creator.Scenario, toolbox.attr_bool, 10
)

toolbox.register(
    "mlco", tools.initRepeat,
    creator.OutputMLC, toolbox.attr_bool, 10
)

toolbox.register(
    "popScen", tools.initRepeat, list,
    toolbox.scenario, n=scen_pop_size
)
toolbox.register(
    "popMLCO", tools.initRepeat, list,
    toolbox.mlco, n=mlco_pop_size
)

toolbox.register("compute_safety_req_value", compute_safety_req_value)

toolbox.register(
    "select", tools.selTournament,
    tournsize=cfg.tournament_selection, fit_attr='fitness'
)

toolbox.register("crossover", tools.cxUniform)

toolbox.register("mutate_mlco", mutate_flat_hetero_individual)
toolbox.register("mutate_scenario", mutate_flat_hetero_individual)

# %%
