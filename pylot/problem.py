"""
Set of variables that define the `problem` structure, especially `creator`
and `toolbox`.

This is the only file that a user has to modify for a given problem to
solve.
"""
from deap import base, creator, tools

import pylot.search_config as cfg
from pylot.problem_utils import (initialize_mlco, mutate_mlco, mutate_scenario,
                                 compute_safety_req_value)
from simulation_manager_cluster import prepare_for_computation, start_computation
# FIXME: This should be imported from problem_utils.py
from src.utils.utility import initialize_hetero_vector

scen_pop_size = cfg.scenario_population_size  # Size of the scenario population
mlco_pop_size = cfg.mlco_population_size  # Size of the MLC output population
min_distance = cfg.min_distance  # Minimum distance between members of an archive

# The list of lower and upper limits for enumerationed types in sceanrio.
scen_enumLimits = cfg.scenario_enumLimits

# Create fitness and individual datatypes.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin,
               safety_req_value=float)
creator.create("Scenario", creator.Individual)
creator.create("OutputMLC", creator.Individual)

toolbox = base.Toolbox()

# Define functions and register them in toolbox.
toolbox.register(
    "scenario", initialize_hetero_vector,
    class_=creator.Scenario, limits=scen_enumLimits
)

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


def compute_safety_value_cs_list(simulator, cs_list, sim_index):
    updated_sim_index = prepare_for_computation(
        cs_list, simulator, sim_index)
    results = start_computation(simulator)
    return updated_sim_index, results


toolbox.register("compute_safety_req_value", compute_safety_req_value)
toolbox.register("compute_safety_cs_list", compute_safety_value_cs_list)

toolbox.register(
    "select", tools.selTournament,
    tournsize=cfg.tournament_selection, fit_attr='fitness'
)

toolbox.register("crossover", tools.cxUniform)

toolbox.register("mutate_mlco", mutate_mlco)
toolbox.register("mutate_scenario", mutate_scenario)
