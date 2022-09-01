"""
Set of variables that define the `problem` structure, especially `creator`
and `toolbox`.

This is the only file that a user has to modify for a given problem to
solve.
"""
# import benchmark.mtq.search_config as cfg
from deap import base, creator, tools
from src.utils.utility import (initialize_hetero_vector,
                               mutate_flat_hetero_individual)

def compute_safety_req_value(simulator, x, y):
    """This is the problem-specific joint fitness evaluation.
    """
    # The MTQ problem.

    cf = 1  # Correction factor that controls the granularity of x and y.

    h_1 = 150
    x_1 = 0.75
    y_1 = 0.75
    s_1 = 1.6
    f_1 = h_1 * \
        (1 - ((16.0/s_1) * pow((x[0]/cf - x_1), 2)) -
         ((16.0/s_1) * pow((y[0]/cf - y_1), 2)))

    h_2 = 50
    x_2 = 0.25
    y_2 = 0.25
    s_2 = 1.0/32.0
    f_2 = h_2 * \
        (1 - ((16.0/s_2) * pow((x[0]/cf - x_2), 2)) -
         ((16.0/s_2) * pow((y[0]/cf - y_2), 2)))

    # result = max(f_1, f_2) - 40.0
    # print(result)

    return max(f_1, f_2) - 100.0
    # return max(f_1, f_2)

def setup_problem(hyperparameters):
    scen_pop_size = hyperparameters['scenario_population_size']  # Size of the scenario population
    mlco_pop_size = hyperparameters['mlco_population_size']  # Size of the MLC output population

    # The list of lower and upper limits.
    enumLimits = hyperparameters['enumLimits']
    # Create fitness and individual datatypes.
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))   # Maximization
    creator.create("Individual", list, fitness=creator.FitnessMax,
                safety_req_value=float)  # Maximization
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

    toolbox.register("compute_safety_req_value", compute_safety_req_value)

    toolbox.register(
        "select", tools.selTournament,
        tournsize=hyperparameters['tournament_selection'], fit_attr='fitness'
    )

    toolbox.register("crossover", tools.cxUniform)

    toolbox.register("mutate_mlco", mutate_flat_hetero_individual)
    toolbox.register("mutate_scenario", mutate_flat_hetero_individual)

    return creator, toolbox
