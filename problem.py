"""
Set of variables that define the `problem` structure, especially `creator`
and `toolbox`.

This is the only file that a user has to modify for a given problem to
solve.
"""

from deap import creator, base, tools
from src.utils.utility import initialize_hetero_vector

SCEN_POP_SIZE = 5  # Size of the scenario population
MLCO_POP_SIZE = 5  # Size of the MLC output population
MIN_DISTANCE = 0.5  # Minimum distance between members of an archive

# The list of lower and upper limits for enumerationed types in sceanrio.
# enumLimits = ['bool', [1, 5], 'bool', [1.35, 276.87]]
# [np.nan, np.nan, (1, 6)]
enumLimits = [[0.0, 1.0]]

# Define the problem's joint fitness function.


def problem_joint_fitness(x, y):
    """This is the problem-specific joint fitness evaluation.
    """
    # MTQ problem.
    cf = 10  # Correction factor that controls the granularity of x and y.

    h_1 = 50
    x_1 = 0.75
    y_1 = 0.75
    s_1 = 1.6
    f_1 = h_1 * \
        (1 - ((16.0/s_1) * pow((x/cf - x_1), 2)) -
         ((16.0/s_1) * pow((y/cf - y_1), 2)))
    h_2 = 150
    x_2 = 0.25
    y_2 = 0.25
    s_2 = 1.0/32.0
    f_2 = h_2 * \
        (1 - ((16.0/s_2) * pow((x/cf - x_2), 2)) -
         ((16.0/s_2) * pow((y/cf - y_2), 2)))

    return max(f_1, f_2)


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

toolbox.register("problem_jfit", problem_joint_fitness)
