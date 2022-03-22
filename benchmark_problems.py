# enumLimits = ['bool', [1, 5], 'bool', [1.35, 276.87]]
# [np.nan, np.nan, (1, 6)]
# enumLimits = [[0.0, 1.0]]


# Define the problem's joint fitness function.

from math import cos, sqrt


def problem_joint_fitness(x, y):
    """This is the problem-specific joint fitness evaluation.
    """
    # The MTQ problem.

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


def problem_joint_fitness(x, y):
    """This is the problem-specific joint fitness evaluation.
    """
    # The Griewangk domain.

    cf = 5.0  # Correction factor that controls the granularity of x and y.

    x_bar = 10.24 * (x / cf) - 5.12
    y_bar = 10.24 * (y / cf) - 5.12

    f = -1.0 - (pow(x_bar, 2) / 4000) - (pow(y_bar, 2) / 4000) + \
        cos(x_bar) * cos(y_bar/(sqrt(2)))

    return f


def problem_joint_fitness(x, y):
    """This is the problem-specific joint fitness evaluation.
    """
    # The OneRidge domain.

    cf = 1.0  # Correction factor that controls the granularity of x and y.

    # if x > 1:
    #     x = 1
    # if y > 1:
    #     y = 1

    f = 1 + 2 * min(x, y) - max(x, y)

    return f


def problem_joint_fitness(x, y):
    """This is the problem-specific joint fitness evaluation.
    """
    # The Booth domain.

    cf = 1.0  # Correction factor that controls the granularity of x and y.

    x_bar = 10.24 * x - 5.12
    y_bar = 10.24 * y - 5.12

    f = -1.0 * pow((x_bar + 2.0 * y_bar - y), 2) - \
        pow((2 * x_bar + y_bar - 5.0), 2)

    return f
