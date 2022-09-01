"""
This module contains the necessary functions to calculate the
boundary-seeking fitness function.
"""
from math import copysign, sqrt

import numpy as np


def wilson(p, n, z=1.96) -> tuple:
    """Calculates the Wilson confidence interval based on the sample
    probability `p` and sample size `n`.
    """
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)

    lower_bound = (centre_adjusted_probability - z *
                   adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z *
                   adjusted_standard_deviation) / denominator
    return (lower_bound, upper_bound)


def estimate_safe_cs_probability(cs_region) -> float:
    """Estimate the probability of finding a safe cs inside a 
    `cs_region`.
    """
    n_evaluated = len(cs_region)
    n_safe = len([cs for cs in cs_region if copysign(
        1.0, cs.safety_req_value) == 1.0])
    return n_safe / n_evaluated


def find_cs_region(center_cs, max_dist, cs_list, dist_matrix_sq: np.array) -> list:
    """Returns a list of evaluated complete solutions with a distance
    from cs less than or equal to max_dist.

    :param cs: the complete solution at the center of the cs_region.
    :param max_dist: the radius of the cs_region.
    :param cs_list: the list of evaluated complete solutions.
    :param dist_matrix_sq: the matrix of pairwise distance between
                            complete solutions.
    """
    # NOTE: We have assumed that the dist_matrix_sq is in squareform. If it is condensed, another formula should be used.
    center_cs_index = cs_list.index(center_cs)
    return [cs for cs in cs_list if dist_matrix_sq[center_cs_index, cs_list.index(cs)] <= max_dist]


def confidence_interval_dist(confidence_interval, target_probability=0.5) -> float:
    """Measures the distance between the edges of a confidence 
    interval and the 'target_probability', if the 'target_probability`
    is not included in the interval.
    """
    if confidence_interval[0] > target_probability:
        return confidence_interval[0] - target_probability
    elif confidence_interval[1] < target_probability:
        return target_probability - confidence_interval[1]
    else:
        return 0


def fitness_function(cs, cs_list: list, dist_matrix: np.array, max_dist: float, w_ci: float = 1.0, w_p: float = 1.0) -> float:
    """Returns a fitness values which measures the distance of
    `cs` from the boundary region. The fitness values also
    relies on the number of complete solutions in the neighbourhood
    of `cs`. The neighbourhood of `cs` is determined
    by `cs`, `pdist_matrix`, and `max_dist`.

    :param cs: the subject of the fitness evlauation.
    :param max_dist: the radius of the cs_region.
    :param cs_list: the list of evaluated complete solutions.
    :param dist_matrix: the matrix of pairwise distance between
                        complete solutions in square form.
    :param w_ci: the weight of the term that focuses on the length
                  of the confidence interval in the fitness function.
    :param w_p: the weight of the term that focuses on the value of
                  the probability in the fitness function.
    """
    # cs_region = find_cs_region(cs, max_dist, cs_list, dist_matrix)
    # p_safe = estimate_safe_cs_probability(cs_region)
    # assert 0 <= p_safe <= 1
    # confidence_interval = wilson(p_safe, len(cs_region))
    # conf_int_dist = confidence_interval_dist(confidence_interval)
    # assert 0 <= conf_int_dist <= 0.5
    # conf_int_len = confidence_interval[1] - confidence_interval[0]
    # assert 0 <= conf_int_len <= 1
    # fitness_value = (1 - 2 * conf_int_dist) * \
    #                 ((w_ci * (1 - conf_int_len)) +
    #                  (w_p * (1 - abs(p_safe - 0.5)))) / (w_ci + w_p)
    # assert 0 <= fitness_value <= 1
    # return fitness_value

    # Testing the updated fitness function.
    cs_region = find_cs_region(cs, max_dist, cs_list, dist_matrix)
    p_safe = estimate_safe_cs_probability(cs_region)
    assert 0 <= p_safe <= 1, 'p_safe can only be between 0 and 1'
    confidence_interval = wilson(p_safe, len(cs_region))
    conf_int_len = confidence_interval[1] - confidence_interval[0]
    assert 0 <= conf_int_len <= 1, 'conf_int_len can only be between 0 and 1'
    fitness_value = max(
        abs(confidence_interval[1] - 0.5), abs(confidence_interval[0] - 0.5))
    assert 0 <= fitness_value <= 0.51, f'fitness_value can only be between 0 and 0.5, current value={fitness_value}'
    return fitness_value
