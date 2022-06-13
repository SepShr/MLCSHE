from deap import base, creator
from fitness_function import estimate_safe_cs_probability
import unittest


class TestEstimateSafeCSProbability(unittest.TestCase):
    def test_nominal_1(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMin, safety_req_value=float)
        ind_1 = creator.Individual([1, 0, 5.0, 1, 1, 21, 4])
        ind_1.safety_req_value = 5.01

        ind_2 = creator.Individual([4, 1, -7.8, 8, 1, 2, 2])
        ind_2.safety_req_value = -3.52

        ind_3 = creator.Individual([2, -1, -9.8, 10, 3, 2, 91])
        ind_3.safety_req_value = -0.8

        cs_region = [ind_1, ind_2, ind_3]

        computed_probability = estimate_safe_cs_probability(cs_region)

        expected_probability = 1/3

        self.assertEqual(computed_probability, expected_probability)
