from deap import base, creator
import unittest
from problem_utils import mutate_scenario


class TestMutateScenario(unittest.TestCase):
    def test_mutate_scenario_testInput1(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMin, safety_req_value=float)
        creator.create("Scenario", creator.Individual)
        lmt = [[1, 5], None, None]
        a = creator.Scenario([1, 2.5, True])
        mut_bit_pb = 1
        mut_guass_mu = 0
        mut_guass_sig = 1
        mut_guass_pb = 1
        mut_int_pb = 1
        mutate_scenario_output = mutate_scenario(
            a, lmt, mut_bit_pb, mut_guass_mu,
            mut_guass_sig, mut_guass_pb, mut_int_pb
        )

        self.assertNotEqual(mutate_scenario_output, a)
        self.assertEqual(type(mutate_scenario_output), type(a))
