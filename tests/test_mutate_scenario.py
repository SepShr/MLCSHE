from deap import creator
from src.main.ICCEA import ICCEA
import unittest
import problem
from problem_utils import mutate_scenario


class TestMutateScenario(unittest.TestCase):
    def test_mutate_scenario_testInput1(self):

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
