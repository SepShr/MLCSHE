from deap import creator
from CCEA import mutateScenario
import unittest

class Test_TestMutateScenario(unittest.TestCase):
    def test_mutateScenario_testInput1(self):
        lmt = [[1, 5], None, None]
        a = creator.Scenario([1, 2.5, True])
        mut_bit_pb = 1
        mut_guass_mu = 0
        mut_guass_sig = 1
        mut_guass_pb = 1
        mut_int_pb = 1
        mutateScenario_output = mutateScenario(a, lmt, mut_bit_pb, mut_guass_mu,\
             mut_guass_sig, mut_guass_pb, mut_int_pb)
        self.assertNotEqual(mutateScenario_output, a)
        self.assertEqual(type(mutateScenario_output), type(a))