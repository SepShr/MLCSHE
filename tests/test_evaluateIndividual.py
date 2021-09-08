from deap import creator
from CCEA import evaluateIndividual
import unittest

class Test_TestEvaluateIndividual(unittest.TestCase):
    def test_evaluateIndividual_testInput1(self):
        ind1 = creator.Scenario([1, False, 5.0])
        cs1 = creator.Individual([ind1, 8])
        cs1.fitness.values = (2.5,)
        ind2 = creator.Scenario([4, True, -7.8])
        cs2 = creator.Individual([ind2, 2])
        cs2.fitness.values = (1,)
        cs3 = creator.Individual([ind1, -1])
        cs3.fitness.values = (-7.5,)
        css = [cs1, cs2, cs3]
        index = 0
        
        expected_value = (2.5,)
        self.assertEqual(evaluateIndividual(ind1, css, index), expected_value)
