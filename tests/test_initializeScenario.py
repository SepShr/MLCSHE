from deap import creator
from CCEA import initializeScenario
import unittest

class Test_TestInitializeScenario(unittest.TestCase):
    def test_initializeScenario_testInput1(self):
        lmt = ['bool', [1, 5], 'bool', [1.35, 276.87]]
        c = creator.Scenario
        scenario = initializeScenario(c, lmt)
        self.assertEqual(len(scenario), len(lmt))
        self.assertEqual(type(scenario), c)