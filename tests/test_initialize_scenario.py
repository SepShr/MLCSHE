from deap import creator
from CCEA import initialize_scenario
import unittest

class TestInitializeScenario(unittest.TestCase):
    def test_initialize_scenario_testInput1(self):
        lmt = ['bool', [1, 5], 'bool', [1.35, 276.87]]
        c = creator.Scenario
        scenario = initialize_scenario(c, lmt)

        self.assertEqual(len(scenario), len(lmt))
        
        self.assertEqual(type(scenario), c)