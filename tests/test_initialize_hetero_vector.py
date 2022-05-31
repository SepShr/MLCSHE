from deap import creator
from src.utils.utility import initialize_hetero_vector
import unittest
import problem
import search_config as cfg


class TestInitializeHeteroVector(unittest.TestCase):
    def test_initialize_hetero_vector_testInput1(self):
        lmt = cfg.scenario_enumLimits
        c = creator.Scenario
        scenario = initialize_hetero_vector(limits=lmt, class_=c)

        self.assertEqual(len(scenario), len(lmt))

        self.assertEqual(type(scenario), c)
