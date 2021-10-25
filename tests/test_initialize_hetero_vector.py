from deap import creator
from src.utils.utility import initialize_hetero_vector
import unittest


class TestInitializeHeteroVector(unittest.TestCase):
    def test_initialize_hetero_vector_testInput1(self):
        lmt = ['bool', [1, 5], 'bool', [1.35, 276.87]]
        c = creator.Scenario
        scenario = initialize_hetero_vector(c, lmt)

        self.assertEqual(len(scenario), len(lmt))

        self.assertEqual(type(scenario), c)
