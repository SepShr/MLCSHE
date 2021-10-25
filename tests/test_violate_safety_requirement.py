from deap import creator
from src.utils.utility import violate_safety_requirement
import unittest

class TestViolateSafetyRequirement(unittest.TestCase):
    def test_violate_safety_requirement_testInput1(self):
        c1 = creator.Individual([1, 'a', 2.7])
        c1.fitness.values = (2,)

        c2 = creator.Individual([1.3])
        c2.fitness.values = (-2,)

        self.assertTrue(violate_safety_requirement(c1))
        self.assertFalse(violate_safety_requirement(c2))