from CCEA import individual_in_list
from deap import creator
import unittest

class TestIndividualInList(unittest.TestCase):
    def test_individual_in_list_testInput1(self):
        c1 = creator.Individual([[1, False, 5.0], [[8, 'a'], [2, 'b']]])
        c2 = creator.Individual([[2, True, 0.24], [[-2, 'e'], [10, 'f']]])
        c3 = creator.Individual([[2, True, 0.24], [[1, 'a'], [21, 'd']]])
        cset = [c1, c2]

        self.assertTrue(individual_in_list(c1, cset))
        self.assertFalse(individual_in_list(c3, cset))