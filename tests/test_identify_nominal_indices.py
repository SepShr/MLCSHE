from deap import creator
from CCEA import identify_nominal_indices
import unittest

class Test_TestIdentifyNominalIndices(unittest.TestCase):
    def test_identify_nominal_indices_testInput1(self):
        test_list = [1, False, 4.5, 4, True, 8.5, 1, 2, 3, 3, True, 2.2]
        expected_list = [0, 1, 3, 4, 6, 7, 8, 9, 10]

        self.assertEqual(identify_nominal_indices(test_list), expected_list)