from deap import creator
from CCEA import flatten
import unittest

class Test_TestFlatten(unittest.TestCase):
    def test_flatten_testInput1(self):
        test_list = [[1, False, 4.5], [4, True, 8.5],1, [2, 3], [3, True, 2.2]]
        expected_flat_list = [1, False, 4.5, 4, True, 8.5, 1, 2, 3, 3, True, 2.2]

        self.assertEqual(flatten(test_list), expected_flat_list)