from CCEA import gather_values_in_list
import unittest

class Test_TestGatherValuesInList(unittest.TestCase):
    def test_gather_values_in_list_testInput1(self):
        test_data = [[1, 0, 5.0, 1, 1, 21, 4], [4, 1, -7.8, 8, 1, 2, 2],
                        [2, -1, -9.8, 10, 3, 2, 91]]
        index = 3

        expected_list = [1, 8, 10]

        self.assertEqual(gather_values_in_list(test_data, index), \
            expected_list)
