import numpy as np
from CCEA import gather_values_in_np_array
import unittest

class Test_TestGatherValuesInNpArray(unittest.TestCase):
    def test_gather_values_in_np_array_testInput1(self):
        test_data = [
            [1, 0, 5.0, 1, 1, 21, 4], 
            [4, 1, -7.8, 8, 1, 2, 2],
            [2, -1, -9.8, 10, 3, 2, 91]
        ]
        index = 3
        output_array = gather_values_in_np_array(test_data, index)
        
        expected_array = np.zeros(3)
        expected_array[0] = 1
        expected_array[1] = 8
        expected_array[2] = 10
        
        

        self.assertIsNone(
            np.testing.assert_array_equal(output_array, expected_array))