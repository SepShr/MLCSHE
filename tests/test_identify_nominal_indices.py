from CCEA import identify_nominal_indices
import unittest

class TestIdentifyNominalIndices(unittest.TestCase):
    def test_identify_nominal_indices_testInput1(self):
        test_list = [
            1, False, 4.5, 4, True, 8.5,
            1, 2, 3, 3, True, 2.2
        ]
        output_list = identify_nominal_indices(test_list)
        expected_list = [0, 1, 3, 4, 6, 7, 8, 9, 10]

        self.assertEqual(output_list, expected_list)