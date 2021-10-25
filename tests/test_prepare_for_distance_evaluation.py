from src.main.ICCEA import prepare_for_distance_evaluation
import unittest

class TestPrepareForDistanceEvaluation(unittest.TestCase):
    def test_prepare_for_distance_evaluation_testInput1(self):
        test_list = [
            [1, False, 4.5], [4, True, 8.5],
            1, [2, 3], [3, True, 2.2]
        ]
        
        expected_flat_list = [
            1, False, 4.5, 4, True, 8.5,
            1, 2, 3, 3, True, 2.2
        ]

        expected_nominal_values_indices = [
            0, 1, 3, 4, 6, 7, 8, 9, 10
        ]

        self.assertEqual(
            prepare_for_distance_evaluation(test_list),
            (expected_flat_list, expected_nominal_values_indices)
        )