from src.utils.utility import measure_heom_distance
import unittest
import problem


class TestMeasureHeomDistance(unittest.TestCase):
    def test_measure_heom_distance_testInput1(self):
        test_data = [
            [1, 0, 5.0, 1, 1, 21, 4],
            [4, 1, -7.8, 8, 1, 2, 2],
            [2, -1, -9.8, 10, 3, 2, 91]
        ]
        category_indices = [0, 3, 6]
        output_value = measure_heom_distance(test_data, category_indices)

        # expected_value = [0.0, 2.235618758750633, 2.5]
        expected_value = [0.0, 0.8449844660007257, 0.944911182523068]

        self.assertEqual(output_value, expected_value)
        self.assertIsInstance(output_value, list)
