import unittest
import numpy as np
from src.utils.PairwiseDistance import PairwiseDistance


class TestGetDistance(unittest.TestCase):
    def setUp(self) -> None:
        scen_1 = [0, 2, 1, 2, 0, 1, 1]
        scen_2 = [2, 6, 0, 3, 2, 0, 0]
        mlco_1 = [0, 5., 352.5, 102., 176.6,
                  253.9, 396.3, 3.7, 57.1, 509.2, 590.]
        mlco_2 = [1, 253.2, 466., 638.3, 478.1,
                  800., 599.5, 747.6, 800., 166.4, 301.3]
        self.cs_1 = [scen_1, mlco_1]
        self.cs_2 = [scen_1, mlco_2]
        self.cs_3 = [scen_2, mlco_1]
        self.cs_4 = [scen_2, mlco_2]

        self.cat_ix = [0, 1, 2, 3, 4, 5, 6, 7]
        self.num_range = [1, 1, 1, 1, 1, 1, 1, 1, 500,
                          500, 800, 600, 800, 600, 800, 600, 800, 600]

    def test_cs_included_1(self):
        cs_list_1 = [self.cs_1, self.cs_2]

        test_distance = PairwiseDistance(
            cs_list=cs_list_1,
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )

        computed_distance = test_distance.get_distance(self.cs_1, self.cs_2)

        expected_distance = 0.36226375

        self.assertEqual(computed_distance, expected_distance)

    def test_cs_not_included_1(self):
        cs_list_1 = [self.cs_1, self.cs_2]

        test_distance = PairwiseDistance(
            cs_list=cs_list_1,
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )

        computed_distance = test_distance.get_distance(self.cs_3, self.cs_2)

        expected_distance = 0.79976375

        self.assertEqual(computed_distance, expected_distance)
