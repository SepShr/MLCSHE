import unittest
import numpy as np
from src.utils.PairwiseDistance import PairwiseDistance


class TestCalculateCdist(unittest.TestCase):
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

    def test_realistic_cs_list_1(self):
        cs_list_1 = [self.cs_1, self.cs_2]
        cs_list_2 = [self.cs_3, self.cs_4]

        expected_cs_list = [self.cs_1, self.cs_2, self.cs_3, self.cs_4]

        test_distance = PairwiseDistance(
            cs_list=cs_list_1,
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )
        test_distance.update_dist_matrix(cs_list=cs_list_2)
        computed_dist_mtx = test_distance.dist_matrix_sq
        expected_dist_mtx = np.array([
            [0., 0.36226375, 0.4375, 0.79976375],
            [0.36226375, 0., 0.79976375, 0.4375],
            [0.4375, 0.79976375, 0., 0.36226375],
            [0.79976375, 0.4375, 0.36226375, 0.]
        ])

        self.assertTrue(np.allclose(computed_dist_mtx, expected_dist_mtx))
        self.assertEqual(test_distance.cs_list, expected_cs_list)

    def test_null_1(self):
        cs_list_1 = [self.cs_1, self.cs_2]

        expected_cs_list = [self.cs_1, self.cs_2]

        test_distance = PairwiseDistance(
            cs_list=cs_list_1,
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )

        test_distance.update_dist_matrix(cs_list=[])

        self.assertEqual(test_distance.cs_list, expected_cs_list)

        _ = PairwiseDistance(
            cs_list=cs_list_1,
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )

        self.assertTrue(np.allclose(
            test_distance.dist_matrix_sq, _.dist_matrix_sq))
        print(_.dist_matrix_sq)

    def test_null_2(self):
        test_distance = PairwiseDistance(
            cs_list=[], numeric_ranges=self.num_range, categorical_indices=self.cat_ix)

        cs_list = [self.cs_1, self.cs_2]

        test_distance.update_dist_matrix(cs_list)

        expected_dist_mtx = [
            [0., 0.36226375],
            [0.36226375, 0.]
        ]

        self.assertTrue(np.allclose(
            test_distance.dist_matrix_sq, expected_dist_mtx))
        self.assertEqual(test_distance.cs_list, cs_list)
