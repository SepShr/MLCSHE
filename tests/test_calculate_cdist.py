import unittest
import numpy as np
from src.utils.PairwiseDistance import PairwiseDistance


class TestCalculateCdist(unittest.TestCase):
    def test_realistic_cs_list_1(self):
        scen_1 = [0, 2, 1, 2, 0, 1, 1]
        scen_2 = [2, 6, 0, 3, 2, 0, 0]
        mlco_1 = [0, 5., 352.5, 102., 176.6,
                  253.9, 396.3, 3.7, 57.1, 509.2, 590.]
        mlco_2 = [1, 253.2, 466., 638.3, 478.1,
                  800., 599.5, 747.6, 800., 166.4, 301.3]

        cs_1 = [scen_1, mlco_1]
        cs_2 = [scen_1, mlco_2]
        cs_3 = [scen_2, mlco_1]
        cs_4 = [scen_2, mlco_2]

        cs_list_1 = [cs_1, cs_2]

        cs_list_2 = [cs_3, cs_4]

        expected_cs_list = [cs_1, cs_2, cs_3, cs_4]

        cat_ix = [0, 1, 2, 3, 4, 5, 6, 7]
        num_range = [1, 1, 1, 1, 1, 1, 1, 1, 500,
                     500, 800, 600, 800, 600, 800, 600, 800, 600]
        test_distance = PairwiseDistance(
            cs_list=cs_list_1,
            numeric_ranges=num_range,
            categorical_indices=cat_ix
        )
        test_distance.update_dist_matrix(
            new_cs=cs_list_2,
            calculated_vectors=test_distance.vectors,
            dist_matrix=test_distance.dist_matrix_sq,
            num_ranges=num_range,
            cat_indices=test_distance.categorical_indices
        )
        computed_dist_mtx = test_distance.dist_matrix_sq
        expected_dist_mtx = np.array([
            [0., 0.36226375, 0.4375, 0.79976375],
            [0.36226375, 0., 0.79976375, 0.4375],
            [0.4375, 0.79976375, 0., 0.36226375],
            [0.79976375, 0.4375, 0.36226375, 0.]
        ])

        self.assertTrue(np.allclose(computed_dist_mtx, expected_dist_mtx))
        self.assertEqual(test_distance.cs_list, expected_cs_list)
