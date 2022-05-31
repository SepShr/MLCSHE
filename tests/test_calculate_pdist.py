import unittest

import numpy as np
from deap import base, creator
from src.utils.PairwiseDistance import PairwiseDistance


class TestCalculatePdist(unittest.TestCase):
    # def test_nominal_vec(self):
    #     test_vec_list_1 = [
    #         [1, 0, 5.0, 1, 1, 21, 4],
    #         [4, 1, -7.8, 8, 1, 2, 2],
    #         [2, -1, -9.8, 10, 3, 2, 91]
    #     ]
    #     cat_ix_1 = [0, 3, 6]
    #     num_range_1 = [1, 4, 20, 1, 3, 25, 1]
    #     test_distance_1 = PairwiseDistance(
    #         vectors=test_vec_list_1,
    #         numeric_ranges=num_range_1,
    #         categorical_indices=cat_ix_1
    #     )
    #     computed_dist_mtx = test_distance_1.dist_matrix_sq

    #     expected_dist_mtx = np.array([
    #         [0., 0.70625, 0.80208333],
    #         [0.70625, 0., 0.65833333],
    #         [0.80208333, 0.65833333, 0.]
    #     ])

    #     self.assertTrue(np.allclose(
    #         computed_dist_mtx, expected_dist_mtx))

    # def test_realistic_vec_1(self):
    #     test_vec_list_2 = [
    #         [0, 2, 1, 2, 0, 1, 1, 0, 5., 352.5, 102., 176.6,
    #             253.9, 396.3, 3.7, 57.1, 509.2, 590.],
    #         [0, 2, 1, 2, 0, 1, 1, 1, 253.2, 466., 638.3,
    #             478.1, 800., 599.5, 747.6, 800., 166.4, 301.3],
    #         [2, 6, 0, 3, 2, 0, 0, 0, 5., 352.5, 102., 176.6,
    #             253.9, 396.3, 3.7, 57.1, 509.2, 590.],
    #         [2, 6, 0, 3, 2, 0, 0, 1, 253.2, 466., 638.3,
    #             478.1, 800., 599.5, 747.6, 800., 166.4, 301.3]
    #     ]
    #     cat_ix_2 = [0, 1, 2, 3, 4, 5, 6, 7]
    #     num_range_2 = [1, 1, 1, 1, 1, 1, 1, 1, 500,
    #                    500, 800, 600, 800, 600, 800, 600, 800, 600]
    #     test_distance_2 = PairwiseDistance(
    #         vectors=test_vec_list_2,
    #         numeric_ranges=num_range_2,
    #         categorical_indices=cat_ix_2
    #     )
    #     computed_dist_mtx = test_distance_2.dist_matrix_sq
    #     expected_dist_mtx = np.array([
    #         [0., 0.36226375, 0.4375, 0.79976375],
    #         [0.36226375, 0., 0.79976375, 0.4375],
    #         [0.4375, 0.79976375, 0., 0.36226375],
    #         [0.79976375, 0.4375, 0.36226375, 0.]
    #     ])

    #     self.assertTrue(np.allclose(computed_dist_mtx, expected_dist_mtx))

    # def test_realistic_vec_2(self):
    #     test_vec_list_3 = [
    #         [0, 2, 1, 2, 0, 1, 1, 0, 5., 352.5, 102., 176.6,
    #             253.9, 396.3, 3.7, 57.1, 509.2, 590., 1, 253.2, 466., 638.3, 478.1,
    #          800., 599.5, 747.6, 800., 166.4, 301.3, 0, 5., 352.5, 102., 176.6,
    #          253.9, 396.3, 3.7, 57.1, 509.2, 590.],
    #         [0, 2, 1, 2, 0, 1, 1, 1, 253.2, 466., 638.3,
    #             478.1, 800., 599.5, 747.6, 800., 166.4, 301.3, 0, 5., 352.5, 102., 176.6,
    #          253.9, 396.3, 3.7, 57.1, 509.2, 590., 1, 253.2, 466., 638.3, 478.1,
    #          800., 599.5, 747.6, 800., 166.4, 301.3],
    #         [2, 6, 0, 3, 2, 0, 0, 0, 5., 352.5, 102., 176.6,
    #             253.9, 396.3, 3.7, 57.1, 509.2, 590., 1, 253.2, 466., 638.3, 478.1,
    #          800., 599.5, 747.6, 800., 166.4, 301.3, 0, 5., 352.5, 102., 176.6,
    #          253.9, 396.3, 3.7, 57.1, 509.2, 590.],
    #         [2, 6, 0, 3, 2, 0, 0, 1, 253.2, 466., 638.3,
    #             478.1, 800., 599.5, 747.6, 800., 166.4, 301.3, 0, 5., 352.5, 102., 176.6,
    #          253.9, 396.3, 3.7, 57.1, 509.2, 590., 1, 253.2, 466., 638.3, 478.1,
    #          800., 599.5, 747.6, 800., 166.4, 301.3]
    #     ]
    #     cat_ix_3 = [0, 1, 2, 3, 4, 5, 6, 7, 18, 29]
    #     num_range_3 = [1, 1, 1, 1, 1, 1, 1, 1, 500,
    #                    500, 800, 600, 800, 600, 800, 600, 800, 600,
    #                    1, 500, 500, 800, 600, 800, 600, 800, 600, 800, 600,
    #                    1, 500, 500, 800, 600, 800, 600, 800, 600, 800, 600]
    #     test_distance_3 = PairwiseDistance(
    #         vectors=test_vec_list_3,
    #         numeric_ranges=num_range_3,
    #         categorical_indices=cat_ix_3
    #     )
    #     computed_dist_mtx = test_distance_3.dist_matrix_sq

    #     expected_dist_mtx = np.array([
    #         [0., 0.44976375, 0.35, 0.79976375],
    #         [0.44976375, 0., 0.79976375, 0.35],
    #         [0.35, 0.79976375, 0., 0.44976375],
    #         [0.79976375, 0.35, 0.44976375, 0.]
    #     ])

    #     self.assertTrue(np.allclose(computed_dist_mtx, expected_dist_mtx))

    def test_realistic_cs_list_1(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMin, unsafe=bool)

        scen_1 = [0, 2, 1, 2, 0, 1, 1]
        scen_2 = [2, 6, 0, 3, 2, 0, 0]
        mlco_1 = [0, 5., 352.5, 102., 176.6,
                  253.9, 396.3, 3.7, 57.1, 509.2, 590.]
        mlco_2 = [1, 253.2, 466., 638.3, 478.1,
                  800., 599.5, 747.6, 800., 166.4, 301.3]

        test_cs_list_1 = [
            creator.Individual([scen_1, mlco_1]),
            creator.Individual([scen_1, mlco_2]),
            creator.Individual([scen_2, mlco_1]),
            creator.Individual([scen_2, mlco_2])
        ]
        cat_ix_2 = [0, 1, 2, 3, 4, 5, 6, 7]
        num_range_2 = [1, 1, 1, 1, 1, 1, 1, 1, 500,
                       500, 800, 600, 800, 600, 800, 600, 800, 600]
        test_distance_4 = PairwiseDistance(
            cs_list=test_cs_list_1,
            numeric_ranges=num_range_2,
            categorical_indices=cat_ix_2
        )
        computed_dist_mtx = test_distance_4.dist_matrix_sq
        expected_dist_mtx = np.array([
            [0., 0.36226375, 0.4375, 0.79976375],
            [0.36226375, 0., 0.79976375, 0.4375],
            [0.4375, 0.79976375, 0., 0.36226375],
            [0.79976375, 0.4375, 0.36226375, 0.]
        ])

        self.assertTrue(np.allclose(computed_dist_mtx, expected_dist_mtx))
