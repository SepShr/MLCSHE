import unittest

from deap import base, creator
from fitness_function import fitness_function
from src.utils.PairwiseDistance import PairwiseDistance


class TestFitnessFunction(unittest.TestCase):
    def test_nominal_1(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMin, unsafe=bool)

        scen_1 = [0, 2, 1, 2, 0, 1, 1]
        scen_2 = [2, 6, 0, 3, 2, 0, 0]
        mlco_1 = [0, 5., 352.5, 102., 176.6,
                  253.9, 396.3, 3.7, 57.1, 509.2, 590.]
        mlco_2 = [1, 253.2, 466., 638.3, 478.1,
                  800., 599.5, 747.6, 800., 166.4, 301.3]

        cs_1 = creator.Individual([scen_1, mlco_1])
        cs_1.unsafe = True
        cs_2 = creator.Individual([scen_1, mlco_2])
        cs_2.unsafe = True
        cs_3 = creator.Individual([scen_2, mlco_1])
        cs_3.unsafe = False
        cs_4 = creator.Individual([scen_2, mlco_2])
        cs_4.unsafe = False

        test_cs_list = [cs_1, cs_2, cs_3, cs_4]
        cat_ix = [0, 1, 2, 3, 4, 5, 6, 7]
        num_range = [1, 1, 1, 1, 1, 1, 1, 1, 500,
                     500, 800, 600, 800, 600, 800, 600, 800, 600]
        test_distance = PairwiseDistance(
            cs_list=test_cs_list,
            numeric_ranges=num_range,
            categorical_indices=cat_ix
        )

        computed_fv = fitness_function(cs=cs_3, max_dist=0.9, cs_list=test_distance.cs_list,
                                       dist_matrix=test_distance.dist_matrix_sq, w_ci=1, w_p=1)

        expected_fv = 1.300071417640343

        self.assertEqual(computed_fv, expected_fv)
