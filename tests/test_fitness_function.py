import unittest

from deap import base, creator
from src.main.fitness_function import calculate_fitness
from src.utils.PairwiseDistance import PairwiseDistance


class TestFitnessFunction(unittest.TestCase):

    def setUp(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMin, safety_req_value=float)

        self.cat_ix = [0, 1, 2, 3, 4, 5, 6, 7]
        self.num_range = [1, 1, 1, 1, 1, 1, 1,
                          1, 500, 500, 800, 600, 800, 600, 800, 600, 800, 600]

    def test_nominal_1(self):
        scen_1 = [0, 2, 1, 2, 0, 1, 1]
        scen_2 = [2, 6, 0, 3, 2, 0, 0]
        mlco_1 = [0, 5., 352.5, 102., 176.6,
                  253.9, 396.3, 3.7, 57.1, 509.2, 590.]
        mlco_2 = [1, 253.2, 466., 638.3, 478.1,
                  800., 599.5, 747.6, 800., 166.4, 301.3]

        cs_1 = creator.Individual([scen_1, mlco_1])
        cs_1.safety_req_value = -0.6
        cs_2 = creator.Individual([scen_1, mlco_2])
        cs_2.safety_req_value = -8.69
        cs_3 = creator.Individual([scen_2, mlco_1])
        cs_3.safety_req_value = 0.01
        cs_4 = creator.Individual([scen_2, mlco_2])
        cs_4.safety_req_value = 54.0

        test_cs_list = [cs_1, cs_2, cs_3, cs_4]

        test_distance = PairwiseDistance(
            cs_list=test_cs_list,
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )

        computed_fv = calculate_fitness(cs=cs_3, max_dist=0.9, cs_list=test_distance.cs_list,
                                        dist_matrix=test_distance.dist_matrix_sq)

        expected_fv = 0.3499642911798285

        self.assertEqual(computed_fv, expected_fv)

    def test_calculate_fitness_radius_metamorphic(self):
        # randomly generate many complete solutions based on the range and cat_ix
        import random
        num_neighbors = 1000
        cs_list = []
        for i in range(num_neighbors):
            scenario = [random.randint(0, 7)
                        for j in range(7)]  # random scenario
            mlco = [random.randint(0, 7)]  # random mlco[0]
            mlco += [random.randint(0, self.num_range[j+8])
                     for j in range(10)]  # random mlco[1:]
            cs = creator.Individual([scenario, mlco])
            cs.safety_req_value = random.uniform(-10.0, 10.0)
            cs_list.append(cs)

        # compute p-dist
        test_distance = PairwiseDistance(
            cs_list=cs_list,
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )

        # compute fitness using max_dist in [0.1, 0.5, 0.9]
        fv_1 = calculate_fitness(cs=test_distance.cs_list[0], max_dist=0.1, cs_list=test_distance.cs_list,
                                 dist_matrix=test_distance.dist_matrix_sq, w_ci=1, w_p=1)
        fv_5 = calculate_fitness(cs=test_distance.cs_list[0], max_dist=0.5, cs_list=test_distance.cs_list,
                                 dist_matrix=test_distance.dist_matrix_sq, w_ci=1, w_p=1)
        fv_9 = calculate_fitness(cs=test_distance.cs_list[0], max_dist=0.9, cs_list=test_distance.cs_list,
                                 dist_matrix=test_distance.dist_matrix_sq, w_ci=1, w_p=1)

        # metamorphic testing
        self.assertTrue(fv_1 >= fv_5 >= fv_9)

    def test_calculate_fitness_num_samples_metamorphic(self):
        # randomly generate many complete solutions based on the range and cat_ix
        import random
        num_neighbors = 1000
        cs_list = []
        for i in range(num_neighbors):
            scenario = [random.randint(0, 7)
                        for j in range(7)]  # random scenario
            mlco = [random.randint(0, 7)]  # random mlco[0]
            mlco += [random.randint(0, self.num_range[j+8])
                     for j in range(10)]  # random mlco[1:]
            cs = creator.Individual([scenario, mlco])
            cs.safety_req_value = random.uniform(-10.0, 10.0)
            cs_list.append(cs)

        # compute p-dist and fitness value
        test_distance_1 = PairwiseDistance(
            cs_list=cs_list[:100],
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )
        fv_100 = calculate_fitness(cs=test_distance_1.cs_list[0], max_dist=0.5, cs_list=test_distance_1.cs_list,
                                   dist_matrix=test_distance_1.dist_matrix_sq)

        test_distance_2 = PairwiseDistance(
            cs_list=cs_list[:500],
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )
        fv_500 = calculate_fitness(cs=test_distance_2.cs_list[0], max_dist=0.5, cs_list=test_distance_2.cs_list,
                                   dist_matrix=test_distance_2.dist_matrix_sq)

        test_distance_3 = PairwiseDistance(
            cs_list=cs_list[:1000],
            numeric_ranges=self.num_range,
            categorical_indices=self.cat_ix
        )
        fv_1000 = calculate_fitness(cs=test_distance_3.cs_list[0], max_dist=0.5, cs_list=test_distance_3.cs_list,
                                    dist_matrix=test_distance_3.dist_matrix_sq)

        # metamorphic testing
        self.assertTrue(fv_100 >= fv_500 >= fv_1000)
