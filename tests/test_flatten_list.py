from deap import base, creator
import numpy as np
import unittest
from src.utils.utility import flatten_list


class TestFlattenList(unittest.TestCase):
    def test_1(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMin, unsafe=bool)

        scen_1 = [0, 2, 1, 2, 0, 1, 1]
        scen_2 = [2, 6, 0, 3, 2, 0, 0]
        mlco_1 = [0, 5., 352.5, 102., 176.6,
                  253.9, 396.3, 3.7, 57.1, 509.2, 590.]
        mlco_2 = [1, 253.2, 466., 638.3, 478.1,
                  800., 599.5, 747.6, 800., 166.4, 301.3]

        cs_1 = creator.Individual([scen_1, mlco_1]),
        cs_2 = creator.Individual([scen_1, mlco_2]),
        cs_3 = creator.Individual([scen_2, mlco_1]),
        cs_4 = creator.Individual([scen_2, mlco_2])

        test_cs_list = [cs_1, cs_2, cs_3, cs_4]

        flattened_cs_list = [flatten_list(list(cs)) for cs in test_cs_list]

        print(flattened_cs_list)

        expected_list = [
            [0, 2, 1, 2, 0, 1, 1, 0, 5., 352.5, 102., 176.6,
                253.9, 396.3, 3.7, 57.1, 509.2, 590.],
            [0, 2, 1, 2, 0, 1, 1, 1, 253.2, 466., 638.3,
                478.1, 800., 599.5, 747.6, 800., 166.4, 301.3],
            [2, 6, 0, 3, 2, 0, 0, 0, 5., 352.5, 102., 176.6,
                253.9, 396.3, 3.7, 57.1, 509.2, 590.],
            [2, 6, 0, 3, 2, 0, 0, 1, 253.2, 466., 638.3,
                478.1, 800., 599.5, 747.6, 800., 166.4, 301.3]
        ]

        self.assertTrue(np.allclose(flattened_cs_list, expected_list))
