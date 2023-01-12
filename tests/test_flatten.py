from deap import creator
from src.main.MLCSHE import MLCSHE
import unittest
import problem
import search_config as cfg
from src.utils.PairwiseDistance import PairwiseDistance


class TestFlatten(unittest.TestCase):
    def test_flatten_testInput1(self):
        # make a solver instance
        pdist_cs = PairwiseDistance(
            cs_list=[],
            numeric_ranges=[[0.0, 1.0]],
            categorical_indices=[]
        )
        pairwise_distance_scen = PairwiseDistance(
            cs_list=[],
            numeric_ranges=cfg.scenario_numeric_ranges,
            categorical_indices=cfg.scenario_catgorical_indices
        )

        pairwise_distance_mlco = PairwiseDistance(
            cs_list=[],
            numeric_ranges=cfg.mlco_numeric_ranges,
            categorical_indices=cfg.mlco_categorical_indices
        )
        solver = MLCSHE(
            creator=problem.creator,
            toolbox=problem.toolbox,
            pairwise_distance_cs=pdist_cs,
            pairwise_distance_p1=pairwise_distance_scen,
            pairwise_distance_p2=pairwise_distance_mlco,
            simulator=None,
            first_population_enumLimits=cfg.scenario_enumLimits
        )

        test_list = [
            [1, False, 4.5], [4, True, 8.5],
            1, [2, 3], [3, True, 2.2]
        ]

        expected_flat_list = [
            1, False, 4.5, 4, True,
            8.5, 1, 2, 3, 3, True, 2.2
        ]

        self.assertEqual(solver.flatten(test_list), expected_flat_list)
