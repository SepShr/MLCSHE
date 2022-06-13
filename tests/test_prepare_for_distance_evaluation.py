from src.main.ICCEA import ICCEA
import unittest
import problem
import search_config as cfg
from simulation_runner import Simulator
from src.utils.PairwiseDistance import PairwiseDistance


class TestPrepareForDistanceEvaluation(unittest.TestCase):
    def test_prepare_for_distance_evaluation_testInput1(self):
        # make a solver instance
        simulator = Simulator()
        pdist_cs = PairwiseDistance(
            cs_list=[],
            numeric_ranges=[[0.0, 1.0]],
            categorical_indices=[]
        )
        solver = ICCEA(
            creator=problem.creator,
            toolbox=problem.toolbox,
            simulator=simulator,
            pairwise_distance_cs=pdist_cs,
            first_population_enumLimits=cfg.scenario_enumLimits
        )

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
            solver.prepare_for_distance_evaluation(test_list),
            (expected_flat_list, expected_nominal_values_indices)
        )
