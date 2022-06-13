from deap import creator
from src.main.ICCEA import ICCEA
import unittest
import problem
import search_config as cfg
from simulation_runner import Simulator
from src.utils.PairwiseDistance import PairwiseDistance


class TestFlatten(unittest.TestCase):
    def test_flatten_testInput1(self):
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
            pairwise_distance_cs=pdist_cs,
            simulator=simulator,
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
