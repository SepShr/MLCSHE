from src.main.MLCSHE import MLCSHE
from deap import creator
import unittest
import problem
import search_config as cfg
from src.utils.PairwiseDistance import PairwiseDistance


class TestIndividualInList(unittest.TestCase):
    def test_individual_in_list_testInput1(self):
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

        c1 = creator.Individual([[1, False, 5.0], [[8, 'a'], [2, 'b']]])
        c2 = creator.Individual([[2, True, 0.24], [[-2, 'e'], [10, 'f']]])
        c3 = creator.Individual([[2, True, 0.24], [[1, 'a'], [21, 'd']]])
        cset = [c1, c2]

        self.assertTrue(solver.individual_in_list(c1, cset))
        self.assertFalse(solver.individual_in_list(c3, cset))
