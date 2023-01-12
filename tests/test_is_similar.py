from deap import creator
from src.main.MLCSHE import MLCSHE
import unittest
import problem
import search_config as cfg
from src.utils.PairwiseDistance import PairwiseDistance


class TestIsSimilar(unittest.TestCase):
    def test_is_similar_testInput1(self):
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

        # Test values
        scen1 = creator.Scenario([1, False, 5.0])
        mlco1 = creator.OutputMLC([[8, 'a'], [2, 'b']])
        scen2 = creator.Scenario([4, True, -7.8])
        scen3 = creator.Scenario([-2, False, 4.87])
        mlco2 = creator.OutputMLC([[1, 'a'], [21, 'd']])
        mlco3 = creator.OutputMLC([[-2, 'e'], [10, 'f']])
        arc = [scen2, scen3]
        arc_collab_dict = {str(scen2): mlco2, str(scen3): mlco3}
        min_dist = 0.5
        ficls = type(scen1)

        output_value = solver.is_similar(
            scen1, mlco1, arc, arc_collab_dict, min_dist, ficls
        )

        self.assertFalse(output_value)
