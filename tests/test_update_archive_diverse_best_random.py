from deap import creator
from src.main.ICCEA import ICCEA
import unittest
import random
import problem
import search_config as cfg
from Simulator import Simulator
from src.utils.PairwiseDistance import PairwiseDistance


class TestUpdateArchiveDiverseBestRandom(unittest.TestCase):
    def test_update_archive_diverse_best_random_testInput1(self):
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

        # prepare test input
        scen1 = creator.Scenario([1, False, 5.0])
        scen1.fitness.values = (10.0,)
        mlco1 = creator.OutputMLC([[8, 'a'], [2, 'b']])
        mlco1.fitness.values = (10.0,)
        scen2 = creator.Scenario([4, True, -7.8])
        scen2.fitness.values = (8.5,)
        scen3 = creator.Scenario([-2, False, 4.87])
        scen3.fitness.values = (random.randint(-10, 8),)
        mlco2 = creator.OutputMLC([[1, 'a'], [21, 'd']])
        mlco2.fitness.values = (8.5,)
        mlco3 = creator.OutputMLC([[-2, 'e'], [10, 'f']])
        mlco3.fitness.values = (random.randint(-10, 8),)
        scen4 = creator.Scenario([2, True, 0.24])
        scen4.fitness.values = (random.randint(-10, 8),)
        mlco4 = creator.OutputMLC([[4, 'g'], [-1, 'h']])
        mlco4.fitness.values = (random.randint(-10, 8),)
        pScen = [scen1, scen2, scen3, scen4]
        pMLCO = [mlco1, mlco2, mlco3, mlco4]

        max_archive_size = 1
        min_distance = 0.5

        output_archive_pScen_1 = solver.update_archive_diverse_best_random(
            pScen, max_archive_size, min_distance
        )

        output_archive_pMLCO_1 = solver.update_archive_diverse_best_random(
            pMLCO, max_archive_size, min_distance
        )

        self.assertEqual(output_archive_pScen_1, [scen1])
        self.assertEqual(len(output_archive_pScen_1), 1)

        self.assertEqual(output_archive_pMLCO_1, [mlco1])
        self.assertEqual(len(output_archive_pMLCO_1), 1)

        max_archive_size = 2

        output_archive_pScen_2 = solver.update_archive_diverse_best_random(
            pScen, max_archive_size, min_distance
        )

        output_archive_pMLCO_2 = solver.update_archive_diverse_best_random(
            pMLCO, max_archive_size, min_distance
        )

        for i in range(len(output_archive_pScen_2)):
            self.assertIn(output_archive_pScen_2[i], pScen)

        self.assertLessEqual(len(output_archive_pScen_2), 2)

        for i in range(len(output_archive_pMLCO_2)):
            self.assertIn(output_archive_pMLCO_2[i], pMLCO)

        self.assertLessEqual(len(output_archive_pMLCO_2), 2)

        min_distance = 1.0

        output_archive_pScen_3 = solver.update_archive_diverse_best_random(
            pScen, max_archive_size, min_distance
        )

        output_archive_pMLCO_3 = solver.update_archive_diverse_best_random(
            pMLCO, max_archive_size, min_distance
        )

        self.assertEqual(output_archive_pScen_3, [scen1])
        self.assertEqual(len(output_archive_pScen_3), 1)

        self.assertEqual(output_archive_pMLCO_3, [mlco1])
        self.assertEqual(len(output_archive_pMLCO_3), 1)
