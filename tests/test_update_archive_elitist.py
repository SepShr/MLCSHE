from deap import creator
from src.main.ICCEA import ICCEA
import unittest
import random
import problem
import search_config as cfg
from simulation_runner import Simulator


class TestUpdateArchiveElitist(unittest.TestCase):
    def test_update_archive_elitist_testInput1(self):
        # make a solver instance
        simulator = Simulator()
        solver = ICCEA(
            creator=problem.creator,
            toolbox=problem.toolbox,
            simulator=simulator,
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

        archive_size = 1
        output_archive_pScen_1 = solver.update_archive_elitist(
            pScen, archive_size
        )

        output_archive_PMLCO_1 = solver.update_archive_elitist(
            pMLCO, archive_size
        )

        self.assertEqual(output_archive_pScen_1, [scen1])
        self.assertEqual(len(output_archive_pScen_1), 1)

        self.assertEqual(output_archive_PMLCO_1, [mlco1])
        self.assertEqual(len(output_archive_PMLCO_1), 1)

        archive_size = 2

        output_archive_pScen_2 = solver.update_archive_elitist(
            pScen, archive_size
        )

        output_archive_PMLCO_2 = solver.update_archive_elitist(
            pMLCO, archive_size
        )

        self.assertEqual(output_archive_pScen_2, [scen1, scen2])
        self.assertEqual(len(output_archive_pScen_2), 2)

        self.assertEqual(output_archive_PMLCO_2, [mlco1, mlco2])
        self.assertEqual(len(output_archive_PMLCO_2), 2)
