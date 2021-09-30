from deap import creator
from CCEA import update_archive
import random
import unittest

class TestUpdateArchive(unittest.TestCase):
    def test_update_archive_testInput1(self):
        scen1 = creator.Scenario([1, False, 5.0])
        scen1.fitness.values = (10.0,)
        mlco1 = creator.OutputMLC([[8, 'a'], [2, 'b']])
        mlco1.fitness.values = (10.0,)
        scen2 = creator.Scenario([4, True, -7.8])
        scen2.fitness.values = (random.randint(-10, 10),)
        scen3 = creator.Scenario([-2, False, 4.87])
        scen3.fitness.values = (random.randint(-10,10),)
        mlco2 = creator.OutputMLC([[1, 'a'], [21, 'd']])
        mlco2.fitness.values = (random.randint(-10, 10),)
        mlco3 = creator.OutputMLC([[-2, 'e'], [10, 'f']])
        mlco3.fitness.values = (random.randint(-10,10),)
        scen4 = creator.Scenario([2, True, 0.24])
        scen4.fitness.values = (random.randint(-10,10),)
        mlco4 = creator.OutputMLC([[4, 'g'], [-1, 'h']])
        mlco4.fitness.values = (random.randint(-10,10),)
        pScen = [scen1, scen2, scen3, scen4]
        pMLCO = [mlco1, mlco2, mlco3, mlco4]
        cls = creator.Individual
        cs1 = creator.Individual([scen1, mlco1])
        cs1.fitness.values = (10.0,)
        cs2 = creator.Individual([scen1, mlco2])
        cs2.fitness.values = (random.randint(-10, 10),)
        cs3 = creator.Individual([scen2, mlco1])
        cs3.fitness.values = (random.randint(-10, 10),)
        cs4 = creator.Individual([scen2, mlco2])
        cs4.fitness.values = (random.randint(-10, 10),)
        css = [cs1, cs2, cs3, cs4]
        min_dist = 1

        output_archive = update_archive(
            pScen, pMLCO, css, cls, min_dist
        )

        self.assertIn(scen1, output_archive)