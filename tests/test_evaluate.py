from unittest.case import SkipTest, skip, skipIf
from deap import creator
from src.main.MLCSHE import MLCSHE
import problem
import unittest


class TestEvaluate(unittest.TestCase):

    @unittest.skip('test data not applicable to jfit evaluation.')
    def test_evaluate_testInput1(self):
        scen1 = creator.Scenario([1, False, 5.0])
        mlco1 = creator.OutputMLC([[8, 'a'], [2, 'b']])
        scen2 = creator.Scenario([4, True, -7.8])
        mlco2 = creator.OutputMLC([[1, 'a'], [21, 'd']])
        pScen = [scen1, scen2]
        aScen = [scen1]
        pMLCO = [mlco1, mlco2]
        aMLCO = [mlco1]
        cls = creator.Individual
        k = 2
        css = [[[1, False, 5.0], [[8, 'a'], [2, 'b']]],
               [[1, False, 5.0], [[1, 'a'], [21, 'd']]],
               [[4, True, -7.8], [[8, 'a'], [2, 'b']]],
               [[4, True, -7.8], [[1, 'a'], [21, 'd']]]]

        # FIXME: this function failed, possibly due to an error in evaluate_joint_fitness()
        a, b, c = self.evaluate(pScen, aScen, pMLCO, aMLCO, cls, k)
        self.assertEqual((a, b, c), (css, pScen, pMLCO))

        # Check whether fitness values for complete solutions and
        # individuals have been recorded.
        for cs in a:
            self.assertIsNotNone(cs.fitness.values)

        for scen in b:
            self.assertIsNotNone(scen.fitness.values)

        for mlco in b:
            self.assertIsNotNone(mlco.fitness.values)
