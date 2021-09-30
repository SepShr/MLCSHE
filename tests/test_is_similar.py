from deap import creator
from CCEA import is_similar
import unittest

class TestIsSimilar(unittest.TestCase):
    def test_is_similar_testInput1(self):
        # Test values
        scen1 = creator.Scenario([1, False, 5.0])
        mlco1 = creator.OutputMLC([[8, 'a'], [2, 'b']])
        scen2 = creator.Scenario([4, True, -7.8])
        scen3 = creator.Scenario([-2, False, 4.87])
        mlco2 = creator.OutputMLC([[1, 'a'], [21, 'd']])
        mlco3 = creator.OutputMLC([[-2, 'e'], [10, 'f']])
        arc = [scen2, scen3]
        arc_collab_dict = {str(scen2): mlco2, str(scen3): mlco3}
        min_dist = 1
        ficls = type(scen1)

        output_value = is_similar(
            scen1, mlco1, arc, arc_collab_dict, min_dist, ficls
        )
        
        self.assertFalse(output_value)
