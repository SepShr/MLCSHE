from deap import creator
from CCEA import breed_scenario
import unittest

class TestBreedScenario(unittest.TestCase):
    def test_breed_scenario_testInput1(self):
        p = [
            creator.Scenario([1, 2.0, False]), 
            creator.Scenario([8, 3.5, True]),
            creator.Scenario([70])
        ]
        a = [creator.Scenario([70])]
        lmt = [[1, 5], None, None]
        ts = 2
        cxpb = 1
        mut_bit_pb = 1
        mut_guass_mu = 0
        mut_guass_sig = 1
        mut_guass_pb = 1
        mut_int_pb = 1
        bred_scenarios = breed_scenario(
            p, a, lmt, ts, cxpb,  mut_bit_pb,
            mut_guass_mu, mut_guass_sig, mut_guass_pb, mut_int_pb)
        expected_value = [[1, 2.0, False], [8, 3.5, True]]
        self.assertNotEqual(bred_scenarios, expected_value)
        self.assertEqual(len(bred_scenarios), len(expected_value))
        
if __name__=='__main__':
    unittest.main()