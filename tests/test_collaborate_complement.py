from deap import creator
from CCEA import collaborate_complement
import unittest

class TestCollaborateComplement(unittest.TestCase):
    def test_collaborate_complement_testInput1(self):
        p1 = [creator.Scenario([1]), creator.Scenario([2.5])]
        a1 = [creator.Scenario([1])]
        p2 = [[3]]
        icls = creator.Individual
        fcls = creator.Scenario
        numTest = 2
        complete_solution_set = collaborate_complement(
            p1, a1, p2, numTest, icls, fcls)
        self.assertEqual(complete_solution_set, [[[2.5], [3]]])
    
    def test_collaborate_complement_numTest(self):
        p1 = [creator.Scenario([1]), creator.Scenario([2.5])]
        a1 = [creator.Scenario([1])]
        p2 = [[3]]
        icls = creator.Individual
        fcls = creator.Scenario
        numTest = 1
        complete_solution_set = collaborate_complement(
            p1, a1, p2, numTest, icls, fcls)
        self.assertEqual(complete_solution_set, [])

if __name__=='__main__':
    unittest.main()