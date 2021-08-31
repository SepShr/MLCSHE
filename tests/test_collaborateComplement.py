from deap import creator
from CCEA import collaborateComplement
import unittest

class Test_TestCollaborateComplement(unittest.TestCase):
    def test_collaborateComplement_testInput1(self):
        p1 = [creator.Scenario([1]), creator.Scenario([2.5])]
        a1 = [creator.Scenario([1])]
        p2 = [[3]]
        icls = creator.Individual
        numTest = 2
        complete_solution_set = collaborateComplement(p1, a1, p2, numTest, icls)
        self.assertEqual(complete_solution_set, [[[2.5], [3]]])
    
    def test_collaborateComplement_numTest(self):
        p1 = [creator.Scenario([1]), creator.Scenario([2.5])]
        a1 = [creator.Scenario([1])]
        p2 = [[3]]
        icls = creator.Individual
        numTest = 1
        complete_solution_set = collaborateComplement(p1, a1, p2, numTest, icls)
        self.assertEqual(complete_solution_set, [])

if __name__=='__main__':
    unittest.main()