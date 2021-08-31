from deap import creator
from CCEA import collaborateArchive
import unittest

class Test_TestCollaborateArchive(unittest.TestCase):
    def test_collaborateArchive_testInput1(self):
        a = [creator.Scenario([1]), creator.Scenario(['a'])]
        b = creator.OutputMLC([2])
        complete_solution_set = collaborateArchive(a, b, creator.Individual)
        self.assertEqual(complete_solution_set, [[[1], 2], [['a'], 2]])

if __name__=='__main__':
    unittest.main()