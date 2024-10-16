from deap import creator
from src.utils.utility import collaborate_archive
import pylot.problem as problem
import unittest


class TestCollaborateArchive(unittest.TestCase):
    def test_collaborate_archive_testInput1(self):

        a = [creator.Scenario([1]), creator.Scenario(['a'])]
        b = creator.OutputMLC([2])
        cls = creator.Individual
        fcls = creator.Scenario
        complete_solution_set = collaborate_archive(a, b, cls, fcls)
        self.assertEqual(complete_solution_set, [[[1], 2], [['a'], 2]])


if __name__ == '__main__':
    unittest.main()
