from deap import creator
from src.utils.utility import collaborate
import problem
import unittest


class TestCollaborate(unittest.TestCase):
    def test_collaborate_testInput1(self):
        p1 = [creator.Scenario([1]), creator.Scenario([2.5])]
        a1 = [creator.Scenario([1])]
        p2 = [[3], creator.OutputMLC(["a"])]
        a2 = [creator.OutputMLC(["a"])]
        cls = creator.Individual
        fcls = creator.Scenario
        k = 2
        complete_solution_set = collaborate(a1, p1, a2, p2, cls, fcls, k)
        expected_solution = [[[1], [3]], [[1], ['a']],
                             [[2.5], ['a']], [[2.5], [3]]]
        self.assertEqual(complete_solution_set, expected_solution)

    def test_collaborate_numTest(self):
        p1 = [creator.Scenario([1]), creator.Scenario([2.5])]
        a1 = [creator.Scenario([1])]
        p2 = [[3], creator.OutputMLC(["a"])]
        a2 = [creator.OutputMLC(["a"])]
        cls = creator.Individual
        fcls = creator.Scenario
        k = 1
        expected_solution = [[[1], [3]], [[1], ['a']], [[2.5], ['a']]]
        complete_solution_set = collaborate(a1, p1, a2, p2, cls, fcls, k)
        self.assertEqual(complete_solution_set, expected_solution)


if __name__ == '__main__':
    unittest.main()
