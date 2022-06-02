from deap import base, creator

import unittest

from problem_utils import initialize_mlco


class TestInitObsTraj(unittest.TestCase):
    def setUp(self) -> None:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMin, unsafe=bool)
        creator.create("OutputMLC", creator.Individual)
        self.c = creator.OutputMLC
        self.duration = 500

    def test_1_trajectory(self):
        num_traj = 1
        created_mlco = initialize_mlco(
            class_=self.c, num_traj=num_traj, duration=self.duration)
        self.assertIsInstance(created_mlco, self.c)
        self.assertEqual(len(created_mlco), 1)
        self.assertEqual(len(created_mlco[0]), 11)
        self.assertIn(created_mlco[0][0], [-1, 0, 1])

    def test_2_trajectory(self):
        num_traj = 2
        created_mlco = initialize_mlco(
            class_=self.c, num_traj=num_traj, duration=self.duration)
        self.assertIsInstance(created_mlco, self.c,
                              f'Type of created_mlco is {type(created_mlco)}')
        self.assertEqual(len(created_mlco), 2)
        self.assertEqual(len(created_mlco[0]), 11)
        self.assertEqual(len(created_mlco[1]), 11)
        self.assertIn(created_mlco[0][0], [-1, 0, 1])
        self.assertIn(created_mlco[1][0], [-1, 0, 1])
