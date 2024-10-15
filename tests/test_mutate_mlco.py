import random
from copy import deepcopy
from deap import base, creator
import unittest
from pylot.problem_utils import initialize_mlco, mutate_mlco, mutate_time
from tqdm import trange


class TestMutateMLCO(unittest.TestCase):
    def setUp(self) -> None:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMin, unsafe=bool)
        creator.create("OutputMLC", creator.Individual)
        self.c = creator.OutputMLC

        self.min_duration = 50
        self.duration = 900
        self.frame_width = 800
        self.frame_height = 600
        self.min_bbox_size = 50

        self.guassian_mutation_mean = 0
        self.guassian_mutation_std = 0.125
        self.guassian_mutation_probability = 1
        self.integer_mutation_probability = 1
        self.null_trajectory = [-1, 0, 0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        x_min_lb = 0.0
        x_min_ub = self.frame_width - self.min_bbox_size
        x_max_lb = self.min_bbox_size
        x_max_ub = self.frame_width
        y_min_lb = 0.0
        y_min_ub = self.frame_height - self.min_bbox_size
        y_max_lb = self.min_bbox_size
        y_max_ub = self.frame_height

        self.traj_enum_limit = [
            [-1, 1],
            [0.0, self.duration],
            [0.0, self.duration],
            [x_min_lb, x_min_ub],
            [x_max_lb, x_max_ub],
            [y_min_lb, y_min_ub],
            [y_max_lb, y_max_ub],
            [x_min_lb, x_min_ub],
            [x_max_lb, x_max_ub],
            [y_min_lb, y_min_ub],
            [y_max_lb, y_max_ub],
        ]

    def test_mutate_1_trajectory(self):
        num_traj = 2
        initial_mlco = initialize_mlco(self.c, num_traj, self.duration)

        mutated_mlco = mutate_mlco(initial_mlco, self.guassian_mutation_mean, self.guassian_mutation_std,
                                   self.guassian_mutation_probability, self.integer_mutation_probability, self.traj_enum_limit)

        self.assertNotEqual(mutated_mlco, initial_mlco)
        self.assertIsInstance(mutated_mlco, self.c)

    def test_mutate_time(self):
        # repeat 10000 times considering the randomness in mutate_time()
        for i in trange(10000):
            # org_time = [random.randint(0, self.duration), random.randint(0, self.duration)]
            org_time = [859, 343]
            integer_mutation_probability = random.random()
            mutated_time = mutate_time(time_list=org_time,
                                       mutipb=integer_mutation_probability,
                                       duration=self.duration,
                                       min_duration=self.min_duration)
            # note that org_time can be the same as mutated_time
            self.assertTrue(0 <= mutated_time[0] <= self.duration)
            self.assertTrue(0 <= mutated_time[1] <= self.duration)
            self.assertTrue(mutated_time[1] -
                            mutated_time[0] >= self.min_duration)
