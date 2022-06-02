import unittest

from problem_utils import create_obs_traj


class TestCreateObsTraj(unittest.TestCase):
    def setUp(self) -> None:
        self.duration = 500
        self.null_label = -1
        self.vehicle_label = 0
        self.pedestrian_label = 1
        self.frame_width = 800
        self.frame_height = 600
        self.min_bbox_size = 50

    def test_null_trajectory(self):
        created_trajectory = create_obs_traj(
            obstacle_label=self.null_label,
            duration=self.duration,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            min_bbox_size=self.min_bbox_size
        )
        expected_trajectory = [-1, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertEqual(created_trajectory, expected_trajectory)

    def test_vehicle(self):
        created_trajectory = create_obs_traj(
            obstacle_label=self.vehicle_label,
            duration=self.duration,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            min_bbox_size=self.min_bbox_size
        )

        # Test length of the trajectory vector.
        self.assertEqual(len(created_trajectory), 11)
        # Test the label.
        self.assertEqual(created_trajectory[0], 0)
        # Test time within bounds.
        self.assertTrue(
            0.0 <= created_trajectory[1] <= self.duration, "t0 is not within bounds.")
        self.assertTrue(
            0.0 <= created_trajectory[2] <= self.duration, "t1 is not within bounds.")
        self.assertTrue(created_trajectory[1] <= created_trajectory[2])

        # Test bbox within bounds.
        x_min_lb = 0.0
        x_min_ub = self.frame_width - self.min_bbox_size
        x_min = created_trajectory[3]

        x_max_lb = self.min_bbox_size
        x_max_ub = self.frame_width
        x_max = created_trajectory[4]

        y_min_lb = 0.0
        y_min_ub = self.frame_height - self.min_bbox_size
        y_min = created_trajectory[5]

        y_max_lb = self.min_bbox_size
        y_max_ub = self.frame_height
        y_max = created_trajectory[6]

        self.assertTrue(x_min_lb <= x_min <= x_min_ub,
                        f'x_min at t0 is not within bounds. {x_min} is not between {x_min_lb} and {x_min_ub}')
        self.assertTrue(x_max_lb <= x_max <= x_max_ub,
                        f'x_max at t0 is not within bounds. {x_max} is not between {x_max_lb} and {x_max_ub}')
        self.assertTrue(y_min_lb <= y_min <= y_min_ub,
                        f'y_min at t0 is not within bounds. {y_min} is not between {y_min_lb} and {y_min_ub}')
        self.assertTrue(y_max_lb <= y_max <= y_max_ub,
                        f'y_max at t0 is not within bounds. {y_max} is not between {y_max_lb} and {y_max_ub}')
