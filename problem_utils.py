from copy import deepcopy
from random import randint, uniform

import search_config as cfg

from simulation_runner import run_simulation

total_mlco_messages = cfg.total_mlco_messages
total_obstacles_per_message = cfg.total_obstacles_per_message

# Initialization functions


def initialize_mlco(class_):
    """Initializes an mlco individual.
    """
    # Initialize number of obstacles per message.
    car_per_message = []
    person_per_message = []
    for i in range(total_mlco_messages):
        car_per_message.append(0)
        person_per_message.append(0)

    for i in range(total_mlco_messages):
        car_per_message[i] = randint(0, total_obstacles_per_message-1)
        person_per_message[i] = randint(
            0, total_obstacles_per_message-car_per_message[i]-1)

    not_an_obstacle_list = [0, 0, 0, 0, -1]

    mlco_list = []

    # Create obstacle message lists.
    for i in range(total_mlco_messages):
        obstacle_message = []
        obstacle_label = 0  # 'vehicle'
        car_obstacle_message = create_obstacle_message(
            label=obstacle_label, num=car_per_message[i])
        obstacle_message += car_obstacle_message

        obstacle_label = 1  # 'person'
        person_obstacle_message = create_obstacle_message(
            label=obstacle_label, num=person_per_message[i])
        obstacle_message += person_obstacle_message

        for i in range(total_obstacles_per_message - len(obstacle_message)):
            obstacle_message.append(not_an_obstacle_list)

        mlco_list.append(obstacle_message)
    return class_(mlco_list)


def create_obstacle_message(label, num):
    """
    Creates an `obstacle_message` for a specific obstacle (`label`).
    An obstacle_message is a number `num` of obstacles.
    """
    obstacle_message = []
    for i in range(num):
        obstacle_message.append(create_single_obstacle(obstacle_label=label))
    return obstacle_message


def create_single_obstacle(obstacle_label, frame_width: float = 800, frame_height: float = 600, min_bbox_size: float = 50):
    """Create an obstacle list given a label.
    """
    x_min = uniform(0.0, frame_width - min_bbox_size)
    x_max = uniform(x_min + min_bbox_size, frame_width)
    y_min = uniform(0.0, frame_height - min_bbox_size)
    y_max = uniform(y_min + min_bbox_size, frame_height)

    obstacle_list = [x_min, x_max, y_min, y_max]

    # FIXME: Ensure that the labels exist in OBSTACLE_LABELS.
    obstacle_list.append(obstacle_label)

    return obstacle_list

# Define the problem's joint fitness function.


def problem_joint_fitness(scenario, mlco):
    """Joint fitness evaluation which runs the simulator.
    """
    scenario_deepcopy = deepcopy(scenario)
    scenario_deepcopy = list(scenario_deepcopy)
    mlco_deepcopy = deepcopy(mlco)
    mlco_deepcopy = list(mlco_deepcopy)

    DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = run_simulation(
        scenario_deepcopy, mlco_deepcopy)

    # return DfP_max
    return DfV_max
