'''
## Encodings
mlco = [
        [
            label: 0 -> vehicle, 1 -> person,
            t0: [0, ??],
            t1: [0, ??],
            bbox_t0_x_min: [0, 750], # Assuming a minimum of 50 px for bbox size.
            bbox_t0_y_min: [0, 550],
            bbox_t0_x_max: [50, 800],
            bbox_t0_y_max: [50, 600],
            bbox_t1_x_min: [0, 750],
            bbox_t1_y_min: [0, 550],
            bbox_t1_x_max: [50, 800],
            bbox_t1_y_max: [50, 600],
        ], 
    ...]

- Example:
mlco_1 = [0, 5., 352.5, 102., 176.6, 253.9, 396.3, 3.7, 57.1, 509.2, 590.]
mlco_2 = [1, 253.2, 466., 638.3, 478.1, 800., 599.5, 747.6, 800., 166.4, 301.3]

scen = [
    time_of_day: 0 -> noon, 1 -> sunset, 2 -> night,
    weather: 0-> clear, 1 -> cloudy, 2 -> wet, 3 -> wet cloudy, 4 -> medium rain, 5 -> hard rain, 6 -> soft rain,
    pedestrian: 0 -> 0, 1 -> 18,
    road curve: 0 -> straight, 1 -> right, 2 -> left, 3 -> not clear!,
    road ID: 0, 1, 2,
    road length: 0, 1, 2 (from shortest to longest),
    path: 0 -> follow road, 1 -> 1st exit, 2 -> 2nd exit
    ]

- Example:
scen_1 = [0, 2, 1, 2, 0, 1, 1]
scen_2 = [2, 6, 0, 3, 2, 0, 0]

cs = [scen, mlco]
- A CS is a triple-nested heterogeneous list. --> flatten 3 times.

-Example:
cs = [[scen_1, mlco_1], [scen_2, mlco_2], [scen_1, mlco_2], [scen_2, mlco_1]]
'''
from copy import deepcopy
from deap import tools
import logging
from random import randint, uniform

import search_config as cfg

# from simulation_runner import run_simulation
from src.utils.utility import mutate_flat_hetero_individual

total_mlco_messages = cfg.total_mlco_messages
total_obstacles_per_message = cfg.total_obstacles_per_message
obstacle_enumLimits = cfg.obstacle_label_enum_limits

logger = logging.getLogger(__name__)
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

    not_an_obstacle_list = [0.0, 0.0, 0.0, 0.0, -1]

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


def create_single_obstacle(
        obstacle_label, frame_width: float = cfg.frame_width,
        frame_height: float = cfg.frame_height,
        min_bbox_size: float = cfg.min_boundingbox_size):
    """Create an obstacle list given a label.
    """
    x_min = round(uniform(0.0, frame_width - min_bbox_size), 3)
    x_max = round(uniform(x_min + min_bbox_size, frame_width), 3)
    y_min = round(uniform(0.0, frame_height - min_bbox_size), 3)
    y_max = round(uniform(y_min + min_bbox_size, frame_height), 3)

    obstacle_list = [x_min, x_max, y_min, y_max]

    # FIXME: Ensure that the labels exist in OBSTACLE_LABELS.
    obstacle_list.append(obstacle_label)

    return obstacle_list


def mutate_scenario(
        scenario, scenario_enumLimits,
        mutbpb, mutgmu, mutgsig, mutgpb, mutipb):
    """Mutates a scenario individual.
    """
    return mutate_flat_hetero_individual(scenario, scenario_enumLimits,
                                         mutbpb, mutgmu,
                                         mutgsig, mutgpb, mutipb)


def mutate_mlco(
        mlco, mutgmu, mutgsig, mutgpb, mutipb, enumLimits=obstacle_enumLimits):
    """Mutates a mlco individual."""
    # Use multiprocessing for each obstacle
    num_messages = len(mlco)
    # print('num messages is: ' + str(num_messages))
    num_obstacles_per_message = len(mlco[0])
    # print(f'obs_per_msg is: {num_obstacles_per_message}')

    for i in range(num_messages):
        for j in range(num_obstacles_per_message):
            mlco[i][j] = mutate_mlco_element(
                mlco[i][j], enumLimits,
                mutgmu, mutgsig, mutgpb, mutipb)
            # print(f'mlco[{i}][{j}] is: {mlco[i][j]}')

    return mlco


def mutate_mlco_element(mlco_element, label_enum_limit, mutgmu, mutgsig, mutgpb, mutipb):
    """Mutates an atomic element the mlco message.
#     Currently the shape of the mlco_element is assumed to be:
#     `[flt, flt, flt, flt, int]`

#     NOTE: The implementation must change when the target mlc changes!
    """
    mlco_element_label = [mlco_element[4]]
    # mlco_element_label = mlco_element[4]
    mlco_element_bbox = mlco_element[:4]

    # mutated_label = randint(label_enum_limit[0], label_enum_limit[1])
    mutated_label = list(tools.mutUniformInt(
        mlco_element_label, label_enum_limit[0], label_enum_limit[1], mutipb)[0])

    # if mutated_label == -1:
    if mutated_label[0] == -1:
        return [0.0, 0.0, 0.0, 0.0, -1]
    else:
        if mlco_element_label == -1:
            # if mlco_element_label[0] == -1:
            # return create_single_obstacle(mutated_label)
            return create_single_obstacle(mutated_label[0])
        else:
            mutated_mlco_element = list(tools.mutGaussian(
                mlco_element_bbox, mu=mutgmu, sigma=mutgsig, indpb=mutgpb)[0])

            repaired_mlco_element = repair_obstacle_bbox(mutated_mlco_element)

            for i in range(len(repaired_mlco_element)):
                repaired_mlco_element[i] = round(repaired_mlco_element[i], 3)

            # return repaired_mlco_element + [mutated_label]
            return mutated_mlco_element + mutated_label


def repair_obstacle_bbox(
        obstacle, max_width=cfg.frame_width, max_height=cfg.frame_height, min_bbox_size=cfg.min_boundingbox_size):
    """Checks whether the mutated obstacle is valid. If not, it will
    repair them.

    Assumptions:
    1. Obstacle has the following shape: 
    `[x_min, x_max, y_min, y_max]`

    2. The min bound for x and y is 0.0.
    """
    # Basic check and repair.
    if obstacle[1] < obstacle[0]:
        obstacle[1] = obstacle[0] + min_bbox_size
        # logger.debug('x_max was less than x_min.')

    if obstacle[3] < obstacle[2]:
        obstacle[3] = obstacle[2] + min_bbox_size
        # logger.debug('y_max was less than y_min.')

    # Check and repair the bound of x_min.
    if obstacle[0] < 0.0:
        obstacle[0] = 0.0
        # logger.debug("x_min was out of bound.")

    # Check and repair the bound of x_max.
    if obstacle[1] > max_width:
        obstacle[1] = max_width
        # logger.debug("x_max was out of bound.")

    # Check and repair the bound of y_min.
    if obstacle[2] < 0.0:
        obstacle[2] = 0.0
        # logger.debug("y_min was out of bound.")

    # Check the and repair bound of y_max.
    if obstacle[3] > max_height:
        obstacle[3] = max_height
        # logger.debug("y_max was out of bound.")

    # Check and repair the min_boundingbox_size.
    if obstacle[1] - obstacle[0] < min_bbox_size:
        if obstacle[1] < max_width - min_bbox_size:
            obstacle[1] += min_bbox_size - (obstacle[1] - obstacle[0])
        else:
            obstacle[0] += (obstacle[1] - obstacle[0]) - min_bbox_size
        # logger.debug("bbox width was less than the min_bbox size.")

    if obstacle[3] - obstacle[2] < min_bbox_size:
        if obstacle[3] < max_height - min_bbox_size:
            obstacle[3] += min_bbox_size - (obstacle[3] - obstacle[2])
        else:
            obstacle[2] += (obstacle[3] - obstacle[2]) - min_bbox_size
        # logger.debug("bbox height was less than the min_bbox size.")

    return obstacle


# TEST
# mlco = [[[260.0, 480.0, 300.0, 400.0, 0], [0.0, 0.0, 0.0, 0.0, -1]], [[0.0, 0.0, 0.0, 0.0, -1], [300.0, 490.0, 350.0, 410.0, 0]]]
# obstacle_enumLimits = [-1, 1]
# mutbpb = 0.5
# mutgmu = 0
# mutgsig = 0.125
# mutgpb = 0.5
# mutipb = 0.5

# Define the problem's joint fitness function.


def problem_joint_fitness(simulator, scenario, mlco):
    """Joint fitness evaluation which runs the simulator.
    """
    scenario_deepcopy = deepcopy(scenario)
    scenario_deepcopy = list(scenario_deepcopy)
    mlco_deepcopy = deepcopy(mlco)
    mlco_deepcopy = list(mlco_deepcopy)

    DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = simulator.run_simulation(
        scenario_deepcopy, mlco_deepcopy)

    logger.info('joint_fitness_value={}'.format(DfC_min))

    return DfC_min
