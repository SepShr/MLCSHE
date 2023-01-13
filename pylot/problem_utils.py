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
import copy
from copy import deepcopy
from deap import tools
import logging
from random import randint, random, uniform

import pylot.search_config as cfg
# from simulation_manager_cluster import prepare_for_computation, start_computation

# from simulation_runner import run_simulation
from src.utils.utility import initialize_hetero_vector, mutate_flat_hetero_individual

total_mlco_messages = cfg.total_mlco_messages
total_obstacles_per_message = cfg.total_obstacles_per_message
obstacle_enumLimits = cfg.obstacle_label_enum_limits
obs_traj_enum_limits = [
    [-1, 1], [0.0, 400.0], [0.0, 400.0], [0, 750], [0, 550], [
        50, 800], [50, 600], [0, 750], [0.0, 550], [50, 800], [50, 600]
]

logger = logging.getLogger(__name__)
# Initialization functions


def initialize_mlco(class_, num_traj: int = cfg.num_trajectories, duration: int = cfg.duration):
    pass
    trajectory_list = []
    for _ in range(num_traj):
        traj_label = randint(-1, 1)
        trajectory_list.append(create_obs_traj(traj_label, duration))
    return class_(trajectory_list)


def create_obs_traj(
    obstacle_label: int,
    duration: int = cfg.duration,
    min_duration: int = cfg.min_trajectory_duration,
    frame_width=cfg.frame_width,
    frame_height=cfg.frame_height,
    min_bbox_size=cfg.min_boundingbox_size
):
    """mlco = [
        [
            label: [-1, 1],  # 0 -> vehicle, 1 -> person,
            t0: [0, duration],
            t1: [0, duration],
            bbox_t0_x_min: [0, 750],  # Assuming a minimum of 50 px for bbox size.
            bbox_t0_x_max: [50, 800],
            bbox_t0_y_min: [0, 550],
            bbox_t0_y_max: [50, 600],
            bbox_t1_x_min: [0, 750],
            bbox_t1_x_max: [50, 800],
            bbox_t1_y_min: [0, 550],
            bbox_t1_y_max: [50, 600]
        ], 
    ...]
    """
    if obstacle_label == -1:
        return cfg.null_trajectory
    else:
        # NOTE: No trajectory is started less than 10 time units to the end.
        # t0 = round(uniform(0.0, duration-10.0), 3)
        # t1 = round(uniform(t0, duration), 3)
        t0 = randint(0, duration - min_duration)
        t1 = randint(t0 + min_duration, duration)
        obstacle_trajectory = [obstacle_label, t0, t1]
        # Add the bounding box at t0.
        obstacle_trajectory += create_random_bbox(
            frame_width, frame_height, min_bbox_size)
        # Add the bounding box at t1.
        obstacle_trajectory += create_random_bbox(
            frame_width, frame_height, min_bbox_size)
        return obstacle_trajectory


def create_random_bbox(frame_width=cfg.frame_width,
                       frame_height=cfg.frame_height,
                       min_bbox_size=cfg.min_boundingbox_size):
    # x_min = randint(0, frame_width - min_bbox_size)
    # x_max = randint(x_min + min_bbox_size, frame_width)
    # y_min = randint(0, frame_height - min_bbox_size)
    # y_max = randint(y_min + min_bbox_size, frame_height)

    x_min = round(uniform(0.0, frame_width - min_bbox_size), 3)
    x_max = round(uniform(x_min + min_bbox_size, frame_width), 3)
    y_min = round(uniform(0.0, frame_height - min_bbox_size), 3)
    y_max = round(uniform(y_min + min_bbox_size, frame_height), 3)

    return [x_min, x_max, y_min, y_max]


def mutate_mlco(
        mlco, mutgmu, mutgsig, mutgpb, mutipb, traj_enumLimits=cfg.traj_enum_limits):
    """Mutates a mlco individual."""
    # TODO: Use multiprocessing for each obstacle
    _class = type(mlco)

    mlco_deepcopy = deepcopy(mlco)
    mutated_mlco = []

    for trajectory in mlco_deepcopy:
        trajectory = mutate_traj(
            trajectory, traj_enumLimits,
            mutgmu, mutgsig, mutgpb, mutipb)
        mutated_mlco.append(trajectory)

    return _class(mutated_mlco)


def mutate_traj(mlco_element, traj_enum_limit, mutgmu, mutgsig, mutgpb, mutipb):
    """Mutates an atomic element the mlco message.
    Currently the shape of the mlco_element is assumed to be:
    `[int, flt, flt, int, int, int, int, int, int, int, int]`
    for now:
    `[int, int, int, flt, flt, flt, flt, flt, flt, flt, flt]`

    NOTE: The implementation must change when the target mlc changes!
    """
    mlco_element_label = [mlco_element[0]]

    mlco_element_time = mlco_element[1:3]
    mlco_element_bbox_t0 = mlco_element[3:7]
    mlco_element_bbox_t1 = mlco_element[7:]

    mutated_label = list(tools.mutUniformInt(
        mlco_element_label, traj_enum_limit[0][0], traj_enum_limit[0][1], mutipb)[0])

    # if mutated_label == -1:
    if mutated_label[0] == -1:
        return cfg.null_trajectory
    else:
        if mlco_element_label == -1:
            return create_obs_traj(mutated_label[0])
        else:
            mutated_time = mutate_time(
                mlco_element_time, mutgpb)
            mutated_bbox_t0 = mutate_bbox(
                mlco_element_bbox_t0, mutgmu, mutgsig, mutgpb)
            mutated_bbox_t1 = mutate_bbox(
                mlco_element_bbox_t1, mutgmu, mutgsig, mutgpb)

            return mutated_label + mutated_time + mutated_bbox_t0 + mutated_bbox_t1


def mutate_time(time_list, mutipb, duration: int = cfg.duration, min_duration: int = cfg.min_trajectory_duration):
    # mutated_time = list(tools.mutUniformInt(
    #     time_list, low=, up=, indpb=mutgpb)[0])
    # if mutated_time[0] >= mutated_time[1]:
    #     if mutated_time[0] < 60.0:
    #         mutated_time[1] += 50.0
    #     else:
    #         mutated_time[0] += -50.0

    # initialize mutated_time
    mutated_time = copy.deepcopy(time_list)

    # mutate the start time
    if random() <= mutipb:
        mutated_time[0] = randint(0, duration)

    # mutation the end time
    if random() <= mutipb:
        # no guarantee that "mutated_time[0] + min_duration < duration"
        # therefore, it's better to randomly generate a value and fix it later
        mutated_time[0] = randint(0, duration)

    # Repair values to have min_duration between the start and end times
    if mutated_time[1] - mutated_time[0] < min_duration:
        diff = min_duration - (mutated_time[1] - mutated_time[0])
        if mutated_time[1] + diff <= duration:
            mutated_time[1] += diff
        elif mutated_time[0] - diff >= 0:
            mutated_time[0] -= diff
        else:
            mutated_time[0] = 0
            mutated_time[1] = min_duration

    return mutated_time


def mutate_bbox(bbox: list, mutgmu, mutgsig, mutgpb):
    mutated_bbox = list(tools.mutGaussian(
        bbox, mu=mutgmu, sigma=mutgsig, indpb=mutgpb)[0])
    mutated_bbox = repair_obstacle_bbox(mutated_bbox)
    for i in range(len(mutated_bbox)):
        mutated_bbox[i] = round(mutated_bbox[i], 3)

    return mutated_bbox


def mutate_scenario(
        scenario, scenario_enumLimits,
        mutbpb, mutgmu, mutgsig, mutgpb, mutipb):
    """Mutates a scenario individual.
    """
    return mutate_flat_hetero_individual(scenario, scenario_enumLimits,
                                         mutbpb, mutgmu,
                                         mutgsig, mutgpb, mutipb)


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


def compute_safety_req_value(simulator, scenario, mlco):
    """Joint fitness evaluation which runs the simulator.
    """
    scenario_deepcopy = deepcopy(scenario)
    scenario_deepcopy = list(scenario_deepcopy)
    mlco_deepcopy = deepcopy(mlco)
    mlco_deepcopy = list(mlco_deepcopy)

    DfC_min, DfV_min, DfP_min, DfM_min, DT_max, traffic_lights_max = simulator.run_simulation(
        scenario_deepcopy, mlco_deepcopy)

    # logger.info('safety_req_value={}'.format(DfC_min))
    logger.info('safety_req_value={}'.format(DfV_min))

    # return DfC_min
    return DfV_min


def trajectory_to_obstacle(trajectory, duration):
    """Creates a sequence (list) of obstacles for a given `trajectory`.
    The size of the sequence is determined by `duration`. Where an 
    obstacle does not exist, an empty list is returned.

    A trajectory has the following shape:
    trajectory = [label, t0, t1, x_min_t0, x_max_t0, y_min_t0, y_max_t0,
    x_min_t1, x_max_t1, y_min_t1, y_max_t1]

    and an obstacle has the following shape:
    obstacle = [x_min, x_max, y_min, y_max, label]
    """
    assert len(trajectory) == 11

    obs_label = trajectory[0]
    assert obs_label in [-1, 0, 1]

    # Initialize the obstacle sequence.
    obs_seq = [[] for _ in range(duration)]

    if obs_label == -1:
        pass  # Discard if the label in null, i.e., -1.
    else:
        t0 = trajectory[1]  # Start time of the trajectory.
        t1 = trajectory[2] - 1  # End time of the trajectory.

        num_msg = t1 - t0  # Number of steps between t0 and t1.

        bbox_t0 = trajectory[3:7]  # Start bounding box.
        bbox_t1 = trajectory[7:]  # End bounding box.

        # Calculate the size of the steps for each bbox paramter.
        steps = [(element1 - element2)/num_msg for (element1,
                                                    element2) in zip(bbox_t1, bbox_t0)]

        # Add the start and end bbox to the obstacle sequence.
        obs_seq[t0].append(bbox_t0 + [obs_label])
        obs_seq[t1].append(bbox_t1 + [obs_label])

        # Calculate the in between obstacles using linear interpolation.
        for i in range(t0 + 1, t1):
            obs_seq[i].append(
                [ele1 + ele2 for (ele1, ele2) in zip(obs_seq[i-1][0][1:], steps)] + [obs_label])

    return obs_seq

# NOTE: Use initialize_mlco for testing!


def mlco_to_obs_seq(mlco, duration=cfg.duration):
    # list_of_obs_sequences = []
    # for trajectory in mlco:
    #     list_of_obs_sequences.append(
    #         trajectory_to_obstacle(trajectory, duration))

    list_of_obs_sequences = [trajectory_to_obstacle(
        trajectory, duration) for trajectory in mlco]

    obs_seq_combined = []
    for i in range(duration):
        obs_instance_combined = []
        for sequence in list_of_obs_sequences:
            obs_instance_combined += sequence[i]
        obs_seq_combined.append(obs_instance_combined)

    for obs in obs_seq_combined:
        if obs == []:
            obs.append(cfg.null_obstacle)

    return obs_seq_combined
