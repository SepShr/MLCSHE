"""
This file contains the adaptor classes for the algorithm to interface
with the simulation setup.

The classes and their encode and decode metods have to be modified for
every different problem.
"""
import copy
import os
import pickle
import signal
import subprocess as sub
import time

import numpy as np

WEATHER_PRESETS = ['ClearNoon', 'ClearSunset', 'CloudyNoon',
                   'CloudySunset', 'HardRainNoon', 'HardRainSunset',
                   'MidRainSunset', 'MidRainyNoon', 'SoftRainNoon',
                   'SoftRainSunset', 'WetCloudyNoon', 'WetCloudySunset',
                   'WetNoon', 'WetSunset']

SIMULATION_DURATION = 30  # Simulation duration in seconds.

CONTAINER_NAME = "pylot"

CARLA_PATH = "/home/erdos/workspace/pylot/scripts/run_simulator.sh"

# PYLOT_PATH = "/home/erdos/workspace/pylot/pylot.py"
PYLOT_PATH = "/home/sepehr/AV/MLCSHE/MLCSHE/run_pylot.sh"

# FLAG_FILE = "--flagfile=configs/mlcshe/mlcshe_config.conf"


FINISHED_FILE_PATH = '/home/erdos/workspace/results/finished.txt'


def translate_scenario_list(scenario_list):
    """Sets the value of the `flags.simulator_weather` based on the
    `scenario_list`.
    """
    # FIXME: Exception handling.
    weather_flag = "--simulator_weather=" + \
        WEATHER_PRESETS[scenario_list[0]] + "\n"

    # scenario_flag = weather_flag

    scenario_flag = {}
    scenario_flag['simulator_weather'] = weather_flag

    return scenario_flag


def translate_mlco_list(mlco_list, container_name=CONTAINER_NAME):
    """Transfers the `mlco_list` to `container_name` and updates
    the flags for pylot.
    """
    # Write the mlco_list to a file.
    pickle_to_file(mlco_list, 'mlco_list.pkl')

    container_id = find_container_id(container_name)

    # FIXME: Make the paths relative.
    # Copy the file inside the Pylot container.
    source_path = '/home/sepehr/AV/MLCSHE/MLCSHE/mlco_list.pkl'
    destination_path = \
        '/home/erdos/workspace/pylot/dependencies/mlco/mlco_list.pkl'
    copy_to_container(container_id, source_path, destination_path)

    # Setup the flags to be passed on to pylot.
    mlco_operator_flag = "--lane_detection_type=highjacker\n"
    mlco_list_path_flag = "--mlco_list_path=" + destination_path + "\n"

    mlco_flag = {}
    mlco_flag['lane_detection_type'] = mlco_operator_flag
    mlco_flag['mlco_list_path'] = mlco_list_path_flag

    return mlco_flag


def pickle_to_file(item_to_be_pickled, file_name: str):
    """Pickles an object and adds it to a file.
    """
    file = open(file_name, 'wb')
    pickle.dump(item_to_be_pickled, file)
    file.close()


def start_container(container_name: str = 'pylot'):
    """Starts a docker container with the name `container_name`.
    """
    cmd = ['docker', 'start', container_name]
    docker_start_proc = sub.run(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    return docker_start_proc.returncode


def stop_container(container_name: str = 'pylot'):
    """Starts a docker container with the name `container_name`.
    """
    cmd = ['docker', 'stop', container_name]
    docker_start_proc = sub.run(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    return docker_start_proc.returncode


def copy_to_container(
        container_id: str, source_path: str, destination_path: str):
    """Copies a file from `source_path` to `destination_path`
    inside a `container_id`.
    """
    container_id_with_path = container_id + ":" + destination_path
    copy_cmd = ['docker', 'cp', source_path, container_id_with_path]
    sub.run(copy_cmd)


def find_container_id(container_name: str):
    """Find the ID of a running container, given its name
    (`container_id`).
    """
    # FIXME: Handle exception.

    # Make sure that the Pylot container is running and find the container ID.
    containers = sub.check_output(['docker', 'ps'], universal_newlines=True)
    containers_split = containers.split('\n')
    for i in containers_split:
        if i.__contains__(container_name):
            container_id = i[:12]
    return container_id


def update_config_file(
        simulation_flag, base_config_file="base_config.conf",
        simulation_config_file="mlcshe_config.conf"):
    """Updates `base_config_file` according to `simulation_flag` and
    writes it to `simulation_config_file`.
    """
    # Find the lines that contain proper flags and updates them.
    base_config = open(base_config_file, 'rt')
    simulation_config = open(simulation_config_file, 'wt')
    for line in base_config:
        for key in simulation_flag:
            if line.__contains__(key):
                simulation_config.write(simulation_flag[key])
                break
        else:
            simulation_config.write(line)

    base_config.close()
    simulation_config.close()


def update_sim_config(scenario_list, mlco_list, container_name: str = CONTAINER_NAME):
    """Updates the configuration file that is used by Pylot to run
    the simulation. It adds the flags based on scenario and mlco to
    the configuation file, and copies it into the container.
    """
    # Set flags based on scenario_list and mlco_list.
    scenario_flag = translate_scenario_list(scenario_list)
    mlco_flag = translate_mlco_list(mlco_list)

    # Update the simulation flags.
    simulation_flag = {}
    simulation_flag.update(scenario_flag)
    simulation_flag.update(mlco_flag)

    # Test data.
    # simulation_flag = {'lane_detection_type': "--lane_detection_type=highjacker\n",
    #                    'mlco_list_path': "--mlco_list_path=/home/erdos/workspace/pylot/dependencies/mlco/mlco_list.pkl\n",
    #                    'simulator_weather': "--simulator_weather=RainyNight\n"}

    # Update the config file.
    update_config_file(simulation_flag)

    # Copy the config file to the container.
    container_id = find_container_id(container_name)
    # FIXME: Make the paths relative.
    source_path = "/home/sepehr/AV/MLCSHE/MLCSHE/mlcshe_config.conf"
    destination_path = \
        "/home/erdos/workspace/pylot/configs/mlcshe/mlcshe_config.conf"
    copy_to_container(container_id, source_path, destination_path)


def run_command_in_shell(command, verbose: bool = True):
    """Runs a command in shell via subprocess. Also if verbose is set
    to `True`, it will print out the `stdout` or `stderr` messages,
    depending on the successful run of the process.
    """
    if verbose:
        print(f'Running {command} in shell...')

    proc = sub.Popen(command, shell=True)
    #  stdout=sub.PIPE, stderr=sub.PIPE,

    # # Verify successful execution of the command.
    # if proc.returncode != 0:
    #     print("Process FAILED.\nThe error was:")
    #     for line in proc.stderr.decode("utf-8").split('\n'):
    #         print(line.strip())
    # else:
    #     print("Process successfully executed.")
    #     if verbose:
    #         for line in proc.stdout.decode("utf-8").split('\n'):
    #             print(line.strip())

    return proc


def run_carla(
        container_name: str = CONTAINER_NAME, carla_run_path=CARLA_PATH):
    """Runs the carla simulator which is inside `container_name`.
    """
    carla_run_command = "nvidia-docker exec -it " + \
        container_name + " " + carla_run_path

    carla_proc = run_command_in_shell(carla_run_command)

    return carla_proc


def run_pylot(run_pylot_path: str = PYLOT_PATH):
    """Runs a script that runs pylot inside a container.
    """
    pylot_run_command = run_pylot_path

    pylot_proc = run_command_in_shell(pylot_run_command)

    return pylot_proc

# COMMENTED FOR THE TEST RUN.
# def remove_finished_file(finished_file_path: str = FINISHED_FILE_PATH):
#     """Removes `finished.txt`
#     """
#     cmd = ['docker', 'exec', 'pylot', 'rm', '-rf',
#            finished_file_path]
#     rm_finished_file_proc = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
#     if os.path.exists("finished.txt"):
#         os.remove("finished.txt")


def reset_sim_setup(container_name: str = 'pylot'):
    """Resets the simulation setup, i.e., restarts the container, and
    removes old finished files.
    """
    stop_container(container_name)
    start_container(container_name)

    # COMMENTED FOR THE TEST RUN.
    # remove_finished_file()


def run_carla_and_pylot():
    """Runs carla and pylot and ensures that they are in sync.
    """
    run_carla()
    time.sleep(20)
    run_pylot()
    time.sleep(20)
    # FIXME: Confirm successful Carla and Pylot are in sync.


def scenario_finished():
    """
    """
    return False


def run_single_scenario(scenario_list, mlco_list):
    """Ensures that the simulation setup is ready, updates simulation
    configuration given `scenario_list` and `mlco_list` and, runs the
    simulation, records its output and 
    """
    scenario_list_deepcopy = copy.deepcopy(scenario_list)
    mlco_list_deepcopy = copy.deepcopy(mlco_list)

    reset_sim_setup()
    update_sim_config(scenario_list_deepcopy, mlco_list_deepcopy)
    run_carla_and_pylot()

    # Monitor scenario execution and end it when its over.
    counter = 0

    while (True):
        counter = counter + 1
        time.sleep(1)
        if (counter > SIMULATION_DURATION) or scenario_finished():
            print("End of simulation")
            stop_container()
            break
        else:
            print("Scenario execution in progress...\ncounter = " +
                  str(counter))

    # # Copy the results of the simulation.
    # copy_sim_results()
    # results_file_name = 'Results/' + str(fv)
    # if os.path.exists(results_file_name):
    #     get_values()


def simulate_and_evaluate_vehicle_distance(scenario_list, mlco_list):
    """Runs a simulator using `scenario_list` and `mlco_list`
    and evaluates the result of the simulation given its safety
    requirement metric.
    """
    # Update the configuration of the simulation and the serialized mlco_list
    update_sim_config(scenario_list, mlco_list)

    # Run Carla and Pylot in the docker container with appropriate config
    carla_proc = run_carla()
    pylot_proc = run_pylot()

    # Confirm that the pylot and carla are interfacing.

    # Either measure jfit at run time or assess the report or log after the simulation.
    # extract_jfit_value_from_sim()

    # Option #1: Record information from pylot during run and evaluate that.
    # Option #2: Get a handle on Carla, add a sensor to the vehicle and record the output.

    # Kill the processes.
    os.kill(carla_proc.pid, signal.SIGINT)
    os.kill(pylot_proc.pid, signal.SIGTERM)
    # FIXME: The processes spawned in the docker have a different pid and
    # will not be killed using the above command.

    # Reset the simulation setup.
    reset_sim_setup()


def main():
    scenario_list = [4]
    mlco_list = [
                [0,  [1, 4, 5, 2, 3, 1], [2, 7, 2, 4, 5, 2]],
                [1,  [2, 4, 5, 2, 3, 1], [3, 7, 2, 4, 6, 2]],
                [2,  [3, 4, 5, 2, 3, 1], [4, 7, 2, 4, 5, 2]]
    ]

    run_single_scenario(scenario_list, mlco_list)


if __name__ == "__main__":
    main()

    # ## COMMENTED IMPLEMENTATION ###

    # import carla
    # import erdos
    # from erdos import Operator, Stream
    # import pylot
    # from pylot.utils import Location, Rotation, Transform, Vector3D
    # from pylot.perception.detection.lane import Lane
    # from absl import flags
    # import cv2

    # FLAGS = flags.FLAGS

    # SCENARIO_BOUNDS = [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0],
    #                    [0.0, 100.0], [0.0, 360.0], [-90.0, 90.0],
    #                    [0.0, 100.0], [0.0, 100.0], [0.0, np.inf],
    #                    [0.0, np.inf]]

    # # Carla wrapper that can translate scenario individuals
    # class Environement:
    #     """
    #     This class creates a carla simulation instance and translates
    #     MLCSHE's Scenarios to Carla World parameters.
    #     """

    #     def __init__(self, host="localhost", port=2000, timeout=2.0):
    #         # Get client.
    #         self.client = carla.Client(host, port)
    #         self.client.set_timeout(timeout)
    #         # Get world.
    #         self.world = self.client.get_world()

    #         # self.actor_list = []

    #     # Decode the scenario individual to simulation parameters
    #     def set_scenario(self, scenario_list):
    #         """
    #         Sets the World parameters based on a MLCSHE Scenario.

    #         The order mapping of World parameters to scenario_list indices is
    #         as follows:

    #         [0] Cloudiness amount [0.0, 100.0]
    #         [1] Precipitation (Rain) amount [0.0, 100.0]
    #         [2] Precipitation deposits (Puddles) amount [0.0, 100.0]
    #         [3] Wind intensity [0.0, 100.0]
    #         [4] Sun azimuth [0.0, 360.0]
    #         [5] Sun altitude [-90.0, 90.0]
    #         [6] Fog density [0.0, 100.0]
    #         [7] Fog Distance [0.0, inf)
    #         [8] Wetness intensity [0.0, 100.0]
    #         [9] Fog falloff [0, inf)
    #         """
    #         # Set weather parameters.
    #         cloudiness, precipitation, precipitation_deposits,\
    #             wind_intensity, sun_azimuth_angle, sun_altitude_angle,\
    #             fog_density, fog_distance, wetness, fog_falloff = scenario_list
    #         self.weather = carla.WeatherParameters(
    #             cloudiness=cloudiness,
    #             precipitation=precipitation,
    #             precipitation_deposits=precipitation_deposits,
    #             wind_intensity=wind_intensity,
    #             sun_azimuth_angle=sun_azimuth_angle,
    #             sun_altitude_angle=sun_altitude_angle,
    #             fog_density=fog_density,
    #             fog_distance=fog_distance,
    #             wetness=wetness,
    #             fog_falloff=fog_falloff
    #         )

    #         # Set light parameters.
    #         # Skipped for now.

    #         # Set world settings.
    #         # Skipped for now.

    #         # Set actors.
    #         # Skipped for now.
    #         # self.model_3 = self.blueprint_library.filter("model3")[0]
    #         # self.blueprint_library = self.world.get_blueprint_library()
    #         # self.transform = \
    #               random.choice(self.world.get_map().get_spawn_points())
    #         # self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
    #         # self.actor_list.append(self.vehicle)

    #     # Encode the scenario datatype from the setup to a hetergeneous list.

    #     def get_scenario_list(self) -> list:
    #         """
    #         Returns a scenario_list based on the World parameters.
    #         """
    #         self.weather = self.world.get_weather()
    #         scenario_list = list(self.weather.cloudiness,
    #                              self.weather.precipitation,
    #                              self.weather.precipitation_deposits,
    #                              self.weather.wind_intensity,
    #                              self.weather.sun_azimuth_angle,
    #                              self.weather.sun_altitude_angle,
    #                              self.weather.fog_density,
    #                              self.weather.fog_distance,
    #                              self.weather.wetness,
    #                              self.weather.fog_falloff)

    #         # Actor-related parameters should be added.
    #         # Skipped for now.

    #         return scenario_list

    #     # Reset or reload the world.
    #     def reset(self):
    #         pass
    #         # self.client.apply_batch(
    #               [carla.command.DestroyActor(x) for x in vehicles_list])

    # Implementation meant to handle mlco individual translation.
    # class LaneDetectionHighjackerOperator(erdos.Operator):
    #     """
    #     """

    #     def __init__(self, camera_stream: erdos.ReadStream,
    #                  detected_lanes_stream: erdos.WriteStream, mlco_list, flags):
    #         self.frame_index = 0  # Initialize frame_index
    #         self.mlco_list = mlco_list

    #         camera_stream.add_callback(self.on_camera_frame,
    #                                    [detected_lanes_stream])
    #         self._flags = flags
    #         self._logger = erdos.utils.setup_logging(self.config.name,
    #                                                  self.config.log_file_name)

    #         #

    #     @staticmethod
    #     def connect(camera_stream: erdos.ReadStream):
    #         """Connects the operator to other streams.
    #         Args:
    #             camera_stream (:py:class:`erdos.ReadStream`): The stream on which
    #                 camera frames are received.
    #         Returns:
    #             :py:class:`erdos.WriteStream`: Stream on which the operator sends
    #             :py:class:`~pylot.perception.messages.LanesMessage` messages.
    #         """
    #         detected_lanes_stream = erdos.WriteStream()
    #         return [detected_lanes_stream]

    #     @erdos.profile_method()
    #     def on_camera_frame(self, msg: erdos.Message,
    #                         detected_lanes_stream: erdos.WriteStream):
    #         """Invoked whenever a frame message is received on the stream.
    #         Args:
    #             msg: A :py:class:`~pylot.perception.messages.FrameMessage`.
    #             detected_lanes_stream (:py:class:`erdos.WriteStream`): Stream on
    #                 which the operator sends
    #                 :py:class:`~pylot.perception.messages.LanesMessage` messages.
    #         """
    #         self._logger.debug('@{}: {} received message'.format(
    #             msg.timestamp, self.config.name))
    #         assert msg.frame.encoding == 'BGR', 'Expects BGR frames'

    #         # # Optional: reformat the image data as an RGB numpy array.
    #         # image = cv2.resize(msg.frame.as_rgb_numpy_array(), (512, 256),
    #         #                    interpolation=cv2.INTER_LINEAR)
    #         # image = image / 127.5 - 1.0

    #         # Decode the MLCO individuals and write them to the
    #         # detected_lanes_stream.
    #         detected_lanes = self.decode_mlco()
    #         self._logger.debug('@{}: Detected {} lanes'.format(
    #             msg.timestamp, len(detected_lanes)))
    #         detected_lanes_stream.send(erdos.Message(msg.timestamp,
    #                                                  detected_lanes))

    #         self.frame_index += 1

    #     def decode_mlco(self):
    #         """Translates an element in the `mlco_list` to pylot.Lane format.
    #         Returns:
    #             :py:class:`d
    #         """
    #         decoded_lanes = []

    #         # Assumption: only a maximum of 2-lane road in considered.

    #         mlco_snapshot_list = self.mlco_list[self.frame_index]
    #         for i in range(2):
    #             lane = self.list_to_lane(mlco_snapshot_list[i])
    #             decoded_lanes.append(lane)

    #         return decoded_lanes

    #     @staticmethod
    #     def list_to_transform(side_markings_list):
    #         """
    #         Translates the lane markings information from a list to
    #         Transfrom datatype. Returns a list of transforms.
    #         """
    #         side_markings_transforms = []
    #         for marking in side_markings_list:
    #             marking_location = Location(marking[0], marking[1], marking[2])
    #             marking_rotation = Rotation(marking[3], marking[4], marking[5])
    #             marking_transform = Transform(
    #                 location=marking_location, rotation=marking_rotation)
    #         side_markings_transforms.append(marking_transform)

    #         return side_markings_transforms

    #     @classmethod
    #     def list_to_lane(cls, lane_list):
    #         """
    #         Translates a list of lane-related parameters into Lane format.
    #         """
    #         lane_index = lane_list[0]
    #         left_markings_list = lane_list[1]
    #         right_markings_list = lane_list[2]

    #         # Translate the markings list into transforms.
    #         left_markings_transforms = cls.list_to_transform(left_markings_list)
    #         right_markings_transforms = \
    #           cls.list_to_transform(right_markings_list)

    #         lane = Lane(
    #             lane_index,
    #             left_markings_transforms,
    #             right_markings_transforms)

    #         return lane

    # # NOTE: Should be added to operator_creator.py
    # def add_lane_detection_highjacker(
    #         bgr_camera_stream, mlco_list, name='highjacker_lane_detection'):
    #     """
    #     The function creates a `lane_detection_stream` for pylot using
    #     the `LaneDetectionHighjackerOperator` operator.
    #     """
    #     op_config = erdos.OperatorConfig(name=name,
    #                                      log_file_name=FLAGS.log_file_name,
    #                                      csv_log_file_name=FLAGS.csv_log_file_name,
    #                                      profile_file_name=FLAGS.profile_file_name)
    #     [lane_detection_stream] = erdos.connect(LaneDetectionHighjackerOperator,
    #                                             op_config, [bgr_camera_stream],
    #                                             mlco_list, FLAGS)
    #     return lane_detection_stream
