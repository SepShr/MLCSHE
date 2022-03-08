import copy
import os
import pathlib
import pickle
import subprocess as sub
import time
import logging
from tqdm import trange
from configparser import ConfigParser

from datetime import datetime
from uuid import uuid4

import simulation_config as cfg
from data_handler import get_values

CWD = os.getcwd()

# Setup logger.
logger = logging.getLogger(__name__)

# # Setup configuration parser instance.
# config = ConfigParser()
# config.read('simulation_config.ini')

# # Load configuration.
# container_name = config['CONTAINER']['container_name']
# simulation_results_destination_path = config['RESULTS']['destination_path']
# print(simulation_results_destination_path)
# dest_path = str(simulation_results_destination_path)
# results_file_name = os.path.join(
#     dest_path, 'simulation_log_file_name.py')
# print(results_file_name)

# hello = 'hello'
# print(os.path.join(hello, 'world'))


def translate_scenario_list(scenario_list):
    """Sets the value of the `flags.simulator_weather` based on the
    `scenario_list`.
    """
    scenario_flag = {}

    # Set weather
    weather = ""
    # weather_flag = "--simulator_weather=" + \
    #     WEATHER_PRESETS[scenario_list[0]] + "\n"
    if (scenario_list[0] == 0):  # noon

        if (scenario_list[1] == 0):  # clear
            weather = "ClearNoon"
        if (scenario_list[1] == 1):  # clear
            weather = "CloudyNoon"
        if (scenario_list[1] == 2):  # clear
            weather = "WetNoon"
        if (scenario_list[1] == 3):  # clear
            weather = "WetCloudyNoon"
        if (scenario_list[1] == 4):  # clear
            weather = "MidRainyNoon"
        if (scenario_list[1] == 5):  # clear
            weather = "HardRainNoon"
        if (scenario_list[1] == 6):  # clear
            weather = "SoftRainNoon"
    if (scenario_list[0] == 1):  # sunset

        if (scenario_list[1] == 0):  # clear
            weather = "ClearSunset"
        if (scenario_list[1] == 1):  # clear
            weather = "CloudySunset"
        if (scenario_list[1] == 2):  # clear
            weather = "WetSunset"
        if (scenario_list[1] == 3):  # clear
            weather = "WetCloudySunset"
        if (scenario_list[1] == 4):  # clear
            weather = "MidRainSunset"
        if (scenario_list[1] == 5):  # clear
            weather = "HardRainSunset"
        if (scenario_list[1] == 6):  # clear
            weather = "SoftRainSunset"
    if (scenario_list[0] == 2):  # sunset

        if (scenario_list[1] == 0):  # clear
            weather = "ClearSunset"
        if (scenario_list[1] == 1):  # clear
            weather = "CloudySunset"
        if (scenario_list[1] == 2):  # clear
            weather = "WetSunset"
        if (scenario_list[1] == 3):  # clear
            weather = "WetCloudySunset"
        if (scenario_list[1] == 4):  # clear
            weather = "MidRainSunset"
        if (scenario_list[1] == 5):  # clear
            weather = "HardRainSunset"
        if (scenario_list[1] == 6):  # clear
            weather = "SoftRainSunset"
        scenario_flag['night_time='] = "--night_time=1\n"
        logger.debug('simulator time of day is set to night.')

    scenario_flag['simulator_weather='] = "--simulator_weather=" + \
        weather + "\n"

    logger.debug('simulator_weather set to {}'.format(weather))

    num_of_pedestrians = 0
    if scenario_list[2] == 0:
        num_of_pedestrians = 0
    if scenario_list[2] == 1:
        num_of_pedestrians = 18

    scenario_flag['simulator_num_people='] = "--simulator_num_people=" + \
        str(num_of_pedestrians) + "\n"
    logger.debug('Number of pedestrians is set to {}'.format(
        str(num_of_pedestrians)))

    # Set target speed of ego vehicle.
    # scenario_flag['target_speed='] =  "\n--target_speed=" + str(scenario_list[?] / 3.6)

    # scenario_flag['simulator_weather='] = weather_flag

    logger.debug('scenario_flag={}'.format(scenario_flag))

    return scenario_flag


def translate_mlco_list(mlco_list, container_name=cfg.container_name, file_name=cfg.mlco_file_name, destination_path=cfg.mlco_destination_path):
    """Transfers the `mlco_list` to `container_name` and updates
    the flags for pylot.
    """
    pathlib.Path('temp/').mkdir(parents=True, exist_ok=True)
    mlco_file = os.path.join('temp', file_name)

    # Write the mlco_list to a file.
    pickle_to_file(mlco_list, mlco_file)

    # Copy the file inside the Pylot container.
    source_path = os.path.join(CWD, mlco_file)
    copy_returncode = copy_to_container(
        container_name, source_path, destination_path)

    if copy_returncode != 0:
        logger.error('{} was NOT copied to {}'.format(
            file_name, container_name))
    else:
        logger.debug('{} successfully copied to {}'.format(
            file_name, container_name))

    # Setup the flags to be passed on to pylot.
    # mlco_operator_flag = "--lane_detection_type=highjacker\n"
    mlco_list_path_flag = "--mlco_list_path=" + destination_path + '\n'

    mlco_flag = {}
    # mlco_flag['lane_detection_type='] = mlco_operator_flag
    mlco_flag['mlco_list_path='] = mlco_list_path_flag

    return mlco_flag


def pickle_to_file(item_to_be_pickled, file_name: str):
    """Pickles an object and adds it to a file.
    """
    file = open(file_name, 'wb')
    pickle.dump(item_to_be_pickled, file)
    file.close()


def start_container(container_name: str = cfg.container_name):
    """Starts a docker container with the name `container_name`.
    """
    cmd = ['docker', 'start', container_name]
    docker_start_proc = sub.run(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    # Log the container's successful start or failure.
    return docker_start_proc.returncode


def stop_container(container_name: str = cfg.container_name):
    """Starts a docker container with the name `container_name`.
    """
    cmd = ['docker', 'stop', container_name]
    docker_start_proc = sub.run(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    return docker_start_proc.returncode


def copy_to_container(
        container_name: str, source_path: str, destination_path: str):
    """Copies a file from `source_path` to `destination_path`
    inside a `container_name`.
    """
    # container_id = find_container_id(container_name)
    container_name_with_path = container_name + ":" + destination_path
    copy_cmd = ['docker', 'cp', source_path, container_name_with_path]
    copy_proc = sub.run(copy_cmd)

    return copy_proc.returncode


def copy_to_host(container_name: str, file_name: str, source_path: str, destination_path: str):
    """Copies a file from a `source_path` inside a `container_name` to
    `destination_path`.
    """
    container_name_with_path = container_name + ":" + source_path + file_name
    copy_cmd = ['docker', 'cp', container_name_with_path, destination_path]
    sub.run(copy_cmd)

    container_name_with_path = container_name + \
        ":" + source_path + file_name + "_ex.log"
    copy_cmd = ['docker', 'cp', container_name_with_path, destination_path]
    sub.run(copy_cmd, stderr=sub.PIPE)


def find_container_id(container_name: str):
    """Find the ID of a running container, given its name
    (`container_id`).
    """
    # FIXME: Handle exception.
    container_id = ''
    # Make sure that the Pylot container is running and find the container ID.
    containers = sub.check_output(['docker', 'ps'], universal_newlines=True)
    containers_split = containers.split('\n')
    for i in containers_split:
        if i.__contains__(container_name):
            container_id = i[:12]
    return container_id


def update_config_file(
        simulation_flag, base_config_file=cfg.base_config_file,
        simulation_config_file_name=cfg.simulation_config_file_name):
    """Updates `base_config_file` according to `simulation_flag` and
    writes it to `simulation_config_file`.
    """
    # Create the temp folder if it does not exist.
    pathlib.Path('temp/').mkdir(parents=True, exist_ok=True)
    simulation_config_file = os.path.join('temp', simulation_config_file_name)
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

# FIXME: Maybe add the simulation ID (counter) to the log file's name.


def update_sim_config(scenario_list, mlco_list, container_name: str = cfg.container_name):
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_log_file_name = str(
        timestamp) + "_sim" + '.log'  # + uuid4().hex
    log_file_name = '/home/erdos/workspace/results/' + simulation_log_file_name

    simulation_flag['--log_fil_name='] = "--log_fil_name=" + \
        log_file_name + "\n"

    # Update the config file.
    update_config_file(simulation_flag)

    copy_returncode = copy_to_container(container_name, CWD + cfg.config_source_path,
                                        cfg.config_destination_path)

    if copy_returncode != 0:
        logger.error('{} was NOT copied to {}'.format(
            cfg.config_source_path, container_name))
    else:
        logger.debug('{} successfully copied to {}'.format(
            cfg.config_source_path, container_name))

    return simulation_log_file_name


def run_command_in_shell(command):
    """Runs a command in shell via subprocess. Also if verbose is set
    to `True`, it will print out the `stdout` or `stderr` messages,
    depending on the successful run of the process.
    """
    logger.debug(f'Running {command} in shell.')

    proc = sub.Popen(command, stdout=sub.PIPE, stderr=sub.PIPE, shell=True)

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
        container_name: str = cfg.container_name, carla_run_path=cfg.carla_runner_path):
    """Runs the carla simulator which is inside `container_name`.
    """
    carla_run_command = "nvidia-docker exec -it " + \
        container_name + " " + carla_run_path

    carla_proc = run_command_in_shell(carla_run_command)

    return carla_proc


def run_pylot(run_pylot_path: str = CWD + cfg.pylot_runner_path):
    """Runs a script that runs pylot inside a container.
    """
    pylot_run_command = [run_pylot_path, cfg.container_name]

    # pylot_proc = run_command_in_shell(pylot_run_command)
    pylot_proc = sub.Popen(pylot_run_command, stdout=sub.PIPE, stderr=sub.PIPE)
    return pylot_proc


def remove_finished_file(container_name: str = cfg.container_name, finished_file_path: str = cfg.finished_file_path):
    """Removes `finished.txt` from the container and the workspace.
    """
    cmd = ['docker', 'exec', container_name, 'rm', '-rf',
           finished_file_path]
    rm_finished_file_proc = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    if os.path.exists("finished.txt"):
        os.remove("finished.txt")


def reset_sim_setup(container_name: str = cfg.container_name):
    """Resets the simulation setup, i.e., restarts the container, and
    removes old finished files.
    """
    stop_container(container_name)
    start_container(container_name)
    remove_finished_file()


def run_carla_and_pylot(carla_sleep_time=cfg.carla_timeout, pylot_sleep_time=cfg.pylot_timeout):
    """Runs carla and pylot and ensures that they are in sync.
    """
    logger.info('Running Carla (timeout = {} sec).'.format(carla_sleep_time))
    run_carla()
    time.sleep(carla_sleep_time)
    logger.info('Running Pylot (timeout = {} sec).'.format(pylot_sleep_time))
    run_pylot()
    time.sleep(pylot_sleep_time)
    # FIXME: Confirm successful Carla and Pylot are in sync.


def scenario_finished():
    """
    Check whether the scenario is finished or not.
    """
    cmd = [cfg.script_directory +
           'copy_pylot_finished_file.sh', cfg.container_name]
    sub.run(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    if os.path.exists(cfg.base_directory + "finished.txt"):
        return True
    return False


# def run_simulation(scenario_list, mlco_list):
#     """Ensures that the simulation setup is ready, updates simulation
#     configuration given `scenario_list` and `mlco_list` and, runs the
#     simulation, records its output and
#     """
#     scenario_list_deepcopy = copy.deepcopy(scenario_list)
#     logger.info('Scenario individual considered for simulation is {}'.format(
#         scenario_list_deepcopy))
#     mlco_list_deepcopy = copy.deepcopy(mlco_list)
#     logger.info('Mlco individual considered for simulation is {}'.format(
#         mlco_list_deepcopy))

#     # Reset the simulation setup.
#     logger.debug("Resetting the simulation setup.")
#     reset_sim_setup()
#     # Update the configuration of the simulation and the serialized mlco_list
#     simulation_log_file_name = update_sim_config(
#         scenario_list_deepcopy, mlco_list_deepcopy)
#     logger.debug("Simulation configuration is updated.")
#     # Run Carla and Pylot in the docker container with appropriate config
#     run_carla_and_pylot()

#     # Monitor scenario execution and end it when its over.
#     counter = 0

#     for counter in trange(cfg.simulation_duration):
#         time.sleep(1)
#         if scenario_finished():
#             break
#     stop_container()
#     logger.info("End of simulation.")

#     # Copy the results of the simulation.
#     copy_to_host(cfg.container_name, simulation_log_file_name,
#                  cfg.simulation_results_source_directory, cfg.simulation_results_destination_path)
#     results_file_name = 'results/' + simulation_log_file_name
#     if os.path.exists(results_file_name):
#         logger.debug(
#             'Found the results of simulation in {}'.format(results_file_name))
#         DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = get_values(
#             simulation_log_file_name)
#         logger.info(
#             f'{DfC_min}, {DfV_max}, {DfP_max}, {DfM_max}, {DT_max}, {traffic_lights_max}')
#         return DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max
#     else:
#         logger.warning(
#             'Did not find the simulation results, i.e., {}'.format(results_file_name))
#         logger.warning("Returning 1000 for all simulation results.")
#         return 1000, 1000, 1000, 1000, 1000, 1000


# def main():
#     scenario_list = [1, 3, 1]
#     mlco_list = [
#                 [200, 250, 210, 260, 0],
#                 [300, 350, 310, 360, 0],
#                 [100, 160, 150, 220, 6]
#     ]
#     #     ],
#     #     [
#     #         [0,  [1, 4, 5, 2, 3, 1], [2, 7, 2, 4, 5, 2]],
#     #         [1,  [2, 4, 5, 2, 3, 1], [3, 7, 2, 4, 6, 2]],
#     #         [2,  [3, 4, 5, 2, 3, 1], [4, 7, 2, 4, 5, 2]]
#     #     ]
#     # ]

#     run_simulation(scenario_list, mlco_list)


# if __name__ == "__main__":
#     main()
