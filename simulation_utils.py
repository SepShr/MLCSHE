import os
import pathlib
import pickle
import subprocess as sub
import time
import logging

from datetime import datetime
# from uuid import uuid4

import simulation_config as cfg

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

    # Select the road.
    road_flag = {}
    road_flag = get_road(scenario_list, road_flag)
    scenario_flag.update(road_flag)

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


def get_values(filename):
    """
    Extract fitness values from simulation output log files.
    """
    file_name = 'results/' + filename

    assert os.path.exists(
        file_name), 'The file {} does not exist!'.format(file_name)

    DfC_min = 1
    DfV_min = 1
    DfP_min = 1
    DfM_min = 1
    DT_max = -1
    traffic_lights_max = 1
    first = True
    distance_Max = -1
    file_name_ex = file_name+'_ex.log'

    if os.path.exists(file_name_ex):
        with open(file_name_ex, "r") as file_handler_ex:
            # file_handler_ex = open(file_name_ex, "r")
            for line_ex in file_handler_ex:  # using sensors to get the data
                if "red_light" in line_ex:
                    logger.info("Red_light invasion")
                    traffic_lights_max = 0
                if "lane" in line_ex:
                    logger.info("lane invasion")
                    DfC_min = 0
                if "sidewalk" in line_ex:
                    logger.info("sidewalk invasion")
                    DfC_min = 0
                if "vehicle" in line_ex:
                    logger.info("vehicle collision")
                    DfV_min = 0

    with open(file_name, "r") as file_handler:
        for line in file_handler:
            line_parts = line.split('>')
            clean_line_parts = line_parts[1].replace('DfC:', '').replace(
                'DfV:', '').replace('DfP:', '').replace('DfM:', '').replace('DT:', '')
            double_parts = clean_line_parts.split(',')
            DfC = float(double_parts[0])
            DfV = float(double_parts[1])
            DfP = float(double_parts[2])
            DfM = float(double_parts[3])
            DT = float(double_parts[4])

            if DT < 0:
                DT_max = 1
                break

            if first:
                first = False
                distance_Max = DT

            DfC = 1 - (DfC / 1.15)  # normalising
            if DfV > 1:
                DfV = 1
            if DfP > 1:
                DfP = 1
            if DfM > 1:
                DfM = 1

            distance_travelled = distance_Max - DT
            normalised_distance_travelled = distance_travelled/distance_Max
            if DfC < DfC_min:
                DfC_min = DfC
            if DfV < DfV_min:
                DfV_min = DfV
            if DfM < DfM_min:
                DfM_min = DfM
            if DfP < DfP_min:
                DfP_min = DfP
            if normalised_distance_travelled > DT_max:
                DT_max = normalised_distance_travelled

    return DfC_min, DfV_min, DfP_min, DfM_min, DT_max, traffic_lights_max


# spawn points for different types of roads
def get_road(fv, file_contents):
    """
    Returns a dictionary of road-related flags.

    `fv` is the scenario_list while `file_contents` is an empty dictionary.
    """
    no_of_signals = 0
    if fv[3] == 0:  # straight Road
        if fv[4] == 0:  # Road ID
            file_contents["simulator_town="] = "--simulator_town=1\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=29\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=31\n"
                file_contents["goal_location="] = "--goal_location=182.202, 330.639991, 0.300000\n"
            elif (fv[5] == 1):
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=31\n"
                file_contents["goal_location="] = " --goal_location=234.0, 330.639991, 0.300000\n"
            elif (fv[5] == 2):
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=31\n"
                file_contents["goal_location="] = "--goal_location=284.0, 330.639991, 0.300000\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=35\n"

        if fv[4] == 1:  # Road ID
            file_contents["simulator_town="] = "--simulator_town=1\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=163\n"
                file_contents["goal_location="] = "--goal_location=396.310547, 190.713867, 0.300000\n"
            elif (fv[5] == 1):
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=64\n"
                file_contents["goal_location="] = "--goal_location=395.959991,204.169998, 0.300000\n"
            elif (fv[5] == 2):
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=64\n"
                file_contents["goal_location="] = "--goal_location=395.959991,254.169998, 0.300000\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=65\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=130\n"
        if fv[4] == 2:  # Road ID
            file_contents["simulator_town="] = "--simulator_town=3\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=227\n"
                file_contents["goal_location="] = "--goal_location=245.583176, 1.595174, 0.300000\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=117\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=216\n"
            file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=119\n"
        if fv[4] == 3:  # Road ID
            file_contents["simulator_town="] = "--simulator_town=3\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=1\n"
                file_contents["goal_location="] = "--goal_location=126.690155, 8.264045, 0.275307\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=147\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=104\n"
            file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=148\n"

    if fv[3] == 1:  # left turn Road
        if fv[4] == 0:
            file_contents["simulator_town="] = "--simulator_town=1\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=125\n"
                file_contents["goal_location="] = "--goal_location=396.349457, 300.406677, 0.300000\n"
            if (fv[5] == 1):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=125\n"
                file_contents["goal_location="] = "--goal_location=396.449991, 230.409991, 0.300000\n"

            if (fv[5] == 2):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=125\n"
                file_contents["goal_location="] = "--goal_location=396.449991, 180.409991, 0.300000\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=47\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=126\n"

        if fv[4] == 1:
            file_contents["simulator_town="] = "--simulator_town=1\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=108\n"
                file_contents["goal_location="] = "--goal_location=22.179979, 330.459991, 0.300000\n"
            if (fv[5] == 1):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=108\n"
                file_contents["goal_location="] = "--goal_location=22.179979, 380.459991, 0.300000\n"

            if (fv[5] == 2):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=108\n"
                file_contents["goal_location="] = "--goal_location=22.179979, 330.459991, 0.300000\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=123\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=54\n"
        if fv[4] == 2:
            no_of_signals = 1
            file_contents["simulator_town="] = "--simulator_town=3\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=71\n"
                file_contents["goal_location="] = "--goal_location=-84.70,-158.58,0.275307\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=130\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=50\n"
            file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=133\n"
        if fv[4] == 3:
            no_of_signals = 1
            file_contents["simulator_town"] = "--simulator_town=3\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=70\n"
                file_contents["goal_location="] = "--goal_location=-88.20,-158.58,0.275307\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=133\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=50\n"
            file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=130\n"

    if fv[3] == 2:  # right turn Road

        if fv[4] == 0:
            file_contents["simulator_town="] = "--simulator_town=1\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=187\n"
                file_contents["goal_location="] = "--goal_location=392.470001, 19.920038, 0.300000\n"
            if (fv[5] == 1):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=187\n"
                file_contents["goal_location="] = "--goal_location=392.470001, 59.920038, 0.300000\n"
            if (fv[5] == 2):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=187\n"
                file_contents["goal_location="] = "--goal_location=392.470001, 109.920038, 0.300000\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=181\n"
            # check me
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=184\n"
        if fv[4] == 1:
            file_contents["simulator_town="] = "--simulator_town=1\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=57\n"
                file_contents["goal_location="] = "--goal_location=2.009914, 295.863309, 0.300000\n"
            if (fv[5] == 1):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=57\n"
                file_contents["goal_location="] = "--goal_location=55.009914, 295.863309, 0.300000\n"
            if (fv[5] == 2):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=57\n"
                file_contents["goal_location="] = "--goal_location=108.009914, 295.863309, 0.300000\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=67\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=108\n"

        if fv[4] == 2:
            file_contents["simulator_town="] = "--simulator_town=5\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=244\n"
                file_contents["goal_location="] = "--goal_location=-230.40, -84.75, 0.300000\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=301\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=282\n"
            file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=293\n"
        if fv[4] == 3:
            file_contents["simulator_town="] = "--simulator_town=3\n"
            if (fv[5] == 0):  # road length
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=57\n"
                file_contents["goal_location="] = "--goal_location=-36.630997, -194.923615, 0.275307\n"
            file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=127\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=264\n"
            file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=128\n"

    if fv[3] == 3:  # road id

        if fv[4] == 0:
            no_of_signals = 1
            file_contents["simulator_town="] = "--simulator_town=5\n"
            if fv[6] == 0:  # Follow Road
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=-220.048904, -3.915073, 0.300000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=137\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=59\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=60\n"
            if fv[6] == 1:  # 1st exit
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=-184.585892, -53.541496, 0.600000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=137\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=59\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=60\n"
            if fv[6] == 2:  # 2nd exit
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=-191.561127, 36.201321, 0.600000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=138\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=60\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=59\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=127\n"
        # -----
        if fv[4] == 1:
            no_of_signals = 1
            file_contents["simulator_town="] = "--simulator_town=5\n"
            if fv[6] == 0:  # Follow Road
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=-188.079239, -29.370184, 0.600000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=38\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=34\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=33\n"

            if fv[6] == 1:  # 1st exit
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=-159.701355, 6.438059, 0.300000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=37\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=33\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=34\n"
            if fv[6] == 2:  # 2nd exit
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=-220.040390, -0.415084, 0.600000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=38\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=34\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=33\n"
            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=131\n"

        if fv[4] == 2:
            no_of_signals = 1
            file_contents["simulator_town="] = "--simulator_town=5\n"
            if fv[6] == 0:  # Follow Road
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=-44.200703, -41.710579, 0.450000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=44\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=275\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=274\n"
            if fv[6] == 1:  # 1st exit
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=0.556072, 6.047980, 0.300000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=44\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=275\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=274\n"
            if fv[6] == 2:  # 2nd exit
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=-81.402634, -0.752534, 0.300000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=43\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=274\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=275\n"

            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=206\n"
        if fv[4] == 3:
            no_of_signals = 1
            file_contents["simulator_town="] = "--simulator_town=5\n"
            if fv[6] == 0:  # Follow Road
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=24.741877, -52.334110, 0.300000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=75\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=77\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=78\n"
            if fv[6] == 1:  # 1st exit
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=1.051966, -94.892189, 0.300000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=75\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=77\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=78\n"
            if fv[6] == 2:  # 2nd exit
                if (fv[5] == 0):  # road length
                    file_contents["goal_location="] = "--goal_location=52.13, -87.77, 0.300000\n"
                file_contents["simulator_spawn_point_index="] = "--simulator_spawn_point_index=76\n"
                file_contents["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=78\n"
                file_contents["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=77\n"

            file_contents["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=218\n"

    return file_contents
