import concurrent.futures
from io import BytesIO
import logging
from pathlib import Path
import pickle
import tarfile
import time
from typing import Container
import docker
import os
import threading
from datetime import datetime
from docker.errors import DockerException, NotFound, APIError
from docker.types import Mount, LogConfig
import platform
# import tracemalloc
import gc
from collections import deque

from problem_utils import mlco_to_obs_seq
from simulation_utils import translate_scenario_list
from tqdm import trange

import simulation_config as cfg

# Setup logger.
logger = logging.getLogger(__name__)


class SimJob():
    """
    Creates a distinct job
    :param sim_Name: Simulation name (must be unique)
    :param templates: files to be copied from your host to the container
    :param command: command to be appended at containers entry point
    """
    # def __init__(self, sim_Name, templates: Path, command=None) -> None:

    def __init__(self, sim_Name, complete_solution, command=None, templates=None) -> None:
        self.templates = templates
        self.sim_name = sim_Name
        self.command = command

        self.complete_solution = complete_solution
        self.scenario = self.complete_solution[0]
        self.mlco = self.complete_solution[1]

    def __str__(self) -> str:
        return self.sim_name


class ContainerSimManager():
    def __init__(self,
                 data_directory: Path,
                 output_directory: Path,
                 docker_container_url: str = cfg.container_img_name,
                 docker_repo_tag: str = cfg.docker_repo_tag
                 ) -> None:
        """
        :param docker_container_url: Simulation container URL at container registry
        :param max_workers: number of parallel workers
        :param data_directory: path to the output directory on your host
        :param docker_repo_tag: container tag, Default: latest
        """
        self._data_directory = data_directory
        self._container_prefix = 'PylotSim'
        self._docker_image_name = docker_container_url + ':' + docker_repo_tag
        self._output_directory = output_directory
        self._monitoring_frequency = 300
        self._minimum_runtime = 300
        self._maximum_inactivity_time = 30 * 60
        self.job_list = []
        self._logger = logging.getLogger(__name__)

    def add_sim_job(self, job: SimJob) -> None:
        """
        Adds a simulation job into the queue
        :param job: simulation job to be added
        """
        self.job_list.append(job)

    def _process_sim_job(self, sim_job: SimJob):
        """
        Triggers processing steps for a single job.
        1. _init_simulation
        2. _run_docker_container
        3. _cleanup_sim_objects
        :param sim_job: SimJob to be processed
        :return: True if processing succeeded, False otherwise.
        """
        input_data_folder, output_folder, simulation_log_file_name = self._init_simulation(
            sim_job=sim_job)

        container = self._run_docker_container(
            container_name=sim_job.sim_name, working_dir=input_data_folder, command=sim_job.command)

        # run_pylot_docker(container=container)
        container.exec_run(
            "python3 /home/erdos/workspace/pylot/pylot.py --flagfile=/mnt/data/mlcshe_config.conf",
            detach=True,
            environment=[
                'PYTHONPATH="$PYTHONPATH:/home/erdos/workspace/pylot/dependencies/:/home/erdos/workspace/pylot/dependencies/lanenet:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/agents/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/:/home/erdos/workspace/scenario_runner"']
        )
        time.sleep(10)

        for counter in trange(cfg.simulation_duration):
            time.sleep(1)
            if self.scenario_finished_docker(container=container, output_folder=output_folder):
                print('found the finished file, breaking the for loop.')
                break

        container.stop()

        self._logger.debug("End of simulation.")

        self.get_results(container, simulation_log_file_name,
                         cfg.simulation_results_source_directory, output_folder)
        results_file_name = output_folder.joinpath(
            simulation_log_file_name).joinpath(simulation_log_file_name)
        if os.path.exists(results_file_name):
            self._logger.debug(
                'Found the results of simulation in {}'.format(results_file_name))
            DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = self._get_values(
                simulation_log_file_name, output_folder)

            safety_req_values = {
                'distance_from_center': DfC_min,
                'distance_from_vehicle': DfV_max,
                'distance_from_pedestrian': DfP_max,
                'distance_from_obstacles': DfM_max,
                'distance_traveled': DT_max,
                'violated_traffic_lights': traffic_lights_max
            }

            self._logger.info('safety_req_values={}'.format(
                safety_req_values))
            container.remove()

            return DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max
        else:
            self._logger.error(
                'Did not find the simulation results, i.e., {}'.format(results_file_name))
            self._logger.error("Returning 1000 for all simulation results.")
            return 1000, 1000, 1000, 1000, 1000, 1000

    def _init_simulation(self, sim_job):
        """
        Initialize simulation. May be overridden with custom function.
        - Create output folders
        - Copy file templates
        - ...
        :param sim_job: SimJob to be processed
        :return: Path to working directory on your host's filesystem for this SimJob
        """

        # prepare your data for your scenario here
        # print('in _init_simulation...')
        output_folder = self._output_directory.joinpath(sim_job.sim_name)
        try:
            with threading.Lock():
                output_folder.mkdir(exist_ok=False, parents=True)
                input_data_folder,  simulation_log_file_name = self.update_sim_config(
                    sim_job.scenario, sim_job.mlco, sim_job.sim_name, self._data_directory)
            return input_data_folder, output_folder, simulation_log_file_name
        except Exception as e:
            self._logger.warning(e)
            return None

    def _run_docker_container(self, container_name, working_dir, command):
        """
        Triggers the simulation run in a separate Docker container.
        :param container_name: the container's name
        :param working_dir: working directory on your host's file system
        :param command: container command (eg. name of script, cli arguments ...) Must match to your docker entry point.
        """
        docker_client = docker.from_env()
        docker_image = docker_client.images.get(self._docker_image_name)
        # print('in _run_docker_container...')
        try:
            system_platform = platform.system()
            if system_platform == "Windows":
                # container = self._docker_client.containers.run(
                container = docker_client.containers.run(
                    image=docker_image,
                    command=command,
                    detach=True,
                    mounts=[Mount(
                        target='/mnt/data',
                        source=str(working_dir.resolve()),
                        type='bind'
                    )],
                    # working_dir='/simulation',
                    name=container_name,
                    environment={
                        # If you need add your environment variables here
                        'PYTHONPATH="$PYTHONPATH:$PYLOT_HOME/dependencies/lanenet/"'
                    },
                    log_config=LogConfig(type=LogConfig.types.JSON, config={
                        'max-size': '500m',
                        'max-file': '3'
                    }),
                    runtime='nvidia'
                )
            else:
                # user_id = os.getuid()
                container = docker_client.containers.run(
                    image=docker_image,
                    command=command,
                    detach=True,
                    mounts=[Mount(
                        target='/mnt/data',
                        source=str(working_dir.resolve()),
                        type='bind'
                    )],
                    # working_dir='/mnt/data',
                    name=container_name,
                    environment=[
                        # If you need add your environment variables here
                        'PYTHONPATH="$PYTHONPATH:$PYLOT_HOME/dependencies/lanenet/"'
                    ],
                    # environment=[
                    #     'PYTHONPATH="$PYTHONPATH:/home/erdos/workspace/pylot/dependencies/:/home/erdos/workspace/pylot/dependencies/lanenet:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/agents/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/:/home/erdos/workspace/scenario_runner"'
                    # ],
                    log_config=LogConfig(type=LogConfig.types.JSON, config={
                        'max-size': '500m',
                        'max-file': '3'
                    }),
                    # ports={'22': 20022},
                    runtime='nvidia'
                    # user=user_id
                )
            time.sleep(10)
            return container
        except DockerException as e:
            self._logger.warning(f'Error in run {container_name}: {e}.')
            return False

    def _get_values(self, results_name: str, results_directory: Path):
        """
        Extract fitness values from simulation output log files.
        """
        # file_name = 'results/' + filename

        sim_results = results_directory.joinpath(
            results_name).joinpath(results_name)
        file_name_ex = results_name+'_ex.log'
        sim_results_ex = results_directory.joinpath(
            file_name_ex).joinpath(file_name_ex)

        assert os.path.exists(
            sim_results), 'The file {} does not exist!'.format(sim_results)

        DfC_min = 1
        DfV_min = 1
        DfP_min = 1
        DfM_min = 1
        DT_max = -1
        traffic_lights_max = 1
        first = True
        distance_Max = -1

        if os.path.exists(sim_results_ex):
            with open(sim_results_ex, "r") as file_handler_ex:
                # file_handler_ex = open(file_name_ex, "r")
                for line_ex in file_handler_ex:  # using sensors to get the data
                    if "red_light" in line_ex:
                        self._logger.info("Red_light invasion")
                        traffic_lights_max = 0
                    if "lane" in line_ex:
                        self._logger.info("lane invasion")
                        # DfC_min = 0
                        DfC_min = -1  # To record safety violation.
                    if "sidewalk" in line_ex:
                        self._logger.info("sidewalk invasion")
                        DfC_min = 0
                    if "vehicle" in line_ex:
                        self._logger.info("vehicle collision")
                        DfV_min = 0

        with open(sim_results, "r") as file_handler:
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

                if DfC_min == 0:
                    DfC_min = -1  # To record safety violation.

        return DfC_min, DfV_min, DfP_min, DfM_min, DT_max, traffic_lights_max

    def update_sim_config(self, scenario_list, mlco_list, simulation_id: str, input_directory_base: Path):
        """Updates the configuration file that is used by Pylot to run
        the simulation. It adds the flags based on scenario and mlco to
        the configuation file.
        """
        # input_data_folder = Path('temp', simulation_id)
        input_data_folder = Path().cwd().joinpath(
            input_directory_base).joinpath(simulation_id)
        # Set flags based on scenario_list and mlco_list.
        scenario_flag = translate_scenario_list(scenario_list)
        mlco_flag, mlco_file = self.translate_mlco_list(
            mlco_to_obs_seq(mlco_list), input_data_folder)

        # Update the simulation flags.
        simulation_flag = {}
        simulation_flag.update(scenario_flag)
        simulation_flag.update(mlco_flag)

        # Setup log file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        simulation_log_file_name = str(
            timestamp) + '_sim_' + simulation_id + '.log'  # + uuid4().hex
        log_file_name = '/home/erdos/workspace/results/' + simulation_log_file_name

        simulation_flag['--log_fil_name='] = "--log_fil_name=" + \
            log_file_name + "\n"

        # Update the config file.
        config_file = self.update_config_file(
            simulation_flag, input_data_folder)

        return input_data_folder, simulation_log_file_name

    @staticmethod
    def update_config_file(
        simulation_flag,
        output_directory: Path = 'temp',
        base_config_file=cfg.base_config_file,
        simulation_config_file_name=cfg.simulation_config_file_name
    ):
        """Updates `base_config_file` according to `simulation_flag` and
        writes it to `simulation_config_file`.
        """
        # Create the temp folder if it does not exist.
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        simulation_config_file = os.path.join(
            output_directory, simulation_config_file_name)
        # Find the lines that contain proper flags and updates them.
        with open(base_config_file, 'rt') as base_config:
            with open(simulation_config_file, 'wt') as simulation_config:
                for line in base_config:
                    for key in simulation_flag:
                        if line.__contains__(key):
                            simulation_config.write(simulation_flag[key])
                            break
                    else:
                        simulation_config.write(line)

        return output_directory.joinpath(simulation_config_file_name)

    @staticmethod
    def translate_mlco_list(
        mlco_list,
        output_directory: Path = 'temp',
        file_name=cfg.mlco_file_name,
        destination_path='/mnt/data/'
        # destination_path=cfg.mlco_destination_path
    ):
        """Transfers the `mlco_list` to `container_name` and updates
        the flags for pylot.
        """
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        mlco_file = os.path.join(output_directory, file_name)

        # Write the mlco_list to a file.
        with open(mlco_file, 'wb') as mf:
            pickle.dump(mlco_list, mf)

        # Setup the flags to be passed on to pylot.
        mlco_list_path_flag = "--mlco_list_path=" + destination_path + file_name + '\n'

        mlco_flag = {}
        # mlco_flag['lane_detection_type='] = mlco_operator_flag
        mlco_flag['mlco_list_path='] = mlco_list_path_flag

        return mlco_flag, Path(mlco_file)

    def get_results(self, container: Container, file_name: str, source_path: str, destination_path: str):
        """Copies the results from the container to host.
        """
        self.copy_to_host(container, file_name, source_path, destination_path)
        try:
            self.copy_to_host(container, file_name+'_ex.log',
                              source_path, destination_path)
        except Exception as e:
            # self._logger.warn(e)
            ex_file_name = file_name + '_ex.log'
            self._logger.debug(f'{ex_file_name} not found in {container.name}')
            pass

    def scenario_finished_docker(self, container: Container, output_folder: Path):
        try:
            finished_file_src_dir = '/home/erdos/workspace/results/'
            finished_file_name = 'finished.txt'
            self.copy_to_host(container, finished_file_name,
                              finished_file_src_dir, output_folder)
            expected_finished_file_path = Path(output_folder).joinpath(
                finished_file_name).joinpath(finished_file_name)
            if os.path.exists(expected_finished_file_path):
                self._logger.info(
                    f'finished file found at {expected_finished_file_path}')
                return True
            else:
                return False

        except Exception as e:
            # print(e)
            return False

    # @staticmethod
    def copy_to_host(self, container: Container, file_name: str, source_path: str, destination_path: str):
        assert file_name is not None, "file_name cannot be None."

        self._logger.debug('Trying to copy {} from {} to {}'.format(
            file_name, source_path, destination_path))
        strm, stat = container.get_archive(source_path+file_name)
        self._logger.debug(stat)
        dst_file_path = Path(destination_path).joinpath(file_name)

        # OPTION 1
        try:
            for d in strm:
                pw_tar = tarfile.TarFile(fileobj=BytesIO(d))
                pw_tar.extractall(dst_file_path)
        except Exception as e:
            self._logger.error(e)
            pass

        # # OPTION 2
        # try:
        #     for d in strm:
        #         self._logger.debug('in for loop...')
        #         with tarfile.TarFile(fileobj=BytesIO(d)) as pw_tar:
        #             pw_tar.extractall(dst_file_path)
        # except Exception as e:
        #     self._logger.warn(e)
        #     pass

        # # OPTION3
        # dst_tar_file = str(dst_file_path)+'.tar'
        # try:
        #     with open(dst_tar_file, 'wb') as f:
        #         for chunk in strm:
        #             f.write(chunk)
        # except Exception as e:
        #     self._logger.warn(
        #         f'Could not create {dst_tar_file}. Exception raised: {e}')

        # tar_file = tarfile.TarFile(Path(dst_tar_file))
        # try:
        #     with tarfile.open(fileobj=tar_file) as tar:
        #         tar.extractall(dst_file_path)
        # except Exception as e:
        #     self._logger.warn(
        #         f'Could not extract {dst_tar_file}. Exception raised: {e}')

        # # OPTION4
        # try:
        #     with open(dst_file_path, 'wb') as f:
        #         for chunk in strm:
        #             f.write(chunk)
        # except Exception as e:
        #     self._logger.warn(
        #         f'Could not create {dst_file_path}. Exception raised: {e}')

        # FIXME: Can use shutil.move() to move each extracted file to its parent directory and remove the extracted directory using shutil.rmtree()


MAX_JOBS_IN_QUEUE = 10


def start_computation(sim_manager, max_workers: int = cfg.max_workers):
    # # OPTION 1 (ALREADY WORKING, ALMOST!)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(sim_manager._process_sim_job, sim_job): sim_job
    #                for sim_job in sim_manager.job_list}
    #     logger.info('jobs submitted')

    #     for future in concurrent.futures.as_completed(futures):
    #         logger.info(f'Run {futures[future]} did finish')
    #     results = [list(f.result()) for f in futures]

    #     del futures
    # sim_manager.job_list.clear()
    # gc.collect()

    # OPTION 2 (BATCH JOBS)
    results = []
    jobs_left = len(sim_manager.job_list)
    jobs_iter = iter(sim_manager.job_list)
    job_queue = deque(sim_manager.job_list)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        while jobs_left > 0:
            batch_results = []
            futures = {}
            for this_job in jobs_iter:
                submitted_job = executor.submit(
                    sim_manager._process_sim_job, this_job)
                logger.info(f'{this_job} submitted')

                futures[submitted_job] = this_job

                if len(futures) >= MAX_JOBS_IN_QUEUE:
                    break

            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    logger.info(f'Run {futures[future]} did finish')
                    jobs_left -= 1  # one down - many to go...
                else:
                    logger.warn(
                        f'Run {futures[future]} did not yield any results!')

            batch_results = [list(f.result()) for f in futures]
            logger.info(f'batch_results={batch_results}')
            results += batch_results
            logger.info(f'results={results}')

            del batch_results
            del futures
            gc.collect()

    sim_manager.job_list.clear()
    gc.collect()
    return results


def prepare_for_computation(cs_list, sim_manager, job_index, sim_job_cmd=cfg.sim_job_command):
    for i, cs in zip(range(job_index, job_index + len(cs_list) + 1), cs_list):
        sim_manager.add_sim_job(SimJob(f'pylot_{i}', cs, sim_job_cmd))
    # logger.info('{} simulation jobs created.'.format(
    #     len(sim_manager.job_list)))
    # print(f'jobs are: {sim_manager.job_list}')
    return job_index + len(cs_list)


def main():

    i = 1
    scenario_1 = [0, 2, 1, 2, 0, 1, 1]
    mlco_1 = [[-1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0, 531, 795, 476.486, 793.544, 90.017, 419.017, 371.84, 728.426, 258.468, 321.834]]

    scenario_2 = [2, 0, 1, 3, 0, 0, 1]
    mlco_2 = [[1, 369, 441, 315.297, 425.586, 152.685, 319.508, 620.136, 704.202, 75.108, 232.678], [
        0, 136, 653, 575.878, 796.938, 416.837, 474.91, 33.195, 86.102, 57.608, 458.03]]

    scenario_3 = [1, 6, 1, 1, 0, 0, 1]
    mlco_3 = [[-1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [-1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    cs_list = [
        [scenario_1, mlco_1],
        [scenario_2, mlco_2],
        [scenario_3, mlco_3]
    ]

    input_directory = Path.cwd().joinpath('temp').joinpath('test')
    output_directory = Path.cwd().joinpath('results').joinpath('test')
    # FIXME: input and output directories should include the search folder as well.

    sim_manager = ContainerSimManager(
        input_directory, output_directory)

    cmd = ["/home/erdos/workspace/pylot/scripts/run_simulator.sh"]

    # cmd = [
    #     "/home/erdos/workspace/pylot/scripts/run_simulator.sh",
    #     "sleep 10",
    #     "python3 /home/erdos/workspace/pylot/pylot.py --flagfile=/mnt/data/mlcshe_config.conf",
    #     "sleep 10"
    # ]

    prepare_for_computation(
        cs_list=cs_list, sim_manager=sim_manager, sim_job_cmd=cmd, job_index=1)

    # Series computation.
    # while len(sim_manager.job_list) > 0:
    #     sim_manager._process_sim_job(sim_manager.job_list.pop())

    # Single job computation.
    # sim_manager._process_sim_job(sim_manager.job_list.pop())

    # Parallel computation.
    start_computation(sim_manager=sim_manager)


if __name__ == "__main__":
    main()
