from concurrent import futures
import copy
from itertools import zip_longest
import logging
import os
import time
from pathlib import Path

from tqdm import trange

import simulation_config as cfg
from Simulator import Simulator
from simulation_utils import (copy_to_host, reset_sim_setup, run_carla_and_pylot,
                              scenario_finished, stop_container,
                              update_sim_config, get_values, pickle_to_file)

from src.utils.utility import setup_logger


class SimJob():
    """
    Creates a distinct job
    :param sim_Name: Simulation name (must be unique)
    :param templates: files to be copied from your host to the container
    """

    def __init__(self, scenario_list, mlco_list, sim_name, templates: Path) -> None:
        # self.templates = templates
        self.scenario_list = scenario_list
        self.mlco_list = mlco_list
        self.sim_name = sim_name
        # self.sim_name = uuid4().hex

    def translate_mlco_list(mlco_list, file_name=cfg.mlco_file_name, destination_path=cfg.mlco_destination_path):
        """Transfers the `mlco_list` to `container_name` and updates
        the flags for pylot.
        """
        Path('temp/').mkdir(parents=True, exist_ok=True)
        mlco_file = os.path.join('temp', file_name)

        # Write the mlco_list to a file.
        pickle_to_file(mlco_list, mlco_file)

        # Setup the flags to be passed on to pylot.
        # mlco_operator_flag = "--lane_detection_type=highjacker\n"
        mlco_list_path_flag = "--mlco_list_path=" + destination_path + '\n'

        mlco_flag = {}
        # mlco_flag['lane_detection_type='] = mlco_operator_flag
        mlco_flag['mlco_list_path='] = mlco_list_path_flag

        return mlco_flag

    def __str__(self) -> str:
        return self.sim_name


class SimulationManager():
    def __init__(self,
                 workers: list = []
                 ) -> None:
        """
        :param container_image_url: Simulation container URL at container registry
        :param max_workers: number of parallel workers
        :param data_directory: path to the output directory on your host
        :param container_image_tag: container tag, Default: latest
        """
        # self._data_directory = data_directory
        self._available_workers = workers
        if len(self._available_workers) == 0:
            raise ValueError("Must have at least one worker available.")
        else:
            self._max_workers = len(workers)
            self._process_pool_executor = futures.ProcessPoolExecutor(
                max_workers=self._max_workers)
        self.job_list = []
        self.simulation_counter = 0
        self._logger = logging.getLogger(__name__)  # Setup logger

    def add_sim_job(self, job: SimJob) -> None:
        """
        Adds a simulation job into the queue
        :param job: simulation job to be added
        """
        self.job_list.append(job)

    def start_computation(self):
        """
        Starts computation of all jobs inside the queue.
        Jobs must be added before calling this function
        """
        # FIXME: Monitor the state of the running containers.
        with self._thread_pool_executor as executor:
            futures = {executor.submit(self._process_sim_job, sim_job): sim_job
                       for sim_job in self.job_list}
            for future in futures.as_completed(futures):
                self._logger.info(f'Run {futures[future]} did finish')

        # Using multiprocessing. Might be better able to handle GIL.
        # self._multiprocessing_pool = multiprocessing.Pool(
        #     len(self._available_workers))
        # results = self._multiprocessing_pool.map(self._process_sim_job, self.job_list)
        # for r in results:
        #     pass

    # def _process_sim_job(self, sim_job: SimJob) -> None:
    def _process_sim_job(self, sim_job: SimJob, simulator: Simulator) -> None:
        """
        Triggers processing steps for a single job.
        1. _init_simulation
        2. _run_docker_container
        3. _cleanup_sim_objects
        :param sim_job: SimJob to be processed
        :return: True if processing succeeded, False otherwise.
        """
        sim_paths = self._init_simulation(sim_job=sim_job)
        if sim_paths is None:
            self._logger.error(
                f'Error during initialization for simulation {sim_job}')
            return False

        try:
            (
                working_dir,
                *file_objects
            ) = sim_paths
        except:
            try:
                working_dir = sim_paths
            except:
                self._logger.error(
                    f'Error during initialization for simulation {str(sim_job)}')
                return False

        self._run_docker_container(
            container_name=sim_job.sim_name, working_dir=working_dir, command=sim_job.command)
        self.cleanup_sim_objects(sim_job=sim_job, file_objects=file_objects)
        return True

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
        output_folder_name = f'job_{sim_job.sim_name}'
        working_dir = self._data_directory.joinpath(output_folder_name)
        try:
            with self._io_lock:
                working_dir.mkdir(exist_ok=False, parents=True)
                # if you need additional files in your simulation e.g. config files, data, add them here example_monte_carlo_pi here

                # translate scenario
                # translate mlco
                # update the config file
                # add the mlco.pkl and config.conf to the working_dir.
                # bind mount the directory to the container

            return working_dir
        except Exception as e:
            self._logger.warning(e)
            return None

    def run_simulation(self, scenario_list, mlco_list):
        """Ensures that the simulation setup is ready, updates simulation
        configuration given `scenario_list` and `mlco_list` and, runs the
        simulation, records its output and 
        """
        self.simulation_counter += 1

        simulation_fitness_values = {}

        # logger.debug('Simulation number is: {}'.format(
        #     self.simulation_counter))
        self._logger.debug('simulation_counter={}'.format(
            self.simulation_counter))

        scenario_list_deepcopy = copy.deepcopy(scenario_list)
        # logger.debug('Scenario individual considered for simulation is {}'.format(
        #     scenario_list_deepcopy))
        self._logger.debug('scenario={}'.format(
            scenario_list_deepcopy))
        mlco_list_deepcopy = copy.deepcopy(mlco_list)
        # logger.debug('Mlco individual considered for simulation is {}'.format(
        #     mlco_list_deepcopy))
        self._logger.debug('mlco={}'.format(mlco_list_deepcopy))

        # Reset the simulation setup.
        self._logger.debug("Resetting the simulation setup.")
        reset_sim_setup()
        # Update the configuration of the simulation and the serialized mlco_list
        simulation_log_file_name = update_sim_config(
            scenario_list_deepcopy, mlco_list_deepcopy, str(self.simulation_counter))
        self._logger.debug("Simulation configuration updated.")
        # Run Carla and Pylot in the docker container with appropriate config
        run_carla_and_pylot()

        # Monitor scenario execution and end it when its over.
        counter = 0

        for counter in trange(cfg.simulation_duration):
            time.sleep(1)
            if scenario_finished():
                break
        stop_container()
        self._logger.debug("End of simulation.")

        # Copy the results of the simulation.
        copy_to_host(cfg.container_name, simulation_log_file_name,
                     cfg.simulation_results_source_directory, cfg.simulation_results_destination_path)
        results_file_name = 'results/' + simulation_log_file_name
        if os.path.exists(results_file_name):
            self._logger.debug(
                'Found the results of simulation in {}'.format(results_file_name))
            DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = get_values(
                simulation_log_file_name)

            simulation_fitness_values = {
                'distance_from_center': DfC_min,
                'distance_from_vehicle': DfV_max,
                'distance_from_pedestrian': DfP_max,
                'distance_from_obstacles': DfM_max,
                'distance_traveled': DT_max,
                'violated_traffic_lights': traffic_lights_max
            }

            self._logger.info('simulation_fitness_values={}'.format(
                simulation_fitness_values))

            return DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max
        else:
            self._logger.error(
                'Did not find the simulation results, i.e., {}'.format(results_file_name))
            self._logger.error("Returning 1000 for all simulation results.")
            return 1000, 1000, 1000, 1000, 1000, 1000


def add_sim_job(cs, simulator):
    simulator.add_sim_job(cs)


def get_safety_req_value(simulator):
    simulator.start_computation


def main(cs_list):
    simulator_1 = Simulator()
    simulator_2 = Simulator()
    simulators_list = [simulator_1, simulator_2]


if __name__ == "__main__":
    setup_logger('test_sim')
    # main(scenario, mlco)
