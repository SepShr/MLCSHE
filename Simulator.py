import copy
import logging
import os
import time

from tqdm import trange

import simulation_config as cfg
from simulation_utils import (copy_to_host, reset_sim_setup, run_carla_and_pylot,
                              scenario_finished, stop_container,
                              update_sim_config, get_values)


class Simulator():
    def __init__(self):
        self.simulation_counter = 0

        # Setup logger.
        self.logger = logging.getLogger(__name__)

    def run_simulation(self, scenario_list, mlco_list):
        """Ensures that the simulation setup is ready, updates simulation
        configuration given `scenario_list` and `mlco_list` and, runs the
        simulation, records its output and 
        """
        self.simulation_counter += 1

        simulation_fitness_values = {}

        self.logger.debug('simulation_counter={}'.format(
            self.simulation_counter))

        scenario_list_deepcopy = copy.deepcopy(scenario_list)
        # logger.debug('Scenario individual considered for simulation is {}'.format(
        #     scenario_list_deepcopy))
        self.logger.debug('scenario={}'.format(
            scenario_list_deepcopy))
        mlco_list_deepcopy = copy.deepcopy(mlco_list)
        # logger.debug('Mlco individual considered for simulation is {}'.format(
        #     mlco_list_deepcopy))
        self.logger.debug('mlco={}'.format(mlco_list_deepcopy))

        # Reset the simulation setup.
        self.logger.debug("Resetting the simulation setup.")
        reset_sim_setup()
        # Update the configuration of the simulation and the serialized mlco_list
        simulation_log_file_name = update_sim_config(
            scenario_list_deepcopy, mlco_list_deepcopy, str(self.simulation_counter))
        self.logger.debug("Simulation configuration updated.")
        # Run Carla and Pylot in the docker container with appropriate config
        run_carla_and_pylot()

        # Monitor scenario execution and end it when its over.
        counter = 0

        for counter in trange(cfg.simulation_duration):
            time.sleep(1)
            if scenario_finished():
                break
        stop_container()
        self.logger.debug("End of simulation.")

        # Copy the results of the simulation.
        copy_to_host(cfg.container_name, simulation_log_file_name,
                     cfg.simulation_results_source_directory, cfg.simulation_results_destination_path)
        results_file_name = 'results/' + simulation_log_file_name
        if os.path.exists(results_file_name):
            self.logger.debug(
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

            self.logger.info('simulation_fitness_values={}'.format(
                simulation_fitness_values))

            return DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max
        else:
            self.logger.error(
                'Did not find the simulation results, i.e., {}'.format(results_file_name))
            self.logger.error("Returning 1000 for all simulation results.")
            return 1000, 1000, 1000, 1000, 1000, 1000
