import copy
import logging
import os
import time

from tqdm import trange

import simulation_config as cfg
from data_handler import get_values
from simulation_utils import (copy_to_host, reset_sim_setup, run_carla_and_pylot,
                              scenario_finished, stop_container,
                              update_sim_config)

# Setup logger.
logger = logging.getLogger(__name__)


def run_simulation(scenario_list, mlco_list):
    """Ensures that the simulation setup is ready, updates simulation
    configuration given `scenario_list` and `mlco_list` and, runs the
    simulation, records its output and 
    """
    scenario_list_deepcopy = copy.deepcopy(scenario_list)
    logger.info('Scenario individual considered for simulation is {}'.format(
        scenario_list_deepcopy))
    mlco_list_deepcopy = copy.deepcopy(mlco_list)
    logger.info('Mlco individual considered for simulation is {}'.format(
        mlco_list_deepcopy))

    # Reset the simulation setup.
    logger.debug("Resetting the simulation setup.")
    reset_sim_setup()
    # Update the configuration of the simulation and the serialized mlco_list
    simulation_log_file_name = update_sim_config(
        scenario_list_deepcopy, mlco_list_deepcopy)
    logger.debug("Simulation configuration is updated.")
    # Run Carla and Pylot in the docker container with appropriate config
    run_carla_and_pylot()

    # Monitor scenario execution and end it when its over.
    counter = 0

    for counter in trange(cfg.simulation_duration):
        time.sleep(1)
        if scenario_finished():
            break
    stop_container()
    logger.info("End of simulation.")

    # Copy the results of the simulation.
    copy_to_host(cfg.container_name, simulation_log_file_name,
                 cfg.simulation_results_source_directory, cfg.simulation_results_destination_path)
    results_file_name = 'results/' + simulation_log_file_name
    if os.path.exists(results_file_name):
        logger.debug(
            'Found the results of simulation in {}'.format(results_file_name))
        DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = get_values(
            simulation_log_file_name)
        logger.info(
            f'{DfC_min}, {DfV_max}, {DfP_max}, {DfM_max}, {DT_max}, {traffic_lights_max}')
        return DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max
    else:
        logger.warning(
            'Did not find the simulation results, i.e., {}'.format(results_file_name))
        logger.warning("Returning 1000 for all simulation results.")
        return 1000, 1000, 1000, 1000, 1000, 1000


# Test values
# scenario_list = [1, 3, 1]
# mlco_list = [
#             [200, 250, 210, 260, 0],
#             [300, 350, 310, 360, 0],
#             [100, 160, 150, 220, 6]
# ]

def main(scenario_list, mlco_list):
    run_simulation(scenario_list, mlco_list)


if __name__ == "__main__":
    main()
