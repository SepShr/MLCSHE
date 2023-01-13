"""
Genetic Algorithm Search Runner for Pylot.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
sys.path.append(os.path.dirname(__file__))  # nopep8

import logging
import pathlib
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pylot.search_config as cfg
from GASearch import GASearch
from src.utils.utility import setup_logger


def main(sim_budget, pop_size, max_gen):
    # Setup directories
    # Get current timestamp to use as a unique ID.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = str(timestamp) + '_GA_Pylot'
    input_directory = Path.cwd().joinpath('temp').joinpath(output_dir_name)
    output_directory = Path.cwd().joinpath('results').joinpath(output_dir_name)

    output_dir = pathlib.Path('results').joinpath(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output folder.

    # Setup logger.
    setup_logger(file_name='GA', output_directory=output_dir, file_log_level='DEBUG',
                 stream_log_level='INFO')

    logger = logging.getLogger('GA')
    logger.info("Starting GA search.")
    logger.info(f'sim_budget={sim_budget}')
    logger.info(f'pop_size={pop_size}')
    logger.info(f'max_gen={max_gen}')

    # Instantiate the random search object.
    ga_search = GASearch(scen_enumLimits=cfg.scenario_enumLimits, radius=cfg.region_radius,
                         sim_input_dir=input_directory, sim_output_dir=output_directory)

    # Run the random search.
    ga_search.run(pop_size=pop_size, max_gen=max_gen, max_evals=sim_budget)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('sim_budget', nargs='?', type=int,
                           default=cfg.max_num_evals, help="simulation budget")
    argparser.add_argument('pop_size', nargs='?', type=int,
                           default=20, help="popuation size")
    argparser.add_argument('max_gen', nargs='?', type=int,
                           default=100, help="max number of generations")
    args = argparser.parse_args()
    main(sim_budget=args.sim_budget, pop_size=args.pop_size, max_gen=args.max_gen)
