import pathlib
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import search_config as cfg
from RandomSearch import RandomSearch
from src.utils.utility import setup_logger


def main(sim_budget):
    # Setup directories
    # Get current timestamp to use as a unique ID.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = str(timestamp) + '_RS_Pylot'
    input_directory = Path.cwd().joinpath('temp').joinpath(output_dir_name)
    output_directory = Path.cwd().joinpath('results').joinpath(output_dir_name)

    output_dir = pathlib.Path('results').joinpath(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output folder.

    # Setup logger.
    setup_logger(file_name='RS', output_directory=output_dir, file_log_level='DEBUG',
                 stream_log_level='INFO')

    # Instantiate the random search object.
    random_search = RandomSearch(scen_enumLimits=cfg.scenario_enumLimits, radius=cfg.region_radius,
                                 sim_input_dir=input_directory, sim_output_dir=output_directory)

    # Run the random search.
    random_search.run(sim_budget)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('sim_budget', type=int,
                           default=cfg.max_num_evals, help="simulation budget")
    args = argparser.parse_args()
    main(args.sim_budget)
