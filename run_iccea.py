"""
iCCEA Runner.
"""

from src.main.ICCEA import ICCEA
import problem
import logging
from datetime import datetime

# NOTE: ICCEA is an algorithm, which is independent of a problem structure
solver = ICCEA(
    creator=problem.creator,
    toolbox=problem.toolbox,
    # more parameters can be added to better define the problem
    enumLimits=problem.enumLimits
)

# Search hyperparameters
num_gen = 2
seed = 10
# Evolution hyperparameters
ts = 2
cxpb = 0.5
mut_bit_pb = 1
mut_guass_mu = 0
mut_guass_sig = 0.125
mut_guass_pb = 0.5
mut_int_pb = 0.5

hyperparameters = [
    ts,
    cxpb,
    mut_bit_pb,
    mut_guass_mu,
    mut_guass_sig,
    mut_guass_pb,
    mut_int_pb
]

logger = logging.getLogger('')

formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

now = datetime.now().strftime("%Y-%m-%d_%H:%M")
log_file_name = 'results/' + str(now) + '_CCEA' + '.log'
logging.basicConfig(filename=log_file_name,
                    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
# file_handler = logging.FileHandler(log_file_name)
# file_handler.setFormatter(formatter)
# file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

# logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("CCEA search started.")
logger.info('Maximum number of generations is: {}'.format(num_gen))
logger.info('Random seed is set to: {}'.format(seed))

# User does not need to modify anything but `problem.py`
solution = solver.solve(
    max_gen=num_gen, hyperparameters=hyperparameters, seed=seed)

# print(f'solution={solution}')
