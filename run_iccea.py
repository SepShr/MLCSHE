"""
iCCEA Runner.
"""

from src.main.ICCEA import ICCEA
from src.utils.utility import setup_logger
import problem
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

# Setup logger.
setup_logger(file_log_level='DEBUG', stream_log_level='INFO')

# User does not need to modify anything but `problem.py`
solution = solver.solve(
    max_gen=num_gen, hyperparameters=hyperparameters, seed=seed)

# print(f'solution={solution}')
