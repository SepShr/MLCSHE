"""
iCCEA Runner.
"""

from src.main.ICCEA import ICCEA
import problem

# NOTE: ICCEA is an algorithm, which is independent of a problem structure
solver = ICCEA(
    creator=problem.creator,
    toolbox=problem.toolbox,
    # more parameters can be added to better define the problem
    enumLimits=problem.enumLimits
)

# User does not need to modify anything but `problem.py`
solution = solver.solve(max_gen=250)

# print(f'solution={solution}')

# FIXME: Add Carla and Pylot instantiation and setup
'''
simulator1 = environment()
simulator1.set_scenario(individual1)

simulator2 = mysimulator()
simulator3 = mysimulator()
simulator4 = mysimulator()
'''
