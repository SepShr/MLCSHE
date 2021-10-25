"""
iCCEA Runner.
"""

from src.main.ICCEA import ICCEA
import problem

# NOTE: that ICCEA is an algorithm, which is independent of a problem structure
solver = ICCEA(
    creator=problem.creator,
    toolbox=problem.toolbox,
    enumLimits=problem.enumLimits  # more parameters can be added to better define the problem
)

# User does not need to modify anything but `problem.py`
solution = solver.solve()

print(f'solution={solution}')
