"""
This file contains the adaptor classes for the algorithm to interface
with the simulation setup.

The classes and their encode and decode metods have to be modified for
every different problem.
"""


class Scenario(object):
    def __init__(self, scenario_params):
        # self.weather = scenario_params.weather
        # self.road = scenario_params.road
        self.world = scenario_params.world
        self.ego_vehicle = scenario_params.ego_vehicle

    # Encode the scenario datatype from the setup to a hetergeneous list.
    def encode(self) -> list:
        pass

    # Decode the scenario individual to simulation parameters
    def decode(self):
        pass

    # Initialize a scenario indiviudal
    def initialize(self):
        pass


class Mlco(object):
    def __init__(self):
        pass

    def as_simulator_mlco(self):
        # """ Retrieves the rotation as an instance of a simulator rotation.
        # Returns:
        #     An instance of a simulator class representing the rotation.
        # """
        # from pylot import LaneMessage
        # return Rotation(self.pitch, self.yaw, self.roll)
        pass

    # Encode the simulation setup's mlco datatype to mlco individual.
    def encode(self) -> list:
        pass

    # Decode the mlco individual to the simulation's MLC datatype.
    def decode(self):
        pass

    # Initialize an mlco indiviudal
    def initialize(self):
        pass
