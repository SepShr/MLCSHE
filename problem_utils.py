"""
This file contains the adaptor classes for the algorithm to interface
with the simulation setup.

The classes and their encode and decode metods have to be modified for
every different problem.
"""
import carla


class Scenario(object):  # Check inheritence.
    def __init__(self, scenario_list):
        # Get client.
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        # Get world.
        self.world = self.client.get_world()
        # Set weather parameters.
        self.weather = carla.WeatherParameters(
            cloudiness=scenario_list[0],
            precipitation=scenario_list[1],
            sun_altitude_angle=scenario_list[2]
        )  # FIXME: Provide exact mapping between list values and parameters.

        # Set light parameters.
        # Skipped for now.

        # Set world settings.
        # Skipped for now.

        # Set actors.
        # Skipped for now.
        # self.world.spawn_actor(}

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
