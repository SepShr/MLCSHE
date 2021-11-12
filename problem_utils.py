"""
This file contains the adaptor classes for the algorithm to interface
with the simulation setup.

The classes and their encode and decode metods have to be modified for
every different problem.
"""
import carla
import numpy as np

SCENARIO_BOUNDS = [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0],
                   [0.0, 100.0], [0.0, 360.0], [-90.0, 90.0],
                   [0.0, 100.0], [0.0, 100.0], [0.0, np.inf],
                   [0.0, np.inf]]


class Environement:
    """
    This class creates a carla simulation instance and translates
    MLCSHE's Scenarios to Carla World parameters.
    """

    def __init__(self, host="localhost", port=2000, timeout=2.0):
        # Get client.
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        # Get world.
        self.world = self.client.get_world()

        # self.actor_list = []

    # Decode the scenario individual to simulation parameters
    def set_scenario(self, scenario_list):
        """
        Sets the World parameters based on a MLCSHE Scenario.

        The order mapping of World parameters to scenario_list indices is
        as follows:

        [0] Cloudiness amount [0.0, 100.0]
        [1] Precipitation (Rain) amount [0.0, 100.0]
        [2] Precipitation deposits (Puddles) amount [0.0, 100.0]
        [3] Wind intensity [0.0, 100.0]
        [4] Sun azimuth [0.0, 360.0]
        [5] Sun altitude [-90.0, 90.0]
        [6] Fog density [0.0, 100.0]
        [7] Fog Distance [0.0, inf)
        [8] Wetness intensity [0.0, 100.0]
        [9] Fog falloff [0, inf)
        """
        # Set weather parameters.
        cloudiness, precipitation, precipitation_deposits,\
            wind_intensity, sun_azimuth_angle, sun_altitude_angle,\
            fog_density, fog_distance, wetness, fog_falloff = scenario_list
        self.weather = carla.WeatherParameters(
            cloudiness=cloudiness,
            precipitation=precipitation,
            precipitation_deposits=precipitation_deposits,
            wind_intensity=wind_intensity,
            sun_azimuth_angle=sun_azimuth_angle,
            sun_altitude_angle=sun_altitude_angle,
            fog_density=fog_density,
            fog_distance=fog_distance,
            wetness=wetness,
            fog_falloff=fog_falloff
        )

        # Set light parameters.
        # Skipped for now.

        # Set world settings.
        # Skipped for now.

        # Set actors.
        # Skipped for now.
        # self.model_3 = self.blueprint_library.filter("model3")[0]
        # self.blueprint_library = self.world.get_blueprint_library()
        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        # self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        # self.actor_list.append(self.vehicle)

    # Encode the scenario datatype from the setup to a hetergeneous list.

    def get_scenario_list(self) -> list:
        """
        Returns a scenario_list based on the World parameters.
        """
        self.weather = self.world.get_weather()
        scenario_list = list(self.weather.cloudiness,
                             self.weather.precipitation,
                             self.weather.precipitation_deposits,
                             self.weather.wind_intensity,
                             self.weather.sun_azimuth_angle,
                             self.weather.sun_altitude_angle,
                             self.weather.fog_density,
                             self.weather.fog_distance,
                             self.weather.wetness,
                             self.weather.fog_falloff)

        # Actor-related parameters should be added.
        # Skipped for now.

        return scenario_list

    # Reset or reload the world.
    def reset(self):
        pass
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])


class MlcoAdapter:
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
