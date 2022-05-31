import unittest

from deap import base, creator
from simulation_utils import translate_scenario_list


class TestTranslateScenarioList(unittest.TestCase):
    def test_translate_scenario_list_testInput1(self):
        # Prepare test input
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        creator.create("Scenario", creator.Individual)
        scenario_list = creator.Scenario([0, 4, 0, 3, 0, 0, 1])

        translated_scenario_list = translate_scenario_list(scenario_list)

        expected_value = {}
        expected_value['simulator_weather='] = '--simulator_weather=MidRainyNoon\n'
        expected_value['simulator_num_people='] = "--simulator_num_people=0\n"
        expected_value["simulator_town="] = "--simulator_town=5\n"
        expected_value["goal_location="] = "--goal_location=-184.585892, -53.541496, 0.600000\n"
        expected_value["simulator_spawn_point_index="] = "--simulator_spawn_point_index=137\n"
        expected_value["vehicle_in_front_spawn_point="] = "--vehicle_in_front_spawn_point=59\n"
        expected_value["vehicle_in_adjacent_spawn_point="] = "--vehicle_in_adjacent_spawn_point=60\n"
        expected_value["vehicle_in_opposite_spawn_point="] = "--vehicle_in_opposite_spawn_point=127\n"

        self.assertEqual(translated_scenario_list, expected_value)


if __name__ == '__main__':
    unittest.main()
