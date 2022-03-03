import unittest

from deap import base, creator
from simulation_utils import translate_scenario_list


class TestTranslateScenarioList(unittest.TestCase):
    def test_translate_scenario_list_testInput1(self):
        # Prepare test input
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        creator.create("Scenario", creator.Individual)
        scenario_list = creator.Scenario([0, 4, 0])

        translated_scenario_list = translate_scenario_list(scenario_list)

        expected_value = {}
        expected_value['simulator_weather='] = '--simulator_weather=MidRainyNoon\n'
        expected_value['simulator_num_people='] = "--simulator_num_people=0\n"

        self.assertEqual(translated_scenario_list, expected_value)


if __name__ == '__main__':
    unittest.main()
