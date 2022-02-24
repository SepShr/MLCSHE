base_directory = ""
script_directory = 'scripts/'
container_name = "pylot"
carla_timeout = 10
pylot_timeout = 20
simulation_duration = 20  # Simulation duration in seconds.
mlco_file_name = 'mlco_list.pkl'
mlco_destination_path = '/home/erdos/workspace/pylot/dependencies/mlco/mlco_list.pkl'
base_config_file = "configs/base_config.conf"
# base_config_file = "configs/new_base_config.conf"
config_source_path = "/temp/mlcshe_config.conf"
config_destination_path = "/home/erdos/workspace/pylot/configs/mlcshe/mlcshe_config.conf"
simulation_config_file_name = "mlcshe_config.conf"
simulation_results_source_directory = "/home/erdos/workspace/results/"
simulation_results_destination_path = "./results/"
# The path to the finished.txt file in the container.
finished_file_path = '/home/erdos/workspace/results/finished.txt'
pylot_runner_path = '/scripts/run_pylot.sh'
carla_runner_path = "/home/erdos/workspace/pylot/scripts/run_simulator.sh"
