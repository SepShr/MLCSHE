base_directory = ""
container_name = "pylot_main"
base_config_file_name = "base_config.conf"
config_source_path = "/home/sepehr/AV/MLCSHE/MLCSHE/mlcshe_config.conf"
config_destination_path = "/home/erdos/workspace/pylot/configs/mlcshe/mlcshe_config.conf"
simulation_config_file_name = "mlcshe_config.conf"
simulation_duration = 100  # Simulation duration in seconds.
simulation_results_source_directory = "/home/erdos/workspace/results/"
simulation_results_destination_path = "./results/"
# The path to the finished.txt file in the container.
finished_file_path = '/home/erdos/workspace/results/finished.txt'
pylot_runner_path = "/home/sepehr/AV/MLCSHE/MLCSHE/run_pylot.sh"
carla_runner_path = "/home/erdos/workspace/pylot/scripts/run_simulator.sh"
