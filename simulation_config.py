base_directory = ""
<<<<<<< HEAD
container_name = "pylot_main"
simulation_duration = 100  # Simulation duration in seconds.
=======
container_name = "pylot"
>>>>>>> e0202fe40f620bbd2b756a242622aa152604179a
base_config_file_name = "base_config.conf"
config_source_path = "/mlcshe_config.conf"
config_destination_path = "/home/erdos/workspace/pylot/configs/mlcshe/mlcshe_config.conf"
simulation_config_file_name = "mlcshe_config.conf"
simulation_results_source_directory = "/home/erdos/workspace/results/"
simulation_results_destination_path = "./results/"
# The path to the finished.txt file in the container.
finished_file_path = '/home/erdos/workspace/results/finished.txt'
pylot_runner_path = "/run_pylot.sh"
carla_runner_path = "/home/erdos/workspace/pylot/scripts/run_simulator.sh"
