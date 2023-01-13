from datetime import datetime
from pathlib import Path

# Setup directories
# Get current timestamp to use as a unique ID.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir_name = str(timestamp) + '_MLCSHE_Pylot'
input_directory = Path.cwd().joinpath('temp').joinpath(output_dir_name)
output_directory = Path.cwd().joinpath('results').joinpath(output_dir_name)

# Search hyperparameters
scenario_population_size = 10  # Size of the scenario population
mlco_population_size = 10  # Size of the MLC output population
min_distance = 0.4  # Minimum distance between members of an archive
region_radius = 0.3  # The radius of the region for fitness evaluations
number_of_generations = 30
# random_seed = 10
max_num_evals = 2500
update_archive_strategy = 'bestRandom'
fitness_function_target_probability = 0.9

# Evolution hyperparameters
tournament_selection = 3
crossover_probability = 0.5
guassian_mutation_mean = 0
guassian_mutation_std = 0.1 * 400
mutation_rate = 1
guassian_mutation_probability = mutation_rate
integer_mutation_probability = mutation_rate
bitflip_mutation_probability = mutation_rate
population_archive_size = 3

# Problem-specific parameters
scenario_enumLimits = [[0, 2], [0, 6], [0, 1], [0, 3], [0, 3], [0, 0], [0, 2]]
total_mlco_messages = 900
total_obstacles_per_message = 3
frame_width = 800.0
frame_height = 480.0
min_boundingbox_size = 50.0
obstacle_label_enum_limits = [-1, 1]
null_obstacle = [0.0, 0.0, 0.0, 0.0, -1]
num_trajectories = 2
duration = 900
min_trajectory_duration = 50
null_trajectory = [-1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

scenario_numeric_ranges = [1, 1, 1, 1, 1, 1, 1]
scenario_catgorical_indices = [0, 1, 2, 3, 4, 5, 6]

trajectory_numeric_ranges = [1, total_mlco_messages, total_mlco_messages,
                             frame_width, frame_height, frame_width, frame_height,
                             frame_width, frame_height, frame_width, frame_height]
categorical_indices = [0, 1, 2, 3, 4, 5, 6, 7]

mlco_numeric_ranges = trajectory_numeric_ranges
mlco_categorical_indices = [7]

numeric_ranges = [1, 1, 1, 1, 1, 1, 1,
                  1, total_mlco_messages, total_mlco_messages,
                  frame_width, frame_height, frame_width, frame_height,
                  frame_width, frame_height, frame_width, frame_height]
for i in range(num_trajectories - 1):
    numeric_ranges += trajectory_numeric_ranges
    mlco_numeric_ranges += trajectory_numeric_ranges

    categorical_indices.append(categorical_indices[-1] + 11)
    mlco_categorical_indices.append(mlco_categorical_indices[-1] + 11)

x_min_lb = 0.0
x_min_ub = frame_width - min_boundingbox_size
x_max_lb = min_boundingbox_size
x_max_ub = frame_width
y_min_lb = 0.0
y_min_ub = frame_height - min_boundingbox_size
y_max_lb = min_boundingbox_size
y_max_ub = frame_height

traj_enum_limits = [
    [-1, 1],
    [0.0, duration],
    [0.0, duration],
    [x_min_lb, x_min_ub],
    [x_max_lb, x_max_ub],
    [y_min_lb, y_min_ub],
    [y_max_lb, y_max_ub],
    [x_min_lb, x_min_ub],
    [x_max_lb, x_max_ub],
    [y_min_lb, y_min_ub],
    [y_max_lb, y_max_ub],
]
