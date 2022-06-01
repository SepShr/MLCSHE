# Search hyperparameters
scenario_population_size = 5  # Size of the scenario population
mlco_population_size = 5  # Size of the MLC output population
min_distance = 0.5  # Minimum distance between members of an archive
number_of_generations = 5
random_seed = 10
max_num_evals = 51200

# Evolution hyperparameters
tournament_selection = 2
crossover_probability = 0.5
guassian_mutation_mean = 0
guassian_mutation_std = 0.125
guassian_mutation_probability = 0.5
integer_mutation_probability = 0.5
bitflip_mutation_probability = 1

# Problem-specific parameters
scenario_enumLimits = [[0, 2], [0, 6], [0, 1], [0, 3], [0, 3], [0, 0], [0, 2]]
total_mlco_messages = 700
total_obstacles_per_message = 3
frame_width = 800.0
frame_height = 480.0
min_boundingbox_size = 50.0
obstacle_label_enum_limits = [-1, 1]
null_obstacle = [0.0, 0.0, 0.0, 0.0, -1]
