# Search hyperparameters
scenario_population_size = 15  # Size of the scenario population
mlco_population_size = 15  # Size of the MLC output population
min_distance = 0.5  # Minimum distance between members of an archive
number_of_generations = 100
random_seed = None
max_num_evals = 51200

# Evolution hyperparameters
tournament_selection = 2
crossover_probability = 0.5
guassian_mutation_mean = 0
guassian_mutation_std = 0.0125
guassian_mutation_probability = 0.25
integer_mutation_probability = 0.5
bitflip_mutation_probability = 1

# Problem-specific parameters
enumLimits = [[0.0, 1.0]]
categorical_indices = []
numeric_ranges = [1, 1]
