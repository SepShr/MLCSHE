# Search hyperparameters
scenario_population_size = 10  # Size of the scenario population
mlco_population_size = 10  # Size of the MLC output population
min_distance = 0.5  # Minimum distance between members of an archive
number_of_generations = 5
random_seed = 10

# Evolution hyperparameters
tournament_selection = 2
crossover_probability = 0.5
guassian_mutation_mean = 0
guassian_mutation_std = 0.125
guassian_mutation_probability = 0.5
integer_mutation_probability = 0.5
bitflip_mutation_probability = 1

# Problem-specific parameters
enumLimits = [[0.0, 1.0]]
