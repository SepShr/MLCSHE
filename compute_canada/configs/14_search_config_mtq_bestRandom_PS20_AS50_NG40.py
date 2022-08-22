from math import floor
# Search hyperparameters
scenario_population_size = 20
mlco_population_size = 20
min_distance = 0.5  # Minimum distance between members of an archive
region_radius = 0.05  # The radius of the region for fitness evaluations
number_of_generations = 40
random_seed = None
max_num_evals = 51200
update_archive_strategy = 'bestRandom'

# Evolution hyperparameters
tournament_selection = 2
crossover_probability = 0.5
guassian_mutation_mean = 0
guassian_mutation_std = 0.1
guassian_mutation_probability = 1
integer_mutation_probability = 1
bitflip_mutation_probability = 1
population_percentage = 50
population_archive_size = floor(
    scenario_population_size * (population_percentage/100))

# Output Directory Name
output_dir_name = 'CCEA_MTQ' + '_' + update_archive_strategy + '_PS' + \
    str(scenario_population_size) + '_AS' + \
    str(population_archive_size) + '_NG' + str(number_of_generations)

# Problem-specific parameters
enumLimits = [[0.0, 1.0]]
categorical_indices = []
numeric_ranges = [1, 1]

numeric_ranges_scen = [1]
numeric_ranges_mlco = [1]
