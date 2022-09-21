from math import floor
# Search hyperparameters
scenario_population_size = 10  # Size of the scenario population
mlco_population_size = 10  # Size of the MLC output population
min_distance = 0.3  # Minimum distance between members of an archive
region_radius = 0.5  # The radius of the region for fitness evaluations
number_of_generations = 40
random_seed = None
max_num_evals = 51200
update_archive_strategy = 'bestRandom'

# Evolution hyperparameters
tournament_selection = 2
crossover_probability = 0.5
guassian_mutation_mean = 0
guassian_mutation_std = 0.1
mutation_rate = 1
guassian_mutation_probability = mutation_rate
integer_mutation_probability = mutation_rate
bitflip_mutation_probability = mutation_rate
population_archive_size = floor(scenario_population_size * 0.25)

# Output Directory Name
output_dir_name = 'CCEA_ONEMAX_MD' + str(min_distance*10) + '_MR' + str(mutation_rate*10) + '_STD' + \
    str(guassian_mutation_std*100) + '_CR' + str(crossover_probability*10) + \
    '_TS' + str(tournament_selection) + '_RR' + str(region_radius*100)

# Problem-specific parameters
enumLimits = [[0, 1], [0, 1], [0, 1], [0, 1], [
    0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
categorical_indices = [0, 1, 2, 3, 4, 5, 6, 7,
                       8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
numeric_ranges = []

categorical_indices_scen = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
categorical_indices_mlco = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
numeric_ranges_scen = []
numeric_ranges_mlco = []