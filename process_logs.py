from copy import deepcopy
from itertools import combinations, product
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from collections import OrderedDict
from deap import base, creator
from scipy.stats import mannwhitneyu

import search_config as cfg
from src.utils.PairwiseDistance import PairwiseDistance

# plt.xkcd()
plt.style.use('bmh')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14


def read_files(files: list) -> dict:
    """Reads the files listed in `files` and returns a dictionary of
    the form {run_number: [fitness values]}.
    """
    results_dict = {}
    for run_number, file in enumerate(files, 1):
        with open(file, 'rb') as f:
            cs_archive = pickle.load(f)

        results_dict[f'run_{run_number}'] = cs_archive

    return results_dict


def postprocess_results(results: dict, max_fitness: float = 1.0, min_distance: float = 0.0, verbose: bool = False) -> dict:
    """Gets a `results` dict and returns a list of distinct
    complete solutions that have fitness values higher than
    `max_fitness` and pairwise distance higher than `min_distance`.
    """
    distinct_trimmed_fitness_dict = {}
    # total_fitness_dict = {}

    # for run_number, file in enumerate(files, 1):
    #     if verbose:
    #         print(f'Processing file {file} ...')

    #     with open(file, 'rb') as f:
    #         cs_archive = pickle.load(f)
    for run_number, cs_archive in results.items():
        # total_fitness_dict[run_number] = [cs.fitness.values[0]
        #                                   for cs in cs_archive]

        # Sort and trim the cs_archive based on the `max_fitness` value.
        boundaries = sort_trim(cs_archive, max_fitness)

        # Remove similar complete solutions from `cs_archive` based on `min_distance`.
        distinct_boundaries = remove_similar(boundaries, min_distance)

        if verbose:
            print(f'len(cs_archive) = {len(cs_archive)}')
            print(f'len(boundaries) = {len(boundaries)}')
            print(f'propotion = {len(boundaries) / len(cs_archive)}')
            print(f'len(distinct_boundaries) = {len(distinct_boundaries)}')

        distinct_trimmed_fitness_dict[run_number] = [cs.fitness.values[0]
                                                     for cs in distinct_boundaries]

    return distinct_trimmed_fitness_dict
    # , total_fitness_dict


def remove_similar(cs_archive: list, min_distance: 0.0) -> list:
    """Remove similar complete solutions from `cs_archive` based on
    their pairwise distance.
    """
    # Compute pairwise distance between all complete solutions solutions.
    pdist = PairwiseDistance(
        cs_archive, cfg.numeric_ranges, cfg.categorical_indices)

    # Remove solutions with pairwise distance lower than `min_distance`
    cs_archive_distinct = []
    for i in range(len(cs_archive)):
        for j in range(i + 1, len(cs_archive)):
            if pdist.get_distance(cs_archive[i], cs_archive[j]) >= min_distance:
                cs_archive_distinct.append(cs_archive[i])
                break

    return cs_archive_distinct


def sort_trim(cs_archive: list, max_fitness: float) -> list:
    """Sort and trim the `cs_archive` based on the `max_fitness` value.
    """
    # Sort the cs_archive from low to high fitness.
    cs_archive_sorted = sorted(
        cs_archive, key=lambda x: x.fitness.values[0])

    # Trim the cs archive
    for cs in cs_archive_sorted:
        if cs.fitness.values[0] >= max_fitness:
            cs_archive_trimmed = cs_archive_sorted[:cs_archive_sorted.index(
                cs)]
            break

    return cs_archive_trimmed


def draw_boxplot(data: dict, max_fitness: float = 1.0, min_distance: float = 0.0, x_label: str = 'X', y_label: str = 'Y', title: str = 'Title', output_dir: str = './results'):
    """Draw a boxplot for the `data` dictionary.
    """
    x_labels, values = list(data.keys()), list(data.values())

    # Draw the boxplot.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(values, showmeans=True, meanline=True)
    ax.set_title(
        f'Boxplot of {title} (max_fitness = {max_fitness}, min_distance = {min_distance})')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(x_labels)

    # Save the figure.
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f'rq1_{title}_mf{max_fitness}_md{min_distance}.png'
    pdf_path = out_dir / f'rq1_{title}_mf{max_fitness}_md{min_distance}.pdf'
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    print(f'Figure saved to {png_path}')
    print(f'Figure saved to {pdf_path}')


def calculate_dbs_per_search(data: dict):
    """Calculate the Size of Estimated Boundary (dbs) for each run.
    """
    dbs_results = []
    for fitness_values in data.values():
        dbs_results.append(len(fitness_values))

    return dbs_results


def calculate_dbs(data: dict):
    dbs_results = {}
    # print(f'key = {list(list(data.values())[0].keys())[0]}')
    if list(list(data.values())[0].keys())[0].__contains__('interval'):
        for search_method, interval_results in data.items():
            dbs_results[search_method] = {}
            for interval, results in interval_results.items():
                # Calculate the Size of Estimated Boundary (dbs) for each run.
                dbs_results_per_interval = [len(fitness_values)
                                            for fitness_values in results.values()]
                dbs_results[search_method][interval] = dbs_results_per_interval
    else:
        for search_method, results in data.items():
            # Calculate the Size of Estimated Boundary (dbs) for each run.
            dbs_results_per_search = [len(fitness_values)
                                      for fitness_values in results.values()]
            dbs_results[search_method] = dbs_results_per_search

    return dbs_results


def calcluate_afv(data: dict):
    afv_results = {}
    for search_method, results in data.items():
        # Calculate the Average Fitness of Estimated Boundary (afv) for each run.
        afv_results_per_search = []
        for fitness_values in results.values():
            afv_results_per_search.append(np.mean(fitness_values))
        afv_results[search_method] = afv_results_per_search
    return afv_results


def perform_mannwhitneyu_test(data: dict, parameter_name: str = 'Y', alpha: float = 0.01):
    # We use Mann-Whitney U test to compare the results of different search methods.
    # The null hypothesis is that the two samples are drawn from the same distribution.
    # The alternative hypothesis is that the two samples are drawn from different distributions.
    # If the p-value is less than the significance level (alpha), we reject the null hypothesis.
    # Otherwise, we fail to reject the null hypothesis.

    # Perform the Mann-Whitney U test.
    print(f'Mann-Whitney U test for {parameter_name}')
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            search_method1, search_method2 = list(
                data.keys())[i], list(data.keys())[j]
            stat, p = mannwhitneyu(data[search_method1],
                                   data[search_method2])
            print(
                f'{search_method1} vs {search_method2}: stat={stat}, p={p}')

            if p > alpha:
                print('Same distribution (fail to reject H0)')
            else:
                print('Different distribution (reject H0)')


def analyze_statistics(data: dict, parameter_name: str = 'Y', alpha: float = 0.01):
    print(f'\nMann-Whitney U test for {parameter_name}')
    print('***************************************')

    for search_method_1, search_method_2 in product(data.keys(), repeat=2):
        if search_method_1 == search_method_2:
            continue
        print(f'{search_method_1} vs {search_method_2}:')
        mwu_results = pg.mwu(data[search_method_1], data[search_method_2])
        print(mwu_results)

        u_value = mwu_results['U-val'].values[0]
        p_value = mwu_results['p-val'].values[0]
        cles_value = mwu_results['CLES'].values[0]

        print(f'U-value: {u_value}, p-value: {p_value}, CLES: {cles_value}')

        if p_value > alpha:
            print('Same distribution (fail to reject H0)')
        else:
            print('Different distribution (reject H0)')
        print('----------------------------------------')
    print('\n')


def answer_rq1(processed_results: dict, max_fitness: float = 1.0, min_distance: float = 0.0, output_dir: str = './results'):
    """Answer RQ1: How effective MLCSHE is compared to baseline approaches?"""
    # Part 1.1: Calculate and draw boxplots for dbs.
    dbs_results = calculate_dbs(processed_results)
    # print(dbs_results)
    draw_boxplot(dbs_results, max_fitness, min_distance, x_label='Search Method',
                 y_label='DBS', title='DBS', output_dir=output_dir)

    # Part 1.2: Perform statistical test of dbs results.
    # Perform Mann-Whitney U test and measure effect size of dbs results.
    analyze_statistics(dbs_results, 'dbs')

    #  Part 2.1: Calculate and draw boxplots for afv values.
    afv_results = calcluate_afv(processed_results)
    draw_boxplot(afv_results, max_fitness, min_distance, x_label='Search Method',
                 y_label='AFV', title='AFV', output_dir=output_dir)

    # Part 2.2: Perform statistical test of afv results.
    # Perform Mann-Whitney U test and measure effect size of afv results.
    analyze_statistics(afv_results, 'afv')


def divide_results_into_intervals(results: dict, intervals: int = 10) -> dict:
    """Returns a dictionary with the results divided into intervals.
    The format of the returned dictionary is as follows:\n
    {
        'interval_1': {
            'run_1': [fitness_values],
            'run_2': [fitness_values],
            ...
        },
        ...
    }
    """
    divided_results = {f'interval_{i}': {} for i in range(1, intervals + 1)}
    for run, cs_archive in results.items():
        # Divide the cs_archive into intervals.
        # Each interval contains members of previous intervals, as well as its interval's members.
        interval_step = len(cs_archive) // intervals
        for i in range(1, intervals + 1):
            divided_results[f'interval_{i}'][run] = cs_archive[(i - 1) *
                                                               interval_step:i * interval_step]

    # Add the members of previous intervals to the current interval.
    for i in range(2, intervals + 1):
        for run, cs_archive in divided_results[f'interval_{i}'].items():
            divided_results[f'interval_{i}'][run] = [
                *divided_results[f'interval_{i-1}'][run], *cs_archive]

    return divided_results


def draw_progression_plot(data: pd.DataFrame, max_fitness: float = 1.0, min_distance: float = 0.0, x_label: str = 'X', y_label: str = 'Y', title: str = 'Title', intervals: int = 10, output_dir: str = './results'):
    """Draw plot with error bars."""
    # x_labels, values = list(data.keys()), list(data.values())
    interval_step = 100//intervals
    x_labels = [*range(interval_step, 100+interval_step, interval_step)]
    print(x_labels)

    # Draw the boxplot.
    fig, ax = plt.subplots(figsize=(8, 6))
    # print(data)
    # ax.errorbar(data['interval'], data['mean'],
    #             yerr=data['err'], fmt='o')
    lines = []
    for search_method in data.search_method.unique():
        data_per_search_method = data[data['search_method'] == search_method]
        # Draw fill_between.
        ax.fill_between(
            data_per_search_method['interval'], data_per_search_method['min'], data_per_search_method['max'], alpha=0.2)
        # Draw line plot.
        line = ax.plot(data_per_search_method['interval'],
                       data_per_search_method['mean'], label=search_method, linewidth=2)
        lines += line

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left")
    ax.set_title(
        f'Progression of {title} over {intervals} intervals\n(max_fitness = {max_fitness}, min_distance = {min_distance})')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_xticklabels(x_labels)

    # Save the figures.
    # plt.show()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f'rq2_{title}_mf{max_fitness}_md{min_distance}.png'
    pdf_path = out_dir / f'rq2_{title}_mf{max_fitness}_md{min_distance}.pdf'
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    print(f'Figure saved to {png_path}')
    print(f'Figure saved to {pdf_path}')


def answer_rq2(processed_results: dict, max_fitness: float = 1.0, min_distance: float = 0.0, intervals: int = 10, output_dir: Path = './results/'):
    # pass

    # Part 1: Calculate and draw scatter plots for dbs.
    dbs_results = calculate_dbs(processed_results)

    # for search_method, interval_values in dbs_results.items():
    #     # for interval, value in interval_values.items():
    #     for i in range(2, intervals+1):
    #         dbs_results[search_method][f'interval_{i}'] = [dbs_results[search_method][f'interval_{i}'][j] +
    #                                                        dbs_results[search_method][f'interval_{i-1}'][j] for j in range(len(dbs_results[search_method][f'interval_{i}']))]
    # print(dbs_results)
    dbs_df = pd.DataFrame()
    for search_method, interval_values in dbs_results.items():
        for interval, value in interval_values.items():
            dbs_df = pd.concat([dbs_df, pd.DataFrame(
                [{'search_method': search_method, 'interval': int(interval.split('_')[-1]), 'max': np.max(value), 'min': np.min(value), 'mean': np.mean(value), 'err': np.ptp(value)/2}])])
    dbs_df.reset_index(drop=True, inplace=True)

    # rq2_df = rq2_df.append(
    #     {'search_method': search_method, 'interval': int(interval.split('_')[-1]), 'value': value}, ignore_index=True)

    # print(dbs_df)

    draw_progression_plot(dbs_df, max_fitness, min_distance,
                          x_label='Search Method', y_label='dbs', title='dbs', intervals=intervals, output_dir=output_dir)


def main(max_fitness: float = 0.2, min_distance: float = 0.4, intervals: int = 10, output_dir: str = './results/diagrams/'):
    # Create fitness and individual datatypes.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin,
                   safety_req_value=float)
    creator.create("Scenario", creator.Individual)
    creator.create("OutputMLC", creator.Individual)

    print('Importing results...')

    # Import the results.
    mlcshe_results = [
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221002_184723_CCEA_Pylot\20221002_184723_CCEA_Pylot\20221002_184724_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221005_220401_CCEA_Pylot\20221005_220401_CCEA_Pylot\20221005_220401_cs_archive_pickle.log',
        # r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221009_230946_CCEA_Pylot\20221009_230946_CCEA_Pylot\20221009_230946_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221012_203758_CCEA_Pylot\20221012_203758_CCEA_Pylot\20221012_203758_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221016_234557_CCEA_Pylot\20221016_234557_CCEA_Pylot\20221016_234557_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221020_203533_CCEA_Pylot\20221020_203533_CCEA_Pylot\20221020_203533_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221024_144711_CCEA_Pylot\20221024_144711_CCEA_Pylot\20221024_144711_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221027_162715_CCEA_Pylot\20221027_162715_CCEA_Pylot\20221027_162715_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221101_211509_CCEA_Pylot\20221101_211509_CCEA_Pylot\20221101_211509_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221104_191808_CCEA_Pylot\20221104_191808_CCEA_Pylot\20221104_191808_cs_archive_pickle.log',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_MLCSHE_Results\20221208_172525_CCEA_Pylot\20221208_172525_CCEA_Pylot\20221208_172525_cs_archive_pickle.log'
    ]
    rs_results = [
        # r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20220926_010645_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221003_090030_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221005_203358_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221008_121823_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221010_214517_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221016_191947_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221019_093933_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221025_082242_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221027_191843_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221104_093109_RS_Pylot\_RS_cs.pkl',
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_RS_Results\20221107_070910_RS_Pylot\_RS_cs.pkl'
    ]

    ga_results = [
        r'C:\Users\sepeh\Projects\MLCSHE\results\Pylot_GA_Results\20221209_211827_GA_Pylot\_GA_cs.pkl'
    ]

    mlcshe_results_dict = read_files(mlcshe_results)
    print('MLCSHE results read.')
    rs_results_dict = read_files(rs_results)
    print('RS results read.')
    ga_results_dict = read_files(ga_results)
    print('GA Results read.')

    processed_mlcshe_results_rq1 = postprocess_results(
        mlcshe_results_dict, max_fitness, min_distance)
    print('MLCSHE results postprocessed.')
    processed_rs_results_rq1 = postprocess_results(
        rs_results_dict, max_fitness, min_distance)
    print('RS results postprocessed.')
    processed_ga_results_rq1 = postprocess_results(
        ga_results_dict, max_fitness, min_distance)
    print('GA Results postprocessed.')
    print(f'GA Results: {processed_ga_results_rq1}')

    processed_results_rq1 = {}
    processed_results_rq1['MLCSHE'] = processed_mlcshe_results_rq1
    processed_results_rq1['RS'] = processed_rs_results_rq1
    processed_results_rq1['GA'] = processed_ga_results_rq1

    divided_mlcshe_results = divide_results_into_intervals(
        mlcshe_results_dict, intervals)
    divided_rs_results = divide_results_into_intervals(
        rs_results_dict, intervals)
    divided_ga_results = divide_results_into_intervals(
        ga_results_dict, intervals)
    print('Results divided into intervals.')

    processed_mlcshe_results_rq2 = {interval: postprocess_results(
        results_per_interval, max_fitness, min_distance) for interval, results_per_interval in divided_mlcshe_results.items()}
    # print(processed_mlcshe_results_rq2)
    print('MLCSHE results postprocessed.')
    processed_rs_results_rq2 = {interval: postprocess_results(
        results_per_interval, max_fitness, min_distance) for interval, results_per_interval in divided_rs_results.items()}
    print('RS results postprocessed.')
    processed_ga_results_rq2 = {interval: postprocess_results(
        results_per_interval, max_fitness, min_distance) for interval, results_per_interval in divided_ga_results.items()}
    print('GA Results postprocessed.')
    print(f'GA Results: {processed_ga_results_rq2}')

    processed_results_rq2 = {}
    processed_results_rq2['MLCSHE'] = processed_mlcshe_results_rq2
    processed_results_rq2['RS'] = processed_rs_results_rq2
    processed_results_rq2['GA'] = processed_ga_results_rq2

    answer_rq1(processed_results_rq1, max_fitness, min_distance, output_dir)
    answer_rq2(processed_results_rq2, max_fitness,
               min_distance, intervals, output_dir)


if __name__ == '__main__':
    main()


# def process_old_text_logs(f):
#     cs_dict = {}
#     with open(f, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line_parts = line.split('jfit_value=')
#             cs = line_parts[0].strip().split('=')[1][:-2]
#             fv = float(line_parts[1].split(',')[0])
#             cs_dict[cs] = fv

#     print(f'len(cs_dict) = {len(cs_dict)}')

#     cs_dict_sorted = sorted(cs_dict.items(), key=lambda x: x[1], reverse=True)

#     best_cs = {}

#     for cs, fv in cs_dict_sorted:
#         if fv <= fv_threshold:
#             best_cs[cs] = fv

#     print(f'len(best_cs) = {len(best_cs)}')

#     print(f'proportion = {len(best_cs) / len(cs_dict)}')

#     # print(best_cs)


# def process_pickled_cs(f):
#     # Create fitness and individual datatypes.
#     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMin,
#                    safety_req_value=float)
#     creator.create("Scenario", creator.Individual)
#     creator.create("OutputMLC", creator.Individual)

#     cs_list = {}
#     with open(f, 'rb') as f:
#         cs_list = pickle.load(f)

#     print(f'len(cs_dict) = {len(cs_list)}')

#     cs_list_sorted = sorted(
#         cs_list, key=lambda x: x.fitness.values[0], reverse=True)

#     # print(cs_list_sorted[0].fitness.values[0])
#     best_cs = {}

#     for cs in cs_list_sorted:
#         if cs.fitness.values[0] <= fv_threshold:
#             best_cs[cs] = cs.fitness.values[0]

#     print(f'len(best_cs) = {len(best_cs)}')

#     print(f'propotion = {len(best_cs) / len(cs_list)}')

#     # print(best_cs)

# # The complete solutions for the fv threshold value of 0.05:

# # Create fitness and individual datatypes.
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# # Structure initializers
# creator.create("Individual", list, fitness=creator.FitnessMin,
#                safety_req_value=float)  # Minimization
# creator.create("Scenario", creator.Individual)
# creator.create("OutputMLC", creator.Individual)

# scen_1 = creator.Scenario([2, 1, 0, 0, 0, 0, 2])
# mlco_1 = creator.OutputMLC([[0, 154, 204, 21.632, 71.632, 3.676, 86.149, 30.989, 80.989, 43.558, 93.558], [
#                            0, 22, 741, 86.753, 136.753, 14.174, 64.174, 0.0, 50.0, 29.188, 81.438]])
# cs_1 = creator.Individual([scen_1, mlco_1])
# cs_1.fitness.values = (0.0493206295939157,)

# scen_2 = creator.Scenario([2, 4, 1, 3, 0, 0, 0])
# mlco_2 = creator.OutputMLC([[0, 154, 204, 21.632, 71.632, 3.676, 86.149, 30.989, 80.989, 43.558, 93.558], [
#                            1, 563, 741, 56.89, 106.89, 25.94, 75.94, 0.0, 50.0, 7.042, 126.788]])
# cs_2 = creator.Individual([scen_2, mlco_2])
# cs_2.fitness.values = (0.0493206295939157,)

# scen_3 = creator.Scenario([0, 1, 0, 0, 0, 0, 0])
# mlco_3 = creator.OutputMLC([[1, 124, 174, 12.631, 62.631, 17.14, 67.14, 0.0, 50.0, 0.0, 50.0], [
#                            1, 301, 351, 0.0, 50.0, 69.385, 119.385, 0.0, 50.0, 12.626, 62.626]])
# cs_3 = creator.Individual([scen_3, mlco_3])
# cs_3.fitness.values = (0.049118118847713554,)

# scen_4 = creator.Scenario([0, 4, 1, 0, 0, 0, 0])
# mlco_4 = creator.OutputMLC([[1, 691, 741, 0.0, 106.403, 0.0, 86.807, 0.0, 50.222, 0.0, 90.003], [
#                            0, 453, 503, 0.0, 50.0, 0.0, 129.968, 67.924, 120.147, 14.652, 89.676]])
# cs_4 = creator.Individual([scen_4, mlco_4])
# cs_4.fitness.values = (0.049118118847713554,)

# scen_5 = creator.Scenario([0, 1, 1, 0, 2, 0, 2])
# mlco_5 = creator.OutputMLC([[1, 709, 759, 83.276, 133.276, 0.0, 50.0, 37.859, 87.859, 22.062, 72.062], [
#                            0, 317, 503, 0.0, 50.0, 0.0, 128.377, 106.169, 156.169, 49.642, 99.642]])
# cs_5 = creator.Individual([scen_5, mlco_5])
# cs_5.fitness.values = (0.049118118847713554,)


# # Instantiate pairwise distance instance.
# pairwise_distance_cs = PairwiseDistance(
#     cs_list=[],
#     numeric_ranges=cfg.numeric_ranges,
#     categorical_indices=cfg.categorical_indices
# )

# pairwise_distance_cs.update_dist_matrix([cs_1, cs_2, cs_3, cs_4, cs_5])
# print(pairwise_distance_cs)

"""Pairwise Distance Matrix = [
        cs_1        cs_2       cs_3        cs_4        cs_5
cs_1    [0.         0.29932868 0.26417305 0.35070633 0.29881057]
cs_2    [0.29932868 0.         0.31780621 0.28478782 0.46254021]
cs_3    [0.26417305 0.31780621 0.         0.22290838 0.28626642]
cs_4    [0.35070633 0.28478782 0.22290838 0.         0.18689737]
cs_5    [0.29881057 0.46254021 0.28626642 0.18689737 0.        ]
    ]
 """
