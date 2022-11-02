from tokenize import group
from traceback import print_tb
import matplotlib.pyplot as plt
import pickle
from itertools import combinations, product
import glob
import numpy as np
from random import randint
# import scipy.stats as stats
import pingouin as pg
import pandas as pd

from statsmodels.graphics.factorplots import interaction_plot

# plt.xkcd()
plt.style.use('bmh')
plt.rcParams["font.family"] = "Times New Roman"

MODES = ['mins', 'avgs', 'maxs']
# pops = ['cs', 'p1', 'p2']
POPS = ['cs', 'p']  # Since p1 and p2 are similar.
POP_SIZE = ['PS10', 'PS20', 'PS30']
SIZE_OF_SMALLEST_SUBGROUP = 30


def read_logbook(logbook_file):
    """Reads the logbook file from a list of file paths.

    Returns a dictionary of lists of fitness values. 
    """
    # name = '_'.join(logbook_file.split('\\')[-2].split('_')[3:])

    name = '_'.join(logbook_file.split('\\')[-2].split('_')[1:])
    # name = '_'.join(logbook_file.split('\\')[-2].split('_')[:])

    with open(logbook_file, 'rb') as f:
        logbook = pickle.load(f)

    gen = logbook.select("gen")
    gen = gen[0::3]

    fitness_dict_final = {}  # Fitness values at the final generation.
    fitness_dict_gen = {}  # Fitness values at every generation.
    fitness_dict_gen['gen'] = gen

    # Retrieve fitness values for scen, mlco and cs.
    fit_mins = logbook.chapters["fitness"].select("min")
    fit_avgs = logbook.chapters["fitness"].select("avg")
    fit_maxs = logbook.chapters["fitness"].select("max")

    fitness_lists = [fit_mins, fit_avgs, fit_maxs]

    for pop, mode in product(POPS, MODES):
        fitness_dict_gen[f'{pop}_fit_{mode}'] = fitness_lists[MODES.index(
            mode)][POPS.index(pop)::3]

    for key, value in fitness_dict_gen.items():
        if key != 'gen':
            fitness_dict_final[key] = value[-1]

    return name, fitness_dict_gen, fitness_dict_final


def read_log_files(log_files: list):
    """Given a list of log files, it reutrns a dictionary that
    contains dictionaries of fitness value lists for each file.
    """
    extracted_values_gen = {}

    extracted_values_final = {}

    for file in log_files:
        file_name, fit_values_gen_dict, fit_values_final_dict = read_logbook(
            file)
        extracted_values_gen[file_name] = fit_values_gen_dict
        extracted_values_final[file_name] = fit_values_final_dict

    return extracted_values_gen, extracted_values_final


def plot_fit_vs_ngen(run_name: str, fitnesses_per_run: dict):
    """Plots and saves fitness values per number of generation
    for a run.
    """
    # Setup plot.
    fig, ax = plt.subplots()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")

    gen = fitnesses_per_run['gen']

    lines = []

    for fitness_type in fitnesses_per_run:
        if fitness_type != 'gen':
            fitness_list = fitnesses_per_run[fitness_type]
            line = ax.plot(gen, fitness_list, "o-", label=fitness_type)
            lines += line

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best")
    ax.set_title(f'Fitness v. Generation for {run_name}')

    plt.savefig(f'fit_vs_gen_{run_name}.png')

    plt.close('all')


def plot_fit_max(logbooks_dict: dict):
    """Plots and saves the figure containing fitness_maxs for cs
    and pop.
    """
    # Setup plot.
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax1.set_xlabel("Generation")
    ax2.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax2.set_ylabel("Fitness")
    ax1.set_title(f'Max $CS$ Fitness v. Generation')
    ax2.set_title(f'Max $P$ Fitness v. Generation')

    lines_1 = []
    lines_2 = []

    for run_name, fits_dict in logbooks_dict.items():
        gen = fits_dict['gen']
        for fit_type in fits_dict:
            if fit_type == 'cs_fit_maxs':
                fit_list = fits_dict[fit_type]
                line = ax1.plot(gen, fit_list, "o-", label=run_name)
                lines_1 += line
            elif fit_type == 'p_fit_maxs':
                fit_list = fits_dict[fit_type]
                line = ax2.plot(gen, fit_list, "x-", label=run_name)
                lines_2 += line

    labels_1 = [l.get_label() for l in lines_1]
    labels_2 = [l.get_label() for l in lines_2]
    ax1.legend(lines_1, labels_1, loc="best")
    ax2.legend(lines_2, labels_2, loc="best")

    plt.savefig(f'fit_vs_gen_max_fits.png')

    plt.close('all')


def plot_fit_min(logbooks_dict: dict):
    """Plots and saves the figure containing fitness_mins for cs
    and pop.
    """
    # Setup plot.
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax1.set_xlabel("Generation")
    ax2.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax2.set_ylabel("Fitness")
    ax1.set_title(f'Min $CS$ Fitness v. Generation')
    ax2.set_title(f'Min $P$ Fitness v. Generation')

    lines_1 = []
    lines_2 = []

    for run_name, fits_dict in logbooks_dict.items():
        gen = fits_dict['gen']
        for fit_type in fits_dict:
            if fit_type == 'cs_fit_mins':
                fit_list = fits_dict[fit_type]
                line = ax1.plot(gen, fit_list, "o-", label=run_name)
                lines_1 += line
            elif fit_type == 'p_fit_mins':
                fit_list = fits_dict[fit_type]
                line = ax2.plot(gen, fit_list, "x-", label=run_name)
                lines_2 += line

    labels_1 = [l.get_label() for l in lines_1]
    labels_2 = [l.get_label() for l in lines_2]
    ax1.legend(lines_1, labels_1, loc="best")
    ax2.legend(lines_2, labels_2, loc="best")

    plt.savefig(f'fit_vs_gen_min_fits.png')

    plt.close('all')


def plot_fit_max_diff(logbooks_dict):
    """Plots and saves the figure containing fitness max differences
    for cs and pop."""
    # Setup plot.
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    ax1.set_xlabel("Generation")
    ax2.set_xlabel("Generation")
    ax1.set_ylabel("$\Delta$ Fitness")
    ax2.set_ylabel("$\Delta$ Fitness")
    ax1.set_title(f'$\Delta$ Max_Fitness of $CS$ v. Generation')
    ax2.set_title(f'$\Delta$ Max_Fitness of $P$ v. Generation')

    lines_1 = []
    lines_2 = []

    for run_name, fits_dict in logbooks_dict.items():
        gen = fits_dict['gen'][:-1]
        for fit_type in fits_dict:
            if fit_type == 'cs_fit_maxs':
                fit_list = fits_dict[fit_type]
                fit_diff_list = [t - s for s, t in zip(fit_list, fit_list[1:])]
                line = ax1.plot(gen, fit_diff_list, "o-", label=run_name)
                lines_1 += line
            elif fit_type == 'p_fit_maxs':
                fit_list = fits_dict[fit_type]
                fit_diff_list = [t - s for s, t in zip(fit_list, fit_list[1:])]
                line = ax2.plot(gen, fit_diff_list, "x-", label=run_name)
                lines_2 += line

    labels_1 = [l.get_label() for l in lines_1]
    labels_2 = [l.get_label() for l in lines_2]
    ax1.legend(lines_1, labels_1, bbox_to_anchor=(0, 1.08, 1, 0.2),
               loc="lower center", mode="expand", borderaxespad=0, ncol=3)
    ax2.legend(lines_2, labels_2, bbox_to_anchor=(0, 0, 1, 0.2),  bbox_transform=fig.transFigure,
               loc="lower right",  borderaxespad=0, ncol=3)

    plt.savefig(f'fit_vs_gen_max_fit_diffs.png')

    plt.close('all')


def get_parameter_modalities(fitness_dict: dict, parameter_type: str):
    """Gets the different values of a parameter type from a dict.
    """
    parameter_modalities = set()
    for key in fitness_dict.keys():
        for i in key.split('_'):
            if i.__contains__(parameter_type):
                parameter_modalities.add(i[2:])

    return parameter_modalities


def get_max_fit_per_type(fitness_dict: dict, parameter_type: str):
    """Sorts the values and keys of a fitness dict according to
    the given parameter type and its values.
    """
    new_dict = {}

    modalities = get_parameter_modalities(fitness_dict, parameter_type)

    for modality in modalities:
        new_dict[parameter_type+modality] = dict()
        new_dict[parameter_type+modality]['p'] = list()
        new_dict[parameter_type+modality]['cs'] = list()
        for key, val in fitness_dict.items():
            max_cs_values_per_modality = []
            max_p_values_per_modality = []
            if key.__contains__(parameter_type+modality):
                for k, v in val.items():
                    if k.__contains__('p_') and k.__contains__('max'):
                        max_p_values_per_modality += list(v)
                    elif k.__contains__('cs_') and k.__contains__('max'):
                        max_cs_values_per_modality += list(v)
            new_dict[parameter_type+modality]['p'] += max_p_values_per_modality
            new_dict[parameter_type+modality]['cs'] += max_cs_values_per_modality

    return new_dict


def get_max_fit_per_arcsize(fitness_dict: dict):
    """Sorts the values and keys of a fitness dict according to
    the given parameter type and its values.
    """
    new_dict = {}
    parameter_type = 'AS'
    as_modalities = ['25', '50', '75']
    ps_modalities = ['PS10', 'PS20', 'PS30']

    mapping = {
        'PS10': [2, 5, 7],
        'PS20': [5, 10, 15],
        'PS30': [7, 15, 22]
    }

    for modality in as_modalities:
        new_dict[parameter_type+modality] = dict()
        new_dict[parameter_type+modality]['p'] = list()
        new_dict[parameter_type+modality]['cs'] = list()
        for key, val in fitness_dict.items():
            max_cs_values_per_modality = []
            max_p_values_per_modality = []
            for psm in ps_modalities:
                if key.__contains__(psm) and key.__contains__(parameter_type+str(mapping[psm][as_modalities.index(modality)])):
                    for k, v in val.items():
                        if k.__contains__('p_') and k.__contains__('max'):
                            max_p_values_per_modality += list(v)
                        elif k.__contains__('cs_') and k.__contains__('max'):
                            max_cs_values_per_modality += list(v)
            new_dict[parameter_type+modality]['p'] += max_p_values_per_modality
            new_dict[parameter_type+modality]['cs'] += max_cs_values_per_modality

    return new_dict


def get_max_fit_per_ua_strat(fitness_dict, strategies=['best', 'bestRandom', 'random']):
    new_dict = {}

    for strategy in strategies:
        new_dict[strategy] = dict()
        new_dict[strategy]['p'] = list()
        new_dict[strategy]['cs'] = list()
        for key, val in fitness_dict.items():
            max_cs_values_per_modality = []
            max_p_values_per_modality = []
            if key.__contains__(strategy):
                for k, v in val.items():
                    if k.__contains__('p_') and k.__contains__('max'):
                        max_p_values_per_modality += list(v)
                    elif k.__contains__('cs_') and k.__contains__('max'):
                        max_cs_values_per_modality += list(v)
            new_dict[strategy]['p'] += max_p_values_per_modality
            new_dict[strategy]['cs'] += max_cs_values_per_modality

    return new_dict


def plot_max_fit_vs_popsize_boxplot(processed_logbooks: dict, modalities=None):
    """Plots the maximum fitness vs population size.
    """
    max_fit_per_ps_dict = get_max_fit_per_type(processed_logbooks, 'PS')
    # print(max_fit_per_ps_dict)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Set x and y axis labels.
    for i in range(2):
        axs[i].set_xlabel("Population Size")
        axs[i].set_ylabel("Max Fitness")

    # Set subplot titles.
    axs[0].set_title('Max $CS$ Fitness v. Population Size')
    axs[1].set_title('Max $P$ Fitness v. Population Size')

    data_cs = []
    data_p = []

    if modalities:
        for m in modalities:
            data_cs.append(max_fit_per_ps_dict[m]['cs'])
            data_p.append(max_fit_per_ps_dict[m]['p'])
        # axs[0, 0].set_xticks([y + 1 for y in range(len(data_cs))])
        axs[0].set_xticklabels(modalities)
        # axs[0, 1].set_xticks([y + 1 for y in range(len(data_p))])
        axs[1].set_xticklabels(modalities)
    else:
        for k in max_fit_per_ps_dict.keys():
            data_cs.append(max_fit_per_ps_dict[k]['cs'])
            data_p.append(max_fit_per_ps_dict[k]['p'])
        axs[0].set_xticklabels(list(max_fit_per_ps_dict.keys()))
        axs[1].set_xticklabels(list(max_fit_per_ps_dict.keys()))

    axs[0].boxplot(data_cs)
    axs[1].boxplot(data_p)

    plt.savefig('max_fit_vs_pop_size.png')

    plt.close('all')


def plot_max_fit_vs_arcsize_boxplot(processed_logbooks: dict, modalities=None):
    """Plots the maximum fitness vs population archive proportion.
    """
    # max_fit_per_as_dict = get_max_fit_per_type(processed_logbooks, 'AS')
    max_fit_per_as_dict = get_max_fit_per_arcsize(processed_logbooks)
    # print(max_fit_per_as_dict)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Set x and y axis labels.
    for i in range(2):
        axs[i].set_xlabel("Population Archive Size")
        axs[i].set_ylabel("Max Fitness")

    # Set subplot titles.
    axs[0].set_title(r'Max $CS$ Fitness v. $\frac{Archive}{Population}\%$')
    axs[1].set_title(r'Max $P$ Fitness v. $\frac{Archive}{Population}\%$')

    data_cs = []
    data_p = []

    if modalities:
        for m in modalities:
            data_cs.append(max_fit_per_as_dict[m]['cs'])
            data_p.append(max_fit_per_as_dict[m]['p'])
        # axs[0, 0].set_xticks([y + 1 for y in range(len(data_cs))])
        axs[0].set_xticklabels(modalities)
        # axs[0, 1].set_xticks([y + 1 for y in range(len(data_p))])
        axs[1].set_xticklabels(modalities)
    else:
        for k in max_fit_per_as_dict.keys():
            data_cs.append(max_fit_per_as_dict[k]['cs'])
            data_p.append(max_fit_per_as_dict[k]['p'])
        axs[0].set_xticklabels(list(max_fit_per_as_dict.keys()))
        axs[1].set_xticklabels(list(max_fit_per_as_dict.keys()))

    axs[0].boxplot(data_cs)
    axs[1].boxplot(data_p)

    plt.savefig(f'max_fit_vs_arc_size.png')

    plt.close('all')


def plot_max_fit_vs_uastrat_boxplot(processed_logbooks: dict):
    """Plots the maximum fitness vs population archive proportion.
    """
    max_fit_per_uas_dict = get_max_fit_per_ua_strat(processed_logbooks)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Set x and y axis labels.
    for i in range(2):
        axs[i].set_xlabel("Update Archive Strategy")
        axs[i].set_ylabel("Max Fitness")

    # Set subplot titles.
    axs[0].set_title(f'Max $CS$ Fitness v. Update Archive Strategy')
    axs[1].set_title(f'Max $P$ Fitness v. Update Archive Strategy')

    data_cs = []
    data_p = []

    for k in max_fit_per_uas_dict.keys():
        data_cs.append(max_fit_per_uas_dict[k]['cs'])
        data_p.append(max_fit_per_uas_dict[k]['p'])
    axs[0].set_xticklabels(list(max_fit_per_uas_dict.keys()))
    axs[1].set_xticklabels(list(max_fit_per_uas_dict.keys()))

    axs[0].boxplot(data_cs)
    axs[1].boxplot(data_p)

    plt.savefig(f'max_fit_vs_ua_strat.png')

    plt.close('all')


def plot_min_fit_vs_col_boxplot(processed_logbooks: pd.DataFrame, col: str):
    """Plots the maximum fitness vs population archive proportion.
    """
    data = []
    modalities = sorted(processed_logbooks[col].unique())
    # print(f'{col}_modalities={modalities}')
    print(f'col={col}')
    grouped = processed_logbooks.groupby(by=col)
    for modality in modalities:
        d = grouped.get_group(modality)['cs_fit_min'].to_list()
        # print(d)
        data.append(list(d))
    # grouped = processed_logbooks.groupby(by=col).mean()['cs_fit_min']
    # data = grouped.to_list()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    # Set x and y axis labels.
    ax.set_xlabel(col)
    ax.set_ylabel("Min Fitness")

    # Set subplot titles.
    ax.set_title(f'Min $CS$ Fitness v. {col}')

    # Set X tick labels.
    # ax.set_xticklabels(modalities)

    # ax.boxplot(data)
    ax.boxplot(data, labels=modalities)

    plt.savefig(f'min_fit_vs_{col}_boxplot.png')

    plt.close('all')


def plot_min_fit_vs_col_main_effects(processed_logbooks: pd.DataFrame, col: str):
    """Plots the maximum fitness vs population archive proportion.
    """
    data = []
    modalities = sorted(processed_logbooks[col].unique())
    grouped = processed_logbooks.groupby(by=col).mean()['cs_fit_min']
    data = grouped.to_list()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    # Set x and y axis labels.
    ax.set_xlabel(col)
    ax.set_ylabel("Mean Min Fitness")
    ax.set_xlim(modalities[0], modalities[-1])
    ax.set_ylim(0, 0.52)

    # Set subplot titles.
    ax.set_title(f'Min $CS$ Fitness v. {col}')

    # Set X tick labels.
    # ax.set_xticklabels(modalities)

    ax.plot(modalities, data)

    plt.savefig(f'min_fit_vs_{col}_main_effect.png')

    plt.close('all')


timedout_runs = {
    '133518_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.93799971]), 'p_fit_maxs': np.array([0.93799971])},
    '133536_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95091739]), 'p_fit_maxs': np.array([0.95091739])},
    '134502_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.46924696]), 'p_fit_maxs': np.array([0.46924696])},
    '134532_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.5979337]), 'p_fit_maxs': np.array([0.5979337])},
    '135637_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95849237]), 'p_fit_maxs': np.array([0.95849237])},
    '135642_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.85671923]), 'p_fit_maxs': np.array([0.85671923])},
    '135647_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '140622_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '140645_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.85933136]), 'p_fit_maxs': np.array([0.85933136])},
    '140656_CCEA_MTQ_bestRandom_PS30_AS7_NG40': {'cs_fit_maxs': np.array([0.96137984]), 'p_fit_maxs': np.array([0.96137984])},
    '141653_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.96287965]), 'p_fit_maxs': np.array([0.96287965])},
    '141710_CCEA_MTQ_bestRandom_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.95002307]), 'p_fit_maxs': np.array([0.95002307])},
    '142610_CCEA_MTQ_random_PS30_AS7_NG40': {'cs_fit_maxs': np.array([0.71680479]), 'p_fit_maxs': np.array([0.71680479])},
    '142619_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.8533408]), 'p_fit_maxs': np.array([0.8533408])},
    '142627_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '142628_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.94954613]), 'p_fit_maxs': np.array([0.94954613])},
    '143537_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '143545_CCEA_MTQ_random_PS30_AS7_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '143553_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.82766953]), 'p_fit_maxs': np.array([0.82766953])},
    '143601_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.92642007]), 'p_fit_maxs': np.array([0.92642007])},
    '144509_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.88892401]), 'p_fit_maxs': np.array([0.88892401])},
    '144520_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.96245318]), 'p_fit_maxs': np.array([0.96245318])},
    '144527_CCEA_MTQ_random_PS30_AS7_NG40': {'cs_fit_maxs': np.array([0.67928497]), 'p_fit_maxs': np.array([0.67928497])},
    '144548_CCEA_MTQ_bestRandom_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.95614757]), 'p_fit_maxs': np.array([0.95614757])},
    '145508_CCEA_MTQ_bestRandom_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.95262114]), 'p_fit_maxs': np.array([0.95262114])},
    '145509_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.96113939]), 'p_fit_maxs': np.array([0.96113939])},
    '145951_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.79268857]), 'p_fit_maxs': np.array([0.79268857])},
    '145953_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '151009_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.8742146]), 'p_fit_maxs': np.array([0.8742146])},
    '151024_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.96147591]), 'p_fit_maxs': np.array([0.96147591])},
    '151034_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.85317333]), 'p_fit_maxs': np.array([0.85317333])},
    '152147_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.94939837]), 'p_fit_maxs': np.array([0.94939837])},
    '153221_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.96565041]), 'p_fit_maxs': np.array([0.96565041])},
    '153235_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.69332423]), 'p_fit_maxs': np.array([0.69332423])},
    '154126_CCEA_MTQ_random_PS30_AS7_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '154142_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '154152_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.85561161]), 'p_fit_maxs': np.array([0.85561161])},
    '154156_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.97352534]), 'p_fit_maxs': np.array([0.97352534])},
    '155118_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.91139789]), 'p_fit_maxs': np.array([0.91139789])},
    '155125_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.81106384]), 'p_fit_maxs': np.array([0.81106384])},
    '155128_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95848077]), 'p_fit_maxs': np.array([0.95848077])},
    '155151_CCEA_MTQ_bestRandom_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.94805622]), 'p_fit_maxs': np.array([0.94805622])},
    '155208_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.86531969]), 'p_fit_maxs': np.array([0.86531969])},
    '155219_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '155231_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.94741878]), 'p_fit_maxs': np.array([0.94741878])},
    '155306_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.76907237]), 'p_fit_maxs': np.array([0.76907237])},
    '155314_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.85409909]), 'p_fit_maxs': np.array([0.85409909])},
    '155617_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.77261925]), 'p_fit_maxs': np.array([0.77261925])},
    '155627_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '155728_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95520153]), 'p_fit_maxs': np.array([0.95520153])},
    '155837_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '160123_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.6786974]), 'p_fit_maxs': np.array([0.6786974])},
    '160228_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.87800964]), 'p_fit_maxs': np.array([0.87800964])},
    '160528_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.66462708]), 'p_fit_maxs': np.array([0.66462708])},
    '160535_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '160742_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.92113633]), 'p_fit_maxs': np.array([0.92113633])},
    '160805_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95845675]), 'p_fit_maxs': np.array([0.95845675])},
    '160815_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.87520998]), 'p_fit_maxs': np.array([0.87520998])},
    '160945_CCEA_MTQ_bestRandom_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.95347301]), 'p_fit_maxs': np.array([0.95347301])},
    '161045_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.96105448]), 'p_fit_maxs': np.array([0.96105448])},
    '161203_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.82967245]), 'p_fit_maxs': np.array([0.82967245])},
    '161209_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '161347_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95783616]), 'p_fit_maxs': np.array([0.95783616])},
    '161446_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '161557_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.71467174]), 'p_fit_maxs': np.array([0.71467174])},
    '162435_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.86010903]), 'p_fit_maxs': np.array([0.86010903])},
    '162458_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '163249_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '163255_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.96728708]), 'p_fit_maxs': np.array([0.96728708])},
    '163309_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.93764655]), 'p_fit_maxs': np.array([0.93764655])},
    '163335_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '163344_CCEA_MTQ_best_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.95588482]), 'p_fit_maxs': np.array([0.95588482])},
    '163427_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '163531_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.80372379]), 'p_fit_maxs': np.array([0.80372379])},
    '163621_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.96463016]), 'p_fit_maxs': np.array([0.96463016])},
    '164230_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95381922]), 'p_fit_maxs': np.array([0.95381922])},
    '164629_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '164631_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '165838_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '165905_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95515841]), 'p_fit_maxs': np.array([0.95515841])},
    '165916_CCEA_MTQ_bestRandom_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95776908]), 'p_fit_maxs': np.array([0.95776908])},
    '171700_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '171728_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.73069724]), 'p_fit_maxs': np.array([0.73069724])},
    '172532_CCEA_MTQ_random_PS30_AS7_NG40': {'cs_fit_maxs': np.array([0.494848]), 'p_fit_maxs': np.array([0.494848])},
    '172546_CCEA_MTQ_best_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.95577452]), 'p_fit_maxs': np.array([0.95577452])},
    '173340_CCEA_MTQ_random_PS30_AS15_NG40': {'cs_fit_maxs': np.array([0.858191]), 'p_fit_maxs': np.array([0.858191])},
    '173342_CCEA_MTQ_random_PS30_AS22_NG40': {'cs_fit_maxs': np.array([0.85525746]), 'p_fit_maxs': np.array([0.85525746])}
}


def merge_two_dicts(dict_1: dict, dict_2: dict):
    """Returns a merged dict made up of `dict_1` and `dict_2`.
    """
    return {**dict_1, **dict_2}


def categorize_into_max_subgroups(ungrouped_logbooks: dict):
    """Groups the data into subgroups of the same configuration.
    """
    # Identify subgroups from the data.
    configs = set("_".join(k.split('_')[1:])
                  for k in ungrouped_logbooks.keys())

    # Initialize the subgorups dict.
    subgrouped_dict = {config: {'p': [], 'cs': []} for config in configs}

    # Categorize the values into subgroups.
    for config in configs:
        for key in ungrouped_logbooks.keys():
            if key.__contains__(config):
                subgrouped_dict[config]['p'] += list(
                    ungrouped_logbooks[key]['p_fit_maxs'])
                subgrouped_dict[config]['cs'] += list(
                    ungrouped_logbooks[key]['cs_fit_maxs'])

    # No. of datapoints in each subgroup should be 27 (the lowest size)
    # to be balanced.

    # Pop a number of datapoints from subgroups to make the size same
    # as the smallest.
    for key in subgrouped_dict.keys():
        while len(subgrouped_dict[key]['cs']) > SIZE_OF_SMALLEST_SUBGROUP:
            r_index = randint(0, SIZE_OF_SMALLEST_SUBGROUP - 1)
            subgrouped_dict[key]['cs'].pop(r_index)
            subgrouped_dict[key]['p'].pop(r_index)

    return subgrouped_dict


def categorize_into_min_subgroups(ungrouped_logbooks: dict):
    """Groups the data into subgroups of the same configuration.
    """
    # Identify subgroups from the data.
    configs = set("_".join(k.split('_')[1:])
                  for k in ungrouped_logbooks.keys())

    # Initialize the subgorups dict.
    # subgrouped_dict = {config: {'p': [], 'cs': []} for config in configs}
    subgrouped_dict = {config: [] for config in configs}

    # Categorize the values into subgroups.
    for config in configs:
        for key in ungrouped_logbooks.keys():
            if key.__contains__(config):
                # subgrouped_dict[config]['p'] += list(
                #     ungrouped_logbooks[key]['p_fit_mins'])
                # subgrouped_dict[config]['cs'] += list(
                #     ungrouped_logbooks[key]['cs_fit_mins'])
                subgrouped_dict[config] += list(
                    ungrouped_logbooks[key]['cs_fit_mins'])

    # No. of datapoints in each subgroup should be 27 (the lowest size)
    # to be balanced.

    # # Pop a number of datapoints from subgroups to make the size same
    # # as the smallest.
    # for key in subgrouped_dict.keys():
    #     while len(subgrouped_dict[key]['cs']) > SIZE_OF_SMALLEST_SUBGROUP:
    #         r_index = randint(0, SIZE_OF_SMALLEST_SUBGROUP - 1)
    #         subgrouped_dict[key]['cs'].pop(r_index)
    #         subgrouped_dict[key]['p'].pop(r_index)

    return subgrouped_dict


def compute_avg_per_subgroup(data_dict: dict) -> dict:
    """Returns the average of values for each subgroup.
    """
    return {key: {'p': np.mean(np.array(data_dict[key]['p'])), 'cs': np.mean(
        np.array(data_dict[key]['cs']))} for key in data_dict.keys()}


def compute_rng_per_subgroup(data_dict: dict) -> dict:
    """Returns the range of values for each subgorup.
    """
    return {key: {'p': np.max(np.array(data_dict[key]['p'])) - np.min(np.array(data_dict[key]['p'])), 'cs': np.max(np.array(data_dict[key]['cs'])) - np.min(np.array(data_dict[key]['cs']))} for key in data_dict.keys()}


def compute_var_per_subgroup(data_dict: dict) -> dict:
    """Returns the variance of values for each subgroup.
    """
    return {key: {'p': np.var(np.array(data_dict[key]['p'])), 'cs': np.var(np.array(data_dict[key]['p']))} for key in data_dict.keys()}


def run_anom(data_dict: dict, alpha=0.10, verbose=False):
    """Performs Analysis of Means (ANOM) on the data.
    Returns a list of keys for the subgourps that are signals.
    """
    # FIXME: assertions.
    # Determine No. of subgroups (k) and No. of samples per subgroup (n).
    k = len(data_dict)
    n = SIZE_OF_SMALLEST_SUBGROUP

    # Calculate means per subgroup.
    avgs = compute_avg_per_subgroup(data_dict)

    # Caculate the grand average.
    grand_avg_cs = np.mean(np.array([avgs[key]['cs'] for key in avgs.keys()]))
    grand_avg_p = np.mean(np.array([avgs[key]['p'] for key in avgs.keys()]))

    if verbose:
        print(f'grand_avg_cs={grand_avg_cs}')
        print(f'grand_avg_p={grand_avg_p}')

    # Calculate data ranges per subgroups.
    rngs = compute_rng_per_subgroup(data_dict)

    # Calculate the mean of ranges.
    average_range_cs = np.mean(
        np.array([rngs[key]['cs'] for key in rngs.keys()]))
    average_range_p = np.mean(
        np.array([rngs[key]['p'] for key in rngs.keys()]))
    if verbose:
        print(f'average_range_cs={average_range_cs}')
        print(f'average_range_p={average_range_p}')

    # NOTE: Hereafter, only values for cs are calculated, not p (since they are the same).
    # The following values are calculated according to Tables A and C of "Understanding Industrial Experiments" by D. Wheeler.
    # Assumptions are based on k = 27 and n = 27.
    D2 = 4.008  # d2 based on n=27, ref: Table A.
    D3 = 0.7005  # d3 based on n=27, ref: Table A.

    # Calculate control limits of means.
    a2 = 3 / (np.sqrt(n) * D2)
    avg_ucl_cs = grand_avg_cs + a2 * average_range_cs
    avg_lcl_cs = grand_avg_cs - a2 * average_range_cs
    if verbose:
        print(f'a2={a2}')
        print(f'avg_ucl_cs={avg_ucl_cs}')
        print(f'avg_lcl_cs={avg_lcl_cs}')

    # Calculate control limits of averages.
    d4 = 1 + ((3 * D3) / D2)
    rng_ucl_cs = d4 * average_range_cs
    d3 = 1 - ((3 * D3) / D2)
    rng_lcl_cs = d3 * average_range_cs
    if verbose:
        print(f'd4={d4}')
        print(f'rng_ucl_cs={rng_ucl_cs}')
        print(f'd3={d3}')
        print(f'rng_lcl_cs={rng_lcl_cs}')

    # Find the subgroups with means outside of the control limits.
    avg_cl_outliers_cs = [key for key in avgs.keys(
    ) if avgs[key]['cs'] > avg_ucl_cs or avgs[key]['cs'] < avg_lcl_cs]
    # print(f'avg_cl_outliers_cs={avg_cl_outliers_cs}')
    if verbose:
        print(f'size of avg_cl_outliers_cs={len(avg_cl_outliers_cs)}')

    # Find the subgroups with ranges outside of the control limits.
    rng_cl_outliers_cs = [key for key in rngs.keys(
    ) if rngs[key]['cs'] > rng_ucl_cs or rngs[key]['cs'] < rng_lcl_cs]
    # print(f'rng_cl_outliers_cs={rng_cl_outliers_cs}')
    if verbose:
        print(f'size of rng_cl_outliers_cs={len(rng_cl_outliers_cs)}')

    # delta nu based on n=27, ref: Table B.
    DELTA_NU = np.mean(np.array([15.36, 16.6]))
    # Degree of freedom for k=1, n=27, ref: Table B.
    NU_1 = np.mean(np.array([15.6, 16.9]))
    nu = NU_1 + ((k - 1) * DELTA_NU)
    if verbose:
        print(f'k={k}, n={n}, nu={nu}')

    # Calculate d2_start, ref: Table B.
    d2_star = D2 + (D2 / (4 * nu))
    if verbose:
        print(f'd2_star={d2_star}')

    # Estimate the overall standard deviation of data.
    std_overall_est = average_range_cs / d2_star
    if verbose:
        print(f'std_overall_est={std_overall_est}')

    # Calculate std of subgroup averges.
    std_avgs_est = std_overall_est / np.sqrt(n)
    if verbose:
        print(f'std_avgs_est={std_avgs_est}')

    H = 3.05  # ASSUMPTIONS: nu = inf, k = 27, alpha = 0.10

    avg_udl_cs = grand_avg_cs + H * std_avgs_est
    avg_ldl_cs = grand_avg_cs - H * std_avgs_est
    if verbose:
        print(f'avg_udl_cs={avg_udl_cs}')
        print(f'avg_ldl_cs={avg_ldl_cs}')

    anom_outliers_cs = [key for key in avgs.keys(
    ) if avgs[key]['cs'] > avg_udl_cs or avgs[key]['cs'] < avg_ldl_cs]
    if verbose:
        print(f'anom_outliers_cs={anom_outliers_cs}')
        print(f'size of anom_outliers_cs={len(anom_outliers_cs)}')

    return anom_outliers_cs


def create_design_table_ua_expr(logbooks_dict: dict):
    # Remove unnecessary columns.
    new_dict = {}
    for run, fit_dict in logbooks_dict.items():
        new_dict[run] = {}
        for key, value in fit_dict.items():
            if key == 'cs_fit_maxs':
                new_dict[run]['cs_fit_max'] = value

    # Remove excess runs to make the number of replications per run equal.
    configs = set("_".join(k.split('_')[1:])
                  for k in new_dict.keys())
    subgrouped_dict = {config: [] for config in configs}
    for config in configs:
        for key in new_dict.keys():
            if key.__contains__(config):
                subgrouped_dict[config].append(key)

    for config in subgrouped_dict.keys():
        while len(subgrouped_dict[config]) > SIZE_OF_SMALLEST_SUBGROUP:
            r_index = randint(0, SIZE_OF_SMALLEST_SUBGROUP - 1)
            run_to_be_popped = subgrouped_dict[config].pop(r_index)
            new_dict.pop(run_to_be_popped, None)

    # Add Pop_Size column.
    for key, value in new_dict.items():
        if key.__contains__('PS10'):
            value['Pop_Size'] = 10
        elif key.__contains__('PS20'):
            value['Pop_Size'] = 20
        elif key.__contains__('PS30'):
            value['Pop_Size'] = 30

    # Add Arc_Size coloumn.
    parameter_type = 'AS'
    as_modalities = ['25', '50', '75']
    ps_modalities = ['PS10', 'PS20', 'PS30']

    mapping = {
        'PS10': [2, 5, 7],
        'PS20': [5, 10, 15],
        'PS30': [7, 15, 22]
    }

    for modality in as_modalities:
        for key in new_dict.keys():
            for psm in ps_modalities:
                if key.__contains__(psm) and key.__contains__(parameter_type+str(mapping[psm][as_modalities.index(modality)])):
                    new_dict[key]['Arc_Size'] = int(modality)

    # Add Update Archive Strategy (UA_Strat) coloumn.
    strategies = ['best', 'random', 'bestRandom']
    for strategy in strategies:
        for key in new_dict.keys():
            if key.__contains__(strategy):
                new_dict[key]['UA_Strat'] = strategy

    # Change the fit_max values from a np array to a value.
    for key in new_dict.keys():
        value = new_dict[key]['cs_fit_max']
        new_dict[key]['cs_fit_max'] = value[0]

    return new_dict


def create_design_table_hp_expr(logbooks_dict: dict):
    # Remove unnecessary columns.
    new_dict = {}
    for run, fit_dict in logbooks_dict.items():
        new_dict[run] = {}
        for key, value in fit_dict.items():
            if key == 'cs_fit_mins':
                new_dict[run]['cs_fit_min'] = value

    # Add Min_Dist column.
    for key, value in new_dict.items():
        if key.__contains__('MD3'):
            value['Min_Dist'] = 0.3
        elif key.__contains__('MD5'):
            value['Min_Dist'] = 0.5
        elif key.__contains__('MD7'):
            value['Min_Dist'] = 0.7

    # Add Mut_Rate coloumn.
    for key, value in new_dict.items():
        if key.__contains__('MR5'):
            value['Mut_Rate'] = 0.5
        elif key.__contains__('MR7'):
            value['Mut_Rate'] = 0.7
        elif key.__contains__('MR10'):
            value['Mut_Rate'] = 1.0

    # # Add Mut_Rate coloumn.
    # for key, value in new_dict.items():
    #     if key.__contains__('MR5'):
    #         value['Mut_Rate'] = 0.5
    #     elif key.__contains__('MR7'):
    #         value['Mut_Rate'] = 0.7
    #     elif key.__contains__('MR10'):
    #         value['Mut_Rate'] = 1.0

    # # Add Mut_std coloumn.
    # for key, value in new_dict.items():
    #     if key.__contains__('STD5_'):
    #         value['Mut_std'] = 0.05
    #     elif key.__contains__('STD10'):
    #         value['Mut_std'] = 0.1
    #     elif key.__contains__('STD50'):
    #         value['Mut_std'] = 0.5

    # Add Cx_Rate coloumn.
    for key, value in new_dict.items():
        if key.__contains__('CR3'):
            value['Cx_Rate'] = 0.3
        elif key.__contains__('CR5'):
            value['Cx_Rate'] = 0.5
        elif key.__contains__('CR7'):
            value['Cx_Rate'] = 0.7

    # Add Tour_Size coloumn.
    for key, value in new_dict.items():
        if key.__contains__('TS2'):
            value['Tour_Size'] = 2
        elif key.__contains__('TS3'):
            value['Tour_Size'] = 3
        elif key.__contains__('TS4'):
            value['Tour_Size'] = 4

    # Add Reg_Rad coloumn.
    for key, value in new_dict.items():
        if key.__contains__('RR3'):
            value['Reg_Rad'] = 0.3
        elif key.__contains__('RR5'):
            value['Reg_Rad'] = 0.5
        elif key.__contains__('RR7'):
            value['Reg_Rad'] = 0.7

    # Change the fit_max values from a np array to a value.
    for key in new_dict.keys():
        value = new_dict[key]['cs_fit_min']
        new_dict[key]['cs_fit_min'] = value[0]

    return new_dict


def remove_excess_runs(preprocessed_logbooks: dict):
    # Remove excess runs to make the number of replications per run equal.
    print(
        f'num of elements before removing excess runs = {len(preprocessed_logbooks)}')

    configs = list(set("_".join(k.split('_')[2:])
                       for k in preprocessed_logbooks.keys()))
    print(f'len(configs)={len(configs)}')
    subgrouped_dict = {config: [] for config in configs}
    for config in configs:
        for key in preprocessed_logbooks.keys():
            if key.__contains__(config):
                subgrouped_dict[config].append(key)
    # print(subgrouped_dict)

    num_elem = 0
    for v in subgrouped_dict.values():
        v = list(v)
        num_elem += len(v)
    print(f'num_elem={num_elem}')

    # assert num_elem == len(
    #     preprocessed_logbooks), 'you have botched the subgrouping job!'

    for config in subgrouped_dict.keys():
        if len(subgrouped_dict[config]) < SIZE_OF_SMALLEST_SUBGROUP:
            print(
                f'config {config} has {len(subgrouped_dict[config])} values, i.e., {subgrouped_dict[config]}')
        while len(subgrouped_dict[config]) > SIZE_OF_SMALLEST_SUBGROUP:
            r_index = randint(0, SIZE_OF_SMALLEST_SUBGROUP - 1)
            run_to_be_popped = subgrouped_dict[config].pop(r_index)
            preprocessed_logbooks.pop(run_to_be_popped, None)
            # print(f'run {run_to_be_popped} was popped!')
        # print(f'config {config} has {len(subgrouped_dict[config])} values')

    print(
        f'num of elements after removing excess runs = {len(preprocessed_logbooks)}')

    return preprocessed_logbooks


def plot_interaction_plots(x, trace, response, param_1, param_2):
    """Draw interaction plots.
    """
    fig, ax = plt.subplots()
    interaction_plot(
        x=x,
        trace=trace,
        response=response,
        colors=['forestgreen', 'tomato', 'royalblue'],
        ax=ax
    )

    plt.savefig(f'ix_plt_{param_1}_{param_2}.png')

    plt.close('all')


def main():
    # log_files = [
    #     r'C:\Users\sepeh\Projects\MLCSHE\results\MLCSHE_HP_DOE\hp_results\20220901_075439_CCEA_MTQ_MD3_MR7_STD50_CR5_TS3_RR10\20220901_075439_logbook.log.pkl',
    #     r'C:\Users\sepeh\Projects\MLCSHE\results\MLCSHE_HP_DOE\hp_results\20220901_075440_CCEA_MTQ_MD3_MR7_STD50_CR5_TS3_RR10\20220901_075440_logbook.log.pkl']
    # Read and preprocess logs.
    # log_files = glob.glob(
    #     'C:\\Users\\sepeh\\Projects\\MLCSHE\\results\\mlche_hp_onemax_results\\*\\*.pkl')
    # print(f'len(log_files)={len(log_files)}')
    # logbooks_dict_gen, logbooks_dict = read_log_files(log_files)
    # print(f'len(logbooks_dict)={len(logbooks_dict)}')
    ## logbooks_dict = merge_two_dicts(logbooks_dict, timedout_runs)
    # logbooks_dict = remove_excess_runs(logbooks_dict)
    # subgrouped_logbooks = categorize_into_min_subgroups(logbooks_dict)
    # print(f'len(subgrouped_logbooks)={len(subgrouped_logbooks)}')
    # num_elem = 0
    # for v in subgrouped_logbooks.values():
    #     v = list(v)
    #     num_elem += len(v)
    # print(f'num_elem={num_elem}')
    # print(subgrouped_logbooks)

    # # Cast into a pandas dataframe and export as excel spreadsheet.
    # doe_table = create_design_table_hp_expr(logbooks_dict)
    # doe_dframe = pd.DataFrame.from_dict(doe_table, orient='index')
    # doe_dframe.to_pickle('hp_doe_results.pkl')

    doe_dframe = pd.read_pickle('hp_doe_results.pkl')
    # doe_dframe = pd.read_pickle('hp_doe_results_xrm.pkl')
    # print(doe_dframe)
    cols = list(doe_dframe.keys())
    cols.pop(0)
    # print(cols)

    # # # doe_dframe.to_excel("output.xlsx")

    # Draw main effects and box plots.
    for col in cols:
        plot_min_fit_vs_col_main_effects(doe_dframe, col)
        plot_min_fit_vs_col_boxplot(doe_dframe, col)

    fitness = doe_dframe['cs_fit_min']

    for param_1, param_2 in combinations(cols, 2):
        plot_interaction_plots(
            x=doe_dframe[param_1],
            trace=doe_dframe[param_2],
            response=fitness,
            param_1=param_1,
            param_2=param_2
        )

    # plot_min_fit_vs_col_boxplot(doe_dframe, 'Cx_Rate')

    # # ANOM
    # anom_signals = run_anom(preprocessed_logbooks)
    # print(f'Number of ANOM signals = {len(anom_signals)}')

    # avgs = compute_avg_per_subgroup(preprocessed_logbooks)
    # for k, v in avgs.items():
    #     print(f'average for {k} = {v}')

    # vars = compute_var_per_subgroup(preprocessed_logbooks)
    # for k, v in vars.items():
    #     print(f'var for {k} = {v}')

    # TESTING PENGOUIN
    # all_data_cs = []
    # for key in preprocessed_logbooks.keys():
    #     all_data_cs += preprocessed_logbooks[key]['cs']
    # print(pg.normality(all_data_cs, method="jarque_bera", alpha=0.1))
    # print(pg.normality(
    #     preprocessed_logbooks['CCEA_MTQ_random_PS30_AS15_NG40']['cs'], method="jarque_bera", alpha=0.1))
    # # ax = pg.qqplot(all_data_cs, dist='norm', confidence=0.9)
    # ax = pg.qqplot(
    #     preprocessed_logbooks['CCEA_MTQ_random_PS30_AS15_NG40']['cs'], dist='norm', confidence=0.9)

    # PLOT BOXPLOTS
    # plot_max_fit_vs_popsize_boxplot(logbooks_dict, ['PS10', 'PS20', 'PS30'])
    # plot_max_fit_vs_arcsize_boxplot(logbooks_dict)
    # plot_max_fit_vs_uastrat_boxplot(logbooks_dict)

    # plt.show()


#     # get_max_fit_per_type(logbooks_dict, 'PS')
#     # for run, fitness_lists in logbooks_dict.items():
#     #     plot_fit_vs_ngen(run, fitness_lists)

#     # plot_fit_max(logbooks_dict)
#     # plot_fit_max_diff(logbooks_dict)
if __name__ == "__main__":
    main()

# Template
# '170820_CCEA_MTQ_best_PS20_AS15_NG40': {'cs_fit_maxs': np.array([[0.9635122]]), 'p_fit_maxs': np.array([[0.9635122])}

# Acual PS30 experiments that have timed out.


# Commented code.

# files = [
#         r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220814_210246_CCEA_MTQ_PS30_NG40_AS22\20220814_210246_logbook.log.pkl',
#         r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220815_105548_CCEA_MTQ_PS30_NG40_AS15\20220815_105548_logbook.log.pkl',
#         r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220815_155923_CCEA_MTQ_PS30_NG40_AS7\20220815_155923_logbook.log.pkl',
#         # r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220815_231841_CCEA_MTQ_PS20_NG40_AS15\20220815_231841_logbook.log.pkl',
#         # r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220816_005729_CCEA_MTQ_PS20_NG40_AS10\20220816_005729_logbook.log.pkl',
#         # r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220816_105550_CCEA_MTQ_PS20_NG40_AS5\20220816_105550_logbook.log.pkl',
#         # r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220815_230057_CCEA_MTQ_PS10_NG40_AS7\20220815_230057_logbook.log.pkl',
#         # r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220815_225537_CCEA_MTQ_PS10_NG40_AS5\20220815_225537_logbook.log.pkl',
#         # r'C:\Users\sepeh\Projects\MLCSHE\results\best_random_archive\20220815_225123_CCEA_MTQ_PS10_NG40_AS2\20220815_225123_logbook.log.pkl',
#         r'C:\Users\sepeh\Projects\MLCSHE\results\20220817_093544_CCEA_MTQ_PS10_NG40_AS2_BDA\20220817_093544_logbook.log.pkl',
#         r'C:\Users\sepeh\Projects\MLCSHE\results\20220817_095254_CCEA_MTQ_PS20_NG40_AS15_BDA\20220817_095254_logbook.log.pkl',
#         r'C:\Users\sepeh\Projects\MLCSHE\results\20220817_132716_CCEA_MTQ_PS20_NG40_AS10_BDA\20220817_132716_logbook.log.pkl',
#         r'C:\Users\sepeh\Projects\MLCSHE\results\20220822_090317_CCEA_MTQ_bestRandom_PS10_AS5_NG40\20220822_090317_logbook.log.pkl',
#         r'C:\Users\sepeh\Projects\MLCSHE\results\20220819_180900_CCEA_MTQ_PS30_NG40_AS22_BDA\20220819_180900_logbook.log.pkl'
#     ]
