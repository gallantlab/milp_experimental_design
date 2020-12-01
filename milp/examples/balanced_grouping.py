
import math
import multiprocessing
import os
import time

import numpy as np
import matplotlib.pyplot as plt


def log10_number_of_possible_groupings(I, G):
    size = I / G
    number = math.factorial(I) / (math.factorial(size) ** G * math.factorial(G))
    log10_number = math.log10(number)
    return log10_number

# randomization solution functions

comparison_data = {
    'X_solution': None,
    'global_feature_means': None,
    'I': None,
    'G': None,
    'F': None,
}


def generate_random_grouping(I, G):
    items = np.arange(I)
    np.random.shuffle(items)
    groups = np.split(np.array(items), G)
    return groups


def evaluate_groups(groups, F, global_feature_means):

    d_group = [
        np.abs(F[:, group].T.mean(0) - global_feature_means).sum()
        for group in groups
    ]
    d_total = sum(d_group)

    return {
        'd_total': d_total,
    }


def set_comparison_context(F, X_solution, global_feature_means):
    comparison_data['X_solution'] = X_solution
    comparison_data['global_feature_means'] = global_feature_means
    comparison_data['F'] = F
    comparison_data['I'], comparison_data['G'] = X_solution.shape


def compute_random_run(i):
    groups = generate_random_grouping(
        comparison_data['I'],
        comparison_data['G'],
    )
    group_eval = evaluate_groups(
        groups,
        comparison_data['F'],
        comparison_data['global_feature_means'],
    )
    return {'d': group_eval['d_total'], 'groups': groups}


def compute_randomization_solutions(F, n):

    pool = multiprocessing.Pool(14)
    results = pool.map(compute_random_run, range(int(n)))
    randomization_ds = [result['d'] for result in results]

    return randomization_ds

# redo


def get_group_sum_matrix(I, G):
    size = int(I / G)
    J = np.zeros((I, G))
    for i in range(G):
        J[(i * size):((i + 1) * size), i] = 1
    J /= size
    return J


def random_group_score(I, F, J):
    items = range(I)
    np.random.shuffle(items)
    F_copy = F[:, items]
    score = np.abs(F_copy.dot(J).T - global_feature_means).sum()
    return score


import time


def compute_best_of_n(n):

    I = comparison_data['I']
    G = comparison_data['G']
    F = comparison_data['F']
    global_feature_means = comparison_data['global_feature_means']
    J = get_group_sum_matrix(I, G)

    t_generate = 0
    t_evaluate = 0

    best_d = float('inf')
    best_grouping = None
    randomization_ds = np.zeros(n, dtype=np.float32)
    for i in range(n):

        if i != 0 and i % int(n / 10.0) == 0:
            print('completed', i, 'trials')

        items = list(range(I))
        np.random.shuffle(items)
        F_copy = F[:, items]
        score = np.abs(F_copy.dot(J).T - global_feature_means).sum()

        groups = items
        group_eval = {
            'd_total': score,
        }

        if group_eval['d_total'] < best_d:
            best_d = group_eval['d_total']
            best_grouping = groups
        randomization_ds[i] = group_eval['d_total']
    return {
        'd': best_d,
        'groups': best_grouping,
        'randomization_ds': randomization_ds,

        't_generate': t_generate,
        't_evaluate': t_evaluate,
    }


def compute_best_grouping(groupings):
    best_d = float('inf')
    best = None

    for grouping in groupings:
        if grouping['d'] < best_d:
            best_d = grouping['d']
            best = grouping

    return best


def compute_randomization_solutions(F, n, processes=14):
    pool = multiprocessing.Pool(processes)
    results = pool.map(
        compute_best_of_n,
        [int(np.ceil(n / processes))] * processes
    )
    all_randomization_ds = np.hstack(
        [result['randomization_ds'] for result in results]
    )
    best_grouping = compute_best_grouping(results)
    best_grouping['randomization_ds'] = all_randomization_ds
    return best_grouping


#
# # plotting functions
#

def plot_luminances(F):

    plt.imshow(F)
    plt.gca().set_aspect(20)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.title(
        'item features (predetermined)',
        fontsize=plt.rcParams['figure.titlesize'],
    )
    plt.xlabel('items')
    plt.ylabel('features')


def plot_luminance_distribution(
    F, plot_mean=False, color=None, show_title=True, show_labels=True,
):

    plt.hist(F[0, :], 19, color=color)
    if plot_mean:
        plt.axvline(
            F[0, :].mean(),
            color='k',
            linestyle=':',
            label='mean luminance',
        )
        plt.legend()

    if show_title:
        plt.title(
            'Video Luminance Distribution',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('luminance (A.U.)')
        plt.ylabel('number of videos')


def plot_group_membership(X_solution, F, show_title=True, show_labels=True):

    plt.imshow(X_solution.T)
    plt.gca().set_aspect(30)
    plt.gca().xaxis.set_ticks_position('bottom')
    if show_title:
        plt.title(
            'MILP solution group membership',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('videos')
        plt.ylabel('runs')


def plot_global_feature_values_color(
    X_solution, F, show_title=True, show_labels=True,
):

    I, G = X_solution.shape
    m_solution = F.dot(X_solution) / (I / float(G))

    plt.imshow(m_solution.T, cmap='RdBu', vmin=-0.6, vmax=0.6)
    plt.colorbar()
    plt.gca().xaxis.set_ticks_position('bottom')
    if show_title:
        plt.title(
            'group mean feature values',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('features')
        plt.ylabel('runs')


def plot_global_feature_deviations_color(X_solution, F):

    I, G = X_solution.shape
    m_solution = F.dot(X_solution) / (I / float(G))
    global_feature_mean = F.mean(1)

    D_solution = global_feature_mean[:, np.newaxis] - m_solution
    plt.imshow(D_solution.T, cmap='RdBu', vmin=-0.10, vmax=0.10)
    plt.colorbar()
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.title(
        'group mean deviation from global feature mean',
        fontsize=plt.rcParams['figure.titlesize'],
    )
    plt.xlabel('features')
    plt.ylabel('groups')


def plot_global_feature_deviations_bar(X_solution, F, show_title=True,
                                       show_labels=True, color=None):
    I, G = X_solution.shape
    m_solution = F.dot(X_solution) / (I / float(G))
    global_feature_mean = F.mean(1)
    D_solution = global_feature_mean[:, np.newaxis] - m_solution
    D_solution = D_solution.squeeze()
    plt.bar(np.arange(D_solution.shape[0]), D_solution, color=color)
    plt.axis([-0.5, D_solution.shape[0] - 0.5, -0.14, 0.14])
    if show_title:
        plt.title(
            'MILP Solution Deviation from Perfect Balance',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('runs')
        plt.ylabel('$\mu_g - \mu$')


def plot_grouping_subplots(
    F, X_solution, randomization_ds, n=None, randomization_axis=None
):

    n_cols = 2
    n_rows = 2

    plt.subplot(n_rows, n_cols, 1)
    plot_luminance_distribution(F)
    plt.subplot(n_rows, n_cols, 2)
    plot_group_membership(X_solution, F)
    plt.subplot(n_rows, n_cols, 3)
    plot_global_feature_deviations_bar(X_solution, F)
    plt.subplot(n_rows, n_cols, 4)
    plot_randomization(
        F,
        X_solution,
        randomization_ds,
        n=n,
        axis=randomization_axis,
    )
    plt.show()


def plot_grouping_separate_plots(F, X_solution, randomization_ds, figsize,
                                 n=None, randomization_axis=None,
                                 show_title=True, show_labels=True,
                                 show_legend=True, save_dir=None):
    if save_dir is not None:
        timestamp = str(time.time())
        save_template = os.path.join(save_dir, '{name}' + timestamp + '.png')

    plt.figure(figsize=figsize)
    plot_luminance_distribution(
        F,
        show_title=show_title,
        show_labels=show_labels,
        color='black',
    )
    if save_dir is not None:
        save_path = save_template.format(name='luminance_distribution')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()


    plt.figure(figsize=figsize)
    plot_group_membership(
        X_solution,
        F,
        show_title=show_title,
        show_labels=show_labels,
    )
    if save_dir is not None:
        save_path = save_template.format(name='group_task_pairings')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_global_feature_deviations_bar(
        X_solution,
        F,
        show_title=show_title,
        show_labels=show_labels,
        color='red',
    )
    if save_dir is not None:
        save_path = save_template.format(name='milp_error')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_randomization(
        F,
        X_solution,
        randomization_ds,
        n=n,
        axis=randomization_axis,
        show_title=show_title,
        show_labels=show_labels,
        show_legend=show_legend,
        milp_color='red',
        random_color='orange',
    )
    if save_dir is not None:
        save_path = save_template.format(name='milp_vs_randomization')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()


def plot_randomization(F, X_solution, randomization_ds, axis=None, n=None,
                       milp_color=None, random_color=None, linewidth=3,
                       show_title=True, show_labels=True, show_legend=True,
                       sci_y=True):

    if milp_color is None:
        milp_color = 'red'
    if random_color is None:
        random_color = 'blue'

    I, G = X_solution.shape
    X_groups = [np.nonzero(X_solution[:, g])[0] for g in range(G)]
    global_feature_means = F.mean(1)
    d_milp = evaluate_groups(X_groups, F, global_feature_means)['d_total']
    n = len(randomization_ds)

    plt.axvline(x=d_milp, label='MILP', color=milp_color, linewidth=linewidth)
    plt.axvline(
        x=min(randomization_ds),
        label='random',
        color=random_color,
        linewidth=linewidth,
    )
    plt.hist(
        randomization_ds,
        max(10, min(50, int(n / 100))),
        color=random_color,
    )
    if axis is not None:
        plt.axis(axis)
    if n is not None:
        title = (
            'MILP vs '
            + '1e'
            + str(int(np.log10(1e6)))
            + ' Randomized Solutions'
        )
    else:
        title = 'MILP vs Randomized Solutions'
    if show_title:
        plt.title(title, fontsize=plt.rcParams['figure.titlesize'])
    if show_labels:
        plt.xlabel('total absolute error')
        plt.ylabel('# random solutions')
    if show_legend:
        plt.legend()
    plt.gcf().set_facecolor('white')

    if sci_y:
        plt.ticklabel_format(style='sci', axis='y')

