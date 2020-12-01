
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np


#
# # plotting functions
#

def plot_stimuli_features(F):
    plt.imshow(F)
    plt.gca().set_aspect(10)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xlabel('image')
    plt.ylabel('feature')
    plt.show()


def plot_image_task_pairings(X_solution, show_title=True, show_labels=True):
    plt.imshow(X_solution.T)
    plt.gca().set_aspect(10)
    if show_title:
        plt.title(
            'image task pairings',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.ylabel('task')
        plt.xlabel('stimulus')


def plot_repeats_per_stimulus(X_solution, show_title=True, show_labels=True,
                              color=None):
    if color is None:
        color = 'red'

    # histogram of repeats per stimulus
    if show_title:
        plt.title(
            'number of times each stimulus is repeated',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('number of repeats')
        plt.ylabel('number of stimuli')
    plt.hist(X_solution.sum(1), 11, range=[-0.5, 10.5], color=color)
    plt.yticks([0, 100, 300, 500, 700])
    plt.axis([0, 8, 0, 800])
    plt.xticks([0, 2, 4, 6])


def plot_target_probabilities_for_each_task(
    F, X_solution, show_title=True, show_labels=True, show_colorbar=True
):
    n_tasks, n_stimuli = F.shape
    unique_values = list(np.unique(F))
    n_feature_levels = len(unique_values)
    probability_counts = np.zeros((n_tasks, n_feature_levels))
    for g in range(n_tasks):
        task_stimuli = np.nonzero(X_solution[:, g])
        task_stimuli_features = F[g, task_stimuli].astype(int)
        for feature_value in task_stimuli_features.flat:
            probability_counts[g, unique_values.index(feature_value)] += 1
    plt.imshow(probability_counts.T, cmap='nipy_spectral', vmin=65, vmax=68)
    for i in np.arange(0.5, n_feature_levels - 0.5, 1):
        plt.axhline(i, color='k')
    for i in np.arange(0.5, n_tasks - 0.5, 1):
        plt.axvline(i, color='k')
    if show_title:
        plt.title(
            'stimulus conditions for each task',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('task')
        plt.ylabel('feature value')
    if show_colorbar:
        plt.colorbar()


def plot_error_histogram(X_solution, F, show_title=True, show_labels=True,
                         color=None):
    if color is None:
        color = 'red'

    n_tasks = X_solution.shape[1]

    # feature probabilities for each task
    n_tasks = F.shape[0]
    task_mean_features = np.zeros((n_tasks, n_tasks))
    delta_task_mean_features = np.zeros((n_tasks, n_tasks))
    for f in range(n_tasks):
        task_mean_features[f, :] = F[:, X_solution[:, f]].mean(1)
        delta_task_mean_features[f, :] = task_mean_features[f, :] - F.mean(1)

    non_diag_deltas = delta_task_mean_features[~np.eye(n_tasks, dtype=bool)]
    plt.hist(non_diag_deltas, 100, range=[-0.2, 0.2], color=color)
    abs_delta_sum = np.abs(non_diag_deltas).sum()
    if show_labels:
        plt.xlabel('deviation from target mean')
        plt.ylabel('mean feature values for each task')
    if show_title:
        plt.title(
            'deviation from target feature means within each task (total='
            + str(round(abs_delta_sum, 5))
            + ')',
            fontsize=plt.rcParams['figure.titlesize']
        )
    plt.axis([-0.2, 0.2, 0, 50])

    return {
        'non_diag_deltas': non_diag_deltas,
        'abs_delta_sum': abs_delta_sum,
    }


def plot_stimulus_pairing_solution_summary(X_solution, F):

    n_rows = 2
    n_cols = 2

    plt.subplot(n_rows, n_cols, 1)
    plot_image_task_pairings(X_solution)
    plt.subplot(n_rows, n_cols, 2)
    plot_repeats_per_stimulus(X_solution)
    plt.subplot(n_rows, n_cols, 3)
    plot_target_probabilities_for_each_task(F, X_solution)
    plt.subplot(n_rows, n_cols, 4)
    plot_error_histogram(X_solution, F)


def plot_as_separate_figures(X_solution, F, figsize, show_labels=True,
                            show_title=True, save_dir=None):
    if save_dir is not None:
        timestamp = str(time.time())
        save_template = os.path.join(save_dir, '{name}' + timestamp + '.png')

    plt.figure(figsize=figsize)
    plot_image_task_pairings(
        X_solution,
        show_labels=show_labels,
        show_title=show_title,
    )
    if save_dir is not None:
        save_path = save_template.format(name='image_task_pairings')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()


    plt.figure(figsize=figsize)
    plot_repeats_per_stimulus(
        X_solution,
        show_labels=show_labels,
        show_title=show_title,
    )
    if save_dir is not None:
        save_path = save_template.format(name='repeats_per_stimulus')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_task_target_frequencies(
        F,
        X_solution,
        show_labels=show_labels,
        show_title=show_title,
    )
    if save_dir is not None:
        save_path = save_template.format(name='task_target_frequencies')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_error_histogram(
        X_solution,
        F,
        show_labels=show_labels,
        show_title=show_title,
    )
    if save_dir is not None:
        save_path = save_template.format(name='error_histogram')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()


def get_task_feature_counts(F, X_solution):

    n_tasks = X_solution.shape[1]
    n_features = X_solution.shape[1]
    n_feature_levels = np.unique(F).shape[0]

    task_feature_counts = np.zeros([n_tasks, n_features, n_feature_levels])
    for t in range(n_tasks):
        task_stimuli = np.nonzero(X_solution[:, t])[0]
        for f in range(n_features):
            for value in range(n_feature_levels):
                F_sum = (F[f, task_stimuli] == value).sum()
                task_feature_counts[t, f, value] = F_sum

    return task_feature_counts


def plot_task_feature_counts(F, X_solution):

    task_feature_counts = get_task_feature_counts(F, X_solution)
    n_feature_levels = task_feature_counts.shape[2]
    for i in range(n_feature_levels):
        plt.imshow(
            task_feature_counts[:, :, i],
            cmap='nipy_spectral',
            vmin=20,
            vmax=120,
        )
        plt.xlabel('feature')
        plt.ylabel('task')
        plt.colorbar()
        plt.title(
            'feature value = ' + str(i),
            fontsize=plt.rcParams['figure.titlesize'],
        )
        plt.show()


def plot_average_feature_values(F, X_solution):

    n_feature_levels = np.unique(F).shape[0]
    mean_feature_level = (n_feature_levels - 1) / 2.0
    trials_per_condition = X_solution.sum(0)[0]

    task_feature_counts = get_task_feature_counts(F, X_solution)
    average_feature_values = (
        task_feature_counts[:, :, 0] * 0
        + task_feature_counts[:, :, 1] * 1
        + task_feature_counts[:, :, 2] * 2
    ) / trials_per_condition

    plt.imshow(average_feature_values, cmap='nipy_spectral')
    plt.title(
        'average feature values for each task',
        fontsize=plt.rcParams['figure.titlesize'],
    )
    plt.ylabel('task')
    plt.xlabel('feature')
    plt.colorbar()
    plt.show()

    average_feature_deviations = average_feature_values - F.mean(1)
    diag = np.diag_indices_from(average_feature_deviations)
    average_feature_deviations[diag] = (
        average_feature_values[diag] - mean_feature_level
    )
    plt.imshow(average_feature_deviations, cmap='nipy_spectral')
    plt.title(
        'average feature deviations for each task',
        fontsize=plt.rcParams['figure.titlesize'],
    )
    plt.ylabel('task')
    plt.xlabel('feature')
    plt.colorbar()
    plt.show()


def get_random_solution_stimulus_pairing(
    n_stimuli,
    n_tasks,
    n_trials,
    F,
    feature_probabilities,
    select='equal_usages',
):
    """

    # algorithm

    for t in trials_per_task:
        for c in conditions:
            for g in groups:

    """

    task_stimuli = {t: [] for t in range(n_tasks)}

    stimuli_counts = np.zeros(n_stimuli)
    trials_per_task = int(n_trials / n_tasks)

    # compute the counts of each feature for each task
    feature_counts = {
        feature_value: int(trials_per_task * feature_probability)
        for feature_value, feature_probability in feature_probabilities.items()
    }
    if sum(feature_counts.values()) != trials_per_task:
        keys = list(feature_counts.keys())
        random.shuffle(keys)
        for key in keys:
            feature_counts[key] += 1
            if sum(feature_counts.values()) == trials_per_task:
                break

    # compute the possible stimuli for each feature value and task
    possible_feature_task_stimuli = {}
    for feature_value in feature_probabilities.keys():
        possible_feature_task_stimuli[feature_value] = {}
        for task in range(n_tasks):
            possible_feature_task_stimuli[feature_value][task] = (
                np.nonzero(F[task] == feature_value)[0]
            )

    # for each feature value, choose a stimulus for each task
    # choose the least-used stimulus that satisfies the requirements of the task
    feature_value_iterations = {
        feature_value: 0
        for feature_value in feature_counts.keys()
    }

    feature_distributions = np.zeros([n_tasks, n_tasks])
    target_feature_distribution = F.mean(1)

    while True:

        for feature_value, feature_count in feature_counts.items():
            if (
                feature_value_iterations[feature_value]
                == feature_counts[feature_value]
            ):
                continue

            for task in range(n_tasks):

                current_set = possible_feature_task_stimuli[feature_value][task]
                current_set = [
                    stimulus
                    for stimulus in current_set
                    if stimulus not in task_stimuli[task]
                ]
                if select == 'equal_usages':
                    index = np.argmin(stimuli_counts[current_set])
                    stimulus = current_set[index]
                elif select == 'balanced_features':
                    n = float(len(task_stimuli[task]))
                    if n == 0:
                        new_distributions = F[:, current_set]
                    else:
                        task_distributions = feature_distributions[:, task]
                        new_distributions = (
                            (1 / n) * F[:, current_set]
                            + (n - 1) / (n) * task_distributions[:, np.newaxis]
                        )

                    distribution_errors = (
                        new_distributions
                        - target_feature_distribution[:, np.newaxis]
                    )
                    index = np.argmin(np.abs(distribution_errors).sum(1))
                    feature_distributions[:, task] = new_distributions[:, index]
                    stimulus = current_set[index]

                else:
                    raise Exception()

                task_stimuli[task].append(stimulus)
                stimuli_counts[stimulus] += 1

            feature_value_iterations[feature_value] += 1

        if sum(feature_value_iterations.values()) == trials_per_task:
            break

    solution = np.zeros((n_stimuli, n_tasks), dtype=bool)
    for task, stimuli in task_stimuli.items():
        for stimulus in stimuli:
            solution[stimulus, task] = True

    return solution


def plot_task_target_frequencies(F, X_solution, show_title=True,
                                 show_labels=True):
    n_tasks, n_stimuli = F.shape
    unique_values = list(np.unique(F))
    n_feature_levels = len(unique_values)
    probability_counts = np.zeros((n_tasks, n_feature_levels))
    for g in range(n_tasks):
        task_stimuli = np.nonzero(X_solution[:, g])
        task_stimuli_features = F[g, task_stimuli].astype(int)
        for feature_value in task_stimuli_features.flat:
            probability_counts[g, unique_values.index(feature_value)] += 1

    hist_range = [65 - 0.5, 69 + 0.5]
    n_bins = int(hist_range[-1] - hist_range[0])

    plt.hist(
        probability_counts.flatten(),
        n_bins,
        range=hist_range,
        color='red',
    )

    if show_labels:
        plt.xlabel('number of trials')
        plt.ylabel('(task, target) pairs')
    if show_title:
        plt.title('frequencies of each (task, target) pair')
    plt.xticks([64, 65, 66, 67, 68, 69])
    plt.axis([63.5, 69.5, 0, 35])

