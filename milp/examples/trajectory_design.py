
import copy
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import scipy.stats

import milp.program


def compute_repeats(n_counts, n_nodes, n_trials):

    mean_repeats_per_node = n_trials / float(n_nodes)
    geometric_parameter = 1 / mean_repeats_per_node

    program = milp.program.initialize_program()

    for count in range(1, n_counts):
        milp.program.add_variable(
            program,
            'c_{count}'.format(count=count),
            int,
        )

    # sum of counts equals number of nodes
    A_eq = {}
    for count in range(1, n_counts):
        A_eq['c_{count}'.format(count=count)] = 1
    milp.program.add_constraint(program, A_eq=A_eq, b_eq=n_nodes)

    # weighted sum of counts equals number of trials
    A_eq = {}
    for count in range(1, n_counts):
        A_eq['c_{count}'.format(count=count)] = count
    milp.program.add_constraint(program, A_eq=A_eq, b_eq=n_trials)

    # distribution resembles geometric distribution
    for count in range(1, n_counts):
        weight = count
        coefficients = {'c_{count}'.format(count=count): weight}
        constant = n_nodes * weight
        constant *= -scipy.stats.geom.pmf(count, geometric_parameter)
        milp.program.add_abs_cost_term(program, coefficients, constant)

    solution = milp.program.solve_MILP(program, verbose=False)

    counts = milp.program.get_solution_variable(solution, 'c')

    return counts


def compute_visits_per_location(repeat_counts):
    visits_per_location = []
    for c, count in enumerate(repeat_counts):
        visits_per_location.extend([c] * count)
    np.random.shuffle(visits_per_location)
    visits_per_location = np.array(visits_per_location)
    return visits_per_location


def compute_target_path_length_pdf(distances, mean_trial_distance):

    distribution = scipy.stats.expon(scale=mean_trial_distance)

    ds = np.linspace(0, distances.max(), 20)
    values = distribution.pdf(ds)

    return {
        'lengths': ds,
        'values': values,
        'f_pdf': distribution.pdf,
        'scipy_distribution': distribution,
    }


def compute_target_path_length_pmf(n_bins, distances, pdf):
    d_min = 0
    d_max = distances.max()

    bin_borders = np.linspace(d_min, d_max, n_bins + 1)
    bin_centers = 0.5 * (bin_borders[:-1] + bin_borders[1:])
    bin_weights = np.zeros(n_bins)

    for b in range(n_bins):
        mass, _ = scipy.integrate.quad(
            pdf['f_pdf'],
            bin_borders[b],
            bin_borders[b + 1],
        )
        bin_weights[b] = mass

    bin_weights /= bin_weights.sum()

    return {
        'n_bins': n_bins,
        'bin_centers': bin_centers,
        'bin_borders': bin_borders,
        'bin_weights': bin_weights,
    }


def compute_target_path_length_discrete(distance_pmf, n_trials):

    bin_weights = distance_pmf['bin_weights']
    bin_centers = distance_pmf['bin_centers']

    program = milp.program.initialize_program()

    # variable for tracking count of each bin
    for b in range(distance_pmf['n_bins']):
        milp.program.add_variable(program, 'b_{b}'.format(b=b), int)

    # sum of bin counts equals number of trials
    A_eq = {}
    for b in range(distance_pmf['n_bins']):
        A_eq['b_{b}'.format(b=b)] = 1
    milp.program.add_constraint(program, A_eq=A_eq, b_eq=n_trials)

    # # weighted sum of counts equals number of trials
    # A_eq = {}
    # for count in range(1, n_counts):
    #     A_eq['b_{b}'.format(b=b)] = b
    # milp.program.add_constraint(program, A_eq=A_eq, b_eq=n_trials)

    # distribution resembles bin weights
    for b in range(distance_pmf['n_bins']):
        weight = bin_weights[b]
        coefficients = {'b_{b}'.format(b=b): weight}
        constant = -bin_weights[b] * n_trials * weight
        milp.program.add_abs_cost_term(program, coefficients, constant)

    solution = milp.program.solve_MILP(program)

    bin_counts = milp.program.get_solution_variable(solution, 'b')

    return {
        'n_bins': distance_pmf['n_bins'],
        'bin_centers': np.array(bin_centers),
        'bin_counts': np.array(bin_counts),
        'bin_borders': np.array(distance_pmf['bin_borders']),
    }


def compute_random_paths(n_bins, bin_counts, bin_centers, bin_borders,
                         n_trials, distances, visits_per_location,
                         n_random=100000):

    best_error = float('inf')
    best_path = None
    best_counts = None

    # increase visits to start_node by one to account for starting position
    n_locations = len(visits_per_location)
    p = visits_per_location / float(visits_per_location.sum())
    start_node = np.random.choice(range(n_locations), 1, replace=False, p=p)

    new_old_index_maps = compute_new_old_index_maps(visits_per_location)
    to_old_index = new_old_index_maps['to_old_index']

    for i in range(n_random):

        g = np.arange(n_trials, dtype=int)
        np.random.shuffle(g)
        random_trajectory = [start_node] + list(to_old_index[g])
        random_trajectory = np.array(random_trajectory, dtype=int)
        traj_distances = distances[
            random_trajectory[:-1],
            random_trajectory[1:],
        ]
        actual_counts, _ = np.histogram(traj_distances, bin_borders)

        error = np.sum(np.abs(actual_counts - bin_counts))
        if error < best_error:
            best_path = random_trajectory
            best_counts = actual_counts
            best_error = error

    return {
        'best_path': best_path,
        'best_counts': best_counts,
        'best_error': best_error,
        'n_random': n_random,
    }


def compute_new_old_index_maps(visits_per_location):
    """return a map from new repeated map index to old original index"""
    n_locations = len(visits_per_location)
    n_replicant_locations = visits_per_location.sum()
    to_new_index_first = -100000 * np.zeros(n_locations, dtype=int)
    to_old_index = np.zeros(n_replicant_locations, dtype=int)
    current = 0
    for location in range(n_locations):
        for copy in range(visits_per_location[location]):
            if copy == 0:
                to_new_index_first[location] = current
            to_old_index[current] = location
            current += 1

    return {
        'to_old_index': to_old_index,
        'to_new_index_first': to_new_index_first,
    }


def compute_trajectory(
    visits_per_location,
    distances,
    target_path_length_distribution,
    start_and_end_nodes=None,
    subtour_iterations=1000,
    verbose=True,
):
    """compute trajectory that visits each path a specified number of times"""

    n_locations = len(visits_per_location)

    # decide start and end nodes of path
    if start_and_end_nodes is None:
        p = visits_per_location / float(visits_per_location.sum())
        start_node, end_node = np.random.choice(
            range(n_locations), 2, replace=False, p=p,
        )
    else:
        start_node, end_node = start_and_end_nodes

    # increase visits to start_node by one to account for starting position
    visits_per_location = copy.deepcopy(visits_per_location)
    visits_per_location[start_node] += 1

    nonzero_locations = visits_per_location > 0
    nonzero_visits_per_location = visits_per_location[nonzero_locations]

    # build new replicant graph that includes node copies
    n_replicant_locations = visits_per_location.sum()
    new_old_index_maps = compute_new_old_index_maps(visits_per_location)
    to_new_index_first = new_old_index_maps['to_new_index_first']
    to_old_index = new_old_index_maps['to_old_index']

    replicant_start_node = to_new_index_first[start_node]
    replicant_end_node = to_new_index_first[end_node]

    program = milp.program.initialize_program()

    # variables: p_i,j = whether path from i to j is included
    path_distances = {}
    paths_entering = {i: [] for i in range(n_replicant_locations)}
    paths_leaving = {i: [] for i in range(n_replicant_locations)}
    for from_location in range(n_replicant_locations):
        for to_location in range(n_replicant_locations):

            # skip self-edges
            if to_old_index[from_location] == to_old_index[to_location]:
                continue

            variable = 'p_{i},{j}'.format(i=from_location, j=to_location)
            milp.program.add_variable(program, variable, bool)

            # index paths entering and leaving each location
            paths_entering[to_location].append(variable)
            paths_leaving[from_location].append(variable)

            # index path distances
            path_distances[variable] = distances[
                to_old_index[from_location],
                to_old_index[to_location]
            ]
    path_variables = list(path_distances.keys())

    # constraint: enter and leave every location one time (except start and end)
    for location in range(n_replicant_locations):

        # paths entering
        A_eq = {variable: 1 for variable in paths_entering[location]}
        if location == replicant_start_node:
            b_eq = 0
        else:
            b_eq = 1
        milp.program.add_constraint(program, A_eq=A_eq, b_eq=b_eq)

        # paths leaving
        A_eq = {variable: 1 for variable in paths_leaving[location]}
        if location == replicant_end_node:
            b_eq = 0
        else:
            b_eq = 1
        milp.program.add_constraint(program, A_eq=A_eq, b_eq=b_eq)

    # cost function: use number of path lengths specified by the
    #   target path length distribution
    bin_borders = target_path_length_distribution['bin_borders']
    bin_counts = target_path_length_distribution['bin_counts']
    for b in range(len(bin_counts)):
        coefficients = {}
        for variable, distance in path_distances.items():
            if bin_borders[b] < distance and distance <= bin_borders[b + 1]:
                coefficients[variable] = 1
        if len(coefficients) > 0:
            milp.program.add_abs_cost_term(
                program, coefficients=coefficients, constant=-bin_counts[b],
            )

    # initial solution
    solution = milp.program.solve_MILP(program, verbose=verbose)
    p = milp.program.get_solution_variable(solution, 'p')
    if verbose:
        print()

    # eliminate subtours
    replicant_node_kwargs = {
        'start_node': replicant_start_node,
        'end_node': replicant_end_node,
    }
    connected_components = compute_connected_components(
        p, **replicant_node_kwargs
    )
    for i in range(subtour_iterations):
        if len(connected_components) == 1:
            break

        if verbose:
            cc_lengths = [
                len(connected_component)
                for connected_component in connected_components
            ]
            print(
                '    eliminating subtours iteration '
                + str(i)
                + ' (sizes = '
                + str(cc_lengths)
                + ')'
            )

        for connected_component in connected_components:
            eliminate_subset_subtours(
                program=program,
                n_nodes=n_replicant_locations,
                subset=connected_component,
                path_variables=path_variables,
            )
        solution = milp.program.solve_MILP(program, verbose=False)
        p = milp.program.get_solution_variable(solution, 'p')
        connected_components = compute_connected_components(
            p, **replicant_node_kwargs
        )
    else:
        raise Exception('could not eliminate subtours')

    # convert to original space
    replicant_trajectory = compute_path(p, **replicant_node_kwargs)
    trajectory = [
        to_old_index[replicant_node] for replicant_node in replicant_trajectory
    ]

    if verbose:
        print()
        print('final trajectory:')
        print(trajectory)

    return trajectory


# helper methods


def compute_path(p, start_node, end_node=None):
    """compute path from first node using p matrix"""

    cycle = []

    current_node = start_node
    cycle.append(current_node)

    while True:

        current_node = np.nonzero(p[current_node, :])[0][0]
        cycle.append(current_node)

        # terminate if cycle
        if end_node is None:
            if current_node == start_node:
                break
        else:
            if current_node == end_node:
                break

    return cycle


def compute_connected_components(p, start_node, end_node):
    """compute connected components from p matrix"""
    n_locations = p.shape[0]
    visited_nodes = np.zeros(n_locations, bool)
    components = []

    # noncyclic path
    path = compute_path(p=p, start_node=start_node, end_node=end_node)
    for n in path:
        visited_nodes[n] = True
    components.append(path)

    # cyclic paths
    while visited_nodes.sum() < n_locations:
        next_node = np.nonzero(visited_nodes == 0)[0][0]
        cycle = compute_path(p=p, start_node=next_node)
        for n in cycle:
            visited_nodes[n] = True
        components.append(cycle)

    return components


def eliminate_subset_subtours(program, n_nodes, subset, path_variables):
    """place constraints in program to eliminate subtours within subset

    - number of edges leaving the subset should be less than or equal to
      subset size minus 1
    - see http://examples.gurobi.com/traveling-salesman-problem/
    """

    subset_vars = {}

    # gather edges within subset
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                name = 'p_{i},{j}'.format(i=i, j=j)

                if name in path_variables and i in subset and j in subset:
                    subset_vars[name] = 1

    subset_size = len(set(subset))
    milp.program.add_constraint(
        program=program,
        A_lt=subset_vars,
        b_lt=(subset_size - 1),
    )


def plot_node_repeats_pmf(n_repeats, n_repeats_pmf, show_title=True,
                          show_labels=True):
    plt.plot(n_repeats, n_repeats_pmf, '.k')
    if show_title:
        plt.title('repeats per node probability (discrete pmf)')
    if show_labels:
        plt.xlabel('n_repeats')
        plt.ylabel('probability')


def plot_node_repeats_denormalized(n_repeats, n_repeats_pmf, n_nodes,
                                   show_title=True, show_labels=True):
    plt.plot(n_repeats, n_repeats_pmf * n_nodes, '.k')
    if show_title:
        plt.title('number of repeats per node')
    if show_labels:
        plt.xlabel('number of repeats')
        plt.ylabel('number of nodes')


def plot_target_vs_actual_repeats_per_node_old(counts, n_repeats_pmf, n_nodes,
                                           show_title=True, show_labels=True,
                                           show_legend=True):
    highest_value = max(max(counts), max(n_repeats_pmf * n_nodes))
    for i in range(int(np.floor(highest_value)) + 1):
        plt.axhline(i, color='#BBBBBB', linestyle='dotted')

    plt.plot(n_repeats_pmf * n_nodes, '.k', label='target')
    plt.plot(counts, '.r', label='actual')

    if show_title:
        plt.title('target vs actual repeats per node')
    if show_labels:
        plt.xlabel('n_repeats')
        plt.ylabel('n_nodes')
    if show_legend:
        plt.legend()

def plot_target_vs_actual_repeats_per_node(counts, n_repeats_pmf, n_nodes,
                                           show_title=True, show_labels=True,
                                           show_legend=True,
                                           n_shown_repeats=None):

    if n_shown_repeats is None:
        n_shown_repeats = len(counts)

    if n_shown_repeats is not None:
        counts = counts[:n_shown_repeats]
        n_repeats_pmf = n_repeats_pmf[:n_shown_repeats]

    highest_value = max(max(counts), max(n_repeats_pmf * n_nodes))
    for i in range(int(np.floor(highest_value)) + 1):
        plt.axhline(i, color='#CCCCCC', linestyle='dotted')

    plt.plot(
        counts,
        '.r',
        label='actual',
        zorder=101,
        markersize=10,
    )

    ax = plt.gca()
    for v, value in enumerate(n_nodes * n_repeats_pmf):
        if v == 0:
            label_kwargs = {'label': 'target'}
        else:
            label_kwargs = {}

        rect = matplotlib.patches.Rectangle(
            (v - 0.5, 0),
            1,
            value,
            linewidth=1,
            edgecolor='None',
            facecolor='black',
            zorder=100,
            **label_kwargs
        )
        ax.add_patch(rect)


    # plot errors
    for i in range(n_shown_repeats):
        plt.plot(
            [i, i],
            [counts[i],n_nodes * n_repeats_pmf[i]],
            '-r',
            zorder=1000,
        )

    if show_legend:
        plt.legend(loc='upper right', framealpha=1)


def plot_distance_distribution(distances, show_title=True, show_labels=True,
                               color=None):
    if color is None:
        color = 'black'
    plt.hist(distances.flatten(), 15, color=color)
    if show_title:
        plt.title('distance distribution')
    if show_labels:
        plt.ylabel('number of paths')
        plt.xlabel('distance')


def plot_locations(x, y, colors=None, cmap=None, title=None, show_title=True,
                   show_labels=True, value_colors=None, show_axes=False,
                   show_borders=False, **plot_kwargs):
    if colors is None:
        plt.scatter(x, y, color='blue')
    elif cmap is not None:
        plt.scatter(x, y, c=colors, cmap=cmap, **plot_kwargs)
    elif isinstance(colors, str):
        plt.scatter(x, y, c=colors, **plot_kwargs)
    else:
        for value in set(colors):
            subset = (colors == value)
            plt.plot(x[subset], y[subset], '.', color=value_colors[value])

    # plt.axis('square')
    if show_borders:
        plt.axvline(0, color='grey')
        plt.axvline(1, color='grey')
        plt.axhline(0, color='grey')
        plt.axhline(1, color='grey')
    if show_title and title is not None:
        plt.title(title)
    if show_labels:
        plt.xlabel('x')
        plt.ylabel('y')
    if not show_axes:
        plt.xticks([])
        plt.yticks([])
    plt.axis([-0.1, 1.1, -0.1, 1.1])


def plot_trajectory(trajectory, x, y, visits_per_location=None, show_title=True,
                    show_labels=True, title=None, show_colorbar=True,
                    show_start_and_end=False, nodestyle=None, edgestyle='k'):

    for from_node, to_node in zip(trajectory[:-1], trajectory[1:]):
        plt.plot(
            [x[from_node], x[to_node]], [y[from_node], y[to_node]],
            edgestyle,
        )

    if visits_per_location is not None:
        color_kwargs = {
            'colors': visits_per_location,
            'cmap': 'nipy_spectral',
            'vmax': visits_per_location.max() + 1,
        }
    elif nodestyle is not None:
        color_kwargs = {'colors': nodestyle}
    else:
        color_kwargs = {}

    plot_locations(
        x,
        y,
        s=50,
        zorder=100,
        show_labels=show_labels,
        show_title=show_title,
        **color_kwargs
    )
    if show_colorbar:
        plt.colorbar()
    if show_start_and_end:
        plt.scatter(
            x[trajectory[0]],
            y[trajectory[0]],
            s=50,
            color='cyan',
            linewidth=50,
        )
        plt.scatter(
            x[trajectory[-1]],
            y[trajectory[-1]],
            s=50,
            color='magenta',
            linewidth=50,
        )
    # plt.axis('square')
    if show_title and title is not None:
        plt.title(title)
    if show_labels:
        plt.xlabel('x')
        plt.ylabel('y')


def plot_distance_pdf(distance_pdf, show_title=True, show_labels=True):
    if show_title:
        plt.title('pdf')
    plt.plot(distance_pdf['lengths'], distance_pdf['values'], 'k')
    plt.ylim([0, distance_pdf['values'].max() * 1.1])
    if show_labels:
        plt.xlabel('path length')
        plt.ylabel('pdf density')


def plot_distance_pmf(distance_pmf, show_title=True, show_labels=True):
    plt.plot(
        distance_pmf['bin_centers'],
        distance_pmf['bin_weights'],
        '.r',
        label='bin weights',
    )
    if show_title:
        plt.title('pmf (binned into ' + str(distance_pmf['n_bins']) + ' bins)')
    if show_labels:
        plt.xlabel('path length')
        plt.ylabel('probability')


def plot_discrete_path_distribution(discrete_path_distribution,
                                    show_title=True, show_labels=True):
    plt.plot(
        discrete_path_distribution['bin_centers'],
        discrete_path_distribution['bin_counts'],
        '.k',
    )
    if show_title:
        plt.title('path length distribution')
    if show_labels:
        plt.xlabel('path length')
        plt.ylabel('n_trials')


def plot_target_vs_actual_distribution(trajectory, bin_borders, bin_centers,
                                       bin_counts, n_bins, distances,
                                       show_labels=True, show_title=True,
                                       show_legend=True, color=None,
                                       show_mean=False, axis=None):
    used_distances = []
    for i in range(len(trajectory) - 1):
        distance = distances[trajectory[i], trajectory[i + 1]]
        used_distances.append(distance)
    if color is None:
        color = 'black'
    plt.hist(used_distances, bins=bin_borders, label='actual', color=color)
    plt.plot(
        bin_centers,
        bin_counts,
        '.r',
        label='targets'
    )
    if show_mean:
        plt.axvline(
            np.mean(used_distances),
            color='grey',
            label='mean',
            zorder=100,
        )

    if show_title:
        plt.title('distribution of path lengths')
    if show_labels:
        plt.xlabel('path length')
        plt.ylabel('frequency')
    if show_legend:
        plt.legend()

    if axis is not None:
        plt.axis(axis)


def plot_actual_vs_target_distribution(trajectory, target_distribution,
                                       distances, actual_color='r', title=None,
                                       show_labels=True, show_title=True,
                                       show_legend=True):

    n_bins = target_distribution['n_bins']
    bin_centers = target_distribution['bin_centers']
    bin_borders = target_distribution['bin_borders']
    target_counts = target_distribution['bin_counts']

    # transform target_distribution into list for use with plt.hist()
    discrete_paths_expanded = []
    for i in range(n_bins):
        discrete_paths_expanded += [bin_centers[i]] * target_counts[i]
    plt.hist(
        discrete_paths_expanded,
        n_bins,
        range=[bin_borders[0], bin_borders[-1]],
        color='black',
        label='target',
    )

    # compute all distances used
    used_distances = []
    for i in range(len(trajectory) - 1):
        distance = distances[trajectory[i], trajectory[i + 1]]
        used_distances.append(distance)

    # compute histogram of distances using same bins as target distribution
    actual_counts, bin_borders = np.histogram(
        used_distances,
        10,
        range=[0, bin_borders[-1]],
    )
    plt.plot(
        (bin_borders[:-1] + bin_borders[1:]) * 0.5,
        actual_counts,
        '.',
        color=actual_color,
        markersize=10,
        label='actual'
    )

    # plot errors between target and actual
    for i in range(n_bins):
        plt.plot(
            [bin_centers[i], bin_centers[i]],
            [target_counts[i], actual_counts[i]],
            '-',
            color=actual_color,
        )

    if show_title and title is not None:
        plt.title(title)
    if show_labels:
        plt.xlabel('length')
        plt.ylabel('frequency')
    if show_legend:
        plt.legend()


def plot_best_random_path_distribution(random_paths, bin_centers, bin_counts,
                                       show_title=True, show_labels=True):
    plt.title('best random path')
    plt.plot(bin_centers, bin_counts, '.k', label='target distribution')
    plt.plot(
        bin_centers,
        random_paths['best_counts'],
        '.r',
        label='actual distribution',
    )
    if show_title:
        plt.title(
            'best random path (n_random='
            + str(random_paths['n_random'])
            + ', error = '
            + str(random_paths['best_error'])
            + ')'
        )
    if show_labels:
        plt.xlabel('length')
        plt.ylabel('n_trials')
    if show_legend:
        plt.legend()


def plot_summary(distances, n_repeats, n_repeats_pmf, n_nodes, repeat_counts,
                 x, y, visits_per_location, distance_pdf,
                 discrete_path_distribution, distance_pmf, trajectory,
                 random_paths, figsize, n_shown_repeats, show_title=True,
                 show_labels=True, show_legend=True, show_colorbar=True,
                 save_dir=None):

    if save_dir is not None:
        timestamp = str(time.time())
        save_template = os.path.join(save_dir, '{name}' + timestamp + '.png')

    plt.figure(figsize=figsize)
    plot_distance_distribution(
        distances,
        show_title=show_title,
        show_labels=show_labels,
    )
    if save_dir is not None:
        save_path = save_template.format(name='distance_distribution')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_target_vs_actual_repeats_per_node(
        repeat_counts,
        n_repeats_pmf,
        n_nodes,
        show_title=show_title,
        show_labels=show_labels,
        show_legend=show_legend,
        n_shown_repeats=n_shown_repeats,
    )
    if save_dir is not None:
        save_path = save_template.format(
            name='target_vs_actual_repeats_per_node',
        )
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_actual_vs_target_distribution(
        trajectory,
        target_distribution=discrete_path_distribution,
        distances=distances,
        show_title=show_title,
        show_labels=show_labels,
        show_legend=show_legend,
        title='optimal MILP trajectory',
    )
    if save_dir is not None:
        save_path = save_template.format(name='actual_vs_target_distribution')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_actual_vs_target_distribution(
        random_paths['best_path'],
        target_distribution=discrete_path_distribution,
        distances=distances,
        show_title=show_title,
        show_labels=show_labels,
        show_legend=show_legend,
        actual_color='orange',
        title='best random trajectory',
    )
    if save_dir is not None:
        save_path = save_template.format(
            name='actual_vs_target_distribution_random'
        )
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_trajectory(
        trajectory,
        x=x,
        y=y,
        edgestyle='red',
        nodestyle='black',
        show_title=show_title,
        show_labels=show_labels,
        show_colorbar=False,
    )
    if save_dir is not None:
        save_path = save_template.format(name='trajectory')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

