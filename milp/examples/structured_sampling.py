
import time
import os

import matplotlib.pyplot as plt
import numpy as np


def get_nonleaves(tree):
    nonleaves = []
    for key, value in tree.items():
        if len(value) != 0:
            nonleaves.append(key)
            nonleaves += get_nonleaves(value)

    return nonleaves


def get_leaves(tree):

    leaves = []
    for key, value in tree.items():
        if len(value) == 0:
            leaves.append(key)
        else:
            leaves += get_leaves(value)
    return leaves


def get_leaves_of_nonleaves(tree):
    leaves_of_nonleaves = {}
    for key, value in tree.items():
        if len(value) != 0:
            leaves_of_nonleaves[key] = get_leaves(value)
            leaves_of_nonleaves.update(get_leaves_of_nonleaves(value))
    return leaves_of_nonleaves


def get_compatibility_matrix(leaf_groups, nonleaf_groups, n_question_templates,
                             n_templates_per_group, leaves_of_nonleaves):

    T = n_question_templates  # number of question templates
    G = len(leaf_groups)  # number of leaf groups

    # compatibility of question templates with groups
    C = np.zeros((T, G), dtype=bool)

    # leaf compatibilities
    for i in range(len(leaf_groups)):
        template_indices = np.arange(
            i * n_templates_per_group,
            (i + 1) * n_templates_per_group,
            dtype=int,
        )
        C[template_indices, i] = True

    # nonleaf compatibilities
    for j, nonleaf_group in enumerate(nonleaf_groups):
        i = G + j
        template_indices = np.arange(
            i * n_templates_per_group,
            (i + 1) * n_templates_per_group,
            dtype=int,
        )
        for leaf_group in leaves_of_nonleaves[nonleaf_group]:
            C[template_indices, leaf_groups.index(leaf_group)] = True

    return C


def get_membership_matrix(G, n_nouns_per_leaf_group):

    N = n_nouns_per_leaf_group * G

    M = np.zeros((G, N), dtype=bool)  # membership of nouns in groups

    # noun group membership
    for i in range(G):
        noun_indices = np.arange(
            i * n_nouns_per_leaf_group,
            (i + 1) * n_nouns_per_leaf_group,
        )
        M[i, noun_indices] = True

    return M


def get_coding_matrices(noun_semantic_tree, n_total_trials,
                        n_question_templates, n_nouns_per_leaf_group):

    n_appearances_per_question_template = int(
        n_total_trials / n_question_templates
    )

    assert (
        n_appearances_per_question_template * n_question_templates
        ==
        n_total_trials
    )

    leaf_groups = get_leaves(noun_semantic_tree)
    nonleaf_groups = get_nonleaves(noun_semantic_tree)
    leaves_of_nonleaves = get_leaves_of_nonleaves(noun_semantic_tree)

    T = n_question_templates

    # number of templates evenly divided by number of groups
    n_total_groups = len(leaf_groups) + len(nonleaf_groups)
    n_templates_per_group = T / n_total_groups
    assert int(n_templates_per_group) * int(n_total_groups) == T

    C = get_compatibility_matrix(
        leaf_groups=leaf_groups,
        nonleaf_groups=nonleaf_groups,
        n_question_templates=n_question_templates,
        n_templates_per_group=n_templates_per_group,
        leaves_of_nonleaves=leaves_of_nonleaves,
    )

    M = get_membership_matrix(
        G=len(leaf_groups),
        n_nouns_per_leaf_group=n_nouns_per_leaf_group,
    )

    return C, M


def plot_P(P, show_title=True, show_labels=True):
    print('Figure 5B:')

    plt.imshow(P, cmap='bwr', vmin=-1, vmax=1, aspect='auto')
    plt.gca().xaxis.set_ticks_position('bottom')
    if show_labels:
        plt.xlabel('concrete nouns')
        plt.ylabel('question templates')
    if show_title:
        plt.title(
            'optimal noun x template pairings (matrix P)',
            fontsize=plt.rcParams['figure.titlesize'],
        )

    plt.xticks([0, 299], [1, 300])
    plt.yticks([0, 119], [1, 120])


def plot_uses_per_question_template(P, show_title=True, show_labels=True):
    print('Figure 5E:')
    plt.hist(P.sum(1), 31, range=[-0.5, 30.5], color='red')
    if show_title:
        plt.title(
            'uses per question template',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('number of uses')
        plt.ylabel('number of question templates')
    plt.axis([0, 30.5, 0, 125])


def plot_uses_per_concrete_noun(P, show_title=True, show_labels=True):
    print('Figure 5F:')
    plt.hist(P.sum(0), 16, range=[-0.5, 15.5], color='red')
    if show_title:
        plt.title(
            'uses per concrete noun',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('number of uses')
        plt.ylabel('number of concrete nouns')
    plt.axis([0, 15.5, 0, 150])


def plot_groups_by_question_templates(
    PMT, show_title=True, show_labels=True, show_colorbar=True,
):
    print('Figure 5D:')
    colormap = plt.cm.get_cmap('nipy_spectral')
    reversed_colormap = colormap.reversed()
    reversed_colormap.set_under('w')

    plt.imshow(PMT, cmap=reversed_colormap, aspect='auto', vmin=0.0001, vmax=20)
    plt.gca().xaxis.set_ticks_position('bottom')
    if show_title:
        plt.title(
            'optimal template x group counts (matrix $PM^T$)',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('leaf groups')
        plt.ylabel('question templates')
    if show_colorbar:
        plt.colorbar()
    plt.xticks([0, 9], [1, 10])
    plt.yticks([0, 119], [1, 120])


def plot_noun_group_membership(M, show_title=True, show_labels=True):
    plt.imshow(M.T, cmap='Greys', aspect='auto')
    plt.gca().xaxis.set_ticks_position('bottom')
    if show_title:
        plt.title(
            'group noun membership (matrix M.T)',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('concrete nouns')
        plt.ylabel('leaf groups')


def plot_template_group_compatibility(C, show_title=True, show_labels=True):
    print('Figure 5C:')
    print()
    plt.imshow(C, cmap='Greys', aspect='auto')
    plt.gca().xaxis.set_ticks_position('bottom')
    if show_title:
        plt.title(
            'template x group compatibility (matrix C)',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('leaf groups')
        plt.ylabel('question templates')
    plt.xticks([0, 9], [1, 10])
    plt.yticks([0, 119], [1, 120])


def plot_noun_template_compatibility(CM, show_title=True, show_labels=True,
                                     imshow_kwargs=None):
    print('Figure 5A:')

    if imshow_kwargs is None:
        imshow_kwargs = {'cmap': 'Greys', 'aspect': 'auto'}
    plt.imshow(CM, **imshow_kwargs)
    plt.gca().xaxis.set_ticks_position('bottom')
    if show_title:
        plt.title(
            'noun x template compatibility (matrix CM)',
            fontsize=plt.rcParams['figure.titlesize'],
        )
    if show_labels:
        plt.xlabel('concrete nouns')
        plt.ylabel('question templates')

    plt.xticks([0, 299], [1, 300])
    plt.yticks([0, 119], [1, 120])


def plot_solution_summary(C, M, P):
    CM = C.dot(M)
    PMT = P.astype(int).dot(M.T)

    n_rows = 3
    n_cols = 2

    plt.subplot(n_rows, n_cols, 1)
    plot_noun_template_compatibility(CM)
    plt.subplot(n_rows, n_cols, 2)
    plot_P(P)
    plt.subplot(n_rows, n_cols, 3)
    plot_template_group_compatibility(C)
    plt.subplot(n_rows, n_cols, 4)
    plot_groups_by_question_templates(PMT)
    plt.subplot(n_rows, n_cols, 5)
    plot_uses_per_question_template(P)
    plt.subplot(n_rows, n_cols, 6)
    plot_uses_per_concrete_noun(P)



def plot_solution_summary_as_separate_figures(C, M, P, figsize,
                                              show_title=True,
                                              show_labels=True, save_dir=None):
    CM = C.dot(M)
    PMT = P.astype(int).dot(M.T)

    if save_dir is not None:
        timestamp = str(time.time())
        save_template = os.path.join(save_dir, '{name}' + timestamp + '.png')

    plt.figure(figsize=figsize)
    plot_noun_template_compatibility(
        CM,
        show_title=show_title,
        show_labels=show_labels,
    )
    if save_dir is not None:
        save_path = save_template.format(name='noun_template_compatibility')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_P(
        P,
        show_title=show_title,
        show_labels=show_labels,
    )
    if save_dir is not None:
        save_path = save_template.format(name='P')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_template_group_compatibility(
        C,
        show_title=show_title,
        show_labels=show_labels,
    )
    if save_dir is not None:
        save_path = save_template.format(name='template_group_compatibility')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_groups_by_question_templates(
        PMT,
        show_title=show_title,
        show_labels=show_labels,
        show_colorbar=False,
    )
    if save_dir is not None:
        save_path = save_template.format(
            name='groups_by_question_templates_no_colorbar'
        )
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_groups_by_question_templates(
        PMT,
        show_title=show_title,
        show_labels=show_labels,
    )
    if save_dir is not None:
        save_path = save_template.format(name='groups_by_question_templates')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_uses_per_question_template(
        P,
        show_title=show_title,
        show_labels=show_labels,
    )
    if save_dir is not None:
        save_path = save_template.format(name='uses_per_question_template')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

    plt.figure(figsize=figsize)
    plot_uses_per_concrete_noun(
        P,
        show_title=show_title,
        show_labels=show_labels,
    )
    if save_dir is not None:
        save_path = save_template.format(name='uses_per_concrete_noun')
        plt.savefig(save_path, bbox_inches='tight')
        print('saved to', save_path)
    plt.show()

