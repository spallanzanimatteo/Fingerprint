import argparse

from functools import reduce
from operator import mul
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd

import os
import json
import shutil
import matplotlib.pyplot as plt

from create import create_means, create_covariance_matrices, get_bag
from create import show_data_set, show_bag_on_data


def get_point_cloud_statistics(nD, hg_names, classes,
                               primary_cond,
                               rM_cond, rM, epsrM,
                               dm_cond, rm_cond, rm, epsrm,
                               maxV_cond, maxV, epsV,
                               spectrumK_cond, spectrumK, epsK, spherical,
                               orientation_cond, ax_align):

    nG = len(hg_names)
    nC = len(classes)

    # geometric parameters describing Gaussian distributions
    MultivariateGaussian = namedtuple('MultivariateGaussian', ['mean', 'covariance'])

    means = create_means(nD, nG, nC, primary_cond, rM_cond, rM, epsrM, dm_cond, rm_cond, rm, epsrm)
    cov_matrices = create_covariance_matrices(nD, nG, nC, maxV_cond, maxV, epsV, spectrumK_cond, spectrumK, epsK, spherical, orientation_cond, ax_align)

    stats_dict = {n: {k: None for k in classes} for n in hg_names}
    for ig in range(0, nG):
        hg_n = hg_names[ig]
        for ic in range(0, nC):
            c_n = classes[ic]
            stats_dict[hg_n][c_n] = MultivariateGaussian(mean=means[ig, ic], covariance=cov_matrices[ig, ic])

    return stats_dict


def assemble_data_set(classes, pC, maxN, pN, nS, maxO, hg_names, pG, stats_dict, n_train, n_test):

    # proxy names for the bags
    __N_MAX_BAGS__ = 10000
    assert (n_train + n_test) <= __N_MAX_BAGS__  # do you really want that test bags get mixed up with training bags?
    __N_PAD_ZEROS__ = int(np.ceil(np.log10(__N_MAX_BAGS__ + 1)))
    names = ['k' + str(i + 1).rjust(__N_PAD_ZEROS__, '0') for i in range(0, __N_MAX_BAGS__)]
    np.random.shuffle(names)

    b_train_names = names[:n_train]
    b_test_names = names[len(names) - n_test:]
    b_names = b_train_names + b_test_names

    # generate set
    for i in range(0, n_train + n_test):
        bag = get_bag(classes, pC, maxN, pN, hg_names, pG, nS, maxO, stats_dict)
        bag = pd.concat([pd.DataFrame(data=[b_names[i]] * len(bag), columns=['k']), bag], axis=1)  # attach bag information
        try:
            df = pd.concat([df, bag])
        except UnboundLocalError:
            df = bag

    return b_train_names, b_test_names, df.reset_index().drop(columns=['index'])


def generate_data_set(options, normalise):

    seed = options['seed']
    if seed:
        np.random.seed(seed=seed)  # for replicability

    # classes
    classes = tuple(options['classes'])
    pC = np.array(options['pC'])
    assert len(classes) > 1  # at least two classes
    assert np.sum(pC) == 1.0

    # homogeneous groups
    Qs = [Qi for Qi in options['hgs']]  # categorical variables domains
    Q = list(itertools.product(*Qs))
    hg_names = list(map(lambda hg: '_'.join([str(n) for n in hg]), Q))
    assert len(hg_names) > 1  # at least two homogeneous groups
    assert len(hg_names) == reduce(mul, [len(Qi) for Qi in Qs])
    pG = options['pG']
    assert len(set(classes).symmetric_difference(set(pG.keys()))) == 0  # a (discrete) probability distribution over `Q`
                                                                        # should be defined for every class
    for c in classes:  # check that `pG` actually contains probability distributions over `Q`
        assert len(pG[c]) == len(hg_names)
        assert np.sum(pG[c]) == 1.0

    # bag properties
    maxN = options['maxN']
    assert maxN > 0  # at least one sub-bag in each bag
    pN = np.array(options['pN'])
    assert len(pN) == len(classes)
    assert np.all(0.0 < pN) & np.all(pN <= 1.0)
    nS = options['nS']
    assert nS > 0  # at least one observation in each sub-bag (less than two could cause problems when computing covariance matrix: watch out!)
    maxO = nS * maxN  # maximum number of observations in each bag

    # geometric properties
    nD = options['nD']
    assert nD > 0

    primary_cond = options['primary_cond']
    assert primary_cond in ('G', 'C')
    rM_cond = options['rM_cond']
    assert rM_cond in ('M', 'I')
    rM = options['rM']
    epsrM = options['epsrM']
    dm_cond = options['dm_cond']
    assert dm_cond in ('F', 'm')
    rm_cond = options['rm_cond']
    assert rm_cond in ('F', 'M', 'm', 'I')
    rm = options['rm']
    assert rm <= rM  # primary (major, `M`) scattering should dominate secondary (minor, `m`) scattering
    epsrm = options['epsrm']

    maxV_cond = options['maxV_cond']  # volume factor
    assert maxV_cond in ('F', 'G', 'C', 'I')
    maxV = options['maxV']
    epsV = options['epsV']
    spectrumK_cond = options['spectrumK_cond']  # shape factor
    assert spectrumK_cond in ('F', 'G', 'C', 'I')
    spectrumK = options['spectrumK']
    epsK = options['epsK']
    spherical = options['spherical']
    orientation_cond = options['orientation_cond']  # orientation factor
    assert orientation_cond in ('F', 'G', 'C', 'I')
    ax_align = options['ax_align']

    stats_dict = get_point_cloud_statistics(nD, hg_names, classes,
                                            primary_cond,
                                            rM_cond, rM, epsrM,
                                            dm_cond, rm_cond, rm, epsrm,
                                            maxV_cond, maxV, epsV,
                                            spectrumK_cond, spectrumK, epsK, spherical,
                                            orientation_cond, ax_align)

    b_train_names, b_test_names, df = assemble_data_set(classes, pC,
                                                        maxN, pN, nS, maxO,
                                                        hg_names, pG, stats_dict,
                                                        n_train=options['n_train'], n_test=options['n_test'])

    # expand qualitative categorical variables into one-hot encodings;
    # this must be done with the vision over both training and test points, so that no known level is missed
    x_names = [x for x in df.columns if x.startswith('x')]
    q_names = [q for q in df.columns if q.startswith('q')]
    for q in q_names:
        oh_q = pd.get_dummies(df[q])
        oh_q_mapper = {v: q + '_' + str(int(v)) for v in oh_q.columns}
        oh_q = oh_q.rename(columns=oh_q_mapper)
        try:
            oh = pd.concat([oh, oh_q], axis=1)
        except UnboundLocalError:
            oh = oh_q
    df = pd.concat([df['k'], oh, df[x_names], df['y']], axis=1)

    # split training set from test set
    df_train_idx = df['k'].isin(b_train_names)
    df_train = df[df_train_idx].copy().reset_index().drop(columns=['index'])
    df_test_idx = df['k'].isin(b_test_names)
    df_test = df[df_test_idx].copy().reset_index().drop(columns=['index'])

    # normalise
    if normalise:
        mu = df_train[x_names].apply(np.mean)
        sigma = df_train[x_names].apply(np.std)
        x_train_norm = df_train[x_names].apply(lambda x, m, s: (x - m) / s, args=(mu, sigma), axis=1)
        df_train[x_names] = x_train_norm
        x_test_norm = df_test[x_names].apply(lambda x, m, s: (x - m) / s, args=(mu, sigma), axis=1)
        df_test[x_names] = x_test_norm

    return df_train, df_test


def create_toy_data_set(config_file, normalise=True, save=False):

    with open(config_file, 'r') as fp:
        options = json.load(fp)

    df_train, df_test = generate_data_set(options, normalise=normalise)

    if save:

        data_set_name = os.path.basename(config_file).lstrip('config_').rstrip('.json')
        data_dir = os.path.join(os.path.dirname(config_file), data_set_name)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        shutil.move(config_file, os.path.join(data_dir, os.path.basename(config_file)))  # save configuration file, so that you know which are its generating parameters

        xlsx_writer = pd.ExcelWriter(os.path.join(data_dir, 'data_set.xlsx.'), engine='xlsxwriter')
        df_train.to_excel(xlsx_writer, sheet_name='training', header=True, index=False)
        df_test.to_excel(xlsx_writer, sheet_name='test', header=True, index=False)
        xlsx_writer.save()

        fig_dir = os.path.join(data_dir, 'Figures')
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
        show_data_set(df_train, save_file=os.path.join(fig_dir, 'data_set.eps'))
        show_bag_on_data(df_train, df_test, seed=126, save_file=os.path.join(fig_dir, 'bag_on_data.eps'))

    else:

        show_data_set(df_train)
        show_bag_on_data(df_train, df_test, seed=126)

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--normalise', action='store_true', default=False, help='Shift numeric data by mean and scale by variance (component-wise)')
    parser.add_argument('--save', action='store_true', default=False, help='Save generated data set to disk')
    args = parser.parse_args()

    create_toy_data_set(args.config_file, normalise=args.normalise, save=args.save)
