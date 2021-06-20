# 
# multipleinstancelearning.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2018-2021 Matteo Spallanzani
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# 

import copy
from collections import namedtuple
import pandas as pd
from more_itertools import locate
import numpy as np
import itertools
import misvm
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import os
import pickle

from fingerprint import Fingerprint
from utils import read_cli, get_exp_dir, load_data, show_accuracy


def prepare_mil_data(dt_train, dt_test):

    qx_names = Fingerprint.get_q_vars(dt_train.columns) + Fingerprint.get_x_vars(dt_train.columns)

    # rescale variables to the range [-1, 1]
    x_train = dt_train[qx_names].to_numpy()
    scaling = np.max(np.abs(x_train), axis=0)
    x_train = x_train / scaling
    dt_train_copy = copy.deepcopy(dt_train)
    dt_train_copy[qx_names] = x_train

    x_test = dt_test[qx_names].to_numpy()
    x_test = x_test / scaling
    dt_test_copy = copy.deepcopy(dt_test)
    dt_test_copy[qx_names] = x_test

    Bag = namedtuple('Bag', ['k', 'y', 'np'])

    train_bags = [Bag(k=b[0], y=int(b[1]['y'].mode()), np=b[1][qx_names].to_numpy()) for b in dt_train_copy.groupby('k')]
    b_train = [b.np for b in train_bags]
    y_train = [b.y for b in train_bags]

    test_bags = [Bag(k=b[0], y=int(b[1]['y'].mode()), np=b[1][qx_names].to_numpy()) for b in dt_test_copy.groupby('k')]
    b_test = [b.np for b in test_bags]
    ky_test = pd.DataFrame([(b.k, b.y) for b in test_bags], columns=['k', 'y'])

    return b_train, y_train, b_test, ky_test


def get_data_set_y1_y2(b_train, y_train, y1, y2):

    idx = list(locate(y_train, lambda y: (y == y1) or (y == y2)))

    b_train_reduced = [b_train[i] for i in idx]
    y_train_reduced = [(-1 if y == y1 else 1) for y in [y_train[i] for i in idx]]

    return b_train_reduced, y_train_reduced


def get_fold_idxs(n_items, n_folds):

    fold_sizes = (n_folds - 1) * [n_items // n_folds] + [n_items - (n_folds - 1) * (n_items // n_folds)]
    fold_idxs = [set(range(s, e)) for s, e in zip(np.cumsum(fold_sizes) - fold_sizes, np.cumsum(fold_sizes))]

    return fold_idxs


def get_fold_sets(b_train, y_train, fold_idxs, i_fold):

    idx_test = fold_idxs[i_fold]
    idx_train = set(itertools.chain.from_iterable(fold_idxs)).difference(idx_test)

    fold_b_train = [b_train[i] for i in idx_train]
    fold_y_train = [y_train[i] for i in idx_train]

    fold_b_test = [b_train[i] for i in idx_test]
    fold_y_test = [y_train[i] for i in idx_test]

    return fold_b_train, fold_y_train, fold_b_test, fold_y_test


def tune_C_gamma(b_train, y_train):

    n_folds = 3
    assert n_folds > 1
    C_candidates = [1.0, 10.0, 100.0]
    n_C = len(C_candidates)
    g_candidates = [0.05, 0.1, 0.5]
    n_g = len(g_candidates)

    err_cv = np.zeros([n_folds, n_C, n_g])

    fold_idxs = get_fold_idxs(len(y_train), n_folds)
    for i_fold in range(0, n_folds):

        fold_b_train, fold_y_train, fold_b_test, fold_y_test = get_fold_sets(b_train, y_train, fold_idxs, i_fold)

        for i_C, i_g in itertools.product(list(range(0, n_C)), list(range(0, n_g))):
            print("Fold {} - Testing C = {}, g = {}".format(i_fold, C_candidates[i_C], g_candidates[i_g]))
            fold_model = misvm.MISVM(kernel='rbf', C=C_candidates[i_C], gamma=g_candidates[i_g])
            fold_model.fit(fold_b_train, fold_y_train)

            fold_pr_y = np.sign(fold_model.predict(fold_b_test))
            err_cv[i_fold, i_C, i_g] = len(fold_y_test) - np.sum(np.equal(fold_y_test, fold_pr_y))

    mean_err_cv = np.mean(err_cv, axis=0)
    i_C_opt, i_g_opt = np.unravel_index(np.argmin(mean_err_cv), mean_err_cv.shape)

    return C_candidates[i_C_opt], g_candidates[i_g_opt]


MILResults = namedtuple('MILResults', ['models_1v1', 'raw', 'cm'])


def summarise_experiment_results(models_1v1, ky_test, pr_y):

    ky_test.columns = ['k', 'gt_y']

    raw = pd.concat([ky_test, pd.DataFrame(data=pr_y, columns=['pr_y'])], axis=1)
    cm = confusion_matrix(raw.gt_y, raw.pr_y, labels=models_1v1['classes'])

    return MILResults(models_1v1=models_1v1, raw=raw, cm=cm)


def inspect_experiment_results(results):

    print("\nMULTIPLE INSTANCE LEARNING (MI-SVM)\n")
    show_accuracy(results.models_1v1['classes'], results.cm)


def run_mil_experiment(dt_train, dt_test):

    b_train, y_train, b_test, ky_test = prepare_mil_data(dt_train, dt_test)

    # train one-vs-one models
    classes = np.unique(y_train)
    classifiers = np.array(list(itertools.combinations(classes, 2)))
    pr_y = np.zeros([len(b_test), len(classifiers)])

    models_1v1 = {'classes': classes}
    for im, (y1, y2) in enumerate(classifiers):

        # one-vs-one data sets
        b_train_reduced, y_train_reduced = get_data_set_y1_y2(b_train, y_train, y1, y2)

        # find best hyper-parameters and train model
        C, g = tune_C_gamma(b_train_reduced, y_train_reduced)
        model = misvm.MISVM(kernel='rbf', C=C, gamma=g)
        model.fit(b_train_reduced, y_train_reduced)

        # predict one-vs-one label for each test bag
        pr_y[:, im] = np.sign(model.predict(b_test))
        pr_y[pr_y[:, im] == 1, im] = y2
        pr_y[pr_y[:, im] == -1, im] = y1

        models_1v1[str(y1) + '_' + str(y2)] = model

    pr_y, _ = mode(pr_y, 1)

    results = summarise_experiment_results(models_1v1, ky_test, pr_y)
    inspect_experiment_results(results)

    return results


def save_mil_results(save_dir, results):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # one-vs-one SVM models
    for k, v in results.models_1v1.items():
        if k != 'classes':
            with open(os.path.join(save_dir, 'model_' + k), 'wb') as fp:
                pickle.dump(results.models_1v1[k], fp)

    # bag-level performance
    xlsxwriter = pd.ExcelWriter(os.path.join(save_dir, 'results.xlsx'), engine='xlsxwriter')
    results.raw.to_excel(xlsxwriter, sheet_name='raw', index=False)
    pd.DataFrame(results.cm).to_excel(xlsxwriter, sheet_name='cm', header=False, index=False)
    xlsxwriter.close()


def mil_main(data_set, data_type, save):

    df_train, df_test = load_data(data_set, data_type)

    try:
        results = run_mil_experiment(df_train, df_test)
    except:
        results = None

    if save and results:
        save_dir = os.path.join(get_exp_dir(data_set, data_type), 'MIL')
        save_mil_results(save_dir, results)


if __name__ == '__main__':

    args = read_cli()
    mil_main(args.data_set, args.data_type, args.save)
