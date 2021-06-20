# 
# multistagefingerprint.py
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

import math
import numpy as np
from collections import namedtuple
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import pickle

from fingerprint import Fingerprint
from fingerprint import MultiStageFingerprint

from utils import read_cli, get_exp_dir, load_data, show_accuracy


def onehot2factor(df):
    """Convert one-hot encodings into categorical variables (strings)."""
    q_names = Fingerprint.get_q_vars(df.columns)
    q_factors = list(set([q.split('_')[0] for q in q_names]))
    q_factors.sort()
    for Qi in q_factors:

        # compute string length that can represent all the levels of this factor
        qn = [q for q in df.columns if q.startswith(Qi)]
        n_chars = math.ceil(math.log10(len(qn) + 1))  # 2 digits if >= 10, 3 if >= 100, ...

        # convert one-hot encoding to string
        levels = np.argmax(df.loc[:, df.columns.isin(qn)].to_numpy(), axis=1) + 1
        levels = [str(l).rjust(n_chars, '0') for l in levels]

        # remove one-hot and insert strings
        df = df.drop(labels=qn, axis=1)
        n = int(Qi.replace('q', ''))
        df.insert(n, Qi, levels)

    return df


def prepare_msf_data(df_train, df_test):
    """Load and prepare data for fingerprint method."""

    df_train = onehot2factor(df_train)
    df_test = onehot2factor(df_test)

    Bag = namedtuple('Bag', ['k', 'y', 'df'])

    train_bags = [Bag(k=b[0], y=int(b[1]['y'].mode()), df=b[1].reset_index().drop(columns=['index', 'k', 'y'])) for b in df_train.groupby('k')]
    b_train = [b.df for b in train_bags]
    y_train = [b.y for b in train_bags]

    if 'y' in df_test.columns:
        test_bags = [Bag(k=b[0], y=int(b[1]['y'].mode()), df=b[1].reset_index().drop(columns=['index', 'k', 'y'])) for b in df_test.groupby('k')]
    else:
        test_bags = [Bag(k=b[0], y=-1, df=b[1].reset_index().drop(columns=['index', 'k'])) for b in df_test.groupby(df_test.k)]
    b_test = [b.df for b in test_bags]
    ky_test = pd.DataFrame([(b.k, b.y) for b in test_bags], columns=['k', 'y'])

    return b_train, y_train, b_test, ky_test


FingerprintResults = namedtuple('FingerprintResults', ['model', 'raw', 'cm'])
MultiStageFingerprintResults = namedtuple('MultiStageFingerprintResults', ['model', 'stages'])


def summarise_experiment_results(msf_model, ky_test):

    def summarise_stage_results(model, ky_test, pr_y):
        ky_test.columns = ['k', 'gt_y']

        raw = pd.concat([ky_test, pd.DataFrame(data=pr_y, columns=['pr_y'])], axis=1)
        cm = confusion_matrix(raw.gt_y, raw.pr_y, model.classes_)

        return FingerprintResults(model=model, raw=raw, cm=cm)

    ky_test.columns = ['k', 'gt_y']

    n_stages = len(msf_model.stages_)
    stages = list()
    for i in range(0, n_stages):
        stages.append(summarise_stage_results(msf_model.stages_[i].model, ky_test, msf_model.stages_[i].pr_y))

    return MultiStageFingerprintResults(model=msf_model, stages=stages)


def inspect_experiment_results(results):

    print("\nMULTI-STAGE FINGERPRINT\n")

    for i, stage in enumerate(results.stages):

        print("Stage {} - Factors: {}".format(i, stage.model.q_factors))
        show_accuracy(stage.model.classes_, stage.cm)

        if (i == (len(results) - 1)):
            for k in stage.raw.loc[stage.raw['pr_y'] == -1, 'k'].values:
                print("Bag {} could not be classified. Too few data points!".format(k))


def run_msf_experiment(df_train, df_test, n_pcs, theta, verbose=False):

    # preprocess data
    b_train, y_train, b_test, ky_test = prepare_msf_data(df_train, df_test)

    # apply multi-stage fingerprint
    msf = MultiStageFingerprint(n_pcs, theta, verbose=verbose)
    msf.apply(b_train, y_train, b_test, k_test=ky_test['k'].values)

    # inspect results
    results = summarise_experiment_results(msf, ky_test)
    inspect_experiment_results(results)

    return results


def save_msf_results(save_dir, results):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # multi-stage fingerprint model
    with open(os.path.join(save_dir, 'model'), 'wb') as fp:
        pickle.dump(results.model, fp)

    for i, stage in enumerate(results.stages):

        suffix = '_stage{}'.format(i)

        # fingerprint model
        with open(os.path.join(save_dir, 'model' + suffix), 'wb') as fp:
            pickle.dump(stage.model, fp)

        # bag-level performance
        xlsxwriter = pd.ExcelWriter(os.path.join(save_dir, 'results' + suffix + '.xlsx'), engine='xlsxwriter')
        stage.raw.to_excel(xlsxwriter, sheet_name='raw', index=False)
        pd.DataFrame(stage.cm).to_excel(xlsxwriter, sheet_name='cm', header=False, index=False)
        xlsxwriter.close()


def msf_main(data_set, data_type, save):

    if (data_type != 'categorical'):

        df_train, df_test = load_data(data_set, data_type)

        try:
            results = run_msf_experiment(df_train, df_test, 2, 0.9, verbose=True)
        except:
            results = None

        if save:
            save_dir = os.path.join(get_exp_dir(data_set, data_type), 'MultiStageFingerprint')
            save_msf_results(save_dir, results)


if __name__ == '__main__':

    args = read_cli()
    msf_main(args.data_set, args.data_type, args.save)
