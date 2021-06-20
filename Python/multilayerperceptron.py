# 
# multilayerperceptron.py
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

import pandas as pd
from sklearn.neural_network import MLPClassifier
from collections import namedtuple
from sklearn.metrics import confusion_matrix
import os
import pickle

from utils import read_cli, get_exp_dir, load_data, show_accuracy


def prepare_mlp_data(df_train, df_test):

    x_train = df_train.drop(columns=['k', 'y']).to_numpy()
    y_train = df_train['y'].to_numpy()

    x_test = df_test.drop(columns=['k', 'y']).to_numpy()
    ky_test = df_test[['k', 'y']]

    return x_train, y_train, x_test, ky_test


MLPResults = namedtuple('MLPResults', ['model', 'raw', 'cm', 'raw_vec', 'cm_vec'])


def summarise_experiment_results(model, ky_test, pr_y):

    ky_test.columns = ['k', 'gt_y']

    raw_vec = pd.concat([ky_test, pd.DataFrame(data=pr_y, columns=['pr_y'])], axis=1)
    cm_vec = confusion_matrix(raw_vec.gt_y, raw_vec.pr_y, labels=model.classes_)

    raw = pd.DataFrame(data=[(tb[0], tb[1]['gt_y'].mode().values[0], tb[1]['pr_y'].mode().values[0]) for tb in raw_vec.groupby(['k'])], columns=['k', 'gt_y', 'pr_y'])
    cm = confusion_matrix(raw.gt_y, raw.pr_y, labels=model.classes_)

    return MLPResults(model=model, raw=raw, cm=cm, raw_vec=raw_vec, cm_vec=cm_vec)


def inspect_experiment_results(results):

    print("\nMULTI-LAYER PERCEPTRON\n")

    print("Vector-level performance")
    show_accuracy(results.model.classes_, results.cm_vec)

    print("Bag-level performance")
    show_accuracy(results.model.classes_, results.cm)


def run_mlp_experiment(df_train, df_test):

    # preprocess data
    x_train, y_train, x_test, ky_test = prepare_mlp_data(df_train, df_test)

    # train neural network classifier
    mlp = MLPClassifier(hidden_layer_sizes=(20, 10), activation='tanh', random_state=1000)
    mlp.fit(x_train, y_train)
    pr_y = mlp.predict(x_test)

    # compute results
    results = summarise_experiment_results(mlp, ky_test, pr_y)
    inspect_experiment_results(results)

    return results


def save_mlp_results(save_dir, results):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # scikit-learn model
    with open(os.path.join(save_dir, 'model'), 'wb') as fp:
        pickle.dump(results.model, fp)

    # bag-level performance
    xlsxwriter = pd.ExcelWriter(os.path.join(save_dir, 'results.xlsx'), engine='xlsxwriter')
    results.raw.to_excel(xlsxwriter, sheet_name='raw', index=False)
    pd.DataFrame(results.cm).to_excel(xlsxwriter, sheet_name='cm', header=False, index=False)
    xlsxwriter.close()

    # vector-level performance
    xlsxwriter = pd.ExcelWriter(os.path.join(save_dir, 'results_vec.xlsx'), engine='xlsxwriter')
    results.raw_vec.to_excel(xlsxwriter, sheet_name='raw', index=False)
    pd.DataFrame(results.cm_vec).to_excel(xlsxwriter, sheet_name='cm', header=False, index=False)
    xlsxwriter.close()


def mlp_main(data_set, data_type, save):

    df_train, df_test = load_data(data_set, data_type)

    try:
        results = run_mlp_experiment(df_train, df_test)
    except:
        results = None

    if save:
        save_dir = os.path.join(get_exp_dir(data_set, data_type), 'MLP')
        save_mlp_results(save_dir, results)


if __name__ == '__main__':

    args = read_cli()
    mlp_main(args.data_set, args.data_type, args.save)
