import argparse
import os
import pandas as pd
import numpy as np

from fingerprint.fingerprint import Fingerprint


def read_cli():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set')
    parser.add_argument('--data_type', type=str, help='Numerical-only (`numerical`), categorical-only (`categorical`), mixed (`mixed`)')
    parser.add_argument('--save', action='store_true', default=False, help='Save models and results to disk')
    args = parser.parse_args()

    assert args.data_type in ('numerical', 'categorical', 'mixed')

    return args


def get_exp_dir(data_set, data_type):
    return os.path.join(os.pardir, 'Data', data_set, 'Experiments', data_type.capitalize())


def load_data(data_set, data_type):

    data_file = os.path.join(os.pardir, 'Data', data_set, 'data_set.xlsx')
    df_train = pd.read_excel(data_file, sheet_name='training')
    df_test = pd.read_excel(data_file, sheet_name='test')

    if data_type == 'numerical':
        q_names = Fingerprint.get_q_vars(df_train.columns)
        df_train = df_train.drop(columns=q_names)
        df_test = df_test.drop(columns=q_names)

    elif data_type == 'categorical':
        x_names = Fingerprint.get_x_vars(df_train.columns)
        df_train = df_train.drop(columns=x_names)
        df_test = df_test.drop(columns=x_names)

    return df_train, df_test


def show_accuracy(classes, cm):

    print("Confusion matrix:")
    print(cm)

    per_class = np.diag(cm) / np.sum(cm, axis=1)
    for y, acc in zip(classes, per_class):
        print("Accuracy on class {:2d}: \t{:6.2f}%".format(y, 100.0 * acc))

    overall = np.sum(np.diag(cm)) / np.sum(cm)
    print("Overall accuracy: \t{:6.2f}%\n".format(100.0 * overall))
