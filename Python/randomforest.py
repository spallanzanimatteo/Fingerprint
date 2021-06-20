import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from collections import namedtuple
from sklearn.metrics import confusion_matrix
import os
import pickle

from utils import read_cli, get_exp_dir, load_data, show_accuracy


def prepare_rf_data(df_train, df_test):

    x_train = df_train.drop(columns=['k', 'y']).to_numpy()
    y_train = df_train['y'].to_numpy()

    x_test = df_test.drop(columns=['k', 'y']).to_numpy()
    ky_test = df_test[['k', 'y']]

    return x_train, y_train, x_test, ky_test


RFResults = namedtuple('RFResults', ['model', 'raw', 'cm', 'raw_vec', 'cm_vec'])


def summarise_experiment_results(model, ky_test, pr_y):

    ky_test.columns = ['k', 'gt_y']

    raw_vec = pd.concat([ky_test, pd.DataFrame(data=pr_y, columns=['pr_y'])], axis=1)
    cm_vec = confusion_matrix(raw_vec.gt_y, raw_vec.pr_y, labels=model.classes_)

    raw = pd.DataFrame(data=[(tb[0], tb[1]['gt_y'].mode().values[0], tb[1]['pr_y'].mode().values[0]) for tb in raw_vec.groupby('k')], columns=['k', 'gt_y', 'pr_y'])
    cm = confusion_matrix(raw.gt_y, raw.pr_y, labels=model.classes_)

    return RFResults(model=model, raw=raw, cm=cm, raw_vec=raw_vec, cm_vec=cm_vec)


def inspect_experiment_results(results):

    print("\nRANDOM FOREST\n")

    print("Vector-level performance")
    show_accuracy(results.model.classes_, results.cm_vec)

    print("Bag-level performance")
    show_accuracy(results.model.classes_, results.cm)


def run_rf_experiment(df_train, df_test):

    # preprocess data
    x_train, y_train, x_test, ky_test = prepare_rf_data(df_train, df_test)

    # train neural network classifier
    rf = VotingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy', random_state=1000))], voting='soft')
    rf.fit(x_train, y_train)
    pr_y = rf.predict(x_test)

    # compute results
    results = summarise_experiment_results(rf, ky_test, pr_y)
    inspect_experiment_results(results)

    return results


def save_rf_results(save_dir, results):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

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


def rf_main(data_set, data_type, save):

    df_train, df_test = load_data(data_set, data_type)

    try:
        results = run_rf_experiment(df_train, df_test)
    except:
        results = None

    if save and results:
        save_dir = os.path.join(get_exp_dir(data_set, data_type), 'RF')
        save_rf_results(save_dir, results)


if __name__ == '__main__':

    args = read_cli()
    rf_main(args.data_set, args.data_type, args.save)
