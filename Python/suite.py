import os

from utils import read_cli, get_exp_dir, load_data
from randomforest import run_rf_experiment, save_rf_results
from multilayerperceptron import run_mlp_experiment, save_mlp_results
from multipleinstancelearning import run_mil_experiment, save_mil_results
from multistagefingerprint import run_msf_experiment, save_msf_results


def suite(data_set, data_type, save):

    exp_dir = get_exp_dir(data_set, data_type)
    df_train, df_test = load_data(data_set, data_type)

    # random forest
    try:
        rf_results = run_rf_experiment(df_train, df_test)
    except:
        rf_results = None
    if save and rf_results:
        rf_save_dir = os.path.join(exp_dir, 'RF')
        save_rf_results(rf_save_dir, rf_results)

    # multi-layer perceptron
    try:
        mlp_results = run_mlp_experiment(df_train, df_test)
    except:
        mlp_results = None
    if save and mlp_results:
        mlp_save_dir = os.path.join(exp_dir, 'MLP')
        save_mlp_results(mlp_save_dir, mlp_results)

    # multiple instance learning
    try:
        mil_results = run_mil_experiment(df_train, df_test)
    except:
        mil_results = None
    if save and mil_results:
        mil_save_dir = os.path.join(exp_dir, 'MIL')
        save_mil_results(mil_save_dir, mil_results)

    # multi-stage fingerprint
    if (data_type != 'categorical'):  # fingerprint can't be executed just on categorical data!
        try:
            msf_results = run_msf_experiment(df_train, df_test, 2, 0.9)
        except:
            msf_results = None
        if save and msf_results:
            msf_save_dir = os.path.join(exp_dir, 'MultiStageFingerprint')
            save_msf_results(msf_save_dir, msf_results)


if __name__ == '__main__':

    args = read_cli()
    suite(args.data_set, args.data_type, args.save)
