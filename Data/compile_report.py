import argparse
import os
from collections import OrderedDict, namedtuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from compile import __MODELS__
from compile import show_cm
from compile import show_accuracy


def get_exp_dir(data_set, data_type):
    return os.path.join(data_set, 'Experiments', data_type.capitalize())


def format_cm(cm):
    cm.index = ['Class_{}'.format(i) for i in (cm.index + 1)]
    cm.columns = ['Class_{}'.format(i) for i in (cm.columns + 1)]
    return cm


def load_results(exp_dir):

    models = [m for m in __MODELS__ if m in os.listdir(exp_dir)]
    results_file = 'results.xlsx'
    res_files = OrderedDict({m: os.path.join(exp_dir, m, results_file) for m in models})

    raws = OrderedDict({m: pd.read_excel(xf, sheet_name='raw') for m, xf in res_files.items() if os.path.isfile(xf)})
    cms = OrderedDict({m: format_cm(pd.read_excel(xf, sheet_name='cm', header=None)) for m, xf in res_files.items() if os.path.isfile(xf)})

    if 'MultiStageFingerprint' in models:
        msf_dir = os.path.join(exp_dir, 'MultiStageFingerprint')
        n_stages = len([f for f in os.listdir(msf_dir) if ('model' in f) and ('stage' in f)])
        msf_raws = {'MSF_Stage{}'.format(i): pd.read_excel(os.path.join(msf_dir, 'results_stage{}.xlsx'.format(i)), sheet_name='raw') for i in range(0, n_stages)}
        msf_cms = {'MSF_Stage{}'.format(i): format_cm(pd.read_excel(os.path.join(msf_dir, 'results_stage{}.xlsx'.format(i)), sheet_name='cm', header=None)) for i in range(0, n_stages)}

        raws = OrderedDict({**raws, **msf_raws})
        cms = OrderedDict({**cms, **msf_cms})

    return raws, cms


def load_results_vec(exp_dir):

    models = [m for m in __MODELS__ if m in os.listdir(exp_dir)]
    results_file = 'results_vec.xlsx'
    res_files = OrderedDict({m: os.path.join(exp_dir, m, results_file) for m in models})

    # here, models which don't have results at the vector level are filtered out
    raws = OrderedDict({m: pd.read_excel(xf, sheet_name='raw') for m, xf in res_files.items() if os.path.isfile(xf)})
    cms = OrderedDict({m: format_cm(pd.read_excel(xf, sheet_name='cm', header=None)) for m, xf in res_files.items() if os.path.isfile(xf)})

    return raws, cms


def compile_tables(raws, cms):

    def generate_raw_results_table(raws):

        ky_test = next(iter(raws.values())).filter(items=['k', 'gt_y']).rename(columns={'gt_y': 'y'})
        pr_ys = pd.concat([raw.filter(items=['pr_y']).rename(columns={'pr_y': m}) for m, raw in raws.items()], axis=1)

        return pd.concat([ky_test, pr_ys], axis=1)

    def generate_accuracy_table(cms):

        def process_cm(cm):

            Performance = namedtuple('Performance', ['Overall'] + list(cm.index))
            overall = 100.0 * (np.sum(np.diag(cm)) / np.sum(np.sum(cm)))
            per_class = 100.0 * (np.diag(cm) / np.sum(cm, axis=1))

            return Performance(*[overall, *per_class])

        cms = {m: process_cm(cm) for m, cm in cms.items()}

        return pd.DataFrame(data=[[*v] for v in cms.values()], index=list(cms.keys()), columns=next(iter(cms.values()))._fields)

    tab_raw = generate_raw_results_table(raws)
    tab_acc = generate_accuracy_table(cms)

    return tab_raw, tab_acc


def write_to_file(xlsxfile, tab_raw, tab_acc, cms):

    xlsxwriter = pd.ExcelWriter(xlsxfile, engine='xlsxwriter')

    tab_raw.to_excel(xlsxwriter, sheet_name='raw', index=False)
    tab_acc.to_excel(xlsxwriter, sheet_name='acc', float_format='%6.2f')
    for m, cm in cms.items():
        cm.to_excel(xlsxwriter, sheet_name='cm_' + m)

    xlsxwriter.close()


def show_fingerprint_cms(raws, cms, save_dir):

    assert ('MSF_Stage0' in raws.keys())  # at least the result of one stage should be present
    assert len(set(raws.keys()).symmetric_difference(set(cms.keys()))) == 0  # for each result, there should be both the raw output and the corresponding confusion matrix
    assert len(set(np.unique(raws['MSF_Stage0'].gt_y)).symmetric_difference(set([int(n.lstrip('Class_')) for n in cms['MSF_Stage0'].index]))) == 0  # the classes should be identified unambiguously

    n_stages = len(raws.keys())
    n_classes = len(np.unique(raws['MSF_Stage0'].gt_y))

    terminated = not (-1 in raws['MSF_Stage{}'.format(n_stages - 1)].pr_y.values)  # is there still some unclassified bag at the last stage?

    # for each stage, print an "extended" confusion matrix including "unknown" classifications
    if ((n_stages > 2) or (not terminated)):

        for i in range(0, n_stages):

            stage = 'MSF_Stage{}'.format(i)
            raw = raws[stage]
            cm = cms[stage]

            unknown = raw[raw.pr_y == -1].gt_y.value_counts()
            unknown = np.array([(unknown[j] if j in unknown.index else 0) for j in range(1, 1 + n_classes)], dtype=np.int64)[:, None]
            unknown = pd.DataFrame(data=unknown, index=cm.index, columns=['?'])

            show_cm(cm, unknown=unknown, save_file=os.path.join(save_dir, 'cm_' + stage + '.eps'))

    # print a normal confusion matrix
    if terminated:
        show_cm(cms['MSF_Stage{}'.format(n_stages - 1)], save_file=os.path.join(save_dir, 'cm_MSF.eps'))


def compile_report(data_set, data_type):

    exp_dir = get_exp_dir(data_set, data_type)

    rep_dir = os.path.join(exp_dir, '_REPORT_')
    if not os.path.isdir(rep_dir):
        os.makedirs(rep_dir, exist_ok=True)

    # bag-level results
    raws, cms = load_results(exp_dir)
    tab_raw, tab_acc = compile_tables(raws, cms)

    bag_dir = os.path.join(rep_dir, '_BAG_')
    if not os.path.isdir(bag_dir):
        os.makedirs(bag_dir, exist_ok=True)
    write_to_file(os.path.join(bag_dir, 'report.xlsx'), tab_raw, tab_acc, cms)

    # show bag-level results graphically
    bag_fig_dir = os.path.join(bag_dir, 'Figures')
    if not os.path.isdir(bag_fig_dir):
        os.makedirs(bag_fig_dir, exist_ok=True)

    for m, cm in cms.items():
        if not m.startswith('MSF'):
            show_cm(cm, save_file=os.path.join(bag_fig_dir, 'cm_' + m + '.eps'))

    raws_msf = {m: raw for m, raw in raws.items() if m.startswith('MSF')}
    cms_msf = {m: cm for m, cm in cms.items() if m.startswith('MSF')}
    if len(raws_msf) > 0:
        show_fingerprint_cms(raws_msf, cms_msf, bag_fig_dir)

    show_accuracy(tab_acc, save_file=os.path.join(bag_fig_dir, 'accuracy.eps'))

    # vector-level results
    raws_vec, cms_vec = load_results_vec(exp_dir)
    tab_raw_vec, tab_acc_vec = compile_tables(raws_vec, cms_vec)

    vec_dir = os.path.join(rep_dir, '_VECTOR_')
    if not os.path.isdir(vec_dir):
        os.makedirs(vec_dir, exist_ok=True)
    write_to_file(os.path.join(vec_dir, 'report_vec.xlsx'), tab_raw_vec, tab_acc_vec, cms_vec)

    # show vector-level results graphically
    vec_fig_dir = os.path.join(vec_dir, 'Figures')
    if not os.path.isdir(vec_fig_dir):
        os.makedirs(vec_fig_dir, exist_ok=True)

    for m, cm in cms_vec.items():
        assert not m.startswith('MSF')  # the fingerprint should does not make predictions at the vector level!
        show_cm(cm, save_file=os.path.join(vec_fig_dir, 'cm_' + m + '_vec.eps'))

    show_accuracy(tab_acc_vec, save_file=os.path.join(vec_fig_dir, 'accuracy_vec.eps'))

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set')
    parser.add_argument('--data_type', type=str, help='Numerical-only (`numerical`), categorical-only (`categorical`), mixed (`mixed`)')
    args = parser.parse_args()

    assert args.data_type in ('numerical', 'categorical', 'mixed')

    compile_report(args.data_set, args.data_type)
