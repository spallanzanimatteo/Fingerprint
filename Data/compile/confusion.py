# 
# confusion.py
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
import matplotlib.font_manager as fm
import itertools
import seaborn as sn


def change_face_colour(facecolours, pos, facetype):

    c_correct = np.array([204, 255, 204, 255]) / 255.0  # light green
    c_wrong = np.array([255, 204, 204, 255]) / 255.0  # light red
    c_unknown = np.array([224, 224, 224, 255]) / 255.0  # light gray
    c_total = np.array([240, 240, 240, 255]) / 255.0  # extra-light gray

    if facetype == 'correct':
        facecolours[pos] = c_correct
    elif facetype == 'wrong':
        facecolours[pos] = c_wrong
    elif facetype == 'unknown':
        facecolours[pos] = c_unknown
    elif facetype == 'total':
        facecolours[pos] = c_total


def write_cm_cell(texts, pos, count, fraction, celltype, fs_count, fs_fraction):

    c_correct = np.array([0, 153, 0, 255]) / 255.0  # dark green
    c_wrong = np.array([204, 0, 0, 255]) / 255.0  # dark red
    c_unknown = np.array([32, 32, 32, 255]) / 255.0  # dark gray

    if celltype == 'correct':
        c = c_correct
    elif celltype == 'wrong':
        c = c_wrong
    elif celltype == 'unknown':
        c = c_unknown

    txt_count = "{:d}".format(count)
    opt_count = dict(color=c, ha='center', va='center', fontproperties=fm.FontProperties(weight='bold', size=fs_count))
    pos_count = (texts[pos]._x, texts[pos]._y)
    new_count = dict(x=pos_count[0], y=pos_count[1], text=txt_count, opt=opt_count)

    txt_fraction = "{:.2f}%".format(fraction)
    opt_fraction = dict(color=c, ha='center', va='center', fontproperties=fm.FontProperties(size=fs_fraction))
    pos_fraction = (texts[pos]._x, texts[pos]._y + 0.3)
    new_fraction = dict(x=pos_fraction[0], y=pos_fraction[1], text=txt_fraction, opt=opt_fraction)

    return new_count, new_fraction


def write_tot_cell(texts, pos, count, correct, wrong, fs_count, fs_fraction):

    c_count = np.array([32, 32, 32, 255]) / 255.0  # dark gray
    c_correct = np.array([0, 153, 0, 255]) / 255.0  # dark green
    c_wrong = np.array([204, 0, 0, 255]) / 255.0  # dark red

    txt_count = "{:d}".format(count)
    opt_count = dict(color=c_count, ha='center', va='center', fontproperties=fm.FontProperties(weight='bold', size=fs_count))
    pos_count = (texts[pos]._x, texts[pos]._y)
    new_count = dict(x=pos_count[0], y=pos_count[1], text=txt_count, opt=opt_count)

    txt_correct = "{:.2f}%".format(correct)
    opt_correct = dict(color=c_correct, ha='center', va='center', fontproperties=fm.FontProperties(weight='bold', size=fs_fraction))
    pos_correct = (texts[pos]._x, texts[pos]._y - 0.3)
    new_correct = dict(x=pos_correct[0], y=pos_correct[1], text=txt_correct, opt=opt_correct)

    txt_wrong = "{:.2f}%".format(wrong)
    opt_wrong = dict(color=c_wrong, ha='center', va='center', fontproperties=fm.FontProperties(weight='bold', size=fs_fraction))
    pos_wrong = (texts[pos]._x, texts[pos]._y + 0.3)
    new_wrong = dict(x=pos_wrong[0], y=pos_wrong[1], text=txt_wrong, opt=opt_wrong)

    return new_count, new_correct, new_wrong


def show_cm(cm, unknown=None, save_file=None):

    assert cm.shape[0] == cm.shape[1]

    cm = cm.copy()  # since the table will be modified, these changes should not reflect outside this function's scope
    if unknown is not None:  # extend confusion matrix
        cm['?'] = unknown.values
    cm['Total'] = np.sum(cm.values, axis=1)  # per-class sum

    n_classes = cm.shape[0]
    n_columns = cm.shape[1]
    cm_rel = 100.0 * (cm.values[:, 0:-1] / np.sum(cm.Total))

    correct = 100.0 * (np.diag(cm.values) / cm.Total)
    wrong = 100.0 * ((np.sum(cm.values[:, 0:n_classes], axis=1) - np.diag(cm.values)) / cm.Total)  # exclude unknown from accuracy computation

    # draw!
    fig = plt.figure()
    fig.set_size_inches(9, 6)
    ax = fig.add_subplot(111)

    # font sizes for cell annotations
    fs_count = 18
    fs_fraction = 14

    ax_hm = sn.heatmap(cm, annot=True, cbar=False, linewidths=1.5, ax=ax, square=True)
    facecolours = ax_hm.findobj(QuadMesh)[0].get_facecolors()
    old_texts = ax_hm.collections[0].axes.texts
    new_texts = list()

    for i, j in itertools.product(range(0, n_classes), range(0, n_columns)):

        pos = i * n_columns + j

        if i == j:  # element on the diagonal (correct classifications)
            change_face_colour(facecolours, pos, 'correct')
            new_texts.extend(list(write_cm_cell(old_texts, pos, cm.values[i, i], cm_rel[i, i], 'correct', fs_count, fs_fraction)))

        elif j < n_classes:  # off-diagonal elements (wrong classifications)
            change_face_colour(facecolours, pos, 'wrong')
            new_texts.extend(list(write_cm_cell(old_texts, pos, cm.values[i, j], cm_rel[i, j], 'wrong', fs_count, fs_fraction)))

        elif (j == n_classes) and (n_columns == n_classes + 2):  # unclassified items
            change_face_colour(facecolours, pos, 'unknown')
            new_texts.extend(list(write_cm_cell(old_texts, pos, cm.values[i, j], cm_rel[i, j], 'unknown', fs_count, fs_fraction)))

        else:  # per-class totals
            change_face_colour(facecolours, pos, 'total')
            new_texts.extend(list(write_tot_cell(old_texts, pos, cm.values[i, j], correct[i], wrong[i], fs_count, fs_fraction)))

    # apply the change to square colours
    ax_hm.collections[0].set_facecolors(facecolours)

    # remove old annotations and write the new ones
    for ot in list(old_texts):
        ot.remove()

    for nt in new_texts:
        ax_hm.text(nt['x'], nt['y'], nt['text'], **nt['opt'])

    # annotate axes
    fs_axis_label = 18
    fs_cell_label = 14

    # y axis (true labels)
    ax.set_ylabel("Ground truth", fontdict={'weight': 'bold', 'size': fs_axis_label})
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_yticks(np.arange(0, len(cm.index)) + 0.5)
    ax.set_yticklabels([n.lstrip('Class_') for n in cm.index], fontdict={'weight': 'bold', 'size': fs_cell_label})

    # x axis (predicted labels)
    ax.set_xlabel("Predicted", fontdict={'weight': 'bold', 'size': fs_axis_label})
    ax.xaxis.set_label_coords((n_classes / 2) / n_columns, 1.11)
    ax.set_xticks(np.arange(0, len(cm.columns)) + 0.5)
    ax.set_xticklabels([n.lstrip('Class_') for n in cm.columns], fontdict={'weight': 'bold', 'size': fs_cell_label})
    ax.xaxis.tick_top()

    # hide ticks
    ax.tick_params(length=0)

    plt.tight_layout()

    if save_file:
        assert (save_file.endswith('.eps'))
        plt.savefig(save_file)
        plt.savefig(save_file.replace('.eps', '.png'))  # save also in PNG format
