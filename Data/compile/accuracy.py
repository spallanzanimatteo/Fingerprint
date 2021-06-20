# 
# accuracy.py
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
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt

from .models import __MODELS__

__MODELS2COLOURS__ = {m: n for m, n in zip(__MODELS__, range(0, len(__MODELS__)))}


def show_accuracy(tab_acc, save_file=None, axis_label_size=12):

    # read data about executed models
    models2rows = {m: m for m in tab_acc.index.values if (m in __MODELS__[:-1])}
    if len([m for m in tab_acc.index.values if m.startswith('MSF')]) > 0:
        # the global results of the multi-stage fingerprint are computed after the last stage of the method
        n_stages = len([m for m in tab_acc.index.values if m.startswith('MSF')])
        models2rows['MultiStageFingerprint'] = 'MSF_Stage{}'.format(n_stages - 1)

    models_idx = np.arange(0, len(models2rows))

    # get number of classes
    nC = len([n for n in tab_acc.columns if n.startswith('Class')])

    # get colours for each model
    def get_colours(models2rows):
        cmap = matplotlib.cm.get_cmap('Set2')
        c_bars = cmap(np.array([__MODELS2COLOURS__[m] for m in models2rows.keys()]))
        try:
            c_bars[list(models2rows.keys()).index('MultiStageFingerprint'), :] = np.array([1.0, 0.0, 0.0, 1.0])  # make fingerprint results bright red, so that they stand out
        except ValueError:
            pass
        alpha = 0.2
        c_lines = np.array([0.0, 0.0, 0.0, alpha])  # reference accuracy levels will be printed in the background to ease reading the plots

        return c_bars, c_lines

    c_bars, c_lines = get_colours(models2rows)

    def get_axis_labels(models2acc):
        # shorten names to fit figure space
        axs_labels = list(models2acc.keys())

        try:
            axs_labels[axs_labels.index('MixtComp')] = 'MC'
        except ValueError:
            pass
        try:
            axs_labels[axs_labels.index('MultiStageFingerprint')] = 'MSF'
        except ValueError:
            pass

        return axs_labels

    # draw!
    fig = plt.figure()
    fig.set_size_inches(3 * (1 + nC) + 1, 5)
    axs = []

    fs_label = 14
    angle_label = 90

    # prepare data
    label = 'Overall'
    models2acc = {m: tab_acc.loc[r, label] for m, r in models2rows.items()}

    # prepare canvas
    axs.append(fig.add_subplot(1, 1 + nC, 1))
    axs[0].set_xlim([models_idx[0] - 1, models_idx[-1] + 1])
    axs[0].set_ylim([0.0, 100.0])
    for acc in [20.0, 40.0, 60.0, 80.0]:
        axs[0].hlines(acc, models_idx[0] - 1, models_idx[-1] + 1, colors=c_lines)
    axs[0].set_xticks(models_idx)
    axs[0].set_xticklabels(get_axis_labels(models2acc), size=fs_label,
                           horizontalalignment='right', verticalalignment='center', rotation=angle_label, rotation_mode='anchor')  # shorten the denomination to create nice labels
    axs[0].set_ylabel(r"Accuracy (%)", fontsize=axis_label_size)

    # plot
    axs[0].bar(models_idx, list(models2acc.values()), width=0.5, color=c_bars, zorder=2)
    axs[0].set_title(label, fontsize=axis_label_size)

    for iC in range(0, nC):

        # prepare data
        label = 'Class_{}'.format(1 + iC)
        models2acc = {m: tab_acc.loc[r, label] for m, r in models2rows.items()}

        # prepare canvas
        axs.append(fig.add_subplot(1, 1 + nC, 1 + 1 + iC))
        axs[1 + iC].set_xlim([models_idx[0] - 1, models_idx[-1] + 1])
        axs[1 + iC].set_ylim([0.0, 100.0])
        for acc in [20.0, 40.0, 60.0, 80.0]:
            axs[1 + iC].hlines(acc, models_idx[0] - 1, models_idx[-1] + 1, colors=c_lines)
        axs[1 + iC].set_xticks(models_idx)
        axs[1 + iC].set_xticklabels(get_axis_labels(models2acc), size=fs_label,
                                    horizontalalignment='right', verticalalignment='center', rotation=angle_label, rotation_mode='anchor')  # shorten the denomination to create nice labels

        # plot
        axs[1 + iC].bar(models_idx, list(models2acc.values()), width=0.5, color=c_bars, zorder=2)
        axs[1 + iC].set_title(label.replace('_', ' '), fontsize=axis_label_size)

    if save_file:
        assert (save_file.endswith('.eps'))
        plt.savefig(save_file)
        plt.savefig(save_file.replace('.eps', '.png'))  # save also in PNG format
