# 
# show.py
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
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_colours(nC):

    cmap = matplotlib.cm.get_cmap('viridis')
    colours = cmap(np.linspace(0.0, 1.0, nC))

    return colours


def get_axis_lims(df, apron_ratio):

    # this ensures that all the data points in the training set are shown, and that an eye-pleasing distance is present between them and the image's border
    lims = np.vstack([np.min(df.values, axis=0), np.max(df.values, axis=0)]).T

    apron = apron_ratio * (lims[:, 1] - lims[:, 0])
    lims[:, 0] -= 0.5 * apron
    lims[:, 1] += 0.5 * apron

    lims = np.sign(lims) * np.ceil(np.abs(lims) / 0.5) * 0.5  # I don't want weird values on the axis

    return lims


def show_data_set(dt_train, save_file=None, axis_label_size=20):

    x_names = [n for n in dt_train.columns if n.startswith('x')]
    nD = len(x_names)

    if nD in [2, 3]:

        # count classes
        classes = list(np.unique(dt_train['y'].values))
        classes.sort()
        nC = len(classes)

        # get colours for each class
        colours = get_colours(nC)

        # get extrema for axis
        apron_ratio = 0.1
        lims = get_axis_lims(dt_train[x_names], apron_ratio)

        # select points to be shown
        ps = 2  # point size
        x = dt_train[x_names]
        c = [colours[classes.index(y)] for y in dt_train['y']]

        if nD == 2:

            fig = plt.figure()
            fig.set_size_inches(9, 9)

            ax = fig.add_subplot(111)
            ax.set_xlim(lims[0, :])
            ax.set_ylim(lims[1, :])

            ax.scatter(x['x1'], x['x2'], s=ps, c=c)
            ax.set_xlabel(r"$x_{1}$", fontsize=axis_label_size)
            ax.set_ylabel(r"$x_{2}$", fontsize=axis_label_size)

        elif nD == 3:

            fig = plt.figure()
            fig.set_size_inches(9, 9)

            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(lims[0, :])
            ax.set_ylim(lims[1, :])
            ax.set_zlim(lims[2, :])

            ax.scatter(x['x1'], x['x2'], x['x3'], c=c)
            ax.set_xlabel(r"$x_{1}$", fontsize=axis_label_size)
            ax.set_ylabel(r"$x_{2}$", fontsize=axis_label_size)
            ax.set_zlabel(r"$x_{3}$", fontsize=axis_label_size)

        if save_file:
            assert (save_file.endswith('.eps'))
            plt.savefig(save_file)
            plt.savefig(save_file.replace('.eps', '.png'))  # save also in PNG format


def show_bag_on_data(dt_train, dt_test, save_file=None, seed=42, axis_label_size=12):

    x_names = [n for n in dt_train.columns if n.startswith('x')]
    nD = len(x_names)

    if nD in [2, 3]:

        # count classes
        classes = list(np.unique(dt_train['y'].values))
        classes.sort()
        nC = len(classes)

        # get colours for each class
        colours = get_colours(nC)

        # get extrema for axis
        apron_ratio = 0.1
        lims = get_axis_lims(dt_train[x_names], apron_ratio)

        # select points to be shown
        ps_train = 1  # point size
        xy_train = dt_train[x_names + ['y']]
        c_train = [colours[classes.index(y)] for y in xy_train['y']]
        c_all = np.array([0.0, 0.0, 0.0, 1.0])  # all data set will be shown in black

        ps_test = 4  # point size
        b_test = [b[1] for b in dt_test.groupby('k')]
        np.random.seed(seed)  # for replicability
        ib = np.random.randint(len(b_test))
        x_test = b_test[ib][x_names]
        c_test = np.array([1.0, 0.0, 0.0, 1.0])  # test bags is overlayed in red

        # draw!
        alpha = 0.3
        c_alpha = np.array([1.0, 1.0, 1.0, alpha])  # to enhance overlays

        def overlay_test_bag(ax, bag, ps, c):

            x_names = [n for n in bag.columns if n.startswith('x')]
            nD = len(x_names)

            if nD == 2:
                ax.scatter(bag['x1'], bag['x2'], s=ps, c=c)

            elif nD == 3:
                ax.scatter(bag['x1'], bag['x2'], bag['x3'], s=ps, c=c)

        if nD == 2:
            fig = plt.figure()
            fig.set_size_inches(3 * (1 + nC), 3)

            axs = []  # array of handles to `matplotlib` axis objects
            for i in range(0, 1 + nC):
                axs.append(fig.add_subplot(1, 1 + nC, 1 + i))
                axs[i].set_xlim(lims[0, :])
                axs[i].set_ylim(lims[1, :])

            axs[0].scatter(xy_train['x1'], xy_train['x2'], s=ps_train, c=c_train)
            axs[0].set_xlabel(r"$x_{1}$", fontsize=axis_label_size)
            axs[0].set_ylabel(r"$x_{2}$", fontsize=axis_label_size)
            for iC in range(0, nC):
                x_temp = xy_train.loc[xy_train['y'] == classes[iC], x_names]
                axs[1 + iC].scatter(x_temp['x1'], x_temp['x2'], s=ps_train, c=colours[iC] * c_alpha)
                overlay_test_bag(axs[1 + iC], x_test, ps_test, c_test)
                axs[1 + iC].set_title("Class {}".format(int(1 + iC)), fontsize=axis_label_size)

        elif nD == 3:
            fig = plt.figure()
            fig.set_size_inches(3 * (1 + nC), 3)

            axs = []  # array of handles to `matplotlib` axis objects
            for i in range(0, 1 + nC):
                axs.append(fig.add_subplot(1, 1 + nC, 1 + i, projection='3d'))
                axs[i].set_xlim(lims[0, :])
                axs[i].set_ylim(lims[1, :])
                axs[i].set_zlim(lims[2, :])

            axs[0].scatter(xy_train['x1'], xy_train['x2'], xy_train['x3'], s=ps_train, c=c_train)
            axs[0].set_xlabel(r"$x_{1}$", fontsize=axis_label_size)
            axs[0].set_ylabel(r"$x_{2}$", fontsize=axis_label_size)
            axs[0].set_zlabel(r"$x_{3}$", fontsize=axis_label_size)
            for iC in range(0, nC):
                x_temp = xy_train.loc[xy_train['y'] == classes[iC], x_names]
                axs[1 + iC].scatter(x_temp['x1'], x_temp['x2'], x_temp['x3'], s=ps_train, c=colours[iC] * c_alpha)
                overlay_test_bag(axs[1 + iC], x_test, ps_test, c_test)
                axs[1 + iC].set_title("Class {}".format(int(1 + iC)), fontsize=axis_label_size)

        # make space for x-axis label
        # (https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot)
        plt.gcf().subplots_adjust(bottom=0.2)

        if save_file:
            assert (save_file.endswith('.eps'))
            plt.savefig(save_file)
            plt.savefig(save_file.replace('.eps', '.png'))  # save also in PNG format
