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

from collections import namedtuple
import pandas as pd
import numpy as np
import sys
import copy
from more_itertools import locate

from .fingerprint import Fingerprint


Stage = namedtuple('Stage', ['model', 'pr_y'])


class MultiStageFingerprint(object):

    def __init__(self, n_pcs, theta, verbose=False):
        self.verbose = verbose
        self.n_pcs = n_pcs
        self.theta = theta
        self.stages_ = list()
        self.labels_ = list()

    @staticmethod
    def _test_point_cloud(pc):

        is_ok = False

        if len(pc) > 1:
            if np.linalg.cond(np.cov(pc.T)) < (1 / sys.float_info.epsilon):
                is_ok = True

        return is_ok

    @staticmethod
    def _find_solvable_point_clouds(data):

        solvable = set()

        x_names = Fingerprint.get_x_vars(data.columns)
        if 'q' not in data.columns:
            solvable.update({'q'} if MultiStageFingerprint._test_point_cloud(data[x_names]) else set())
        else:
            solvable.update(set([q for q, df in data.groupby('q') if MultiStageFingerprint._test_point_cloud(df[x_names])]))

        return solvable

    @staticmethod
    def _resolubility_check_training(q_factors, b_train, y_train):

        classes = np.sort(np.unique(y_train))
        sg_training = {y: set() for y in classes}

        for y in classes:
            y_data = pd.concat([b for b, yb in zip(b_train, y_train) if yb == y], axis=0)
            y_data = Fingerprint.compute_hg_codes(y_data, q_factors)
            sg_training[y].update(MultiStageFingerprint._find_solvable_point_clouds(y_data))

        return sg_training

    @staticmethod
    def _resolubility_check_test(q_factors, sg_training, test_bag):

        test_bag = Fingerprint.compute_hg_codes(test_bag, q_factors)
        sg_test = MultiStageFingerprint._find_solvable_point_clouds(test_bag)

        test_bag_is_solvable = False
        for y, sg in sg_training.items():
            if sg_test & sg:
                test_bag_is_solvable = True

        return test_bag_is_solvable

    @staticmethod
    def _count_solvable_test_bags(q_factors, b_train, y_train, b_test):

        sg_training = MultiStageFingerprint._resolubility_check_training(q_factors, b_train, y_train)
        n_solvable = 0
        for b in b_test:
            n_solvable += 1 if MultiStageFingerprint._resolubility_check_test(q_factors, sg_training, b) else 0

        return n_solvable

    @staticmethod
    def _sieve(q_factors, b_train, y_train, b_test, verbose=False):
        """Select the subset of factors which allows to classify most test bags."""
        for q in q_factors:

            q_factors_temp = q_factors.copy()
            q_factors_temp.remove(q)
            n_temp = MultiStageFingerprint._count_solvable_test_bags(q_factors_temp, b_train, y_train, b_test)

            try:
                if n_temp > n_best:
                    n_best = n_temp
                    q_factors_best = q_factors_temp
            except UnboundLocalError:
                n_best = n_temp
                q_factors_best = q_factors_temp

        if verbose:
            print("Factors: {} - {}/{} bags can be classified.".format(q_factors_best, n_best, len(b_test)))

        return q_factors_best

    def _apply_stage(self, q_factors, b_train, y_train, b_test, k_test):

        model = Fingerprint(q_factors, n_pcs=self.n_pcs, theta=self.theta, verbose=self.verbose)
        model.fit(b_train, y_train)
        pr_y = model.predict(b_test, k_test=k_test)

        return model, pr_y

    def apply(self, b_train, y_train, b_test, k_test=None):

        self.stages_ = list()
        self.labels_ = list()

        if k_test is None:
            k_test = list(range(0, len(b_test)))

        pr_y = [-1 for _ in range(0, len(b_test))]

        # select set of bags that still need to be solved
        unsolved = list(locate(pr_y, lambda y_hat: y_hat == -1))
        b_test_filtered = [b_test[i] for i in unsolved]
        k_test_filtered = [k_test[i] for i in unsolved]

        # start using all factors...
        q_factors = Fingerprint.get_q_vars(b_train[0].columns)
        model, pr_y_stage = self._apply_stage(q_factors, b_train, y_train, b_test_filtered, k_test_filtered)

        # update solved bags
        pr_y = copy.deepcopy(pr_y)
        for i, y in enumerate(pr_y_stage):
            pr_y[unsolved[i]] = y
        self.stages_.append(Stage(model=model, pr_y=pr_y))

        while (len(q_factors) > 0) and (len(list(locate(pr_y, lambda y_hat: y_hat == -1))) > 0):

            # select set of bags that still need to be solved
            unsolved = list(locate(pr_y, lambda y_hat: y_hat == -1))
            b_test_filtered = [b_test[i] for i in unsolved]
            k_test_filtered = [k_test[i] for i in unsolved]

            # ...then merge factors to classify unclassified bags
            q_factors = MultiStageFingerprint._sieve(q_factors, b_train, y_train, b_test_filtered, verbose=self.verbose)
            model, pr_y_stage = self._apply_stage(q_factors, b_train, y_train, b_test_filtered, k_test_filtered)

            # update solved bags
            pr_y = copy.deepcopy(pr_y)
            for i, y in enumerate(pr_y_stage):
                pr_y[unsolved[i]] = y
            self.stages_.append(Stage(model=model, pr_y=pr_y))

        self.labels_ = pr_y
