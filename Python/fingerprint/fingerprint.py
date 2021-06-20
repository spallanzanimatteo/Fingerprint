# 
# fingerprint.py
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
import numpy as np
import pandas as pd
import sys


PointCloudStats = namedtuple('PointCloudStats', ['m', 'invS', 'm_mh', 's_mh', 'PCs'])
GroupComparison = namedtuple('GroupComparison', ['crit1', 'crit2'])


class Fingerprint(object):
    """The `fingerprint` classification algorithm.

    Attributes:
        fingerprint_ (dict): for each class, a dictionary describing its
            homogeneous groups; if no homogeneous group was found, or if the
            data points were insufficient to compute their statistics, the
            entry corresponding to the class is an empty dictionary.
        comparisons_ (list): for each bag in the inference (i.e., test) set, a
            `dict` of `dicts`, where the upper layer of keys indexes the class
            and the lower layer of keys indexes the homogenous groups; each
            leaf item is a `GroupComparison` objects reporting the results of
            the comparison of the homogeneous group 'q' between the bag's and
            the class 'y''s fingerprints.

    """
    def __init__(self, q_factors, n_pcs=1, theta=0.9, verbose=False):
        self.verbose = verbose
        self.classes_ = None
        self.q_factors = q_factors
        self.n_pcs = n_pcs
        self.theta = theta
        self.fingerprint_ = dict()
        self.comparisons_ = list()
        self.labels_ = list()

    @staticmethod
    def get_q_vars(colnames):
        return [n for n in colnames if n.startswith('q')]

    @staticmethod
    def get_x_vars(colnames):
        return [n for n in colnames if n.startswith('x')]

    @staticmethod
    def compute_hg_codes(df, q_factors):
        """Build codes of homogeneous groups."""
        q_names = Fingerprint.get_q_vars(df.columns)

        if len(q_factors) == 0:
            df = df.drop(labels=q_names, axis=1).copy()

        else:
            for i in range(0, len(q_factors)):
                try:
                    q = q + df[q_factors[i]]  # join strings
                except UnboundLocalError:
                    q = df[q_factors[i]]
            df = df.drop(labels=q_names, axis=1).copy()
            df.insert(1, 'q', q)

        return df

    def _compute_point_cloud_stats(self, pc):

        assert len(pc) > 0
        stats = None

        if len(pc) > 1:
            # there is more than one data point in the point cloud
            S = np.cov(pc.T)
            if np.linalg.cond(S) < (1 / sys.float_info.epsilon):

                # basic multivariate statistics
                invS = np.linalg.inv(S)
                m = np.mean(pc, axis=0)

                # Mahalanobis distances
                mahal = np.sqrt(np.diag(np.dot(np.dot((pc - m), invS), (pc - m).T)))
                m_mh = np.mean(mahal)
                s_mh = np.std(mahal)

                # PCA
                S_eigvals, S_eigvecs = np.linalg.eig(S)
                idx = np.argsort(S_eigvals)[::-1][0:self.n_pcs]
                PCs = S_eigvecs[:, idx]

                stats = PointCloudStats(m=m, invS=invS, m_mh=m_mh, s_mh=s_mh, PCs=PCs)

        return stats

    def fit(self, b_train, y_train):
        """Compute statistics of homogeneous groups for each class."""
        assert len(b_train) == len(y_train)

        self.classes_ = np.sort(np.unique(y_train))
        assert len(self.classes_) > 1  # the problem is not a classification one!

        for y in self.classes_:

            # merge bags with identical label
            y_data = pd.concat([b for b, yb in zip(b_train, y_train) if yb == y], axis=0)
            y_data = Fingerprint.compute_hg_codes(y_data, self.q_factors)
            assert len(y_data) > 0

            # split class data into homogeneous groups
            x_names = Fingerprint.get_x_vars(y_data.columns)
            if 'q' in y_data.columns:
                y_q_levels = np.unique(y_data['q'])
                y_fingerprint = {q: pc for q, pc in zip(y_q_levels, [y_data.loc[(y_data['q'] == q), x_names].to_numpy() for q in y_q_levels])}
            else:
                y_fingerprint = {q: pc for q, pc in zip(['q'], [y_data[x_names].to_numpy()])}

            # compute statistics (when possible)
            y_fingerprint = {q: self._compute_point_cloud_stats(pc) for q, pc in y_fingerprint.items()}
            y_fingerprint = {q: y_q_stats for q, y_q_stats in y_fingerprint.items() if y_q_stats is not None}

            self.fingerprint_[y] = y_fingerprint

    def _compare_point_clouds(self, y_q_stats, pc):
        assert len(pc) > 0
        group_comparison = None

        if len(pc) > 1:
            S = np.cov(pc.T)
            if np.linalg.cond(S) < 1 / sys.float_info.epsilon:

                y_q_invS = y_q_stats.invS

                # first criterion
                y_q_m = y_q_stats.m
                y_q_m_mh = y_q_stats.m_mh
                y_q_s_mh = y_q_stats.s_mh
                mahal = np.sqrt(np.diag(np.dot(np.dot(pc - y_q_m, y_q_invS), (pc - y_q_m).T)))
                crit1 = np.mean(mahal - y_q_m_mh) / y_q_s_mh
                # crit1 = np.abs(np.mean(mahal - y_q_m_mh) / y_q_s_mh)  # think to what could happen with two Gaussian clouds having the same mean and shape, but one is much more "condensed" than the other
                # crit1 = np.std(mahal - y_q_m_mh) / y_q_s_mh  # compare the standard deviations of the "chi-square"-like distributions

                # second criterion
                y_q_PCs = y_q_stats.PCs
                S_eigvals, S_eigvecs = np.linalg.eig(S)
                idx = np.argsort(S_eigvals)[::-1][0:self.n_pcs]
                PCs = S_eigvecs[:, idx]
                crit2 = np.abs(np.diag(np.dot(y_q_PCs.T, PCs)))

                group_comparison = GroupComparison(crit1=crit1, crit2=crit2)

        return group_comparison

    def _compare_fingerprints(self, k_data):
        x_names = Fingerprint.get_x_vars(k_data.columns)
        comparisons = dict()

        for y in self.classes_:

            comparisons[y] = dict()
            y_fingerprint = self.fingerprint_[y]
            if y_fingerprint != {}:

                # split test bag into homogeneous groups
                if 'q' in k_data.columns:
                    b_q_levels = np.unique(k_data['q'])
                    b_q_levels_sh = set(y_fingerprint.keys()).intersection(set(b_q_levels))
                    b_groups = {q: pc for q, pc in zip(b_q_levels_sh, [k_data.loc[(k_data['q'] == q), x_names].to_numpy() for q in b_q_levels_sh])}
                else:
                    b_groups = {q: pc for q, pc in zip(['q'], [k_data[x_names].to_numpy()])}

                # compute fingerprint criteria for class `y`
                for q, pc in b_groups.items():
                    b_y_q_comp = self._compare_point_clouds(y_fingerprint[q], pc)
                    if b_y_q_comp is not None:
                        comparisons[y][q] = b_y_q_comp

        return comparisons

    def _classify(self, comparisons):
        """Assign a bag to a class.

        Args:
            k (int or str): identifier (ID) of the bag.
            comparisons (dict): for each of the classes, the statistics of the
                comparisons between a bag's fingerprint and the class's fingerprint
                (as returned by method `_compare_fingerprints`)

        Return:
            y_hat (int): the index of the predicted class.

        """
        Criteria = namedtuple('Criteria', ['n', 'crit1', 'crit2'])

        def compute_criteria(comps):
            if len(comps) > 0:
                return Criteria(n=len(comps), crit1=np.mean([gc.crit1 for gc in comps.values()]), crit2=np.mean([gc.crit2 for gc in comps.values()]))
            else:
                return Criteria(n=0, crit1=np.nan, crit2=np.nan)

        criteria = {y: compute_criteria(comps) for y, comps in comparisons.items()}

        # apply criterion 0
        filter0 = {y: crit for y, crit in criteria.items() if crit.n > 0}
        candidates0 = set(filter0.keys())
        assert candidates0.issubset(set(criteria.keys()))

        if len(candidates0) == 0:
            y_hat = -1

        elif len(candidates0) == 1:
            y_hat = candidates0.pop()

        else:
            # apply criterion 1
            ya = min(filter0.items(), key=lambda i: i[1].crit1)[0]
            yb = min({y: crit for y, crit in filter0.items() if (y != ya)}.items(), key=lambda i: i[1].crit1)[0]
            filter1 = {y: crit for y, crit in criteria.items() if (y in [ya, yb])}
            candidates1 = set(filter1.keys())
            assert candidates1.issubset(candidates0)

            if filter1[ya].crit1 < filter1[yb].crit1 * self.theta:
                y_hat = ya

            else:
                # apply criterion 2
                y_hat = max(filter1.items(), key=lambda i: i[1].crit2)[0]

        return y_hat

    def predict(self, b_test, k_test=None):
        """
        Arguments:
            `test_df`: a `list` of `pandas.DataFrame`s.
            `k_test`: optional; a `list` of identifiers (IDs) of the bags to
                be tested; if the `Fingerprint` object is `verbose`, it will
                indicate which bags are having problems.

        Returns:
            `labels`: a `list` of `int`s.
        """

        if k_test is None:
            k_test = list(range(0, len(b_test)))

        self.comparisons_ = list()
        self.labels_ = list()
        for k, df in zip(k_test, b_test):

            df = Fingerprint.compute_hg_codes(df, self.q_factors)

            comparisons = self._compare_fingerprints(df)
            self.comparisons_.append((k, comparisons))

            y_hat = self._classify(comparisons)
            self.labels_.append(y_hat)

            if self.verbose and (y_hat == -1):
                print('Bag {} could not be classified.'.format(k))

        return self.labels_
