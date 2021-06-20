# 
# bag.py
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
import pandas as pd


def get_bag(classes, pC, maxN, pN, hg_names, pG, nS, maxO, stats_dict):

    def get_bag_composition(classes, pC, maxN, pN, hg_names, pG):

        def sample_from_discrete(p):

            x = np.random.rand()
            sum = 0.0
            for i in range(0, len(p)):
                sum += p[i]
                if x < sum:
                    return i

        ic = sample_from_discrete(pC)  # get class

        n_sub_bags = 1 + np.random.binomial(maxN - 1, pN[ic])  # at least one sub-bag!
        pG = pG[classes[ic]]
        for i in range(0, n_sub_bags):
            ig = sample_from_discrete(pG)
            try:
                hg_sel.append(hg_names[ig])
            except UnboundLocalError:
                hg_sel = [hg_names[ig]]

        return classes[ic], hg_sel

    c_n, hg_sel = get_bag_composition(classes, pC, maxN, pN, hg_names, pG)
    for hg_n, c_n in zip(hg_sel, [c_n] * len(hg_sel)):

        # get numerical observations (in Euclidean space)
        stats = stats_dict[hg_n][c_n]
        try:
            x = np.vstack([x, np.random.multivariate_normal(stats.mean, stats.covariance, nS)])
        except UnboundLocalError:
            x = np.random.multivariate_normal(stats.mean, stats.covariance, nS)

        # get categorical observations
        qs = np.array([int(q) for q in hg_n.split('_')])
        qs = np.tile(qs, (nS, 1))
        try:
            q = np.vstack([q, qs])
        except UnboundLocalError:
            q = qs

    # get labels
    y = np.array([classes.index(c_n) + 1] * x.shape[0])  # use numeric class label

    qfr = pd.DataFrame(data=q, columns=['q' + str(i + 1) for i in range(0, q.shape[1])])
    xfr = pd.DataFrame(data=x, columns=['x' + str(i + 1) for i in range(0, x.shape[1])])
    yfr = pd.DataFrame(data=y, columns=['y'])

    df = pd.concat([qfr, xfr, yfr], axis=1)
    assert len(df) <= maxO

    return df
