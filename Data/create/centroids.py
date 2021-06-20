# 
# centroids.py
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
from sklearn.cluster import KMeans
from scipy.stats import ortho_group


def get_directions(nD, n):
    # k-means is a quick way to homogeneously spread the directions in the space (provided it is applied on a uniform distribution on the nD-dimensional sphere)
    x = np.random.randn(1000, nD)
    x = x / np.linalg.norm(x, axis=1)[:, None]
    kmodel = KMeans(n_clusters=n)
    kmodel.fit(x)
    v = kmodel.cluster_centers_ / np.linalg.norm(kmodel.cluster_centers_, axis=1)[:, None]  # normalise the directions
    rot_matrix = ortho_group.rvs(nD)
    v = np.dot(v, rot_matrix)
    return v


def get_primary_scattering(nD, nM, nm, rM_cond, rM, epsrM):
    # nM: number of DoF of the major factor of variation
    assert 0.0 < epsrM <= 1.0  # so that the primary centroids will be in the sphere or radius `2 * epsrM`

    cM = get_directions(nD, nM) * rM
    cM = np.tile(cM, (nm, 1))
    cM = np.reshape(cM, (nm, nM, nD))
    cM = np.transpose(cM, (1, 0, 2))

    # get distances from origin
    if rM_cond == 'M':
        rMs = np.array([rM] * nM)
        if epsrM != 0.0:
            rMs *= epsrM * (2. * np.random.rand(nM) - 1.) + 1.
        rMs = np.tile(rMs, (nm, 1))
        rMs = np.transpose(rMs, (1, 0))

    elif rM_cond == 'I':
        if epsrM != 0.0:
            rM *= epsrM * (2. * np.random.rand() - 1.) + 1.
        rMs = np.array([rM] * nM)
        rMs = np.reshape(rMs, (nM, nm))

    primary_centroids = cM * rMs[:, :, None]
    return primary_centroids


def get_secondary_scattering(nD, nM, nm, dm_cond, rm_cond, rm, epsrm):
    # nM: number of DoF of the major factor of variation
    # nm: number of DoF of the minor factor of variation
    assert 0.0 < epsrm <= 1.0  # so that the secondary scatterings are bound in a sphere of radius `2 * epsrm`

    # get directions
    if dm_cond == 'F':
        for iM in range(0, nM):
            try:
                cm = np.vstack([cm, get_directions(nD, nm)])
            except UnboundLocalError:
                cm = get_directions(nD, nm)
        cm = np.reshape(cm, (nm, nM, nD))
        cm = np.transpose(cm, (1, 0, 2))

    elif dm_cond == 'm':  # major factor of variation has no effect on directions
        cm = get_directions(nD, nm)
        cm = np.tile(cm, (nM, 1))
        cm = np.reshape(cm, (nM, nm, nD))

    # get distances from primary centroids
    if rm_cond == 'F':  # secondary scattering depends on both primary and secondary factors of variation
        rms = np.array([rm] * (nM * nm))
        if epsrm != 0.0:
            rms *= epsrm * (2. * np.random.rand(len(rms)) - 1.) + 1.
        rms = np.reshape(rms, (nM, nm))

    elif rm_cond == 'M':  # secondary scattering depends just on primary factor of variation
        rms = np.array([rm] * nM)
        if epsrm != 0.0:
            rms *= epsrm * (2. * np.random.rand(nM) - 1.) + 1.
        rms = np.tile(rms, (nm, 1))
        rms = np.transpose(rms, (1, 0))

    elif rm_cond == 'm':  # secondary scattering depends just on secondary factor of variation
        rms = np.array([rm] * nm)
        if epsrm != 0.0:
            rms *= epsrm * (2. * np.random.rand(nm) - 1.) + 1.
        rms = np.tile(rms, (nM, 1))

    elif rm_cond == 'I':  # secondary scattering is independent of both primary and secondary factor of variation
        if epsrm != 0.0:
            rm *= epsrm * (2. * np.random.rand() - 1.) + 1.
        rms = np.array([rm] * (nm * nM))
        rms = np.reshape(rms, (nM, nm))

    secondary_scattering = cm * rms[:, :, None]
    return secondary_scattering


def create_means(nD, nG, nC, primary_cond, rM_cond, rM, epsrM, dm_cond, rm_cond, rm, epsrm):

    if primary_cond == 'G':
        nM, nm = nG, nC
    elif primary_cond == 'C':
        nM, nm = nC, nG

    primary_scattering = get_primary_scattering(nD, nM, nm, rM_cond, rM, epsrM)
    secondary_scattering = get_secondary_scattering(nD, nM, nm, dm_cond, rm_cond, rm, epsrm)
    centroids = primary_scattering + secondary_scattering

    if primary_cond == 'C':
        centroids = np.transpose(centroids, (1, 0, 2))

    return centroids
