import numpy as np
from scipy.stats import ortho_group


def get_maxVs(nG, nC, conditioning, maxV, epsV):

    assert 0.0 < epsV <= 1.0  # so that the actual `maxV` for each Gaussian will be in the range `(0, 2 * maxV]`

    if conditioning == 'F':  # each Gaussian is fully dependent on the group/class variable combination; i.e., it will likely have a different volume from other Gaussians
        maxVs = np.array([maxV] * (nG * nC))
        if epsV != 0.0:
            maxVs *= epsV * (2. * np.random.rand(len(maxVs)) - 1.) + 1.
        maxVs = np.reshape(maxVs, (nG, nC))

    elif conditioning == 'G':  # the Gaussians in each cluster of corresponding homogeneous groups will have the same volume
        maxVs= np.array([maxV] * nG)
        if epsV != 0.0:
            maxVs *= epsV * (2. * np.random.rand(nG) - 1.) + 1.
        maxVs = np.tile(maxVs, (nC, 1))
        maxVs = np.transpose(maxVs, (1, 0))

    elif conditioning == 'C':  # the Gaussians composing a specific segment mixture will have the same volume
        maxVs = np.array([maxV] * nC)
        if epsV != 0.0:
            maxVs *= epsV * (2. * np.random.rand(nC) - 1.) + 1.
        maxVs = np.tile(maxVs, (nG, 1))

    elif conditioning == 'I':  # all Gaussians will have the same volume, independently of homogeneous group and segment
        if epsV != 0.0:
            maxV *= epsV * (2. * np.random.rand() - 1.) + 1.
        maxVs = np.array([maxV] * (nG * nC))
        maxVs = np.reshape(maxVs, (nG, nC))


    return maxVs


def get_spectra(nD, nG, nC, conditioning, spectrumK, epsK, spherical=False):

    assert spectrumK > 0
    assert 0.0 < epsK <= 1.0  # so that the actual spectra will be in the range `(e^{-2 * k}, e^{0} = 1]`

    if spherical:  # note that either all te Gaussians are forced to be spherical (even if they belong to different conditioning groups), ore none is forced to
        spectrumK = 0

    _base_eig_exps = - np.linspace(0.0, 1.0, nD)

    if conditioning == 'F':
        spectraK = np.array([spectrumK] * (nG * nC))
        if epsK != 0.0:
            spectraK *= epsK * (2. * np.random.rand(len(spectraK)) - 1.) + 1.
        spectraK = np.reshape(spectraK, (nG, nC))

    elif conditioning == 'G':
        spectraK = np.array([spectrumK] * nG)
        if epsK != 0.0:
            spectraK *= epsK * (2. * np.random.rand(nG) - 1.) + 1.
        spectraK = np.tile(spectraK, (nC, 1))
        spectraK = np.transpose(spectraK, (1, 0))

    elif conditioning == 'C':
        spectraK = np.array([spectrumK] * nC)
        if epsK != 0.0:
            spectraK *= epsK * (2. * np.random.rand(nC) - 1.) + 1.
        spectraK = np.tile(spectraK, (nG, 1))

    elif conditioning == 'I':
        if epsK != 0.0:
            spectrumK *= epsK * (2. * np.random.rand() - 1.) + 1.
        spectraK = np.array([spectrumK] * (nG * nC))
        spectraK = np.reshape(spectraK, (nG, nC))

    spectra = np.exp(spectraK[:, :, None] * _base_eig_exps[None, None, :])

    return spectra


def get_orientations(nD, nG, nC, conditioning, axis_aligned=False):

    def get_rotation_matrix(nD, axis_aligned=False):
        if axis_aligned:
            rot_matrix = np.eye(nD)
        else:
            rot_matrix = ortho_group.rvs(nD)
        return rot_matrix

    rot_matrices = np.zeros((nG, nC, nD, nD))

    if conditioning == 'F':
        for ig in range(0, nG):
            for ic in range(0, nC):
                rot_matrix = get_rotation_matrix(nD, axis_aligned=axis_aligned)
                np.random.shuffle(rot_matrix)  # shuffle rows
                rot_matrices[ig, ic] = rot_matrix

    elif conditioning == 'G':
        for ig in range(0, nG):
            rot_matrix = get_rotation_matrix(nD, axis_aligned=axis_aligned)
            np.random.shuffle(rot_matrix)
            rot_matrices[ig, :, :, :] = rot_matrix

    elif conditioning == 'C':
        for ic in range(0, nC):
            rot_matrix = get_rotation_matrix(nD, axis_aligned=axis_aligned)
            np.random.shuffle(rot_matrix)
            rot_matrices[..., ic, :, :] = rot_matrix

    elif conditioning == 'I':
        rot_matrix = get_rotation_matrix(nD, axis_aligned=axis_aligned)
        np.random.shuffle(rot_matrix)
        rot_matrices[..., :, :] = rot_matrix

    return rot_matrices


def create_covariance_matrices(nD, nG, nC,
                            maxV_cond, maxV, epsV,
                            spectrumK_cond, spectrumK, epsK, spherical,
                            orientation_cond, ax_align):

    cov_matrices = np.zeros((nG, nC, nD, nD))

    maxVs = get_maxVs(nG, nC, maxV_cond, maxV, epsV)
    spectra = get_spectra(nD, nG, nC, spectrumK_cond, spectrumK, epsK, spherical=spherical)
    rot_matrices = get_orientations(nD, nG, nC, orientation_cond, axis_aligned=ax_align)

    for ig in range(0, nG):
        for ic in range(0, nC):
            cov_matrix = np.dot(np.dot(rot_matrices[ig, ic], np.diag(maxVs[ig, ic] * spectra[ig, ic])), rot_matrices[ig, ic].T)
            cov_matrices[ig, ic] = cov_matrix

    return cov_matrices
