# ============================================================================
# ============================================================================
# Copyright (c) 2026 Nghia T. Vo. All rights reserved.
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
# ============================================================================
# Author: Nghia T. Vo
# E-mail:
# Description: Python module for postprocessing methods.
# Contributors:
# ============================================================================

"""
Post-processing methods for correcting spectral and image data.

This module provides functions for applying previously computed calibration
coefficients to 1D spectra and 2D detector images. The corrections map data
from pixel/index space to physical axes such as energy or scattering vector
(q) using polynomial calibration models.

Core functionalities include:
- Correcting energy for 1D spectra and 2D images
- Correcting q-axis for 1D pattern and 2D images
- Generating q-map from per-row calibration coefficients
"""


import multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed


def correct_energy_1d(list_data, list_coef, energy_step):
    """
    Apply an energy-axis correction to a 1D spectrum.

    The function uses a polynomial calibration model to map measured energy
    coordinates to corrected energy coordinates, followed by interpolation
    of the input data onto a uniform energy grid.

    Parameters
    ----------
    list_data : array_like
        1D input spectrum (intensity values).
    list_coef : array_like
        Polynomial calibration coefficients mapping pixel index to measured
        energy. Coefficients must be ordered with the constant term first.
    energy_step : float
        Energy step size per pixel.

    Returns
    -------
    ndarray
        Energy-corrected 1D spectrum. Values outside the calibrated range
        are set to zero.
    """
    width = len(list_data)
    x_list = np.arange(width)
    list_energy = x_list * energy_step
    list_e_mea = np.sum(np.asarray(
        [factor * x_list ** i for i, factor in enumerate(list_coef)]), axis=0)
    list_correct = np.interp(list_energy, list_e_mea, list_data, left=0,
                             right=0)
    return list_correct


def correct_energy_image(image, mat_coef, energy_step, ncore=None,
                         prefer="processes"):
    """
    Apply an energy-axis correction to a 2D image row-by-row.

    Each row of the image is treated as an independent 1D spectrum and
    corrected using its corresponding polynomial calibration coefficients.

    Parameters
    ----------
    image : array_like
        2D input image with shape (height, width), where each row represents
        a spectrum.
    mat_coef : array_like
        2D array of polynomial calibration coefficients with shape
        (height, n_coef). Each row corresponds to one image row.
    energy_step : float
        Energy step size per pixel defining the target uniform energy grid.
    ncore : int or None, optional
        Number of CPU cores used for parallel processing. Automatically
        selected if None.
    prefer : {"threads", "processes"}, optional
        Parallelization backend.

    Returns
    -------
    ndarray
        Energy-corrected 2D image with the same shape as the input image.
        Values outside the calibrated range are set to zero.
    """
    (height, width) = image.shape
    if len(mat_coef) != height:
        raise ValueError(f"Row number of the calibration coefficients: "
                         f" {len(mat_coef)} is not the same as the height "
                         f"value {height}")
    x_list = np.arange(width)
    list_energy = x_list * energy_step
    max_cores = mp.cpu_count()
    if ncore is None:
        ncore = max(1, max_cores - 1)
    else:
        ncore = max(1, min(ncore, max_cores))

    def __correct_e_core(list_data, list_coef, x_list_, list_energy_):
        list_em = np.sum(np.asarray(
            [factor * x_list_ ** i for i, factor in enumerate(list_coef)]),
            axis=0)
        return np.interp(list_energy_, list_em, list_data, left=0, right=0)

    if ncore == 1:
        mat_correct = np.zeros_like(image)
        for ii in range(height):
            mat_correct[ii] = __correct_e_core(image[ii], mat_coef[ii],
                                               x_list, list_energy)
    else:
        results = Parallel(n_jobs=ncore, prefer=prefer)(delayed(
            __correct_e_core)(image[ii], mat_coef[ii], x_list,
                              list_energy) for ii in range(height))
        mat_correct = np.asarray(results)
    return mat_correct


def get_energy_map(height, width, mat_coef):
    """
    Generate a 2D energy-map from per-row polynomial calibration coefficients.

    The E-map provides the E-value corresponding to each pixel position
    in a detector image, based on row-dependent calibration coefficients.

    Parameters
    ----------
    height : int
        Number of rows in the detector image.
    width : int
        Number of pixels per row.
    mat_coef : array_like
        2D array of polynomial calibration coefficients with shape
        (height, n_coef). Each row corresponds to one detector row.

    Returns
    -------
    ndarray
        2D array of shape (height, width) containing q-values for each
        pixel position.
    """
    if len(mat_coef) != height:
        raise ValueError(f"Row number of the calibration coefficients: "
                         f" {len(mat_coef)} is not the same as the height "
                         f"value {height}")
    e_map = np.zeros((height, width))
    x_list = np.arange(width)
    for ii in range(height):
        list_coef = mat_coef[ii]
        e_map[ii] = np.sum(np.asarray(
            [factor * x_list ** i for i, factor in enumerate(list_coef)]),
            axis=0)
    return e_map


def correct_q_1d(list_data, list_coef, q_step):
    """
    Apply a scattering vector (q) correction to a 1D spectrum.

    The function uses a polynomial calibration model to map pixel indices
    to measured q-values and interpolates the data onto a uniform q grid.

    Parameters
    ----------
    list_data : array_like
        1D input spectrum (intensity values).
    list_coef : array_like
        Polynomial calibration coefficients mapping pixel index to measured
        q-values. Coefficients must be ordered with the constant term first.
    q_step : float
        q step size per pixel defining the target uniform q grid.

    Returns
    -------
    ndarray
        q-corrected 1D spectrum. Values outside the calibrated range are set
        to zero.
    """
    width = len(list_data)
    x_list = np.arange(width)
    q_list = np.arange(width) * q_step
    list_q_mea = np.sum(np.asarray(
        [factor * x_list ** i for i, factor in enumerate(list_coef)]), axis=0)
    list_correct = np.interp(q_list, list_q_mea, list_data, left=0, right=0)
    return list_correct


def get_q_map(height, width, mat_coef):
    """
    Generate a 2D q-map from per-row polynomial calibration coefficients.

    The q-map provides the q-value corresponding to each pixel position
    in a detector image, based on row-dependent calibration coefficients.

    Parameters
    ----------
    height : int
        Number of rows in the detector image.
    width : int
        Number of pixels per row.
    mat_coef : array_like
        2D array of polynomial calibration coefficients with shape
        (height, n_coef). Each row corresponds to one detector row.

    Returns
    -------
    ndarray
        2D array of shape (height, width) containing q-values for each
        pixel position.
    """
    if len(mat_coef) != height:
        raise ValueError(f"Row number of the calibration coefficients: "
                         f" {len(mat_coef)} is not the same as the height "
                         f"value {height}")
    q_map = np.zeros((height, width))
    x_list = np.arange(width)
    for ii in range(height):
        list_coef = mat_coef[ii]
        q_map[ii] = np.sum(np.asarray(
            [factor * x_list ** i for i, factor in enumerate(list_coef)]),
            axis=0)
    return q_map


def correct_q_image(image, mat_coef, q_step, ncore=1, prefer="processes"):
    """
    Apply a scattering vector (q) correction to a 2D image row-by-row.

    Each row of the image is treated as an independent 1D spectrum and
    corrected using its corresponding polynomial calibration coefficients.
    The correction can be parallelized over rows.

    Parameters
    ----------
    image : array_like
        2D input image with shape (height, width), where each row represents
        a spectrum.
    mat_coef : array_like
        2D array of polynomial calibration coefficients with shape
        (height, n_coef). Each row corresponds to one image row.
    q_step : float
        q step size per pixel defining the target uniform q grid.
    ncore : int or None, optional
        Number of CPU cores used for parallel processing. If None, use all
        available cores minus one. If set to 1, computation is serial.
    prefer : {"threads", "processes"}, optional
        Parallelization backend.

    Returns
    -------
    ndarray
        q-corrected 2D image with the same shape as the input image.
        Values outside the calibrated range are set to zero.
    """
    (height, width) = image.shape
    if len(mat_coef) != height:
        raise ValueError(f"Row number of the calibration coefficients: "
                         f" {len(mat_coef)} is not the same as the height "
                         f"value {height}")
    x_list = np.arange(width)
    q_list = np.arange(width) * q_step
    max_cores = mp.cpu_count()
    if ncore is None:
        ncore = max(1, max_cores - 1)
    else:
        ncore = max(1, min(ncore, max_cores))

    def __correct_q_core(list_data, list_coef, x_list_, q_list_):
        list_qm = np.sum(np.asarray(
            [factor * x_list_ ** i for i, factor in enumerate(list_coef)]),
            axis=0)
        return np.interp(q_list_, list_qm, list_data, left=0, right=0)

    if ncore == 1:
        mat_correct = np.zeros_like(image)
        for ii in range(height):
            mat_correct[ii] = __correct_q_core(image[ii], mat_coef[ii],
                                               x_list, q_list)
    else:
        results = Parallel(n_jobs=ncore, prefer=prefer)(delayed(
            __correct_q_core)(image[ii], mat_coef[ii], x_list,
                              q_list) for ii in range(height))
        mat_correct = np.asarray(results)
    return mat_correct
