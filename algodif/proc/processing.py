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
# Description: Python module for processing methods.
# Contributors:
# ============================================================================

"""
Calibration and peak-matching methods for 1D and 2D spectral data.

This module provides a collection of functions for calibrating physical
axes, such as energy and scattering vector (q), from 1D spectra and 2D
detector images. The calibration is performed by detecting spectral peaks,
matching them to known reference values, and fitting polynomial mappings
from pixel/index space to physical units.

Core features include:
- Robust peak detection with optional smoothing and subpixel refinement
- Peak matching via polynomial optimization between generated gaussian
  spectrum.
- Energy-axis calibration for spectra and images
- q-axis calibration for generic data and GeRM detector geometries
"""

import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np
import scipy.ndimage as ndi
from scipy import stats
from scipy.optimize import minimize
import algodif.prep.preprocessing as prep
import algodif.util.utility as util


def __cost_function(poly_facts, y1, x2, y2, metric="mean"):
    """
    Compute a cost value between a reference signal and a generated signal.

    The scaling and shifting are defined by a polynomial applied to `x2`,
    followed by interpolation of `y2` onto the transformed coordinates.

    Parameters
    ----------
    poly_facts : array_like
        Polynomial coefficients defining the scale/shift transformation.
        Length defines the polynomial degree.
    y1 : array_like
        Reference signal.
    x2 : array_like
        Coordinate array associated with `y2`.
    y2 : array_like
        Signal to be transformed and compared to `y1`.
    metric : {"mean", "pearson"}, optional
        Cost metric to use. If "mean", computes the mean positive difference.
        Otherwise, uses `1 - Pearson correlation coefficient`.

    Returns
    -------
    float
        Cost value to be minimized.
    """
    degree = len(poly_facts) - 1
    x2_ss = np.sum(np.asarray(
        [factor * x2 ** (degree - i) for i, factor in enumerate(poly_facts)]),
        axis=0)
    y2_ss = np.interp(x2_ss, x2, y2)
    if metric != "mean":
        metric = 1 - stats.pearsonr(y2_ss, y1)[0]
    else:
        metric = np.mean(np.clip((y1 - y2_ss), 0, None))
    return metric


def _find_scale_shift(y1, y2, initial_guess=[1.0, 0.0],
                      bounds=[(0.85, 1.15), (-300, 300)], metric="mean"):
    """
    Estimate scale and shift parameters aligning `y2` to `y1`.
    The parameters are optimized by minimizing a cost function using the
    Nelder–Mead algorithm.

    Parameters
    ----------
    y1 : array_like
        Reference signal.
    y2 : array_like
        Signal to be scaled and shifted.
    initial_guess : array_like, optional
        Initial guess for the scale and shift parameters.
    bounds : sequence of tuple or None, optional
        Bounds for the optimization parameters. If None, no bounds are applied.
    metric : {"mean", "pearson"}, optional
        Cost metric used during optimization.

    Returns
    -------
    ndarray
        Optimized scale and shift parameters.
    """
    x2 = np.arange(len(y2))
    ss0 = np.array(initial_guess)
    if bounds is not None:
        results = minimize(__cost_function, ss0,
                           args=tuple([y1, x2, y2, metric]),
                           method="Nelder-Mead", bounds=bounds,
                           options={"xatol": 1e-8})
    else:
        results = minimize(__cost_function, ss0,
                           args=tuple([y1, x2, y2, metric]),
                           method="Nelder-Mead", options={"xatol": 1e-8})
    return results.x


def _correct_scale_shift(y, poly_facts, left=0, right=0):
    """
    Apply a polynomial-based scale and shift correction to a signal.
    The correction is performed by transforming the coordinate axis and
    interpolating the signal onto the new coordinates.

    Parameters
    ----------
    y : array_like
        Input signal to be corrected.
    poly_facts : array_like
        Polynomial coefficients defining the scale/shift transformation.
    left : float, optional
        Value used for extrapolation to the left.
    right : float, optional
        Value used for extrapolation to the right.

    Returns
    -------
    ndarray
        Corrected signal.
    """
    degree = len(poly_facts) - 1
    x = np.arange(len(y))
    x1 = np.sum(np.asarray(
        [factor * x ** (degree - i) for i, factor in enumerate(poly_facts)]),
        axis=0)
    y_corr = np.interp(x1, x, y, left=left, right=right)
    return y_corr


def _align_peaks_coarse(reference_peaks, measured_peaks, shift_start,
                        shift_stop, shift_step=1, ratio_start=-0.15,
                        ratio_stop=0.15, ratio_step=0.005, metric="pearson"):
    """
    Perform a coarse search to align measured peaks to reference peaks.
    The alignment is achieved by scanning over scale and shift values and
    selecting the combination that minimizes the chosen cost metric.

    Parameters
    ----------
    reference_peaks : array_like
        Reference peak positions.
    measured_peaks : array_like
        Measured peak positions to be aligned.
    shift_start : float
        Start value of the shift range.
    shift_stop : float
        Stop value of the shift range.
    shift_step : float, optional
        Step size for the shift scan.
    ratio_start : float, optional
        Start value of the relative scale range (around 1.0).
    ratio_stop : float, optional
        Stop value of the relative scale range (around 1.0).
    ratio_step : float, optional
        Step size for the scale scan.
    metric : {"mean", "pearson"}, optional
        Cost metric used to evaluate alignment quality.

    Returns
    -------
    tuple of float
        Optimal (scale, shift) values.
    """
    list_shift = np.arange(shift_start, shift_stop + shift_step, shift_step)
    num_shift = len(list_shift)
    list_scale = 1.0 + np.arange(ratio_start, ratio_stop + ratio_step,
                                 ratio_step)
    num_scale = len(list_scale)
    mat_metric = np.ones((num_scale, num_shift), dtype=np.float32)
    for i, scale in enumerate(list_scale):
        for j, shift in enumerate(list_shift):
            measured_peaks1 = _correct_scale_shift(measured_peaks,
                                                   [scale, shift])
            if metric != "mean":
                metric = 1 - stats.pearsonr(reference_peaks,
                                            measured_peaks1)[0]
            else:
                metric = np.mean(
                    np.clip((reference_peaks - measured_peaks1), 0, None))
            mat_metric[i, j] = metric
    (i_pos, j_pos) = np.unravel_index(np.argmin(mat_metric, axis=None),
                                      mat_metric.shape)
    return list_scale[i_pos], list_shift[j_pos]


def _make_gaussian_peaks(discrete_vals, min_val, max_val, step_val, size=5):
    """
    Generate a 1D signal composed of Gaussian-shaped peaks at discrete values.
    Each discrete value is mapped to the nearest grid point and optionally
    smoothed with a Gaussian filter. The final signal is normalized to unity.

    Parameters
    ----------
    discrete_vals : array_like
        Discrete values at which peaks are located.
    min_val : float
        Minimum value of the output grid.
    max_val : float
        Maximum value of the output grid.
    step_val : float
        Step size of the output grid.
    size : int, optional
        Controls the width of the Gaussian smoothing. If zero or negative,
        no smoothing is applied.

    Returns
    -------
    y : ndarray
        Normalized 1D array containing Gaussian-shaped peaks.
    idx : ndarray
        Indices of the peak positions on the output grid.
    order : ndarray
        Indices mapping each peak to its original order in `discrete_vals`.
    """
    x_list = np.arange(min_val, max_val + step_val, step_val)
    y_list = np.zeros_like(x_list)
    list_idx = []
    list_order = []
    for i, x in enumerate(discrete_vals):
        y_list1 = np.zeros_like(x_list)
        pos_min = np.argmin(np.abs(x - x_list))
        y_list1[pos_min] = 1
        list_idx.append(pos_min)
        list_order.append(i)
        if size > 0:
            y_list1 = ndi.gaussian_filter1d(y_list1, size // 2 + 1)
        y_list = np.max(np.vstack((y_list, y_list1)), axis=0)
    y_list = y_list / np.max(y_list)
    return y_list, np.asarray(list_idx), np.asarray(list_order)


def match_peaks(measured_peaks, reference_values, step_size, order=2,
                search_range=5, tol=0.4, size=9, margin=6, scale_range=0.08,
                bounds=None, initial_guess=None, return_check=False):
    """
    Match measured peaks to reference values using polynomial fitting
    optimization. The method converts discrete peaks into Gaussian
    representations, estimates a polynomial transformation,
    and selects matching peaks based on a local similarity criterion.

    Parameters
    ----------
    measured_peaks : array_like
        Peak positions (pixel unit).
    reference_values : array_like
        Reference values such as Q-values or Energies.
    step_size : float
        Conversion factor from pixel to physical units (Q or Energy).
    order : int, optional
        Polynomial order used for the transformation.
    search_range : int or tuple, optional
        Shift search range (in physical units or as a tuple of bounds).
    tol : float, optional
        Tolerance threshold for accepting a peak match, in the range of [0, 1].
    size : int, optional
        Gaussian kernel size used to represent peaks.
    margin : int, optional
        Margin (pixel unit) for extending the grid beyond peak extrema.
    scale_range : float, optional
        Relative scale deviation around unity (1 -/+ scale_range).
    bounds : sequence of tuple or None, optional
        Bounds for the polynomial optimization parameters.
    initial_guess : array_like or None, optional
        Initial guess for the polynomial coefficients.
    return_check : bool, optional
        If True, return data for diagnostic purposes.

    Returns
    -------
    selected_peaks : ndarray
        Subset of `measured_peaks` matched to reference peaks.
    reference_ids : ndarray
        Indices of the matched reference peaks.
    results : dict, optional
        Diagnostic information, returned only if `return_check` is True.
    """
    measured_values = measured_peaks * step_size
    if isinstance(search_range, tuple):
        start = int(search_range[0] / step_size)
        stop = int(search_range[1] / step_size)
    else:
        start = -int(abs(search_range) / step_size)
        stop = abs(start)
    scale_range = np.clip(scale_range, 0.0, 1.0)
    scale_step = scale_range / 32
    min_val = min(np.min(measured_values),
                  np.min(reference_values)) - size * margin * step_size
    max_val = max(np.max(measured_values),
                  np.max(reference_values)) + size * margin * step_size
    meas_gauss_peaks, meas_idx, meas_order = _make_gaussian_peaks(
        measured_values, min_val, max_val, step_size, size=size)
    ref_gauss_peaks, ref_idx, ref_order = _make_gaussian_peaks(
        reference_values, min_val, max_val, step_size, size=size)

    if initial_guess is None:
        scale_val, shift_val = _align_peaks_coarse(
            ref_gauss_peaks, meas_gauss_peaks, start, stop,
            ratio_start=-scale_range, ratio_stop=scale_range,
            ratio_step=scale_step)
        initial_guess = np.zeros(order + 1)
        initial_guess[-2] = scale_val
        initial_guess[-1] = shift_val
    poly_facts = _find_scale_shift(ref_gauss_peaks, meas_gauss_peaks,
                                   initial_guess=initial_guess, bounds=bounds)
    meas_gauss_peaks_corr = _correct_scale_shift(meas_gauss_peaks, poly_facts)
    degree = len(poly_facts) - 1
    list_peaks_ref = np.int16(np.sum(np.asarray(
        [factor * ref_idx ** (degree - i) for i, factor in
         enumerate(poly_facts)]), axis=0))
    selected_peaks = []
    reference_ids = []
    rad = size // 2
    num_idx = len(meas_gauss_peaks_corr)
    for i, peak in enumerate(list_peaks_ref):
        pos = ref_idx[i]
        start_idx = pos - rad
        stop_idx = pos + rad + 1
        if 0 <= start_idx < num_idx and 0 <= stop_idx < num_idx:
            list1 = meas_gauss_peaks_corr[start_idx:stop_idx]
            list2 = ref_gauss_peaks[start_idx:stop_idx]
            val = np.mean(np.abs(list1 - list2))
            if val < tol:
                min_pos = np.argmin(np.abs(peak - meas_idx))
                selected_peaks.append(measured_peaks[meas_order[min_pos]])
                reference_ids.append(i)
    selected_peaks = np.asarray(selected_peaks)
    reference_ids = np.asarray(reference_ids)
    if return_check:
        results = {"x": np.linspace(min_val, max_val, len(meas_gauss_peaks)),
                   "measured_before": meas_gauss_peaks,
                   "measured_after": meas_gauss_peaks_corr,
                   "reference": ref_gauss_peaks,
                   "poly_factors": poly_facts}
        return selected_peaks, reference_ids, results
    return selected_peaks, reference_ids


def calibrate_energy_1d(list_data, reference_energies, energy_step,
                        fit_order=2, zero_left=0, zero_right=-1, sigma=0,
                        smooth_svd_para={"rank": 5,
                                         "lambda_threshold": 0.5},
                        get_peaks_para={"width": 15, "ratio": 0.1,
                                        "sigma": 0, "abs_threshold": None},
                        sub_peaks_para={"radius": 15, "sigma": 1,
                                        "skew_fit": True},
                        match_peaks_para={"tol": 0.4, "size": 31,
                                          "margin": 20,
                                          "search_range": (-5, 5),
                                          "bounds": None,
                                          "initial_guess": None},
                        peaks=None, return_check=False):
    """
    Calibrate a 1D spectrum by polynomial fitting to the energy axis.

    The function detects peak positions in a 1D signal, matches them to
    reference energies, and fits a polynomial mapping from pixel/index
    space to energy space.

    Parameters
    ----------
    list_data : array_like
        1D input signal (e.g. spectrum intensity).
    reference_energies : array_like
        Reference energy values used for calibration (E.g keV).
    energy_step : float
        Energy step per pixel used for peak matching.
    fit_order : int, optional
        Polynomial order of the energy calibration.
    zero_left : int, optional
        Points on the left side of the 1D input to be zeroed before processing.
    zero_right : int, optional
        Index from which data are zeroed on the right side. If negative,
        zeroing is applied from the end.
    sigma : float, optional
        Standard deviation for optional Gaussian smoothing.
    smooth_svd_para : dict or None, optional
        Parameters for SVD-based smoothing. If None,
        no SVD smoothing is applied.
    get_peaks_para : dict or None, optional
        Parameters for peak detection.
    sub_peaks_para : dict or None, optional
        Parameters for subpixel peak refinement.
    match_peaks_para : dict, optional
        Parameters controlling peak matching optimization.
    peaks : array_like or None, optional
        Precomputed peak positions (pixel units). If None, peaks are detected
        from `list_data`.
    return_check : bool, optional
        If True, also return matched peak positions and reference energies.

    Returns
    -------
    list_fact : ndarray
        Polynomial coefficients (constant term first, i.e a0+a1*x+a2*x^2 ..)
        for the energy calibration.
    loc_peaks : ndarray, optional
        Matched peak positions in pixel units. Returned if `return_check`
        is True.
    loc_energies : ndarray, optional
        Corresponding reference energies. Returned if `return_check` is True.
    """
    list_data = np.copy(list_data)
    npoint = len(list_data)
    min_len = fit_order + 1
    zero_left = np.clip(zero_left, 0, npoint - min_len)
    if zero_left > 0:
        list_data[:zero_left] = 0
    if zero_right == 0:
        zero_right == npoint - 1
    list_data[zero_right:] = 0
    list_fact = np.zeros(fit_order + 1, dtype=np.float32)
    if match_peaks_para is None:
        raise ValueError("Please provide parameters of the method for "
                         "matching measured peaks and reference peaks!")
    else:
        match_peaks_para = dict(match_peaks_para)
    loc_peaks, loc_energies = [], []
    if peaks is None:
        if sigma > 0:
            list_data = ndi.gaussian_filter1d(list_data, sigma)
        nmin, nmax = np.min(list_data), np.max(list_data)
        if nmax == nmin:
            if return_check:
                return list_fact, loc_peaks, loc_energies
            else:
                return list_fact
        if smooth_svd_para is not None:
            smooth_svd_para = dict(smooth_svd_para)
            list_data = prep.smooth_filter_based_svd(list_data,
                                                     **smooth_svd_para)
        if get_peaks_para is None:
            raise ValueError("Please provide parameters of the method for "
                             "getting peaks!")
        else:
            get_peaks_para = dict(get_peaks_para)
            peaks = prep.get_peaks_by_ranked_descent(list_data,
                                                     **get_peaks_para)
    else:
        try:
            peaks = np.squeeze(np.asarray(peaks))
        except Exception as error:
            raise ValueError(f"Error: {error}\nPlease provide peak locations "
                             f"(pixel unit) as a list or 1d-array!")
    if len(peaks) > min_len:
        if (peaks[0] - np.floor(peaks[0]) == 0.0) and (
                sub_peaks_para is not None):
            sub_peaks_para = dict(sub_peaks_para)
            peaks = prep.locate_gaussian_peaks_subpixel(list_data, peaks,
                                                        **sub_peaks_para)
        loc_peaks, ref_positions = match_peaks(peaks, reference_energies,
                                               energy_step, order=fit_order,
                                               **match_peaks_para)[:2]
        if len(ref_positions) > min_len:
            list_fact = np.float32(
                np.polyfit(loc_peaks, reference_energies[ref_positions],
                           fit_order, rcond=1e-64)[::-1])
            loc_energies = reference_energies[ref_positions]
    if return_check:
        return list_fact, loc_peaks, loc_energies
    else:
        return list_fact


def calibrate_energy_image(image, reference_energies, energy_step,
                           fit_order=2, zero_left=0, zero_right=-1, sigma=0,
                           smooth_svd_para={"rank": 5,
                                            "lambda_threshold": 0.5},
                           get_peaks_para={"width": 15, "ratio": 0.1,
                                           "sigma": 0, "abs_threshold": None},
                           sub_peaks_para={"radius": 15, "sigma": 0,
                                           "skew_fit": False},
                           match_peaks_para={"tol": 0.4, "size": 31,
                                             "margin": 20,
                                             "search_range": (-5, 5),
                                             "bounds": None,
                                             "initial_guess": None},
                           peaks=None, return_check=False,
                           ncore=None, prefer="processes"):
    """
    Calibrate the energy axis of a 2D image row-by-row.

    Each row of the image is treated as an independent 1D spectrum and
    calibrated using polynomial fitting based on matched reference energies.

    Parameters
    ----------
    image : ndarray
        2D input image with shape (height, width), where each row is a
        spectrum.
    reference_energies : array_like
        Reference energy values used for calibration.
    energy_step : float
        Energy step per pixel used for peak matching.
    fit_order : int, optional
        Polynomial order of the energy calibration.
    zero_left : int, optional
        Number of points on the left side of each row to be zeroed.
    zero_right : int, optional
        Index from which data are zeroed on the right side. If negative,
        zeroing is applied from the end.
    sigma : float, optional
        Standard deviation for optional Gaussian smoothing.
    smooth_svd_para : dict or None, optional
        Parameters for SVD-based smoothing. If None,
        no SVD smoothing is applied.
    get_peaks_para : dict or None, optional
        Parameters for peak detection.
    sub_peaks_para : dict or None, optional
        Parameters for subpixel peak refinement.
    match_peaks_para : dict, optional
        Parameters controlling peak matching optimization.
    peaks : sequence of array_like or None, optional
        Precomputed peak positions for each row. Length must equal image
        height. If None, peaks are detected independently for each row.
    return_check : bool, optional
        If True, also return matched peak positions and reference energies
        for each row.
    ncore : int or None, optional
        Number of CPU cores used for parallel processing. If None, uses
        all available cores minus one.
    prefer : {"processes", "threads"}, optional
        Parallelization backend.

    Returns
    -------
    results : ndarray
        Array of polynomial coefficients for each image row. Coefficients
        are ordered with the constant term first.
    loc_peaks : ndarray, optional
        Return matched peak positions for each row if `return_check` is True.
    loc_energies : ndarray, optional
        Corresponding reference energies for each row. Returned if
        `return_check` is True.
    """
    (height, width) = image.shape
    if peaks is not None:
        if len(peaks) != height:
            raise ValueError(f"Length of list of list of peaks, {len(peaks)}, "
                             f"is not the same as the height, {height}, "
                             f"of the image")
    max_cores = mp.cpu_count()
    if ncore is None:
        ncore = max(1, max_cores - 1)
    else:
        ncore = max(1, min(ncore, max_cores))
    results = Parallel(n_jobs=ncore, prefer=prefer)(
        delayed(calibrate_energy_1d)(
            image[idx], reference_energies, energy_step, fit_order,
            zero_left, zero_right, sigma, smooth_svd_para,
            get_peaks_para, sub_peaks_para, match_peaks_para, peaks,
            return_check) for idx in range(height))
    if return_check:
        return (np.asarray(results[:][0]), np.asarray(results[:][1]),
                np.asarray(results[:][2]))
    else:
        return np.asarray(results)


def calibrate_q_germ_detector_1d(strip_idx, list_data, d_spacing_values,
                                 two_theta, energy_step, energy_guess=None,
                                 fit_order=2, zero_left=0,
                                 zero_right=-1, sigma=0,
                                 smooth_svd_para={"rank": 5,
                                                  "lambda_threshold": 0.5},
                                 get_peaks_para={"width": 15, "ratio": 0.1,
                                                 "sigma": 0,
                                                 "abs_threshold": None},
                                 sub_peaks_para={"radius": 15, "sigma": 0,
                                                 "skew_fit": False},
                                 match_peaks_para={"tol": 0.4, "size": 31,
                                                   "margin": 20,
                                                   "search_range": (-5, 5),
                                                   "bounds": None,
                                                   "initial_guess": None},
                                 peaks=None, return_check=False):
    """
    Calibrate the scattering vector (q) axis for a 1D GERM detector pattern.

    The function detects peak positions in a single GERM detector strip,
    matches them to reference d-spacings, and fits a polynomial mapping from
    pixel/index space to q-space.

    Parameters
    ----------
    strip_idx : int
        Index of the GERM detector strip.
    list_data : array_like
        1D intensity profile measured by the detector strip.
    d_spacing_values : array_like
        Reference d-spacing values used for calibration.
    two_theta : float
        Scattering angle (2θ, in degree) of the detector.
    energy_step : float
        Energy step per pixel used for peak matching.
    energy_guess : array_like or None, optional
        Initial energy estimates corresponding to `d_spacing_values`.
        If None, energies are estimated from detector geometry.
    fit_order : int, optional
        Polynomial order of the q calibration.
    zero_left : int, optional
        Number of points on the left side to be zeroed before processing.
    zero_right : int, optional
        Index from which data are zeroed on the right side. If negative,
        zeroing is applied from the end.
    sigma : float, optional
        Standard deviation for optional Gaussian smoothing.
    smooth_svd_para : dict or None, optional
        Parameters for SVD-based smoothing. If None, no SVD smoothing is
        applied.
    get_peaks_para : dict or None, optional
        Parameters for peak detection.
    sub_peaks_para : dict or None, optional
        Parameters for subpixel peak refinement.
    match_peaks_para : dict, optional
        Parameters controlling peak matching and scale/shift optimization.
    peaks : array_like or None, optional
        Precomputed peak positions (pixel units). If None, peaks are detected
        from `list_data`.
    return_check : bool, optional
        If True, also return matched peak positions and reference q-values.

    Returns
    -------
    list_fact : ndarray
        Polynomial coefficients (constant term first) mapping pixel index
        to q-values.
    loc_peaks : ndarray, optional
        Matched peak positions in pixel units. Returned if `return_check`
        is True.
    loc_q_values : ndarray, optional
        Corresponding reference q-values. Returned if `return_check` is True.
    """
    list_data = np.copy(list_data)
    q_values = 2 * np.pi / d_spacing_values
    npoint = len(list_data)
    min_len = fit_order + 1
    zero_left = np.clip(zero_left, 0, npoint - min_len)
    if zero_left > 0:
        list_data[:zero_left] = 0
    if zero_right == 0:
        zero_right = npoint - 1
    list_data[zero_right:] = 0
    list_fact = np.zeros(fit_order + 1, dtype=np.float32)
    if match_peaks_para is None:
        raise ValueError("Please provide parameters of the method for "
                         "matching measured peaks and reference peaks!")
    else:
        match_peaks_para = dict(match_peaks_para)
    if energy_guess is None:
        energy_guess = np.asarray(
            [util.estimate_energy_germ_detector(
                strip_idx, d, two_theta) for d in d_spacing_values])
    else:
        if len(energy_guess) != len(d_spacing_values):
            raise ValueError("Number of d-spacing values is not the same as "
                             "the number of energy guess")
    loc_peaks, loc_q_values = [], []
    if peaks is None:
        if sigma > 0:
            list_data = ndi.gaussian_filter1d(list_data, sigma)
        nmin, nmax = np.min(list_data), np.max(list_data)
        if nmax == nmin:
            if return_check:
                return list_fact, loc_peaks, loc_q_values
            else:
                return list_fact
        if smooth_svd_para is not None:
            smooth_svd_para = dict(smooth_svd_para)
            list_data = prep.smooth_filter_based_svd(list_data,
                                                     **smooth_svd_para)
        if get_peaks_para is None:
            raise ValueError("Please provide parameters of the method for "
                             "getting peaks!")
        else:
            get_peaks_para = dict(get_peaks_para)
            peaks = prep.get_peaks_by_ranked_descent(list_data,
                                                     **get_peaks_para)
    else:
        try:
            peaks = np.squeeze(np.asarray(peaks))
        except Exception as error:
            raise ValueError(f"Error: {error}\nPlease provide peak locations "
                             f"(pixel unit) as a list or 1d-array!")
    if len(peaks) > min_len:
        if (peaks[0] - np.floor(peaks[0]) == 0.0) and (
                sub_peaks_para is not None):
            sub_peaks_para = dict(sub_peaks_para)
            peaks = prep.locate_gaussian_peaks_subpixel(list_data, peaks,
                                                        **sub_peaks_para)
        loc_peaks, ref_positions = match_peaks(peaks, energy_guess,
                                               energy_step, order=fit_order,
                                               **match_peaks_para)[:2]
        if len(ref_positions) > min_len:
            list_fact = np.float32(
                np.polyfit(np.float64(loc_peaks),
                           np.float64(q_values[ref_positions]),
                           fit_order, rcond=1e-64)[::-1])
            loc_q_values = q_values[ref_positions]
    if return_check:
        return list_fact, loc_peaks, loc_q_values
    else:
        return list_fact


def calibrate_q_germ_detector_image(image, d_spacing_values,
                                    two_theta, energy_step, energy_guess=None,
                                    fit_order=2, zero_left=0,
                                    zero_right=-1, sigma=0,
                                    smooth_svd_para={"rank": 5,
                                                     "lambda_threshold": 0.5},
                                    get_peaks_para={"width": 15, "ratio": 0.1,
                                                    "sigma": 0,
                                                    "abs_threshold": None},
                                    sub_peaks_para={"radius": 15, "sigma": 0,
                                                    "skew_fit": False},
                                    match_peaks_para={"tol": 0.4, "size": 31,
                                                      "margin": 20,
                                                      "search_range": (-5, 5),
                                                      "bounds": None,
                                                      "initial_guess": None},
                                    peaks=None, return_check=False, ncore=None,
                                    prefer="processes"):
    """
    Calibrate the scattering vector (q) axis for a 2D GERM detector image.

    This function applies 1D q-axis calibration independently to each row
    (strip) of a GERM detector image. For each detector strip, peak positions
    are detected or provided, matched to reference d-spacings, and used to
    fit a polynomial mapping from pixel/index space to q-space.

    Parameters
    ----------
    image : array_like
        2D detector image with shape (n_strips, n_pixels), where each row
        corresponds to a GERM detector strip.
    d_spacing_values : array_like
        Reference d-spacing values used for calibration.
    two_theta : float
        Scattering angle (2θ, in degree) of the detector.
    energy_step : float
        Energy step per pixel used for peak matching.
    energy_guess : array_like or None, optional
        Initial energy estimates corresponding to `d_spacing_values`.
        If None, energies are estimated from detector geometry for each strip.
    fit_order : int, optional
        Polynomial order of the q calibration.
    zero_left : int, optional
        Number of points on the left side of each strip to be zeroed before
        processing.
    zero_right : int, optional
        Index from which data are zeroed on the right side. If negative,
        zeroing is applied from the end.
    sigma : float, optional
        Standard deviation for optional Gaussian smoothing applied to each
        strip before peak detection.
    smooth_svd_para : dict or None, optional
        Parameters for SVD-based smoothing. If None, no SVD smoothing is
        applied.
    get_peaks_para : dict or None, optional
        Parameters for peak detection.
    sub_peaks_para : dict or None, optional
        Parameters for subpixel peak refinement.
    match_peaks_para : dict, optional
        Parameters controlling peak matching and scale/shift optimization.
    peaks : array_like or None, optional
        Precomputed peak positions for each detector strip. If provided,
        it must have length equal to the number of strips in `image`, where
        each element is a list or 1D array of peak positions (pixel units)
        for the corresponding strip. If None, peaks are detected from the
        image data.
    return_check : bool, optional
        If True, also return matched peak positions and reference q-values
        for each strip.
    ncore : int or None, optional
        Number of CPU cores used for parallel processing. If None, use
        all available cores minus one.
    prefer : {"processes", "threads"}, optional
        Parallelization backend.

    Returns
    -------
    list_fact : ndarray
        Array of polynomial coefficients for each detector strip, with shape
        (n_strips, fit_order + 1). Coefficients are ordered with the constant
        term first.
    loc_peaks : ndarray, optional
        Matched peak positions (pixel units) for each strip. Returned if
        `return_check` is True.
    loc_q_values : ndarray, optional
        Corresponding reference q-values for each strip. Returned if
        `return_check` is True.
    """
    (height, width) = image.shape
    if peaks is not None:
        if len(peaks) != height:
            raise ValueError(f"Length of list of list of peaks, {len(peaks)}, "
                             f"is not the same as the height, {height}, "
                             f"of the image")
    max_cores = mp.cpu_count()
    if ncore is None:
        ncore = max(1, max_cores - 1)
    else:
        ncore = max(1, min(ncore, max_cores))
    results = Parallel(n_jobs=ncore, prefer=prefer)(
        delayed(calibrate_q_germ_detector_1d)(
            idx, image[idx], d_spacing_values, two_theta, energy_step,
            energy_guess, fit_order, zero_left, zero_right, sigma,
            smooth_svd_para, get_peaks_para, sub_peaks_para, match_peaks_para,
            peaks, return_check) for idx in range(height))
    if return_check:
        return (np.asarray(results[:][0]), np.asarray(results[:][1]),
                np.asarray(results[:][2]))
    else:
        return np.asarray(results)


def calibrate_q_1d(list_data, q_reference, q_step, fit_order=2, zero_left=0,
                   zero_right=-1, sigma=0,
                   smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
                   get_peaks_para={"width": 15, "ratio": 0.1,
                                   "sigma": 0, "abs_threshold": None},
                   sub_peaks_para={"radius": 15, "sigma": 0,
                                   "skew_fit": False},
                   match_peaks_para={"tol": 0.4, "size": 31,
                                     "margin": 20,
                                     "search_range": (-0.5, 0.5),
                                     "bounds": None,
                                     "initial_guess": None},
                   peaks=None, return_check=False):
    """
    Calibrate the scattering vector (q) axis for a 1D pattern.

    The function detects peak positions in a 1D intensity profile, matches
    them to reference q-values, and fits a polynomial mapping from pixel/index
    space to q-space.

    Parameters
    ----------
    list_data : array_like
        1D intensity profile.
    q_reference : array_like
        Reference q-values used for calibration.
    q_step : float
        Step size in q per pixel used for peak matching.
    fit_order : int, optional
        Polynomial order of the q calibration.
    zero_left : int, optional
        Number of points on the left side to be zeroed before processing.
    zero_right : int, optional
        Index from which data are zeroed on the right side. If negative,
        zeroing is applied from the end.
    sigma : float, optional
        Standard deviation for optional Gaussian smoothing.
    smooth_svd_para : dict or None, optional
        Parameters for SVD-based smoothing. If None, no SVD smoothing is
        applied.
    get_peaks_para : dict or None, optional
        Parameters for peak detection.
    sub_peaks_para : dict or None, optional
        Parameters for subpixel peak refinement.
    match_peaks_para : dict, optional
        Parameters controlling peak matching optimization.
    peaks : array_like or None, optional
        Precomputed peak positions (pixel units). If None, peaks are detected
        from `list_data`.
    return_check : bool, optional
        If True, also return matched peak positions and reference q-values.

    Returns
    -------
    list_fact : ndarray
        Polynomial coefficients (constant term first) mapping pixel index
        to q-values.
    loc_peaks : ndarray, optional
        Matched peak positions in pixel units. Returned if `return_check`
        is True.
    loc_q_values : ndarray, optional
        Corresponding reference q-values. Returned if `return_check` is True.
    """
    list_data = np.copy(list_data)
    npoint = len(list_data)
    min_len = fit_order + 1
    zero_left = np.clip(zero_left, 0, npoint - min_len)
    if zero_left > 0:
        list_data[:zero_left] = 0
    if zero_right == 0:
        zero_right = npoint - 1
    list_data[zero_right:] = 0
    list_fact = np.zeros(fit_order + 1, dtype=np.float32)
    if match_peaks_para is None:
        raise ValueError("Please provide parameters of the method for "
                         "matching measured peaks and reference peaks!")
    else:
        match_peaks_para = dict(match_peaks_para)
    loc_peaks, loc_q_values = [], []
    if peaks is None:
        if sigma > 0:
            list_data = ndi.gaussian_filter1d(list_data, sigma)
        nmin, nmax = np.min(list_data), np.max(list_data)
        if nmax == nmin:
            if return_check:
                return list_fact, loc_peaks, loc_q_values
            else:
                return list_fact
        if smooth_svd_para is not None:
            smooth_svd_para = dict(smooth_svd_para)
            list_data = prep.smooth_filter_based_svd(list_data,
                                                     **smooth_svd_para)
        if get_peaks_para is None:
            raise ValueError("Please provide parameters of the method for "
                             "getting peaks!")
        else:
            get_peaks_para = dict(get_peaks_para)
            peaks = prep.get_peaks_by_ranked_descent(list_data,
                                                     **get_peaks_para)
    else:
        try:
            peaks = np.squeeze(np.asarray(peaks))
        except Exception as error:
            raise ValueError(f"Error: {error}\nPlease provide peak locations "
                             f"(pixel unit) as a list or 1d-array!")
    if len(peaks) > min_len:
        if (peaks[0] - np.floor(peaks[0]) == 0.0) and (
                sub_peaks_para is not None):
            sub_peaks_para = dict(sub_peaks_para)
            peaks = prep.locate_gaussian_peaks_subpixel(list_data, peaks,
                                                        **sub_peaks_para)
        loc_peaks, ref_positions = match_peaks(peaks, q_reference,
                                               q_step, order=fit_order,
                                               **match_peaks_para)[:2]
        if len(ref_positions) > min_len:
            list_fact = np.float32(
                np.polyfit(np.float64(loc_peaks),
                           np.float64(q_reference[ref_positions]),
                           fit_order, rcond=1e-64)[::-1])
            loc_q_values = q_reference[ref_positions]
    if return_check:
        return list_fact, loc_peaks, loc_q_values
    else:
        return list_fact


def calibrate_q_image(image, q_reference, q_step, fit_order=2, zero_left=0,
                      zero_right=-1, sigma=0,
                      smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
                      get_peaks_para={"width": 15, "ratio": 0.1,
                                      "sigma": 0, "abs_threshold": None},
                      sub_peaks_para={"radius": 15, "sigma": 0,
                                      "skew_fit": False},
                      match_peaks_para={"tol": 0.4, "size": 31,
                                        "margin": 20,
                                        "search_range": (-0.5, 0.5),
                                        "bounds": None,
                                        "initial_guess": None},
                      peaks=None, return_check=False, ncore=None,
                      prefer="processes"):
    """
    Calibrate the scattering vector (q) axis for a 2D image.

    This function applies 1D q-axis calibration independently to each row
    of the input image. For each row, peak positions are detected or provided,
    matched to reference q-values, and used to fit a polynomial mapping from
    pixel/index space to q-space.

    Parameters
    ----------
    image : array_like
        2D image with shape (n_rows, n_pixels), where each row is treated
        as an independent 1D spectrum.
    q_reference : array_like
        Reference q-values used for calibration.
    q_step : float or array_like
        Step size in q per pixel used for peak matching. If a scalar is
        provided, the same value is used for all rows. If array-like, it
        must have length equal to the number of image rows.
    fit_order : int, optional
        Polynomial order of the q calibration.
    zero_left : int, optional
        Number of points on the left side of each row to be zeroed before
        processing.
    zero_right : int, optional
        Index from which data are zeroed on the right side. If negative,
        zeroing is applied from the end.
    sigma : float, optional
        Standard deviation for optional Gaussian smoothing applied to each
        row before peak detection.
    smooth_svd_para : dict or None, optional
        Parameters for SVD-based smoothing. If None, no SVD smoothing is
        applied.
    get_peaks_para : dict or None, optional
        Parameters for peak detection.
    sub_peaks_para : dict or None, optional
        Parameters for subpixel peak refinement.
    match_peaks_para : dict, optional
        Parameters controlling peak matching optimization.
    peaks : array_like or None, optional
        Precomputed peak positions for each image row. If provided, it must
        have length equal to the number of rows in `image`, where each element
        is a list or 1D array of peak positions (pixel units). If None, peaks
        are detected from the image data.
    return_check : bool, optional
        If True, also return matched peak positions and reference q-values
        for each row.
    ncore : int or None, optional
        Number of CPU cores used for parallel processing. If None, use all
        available cores minus one.
    prefer : {"processes", "threads"}, optional
        Parallelization backend.

    Returns
    -------
    list_fact : ndarray
        Array of polynomial coefficients for each image row, with shape
        (n_rows, fit_order + 1). Coefficients are ordered with the constant
        term first.
    loc_peaks : ndarray, optional
        Matched peak positions (pixel units) for each row. Returned if
        `return_check` is True.
    loc_q_values : ndarray, optional
        Corresponding reference q-values for each row. Returned if
        `return_check` is True.
    """
    (height, width) = image.shape
    if not isinstance(q_step, float):
        if len(q_step) != height:
            raise ValueError(f"Length of list of q-step, {len(q_step)}, "
                             f"is not the same as the height, {height}, "
                             f"of the image")
    else:
        q_step = np.ones(height) * q_step
    if peaks is not None:
        if len(peaks) != height:
            raise ValueError(f"Length of list of list of peaks, {len(peaks)}, "
                             f"is not the same as the height, {height}, "
                             f"of the image")
    max_cores = mp.cpu_count()
    if ncore is None:
        ncore = max(1, max_cores - 1)
    else:
        ncore = max(1, min(ncore, max_cores))
    results = Parallel(n_jobs=ncore, prefer=prefer)(
        delayed(calibrate_q_1d)(
            image[idx], q_reference, q_step[idx], fit_order, zero_left,
            zero_right, sigma, smooth_svd_para, get_peaks_para,
            sub_peaks_para, match_peaks_para, peaks,
            return_check) for idx in range(height))
    if return_check:
        return (np.asarray(results[:][0]), np.asarray(results[:][1]),
                np.asarray(results[:][2]))
    else:
        return np.asarray(results)
