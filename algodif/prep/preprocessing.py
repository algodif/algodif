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
# Description: Python module for preprocessing methods.
# Contributors:
# ============================================================================

"""
Module of methods in the preprocessing stage:

    -   Remove background of 1D signal or 2D image.
    -   Smooth signal/image using an SVD-based method.
    -   Methods for locate peaks with sub-pixel accuracy.
    -   Fit gaussian or skew gaussian to a peak.
    -
"""


import warnings
import multiprocessing as mp
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.linalg import cholesky
from scipy.sparse.linalg import spsolve
from scipy.stats import skewnorm
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def remove_background_based_als(list_data, ld=1e6, weight=0.05, num_iter=10):
    """
    Remove the baseline, i.e. background, using Asymmetric Least Squares (ALS)
    smoothing (Ref. [1]).

    Parameters
    ----------
    list_data : ndarray
        1D array of input signal (fluorescent spectrum or diffraction pattern).
    ld : float, optional
        Smoothing parameter. Higher values yield smoother baselines.
    weight : float, optional
        Asymmetry parameter (0 < weight < 1). Smaller is stronger suppression.
    num_iter : int, optional
        Number of iterations.

    Returns
    -------
    array_like
        Background-removed signal.

    References
    ----------
    [1] : https://doi.org/10.1021/ac051370e
    """
    if list_data.ndim != 1:
        raise ValueError("Input must be a 1D array!!!")
    size = len(list_data)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(size - 2, size),
                     format="csc")
    w = np.ones(size)
    z = 0.0
    for _ in range(num_iter):
        W = sparse.diags(w, 0, shape=(size, size), format="csc")
        Z = W + ld * D.T @ D
        z = spsolve(Z, w * list_data)
        w = weight * (list_data > z) + (1 - weight) * (list_data < z)
    return list_data - z


def remove_background_based_arpls(list_data, ld=1e5, weight=0.05, num_iter=10):
    """
    Remove the baseline using Asymmetrically Reweighted Penalized Least Squares
    (arPLS) smoothing (Ref. [1]).

    Parameters
    ----------
    list_data : ndarray
        1D array of input signal (fluorescent spectrum or diffraction pattern).
    ld : float, optional
        Smoothing parameter. Higher values yield smoother baselines.
    weight : float, optional
        Weight deviation (0 < weight < 1). Smaller means less negative values.
    num_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    array_like
        Background-removed signal.

    References
    ----------
    [1] : https://doi.org/10.1039/C4AN01061B
    """
    if list_data.ndim != 1:
        raise ValueError("Input must be a 1D array!!!")
    size = len(list_data)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(size - 2, size),
                     format="csc")
    H = ld * (D.T @ D)
    w = np.ones(size)
    z = 0.0
    for _ in range(num_iter):
        W = sparse.diags(w, 0, shape=(size, size), format="csc")
        Z = (W + H).toarray()
        try:
            C = cholesky(Z, lower=True)
        except Exception as error:
            raise RuntimeError(f"Cholesky decomposition failed: {error}")
        y_w = w * list_data
        y_tmp = np.linalg.solve(C, y_w)
        z = np.linalg.solve(C.T, y_tmp)
        d = list_data - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        term = 2 * (d - (2 * s - m)) / s
        term = np.clip(term, None, 80)
        w_new = 1.0 / (1 + np.exp(term))
        if np.linalg.norm(w - w_new) / np.linalg.norm(w) < weight:
            break
        w = w_new
    return list_data - z


def remove_image_background_based_als(image, axis=0, ld=1e6, weight=0.05,
                                      num_iter=10, ncore=None,
                                      prefer="processes"):
    """
    Apply 1D background removal method, Asymmetric Least Squares (ALS)
    (Ref. [1]), along an axis of an image.

    Parameters
    ----------
    image : ndarray
        2D array.
    axis : int
        Axis along which the method is applied.
    ld : float, optional
        Smoothing parameter. Higher values yield smoother baselines.
    weight : float, optional
        Asymmetry parameter (0 < weight < 1). Smaller is stronger suppression.
    num_iter : int, optional
        Number of iterations.
    ncore : int, optional
        Number of cpu-cores used for computing. Automatically selected if None.
    prefer : {"threads", "processes"}
        Preferred parallel backend ("threads" for I/O bound tasks or
        "processes" for CPU-bound tasks).

    Returns
    -------
    array_like
        Background-removed image.

    References
    ----------
    [1] : https://doi.org/10.1021/ac051370e
    """

    max_cores = mp.cpu_count()
    if ncore is None:
        ncore = max(1, max_cores - 1)
    else:
        ncore = max(1, min(ncore, max_cores))
    if axis != 0 and axis != 1:
        raise ValueError("Only 2 options: 0 or 1")
    if axis == 1:
        image = np.transpose(image)
    (height, width) = image.shape
    max_val = np.max(image)
    if max_val == 0.0:
        raise ValueError("Check input data. Maximum value is zero!!!")
    img_norm = image / max_val
    img_corr = Parallel(n_jobs=ncore, prefer=prefer)(delayed(
        remove_background_based_als)(img_norm[idx], ld, weight, num_iter) for
                                                     idx in range(height))
    img_corr = np.asarray(img_corr) * max_val
    if axis == 1:
        img_corr = np.transpose(img_corr)
    return img_corr


def remove_image_background_based_arpls(image, axis=0, ld=1e6, weight=0.05,
                                        num_iter=5, ncore=None,
                                        prefer="processes"):
    """
    Apply 1D background removal method, Asymmetrically Reweighted Penalized
    Least Squares (arPLS) (Ref. [1]), along an axis of an image.

    Parameters
    ----------
    image : ndarray
        2D array.
    axis : int
        Axis along which the method is applied.
    ld : float, optional
        Smoothing parameter. Higher values yield smoother baselines.
    weight : float, optional
        Weight deviation (0 < weight < 1). Smaller means less negative values.
    num_iter : int, optional
        Maximum number of iterations.
    ncore : int, optional
        Number of cpu-cores used for computing. Automatically selected if None.
    prefer : {"threads", "processes"}
        Preferred parallel backend ("threads" for I/O bound tasks or
        "processes" for CPU-bound tasks).

    Returns
    -------
    array_like
        Background-removed image.

    References
    ----------
    [1] : https://doi.org/10.1039/C4AN01061B
    """
    max_cores = mp.cpu_count()
    if ncore is None:
        ncore = max(1, max_cores - 1)
    else:
        ncore = max(1, min(ncore, max_cores))
    if axis != 0 and axis != 1:
        raise ValueError("Only 2 options: 0 or 1")
    if axis == 1:
        image = np.transpose(image)
    (height, width) = image.shape
    max_val = np.max(image)
    if max_val == 0.0:
        raise ValueError("Check input data. Maximum value is zero!!!")
    img_norm = image / max_val
    img_corr = Parallel(n_jobs=ncore, prefer=prefer)(delayed(
        remove_background_based_arpls)(img_norm[idx], ld, weight, num_iter) for
                                                     idx in range(height))
    img_corr = np.asarray(img_corr) * max_val
    if axis == 1:
        img_corr = np.transpose(img_corr)
    return img_corr


def get_peaks(list_data, width=15, sigma=0, norm=True, threshold=0.05):
    """
    Detect peaks in a 1D signal using prominence and optional smoothing.

    Parameters
    ----------
    list_data : ndarray
        1D input array (e.g., spectrum or signal).
    width : int, optional
        Minimum width of the peaks.
    sigma : int, optional
        Standard deviation for Gaussian smoothing If larger than 0.
    norm : bool, optional
        If True, normalize the signal to its maximum before peak detection.
    threshold : float, optional
        Minimum prominence of peaks. Default is 0.05.

    Returns
    -------
    array_like
        Indices of detected peaks.
    """
    if list_data.ndim != 1:
        raise ValueError("Input must be a 1D array!!!")
    if sigma > 0:
        list_data = ndi.gaussian_filter1d(list_data, sigma)
    if norm:
        list_data = list_data / np.max(list_data)
    peaks = sig.find_peaks(list_data, width=width, prominence=threshold)[0]
    return peaks


def _locate_peak_subpixel(list_data, method="diff"):
    """
    Locate the position of the maximum value of a 1d-array with sub-pixel
    accuracy.

    Parameters
    ----------
    list_data : array_like
        1D array. Recommended size < 10.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method or a polynomial method.

    Returns
    -------
    float
        Sub-pixel position of the maximum value.
    """
    size = len(list_data)
    pos = np.argmax(list_data)
    pos_b, pos_f, size1 = pos - 1, pos + 1, size - 1
    if method == "poly_fit":
        afact, bfact = np.polyfit(np.arange(size), list_data, 2)[:2]
        if afact != 0.0:
            sub_pos = -bfact / (2 * afact)
            if size1 > sub_pos > 0:
                pos = sub_pos
    else:
        if size1 > pos > 0:
            d1fx = (list_data[pos_f] - list_data[pos_b]) / 2.0
            d2fx = list_data[pos_f] + list_data[pos_b] - 2 * list_data[pos]
            if d2fx != 0.0:
                x_off = -d1fx / d2fx
                pos = pos + x_off
    return pos


def __gauss_function(x, a, b, c, d):
    """
    Supplementary method for fit_gaussian_peak method
    """
    y = a * np.exp(-np.power((x - c) / (2 * b ** 2), 2)) + d
    return y


def fit_gaussian_peak(list_data, sigma=0, return_para=False):
    """
    Fit a Gaussian peak to 1D array.

    Parameters
    ----------
    list_data : 1D array
        The input data to fit.
    sigma : float
        Gaussian filter smoothing before fitting.
    return_para : bool
        Whether to return parameters.

    Returns
    -------
    fit_data : 1D array
        The fitted Gaussian data.
    c : float
        The fitted peak center.
    can_fit : bool
        Whether the fitting was successful.
    (optional) a, b, c, d : float
        Amplitude, width, center, baseline.
    """
    if sigma > 0:
        list_data = ndi.gaussian_filter1d(list_data, sigma)
    npoint = len(list_data)
    mid = npoint // 2
    x_list = np.arange(npoint) - mid
    norm_fact = np.max(list_data)
    can_fit = False
    fit_data = list_data
    if norm_fact == 0.0:
        return fit_data, np.argmax(fit_data), can_fit
    try:
        paras = curve_fit(__gauss_function,
                          x_list, list_data / norm_fact)[0]
        a = paras[0] * norm_fact
        b = paras[1]
        c = paras[2]
        d = paras[3] * norm_fact
        fit_data = __gauss_function(x_list, a, b, c, d)
        peak_pos = mid + c
        can_fit = True
    except (RuntimeError, RuntimeWarning):
        return fit_data, np.argmax(fit_data), can_fit
    if return_para:
        return fit_data, a, b, c, d, can_fit
    else:
        return fit_data, peak_pos, can_fit


def __skewed_gauss_function(x, a, b, c, d, skew):
    """
    Skewed Gaussian function:
    a * skewnorm.pdf(x, skew, loc=c, scale=b) + d
    """
    return a * skewnorm.pdf(x, skew, loc=c, scale=b) + d


def __locate_skew_peak(a, loc=0, scale=1):
    result = minimize_scalar(
        lambda x: -skewnorm.pdf(x, a, loc=loc, scale=scale),
        bounds=(loc - 5*scale, loc + 5*scale), method="bounded")
    return result.x


def fit_skewed_gaussian_peak(list_data, sigma=0, return_para=False):
    """
    Fit a skewed Gaussian peak to a 1D array.

    Parameters
    ----------
    list_data : 1D array
        The input data to fit.
    sigma : float
        Gaussian filter smoothing before fitting.
    return_para : bool
        Whether to return parameters.

    Returns
    -------
    fit_data : 1D array
        The fitted skewed Gaussian data.
    c : float
        The fitted peak center.
    can_fit : bool
        Whether the fitting was successful.
    (optional) a, b, c, d, skew : float
        Amplitude, width, center, baseline, skewness.
    """
    if sigma > 0:
        list_data = ndi.gaussian_filter1d(list_data, sigma)
    npoint = len(list_data)
    mid = npoint // 2
    x_list = np.arange(npoint) - mid
    norm_fact = np.max(list_data)
    can_fit = False
    fit_data = list_data
    if norm_fact == 0.0:
        return fit_data, np.argmax(fit_data), can_fit
    p0 = [1.0, mid, 0.0, 0.0, 0.0]
    bounds = ([0, 1e-6, -mid, -np.inf, -10],
              [np.inf, npoint, mid, np.inf, 10])
    try:
        paras = curve_fit(
            __skewed_gauss_function, x_list, list_data / norm_fact, p0=p0,
            bounds=bounds)[0]
        a = paras[0] * norm_fact
        b, c, d, skew = paras[1], paras[2], paras[3] * norm_fact, paras[4]
        fit_data = __skewed_gauss_function(x_list, a / norm_fact, b, c,
                                           d / norm_fact, skew) * norm_fact
        peak_pos = mid + __locate_skew_peak(skew, loc=c, scale=b)
        can_fit = True
    except (RuntimeError, RuntimeWarning):
        return fit_data, np.argmax(fit_data), can_fit
    if return_para:
        return fit_data, a, b, c, d, skew, can_fit
    else:
        return fit_data, peak_pos, can_fit


def select_gaussian_peaks(list_data, peaks, tol=0.2, width=15, sigma=0,
                          sub_pixel=True, percentile=70, skew_fit=True,
                          show=False):
    """
    Select good peaks from the input data based on Gaussian fitting and
    tolerance criteria.

    Parameters
    ----------
    list_data : array_like
        1d-array.
    peaks : list of int
        Indices of candidate peaks in the data.
    tol : float, optional
        Tolerance for peak fitting accuracy.
    width : int, optional
        Width of a peak to consider for fitting.
    sigma : float, optional
        Standard deviation for Gaussian smoothing.
    sub_pixel : bool, optional
        Return fitted positions if True; otherwise return original indices.
    percentile : float, optional
        Percentile of absolute fit residuals to compare with tol.
        Clipped to [50, 100].
    skew_fit : bool, optional
        Use a skewed Gaussian if True; otherwise use a standard Gaussian.
    show : bool, optional
        Plot each accepted peak and its fit.

    Returns
    -------
    list of int
        Indices of the selected good peaks.
    """
    if sigma > 0:
        list_data = ndi.gaussian_filter1d(list_data, sigma)
    percentile = np.clip(percentile, 50, 100)
    good_peaks = []
    npoint = len(list_data)
    radius = width // 2
    for p in peaks:
        start = max(0, p - radius)
        stop = min(npoint, p + radius + 1)
        if (stop - start) > 3:
            list_sub = list_data[start:stop]
            std = np.std(list_sub)
            if std != 0.0:
                list_norm = (list_sub - np.min(list_sub)) / std
                if skew_fit is True:
                    fit_data, sub_pos, check = fit_skewed_gaussian_peak(
                        list_norm)
                else:
                    fit_data, sub_pos, check = fit_gaussian_peak(list_norm)
                res_val = np.percentile(np.abs(fit_data - list_norm),
                                        percentile)
                sub_pos = sub_pos + start
                if check and res_val < tol:
                    if sub_pixel:
                        good_peaks.append(sub_pos)
                    else:
                        good_peaks.append(p)
                    if show:
                        plt.figure(0)
                        plt.plot(np.arange(len(list_data)), list_data,
                                 color="blue")
                        plt.plot([p], list_data[p], "o", color="red")
                        plt.title(f"Peak location {p}.")
                        plt.figure(1)
                        plt.plot(np.arange(start, stop), list_norm, "-o",
                                 color="blue")
                        plt.plot(np.arange(start, stop), fit_data, color="red")
                        plt.plot([sub_pos, sub_pos],
                                 [np.min(fit_data), np.max(fit_data)],
                                 color="green")
                        plt.title(
                            f"Peak location {p}. Sub {sub_pos}. Check {check}")
                        plt.grid(True)
                        plt.show()
    return np.asarray(good_peaks)


def locate_peaks_subpixel(list_data, peaks, radius=11, sigma=1,
                          method="poly_fit", show=False):
    """
    Locate peaks with sub-pixel accuracy.

    Parameters
    ----------
    list_data : array_like
        1D array.
    peaks : list of int
        Indices of peaks of 1D signal.
    radius : int, optional
        Define the window radius around a peak. Default is 11.
    sigma : float, optional
        Standard deviation for Gaussian smoothing.
    method : {"diff", "poly_fit"}
        Method for finding sub-pixel shift. Two options: a differential
        method or a polynomial method.
    show : bool, optional
        Display the plotting result for each peak.

    Returns
    -------
    list of float
        Sub-pixel positions of peaks.
    """
    peaks_sub = []
    width = len(list_data)
    if sigma > 0:
        list_data = ndi.gaussian_filter1d(list_data, sigma)
    for idx in peaks:
        if (width - radius - 1) > idx > radius:
            list_sub = list_data[idx - radius:idx + radius + 1] + 1
            pos = _locate_peak_subpixel(list_sub, method=method)
            peaks_sub.append(idx - radius + pos)
            if show:
                x_list = np.arange(idx - radius, idx + radius + 1)
                plt.plot(x_list, list_sub, "-o", color="blue")
                plt.plot(np.ones_like(x_list) * (idx - radius + pos),
                         np.linspace(np.min(list_sub), np.max(list_sub),
                                     len(x_list)), color="red")
                plt.title("Sub-pixel peak location {0:.3f}".format(idx + pos))
                plt.grid(True)
                plt.show()
    return np.asarray(peaks_sub)


def locate_gaussian_peaks_subpixel(list_data, peaks, radius=11, sigma=1,
                                   skew_fit=True, show=False):
    """
    Locate peaks with sub-pixel accuracy using Gaussian fitting.

    Parameters
    ----------
    list_data : array_like
        1D array.
    peaks : list of int
        Indices of peaks of 1D signal.
    radius : int, optional
        Define the window radius around a peak. Default is 11.
    sigma : float, optional
        Standard deviation for Gaussian smoothing.
    skew_fit : bool, optional
        Apply skewed Gaussian fitting if True
    show : bool, optional
        Display the plotting result for each peak.

    Returns
    -------
    list of float
        Sub-pixel positions of peaks.
    """
    peaks_sub = []
    npoint = len(list_data)
    if sigma > 0:
        list_data = ndi.gaussian_filter1d(list_data, sigma)
    for p in peaks:
        start = max(0, p - radius)
        stop = min(npoint, p + radius + 1)
        if (stop - start) > 3:
            list_sub = list_data[start:stop]
            std = np.std(list_sub)
            if std != 0.0:
                list_norm = (list_sub - np.min(list_sub)) / std
                if skew_fit is True:
                    fit_data, sub_pos, check = fit_skewed_gaussian_peak(
                        list_norm)
                else:
                    fit_data, sub_pos, check = fit_gaussian_peak(list_norm)
                sub_pos = sub_pos + start
                if show:
                    plt.figure(0)
                    plt.plot(np.arange(len(list_data)), list_data,
                             color="blue")
                    plt.plot([p], list_data[p], "o", color="red")
                    plt.title(f"Peak location {p}.")
                    plt.figure(1)
                    plt.plot(np.arange(start, stop), list_norm, "-o",
                             color="blue")
                    plt.plot(np.arange(start, stop), fit_data, color="red")
                    plt.plot([sub_pos, sub_pos],
                             [np.min(fit_data), np.max(fit_data)],
                             color="green")
                    plt.title(
                        f"Peak location {p}. Sub {sub_pos}. Check {check}")
                    plt.grid(True)
                    plt.show()
                peaks_sub.append(sub_pos)
    return np.asarray(peaks_sub)


def smooth_filter_based_svd(list_data, lambda_threshold=0.5, rank=5):
    """
    Smooth a 1D signal using Singular Value Decomposition (SVD) with
    soft-thresholding.

    Parameters:
    ----------
    list_data : np.ndarray
        The 1D input signal to be smoothed.
    rank : int, optional (default=5)
        Number of singular values (components) to retain for reconstruction.
        Must be a positive integer less than the data length.
    lambda_threshold : float, optional (default=0.5)
        Threshold used for soft-thresholding the singular values.
        Must be non-negative.

    Returns:
    -------
    np.ndarray
        The smoothed signal, scaled to the original amplitude.
    """
    rank = int(rank)
    if rank > len(list_data) // 2:
        raise ValueError("`rank` must not exceed the length of the signal.")
    if lambda_threshold < 0:
        raise ValueError("`lambda_threshold` must be non-negative.")
    max_value = np.max(list_data)
    if max_value != 0:
        norm_data = list_data / max_value
    else:
        return list_data
    npoint = len(norm_data)
    hankel_matrix = np.column_stack(
        [norm_data[i:npoint - rank + i + 1] for i in range(rank)])

    U, singular_values, Vt = np.linalg.svd(hankel_matrix, full_matrices=False)
    singular_values = np.where(singular_values > lambda_threshold,
                               singular_values - lambda_threshold, 0)
    singular_values = singular_values[:rank]
    reconstructed_matrix = np.dot(U[:, :rank] * singular_values, Vt[:rank, :])
    smoothed_data = np.zeros(npoint)
    for i in range(rank):
        for j in range(npoint - rank + 1):
            smoothed_data[i + j] += reconstructed_matrix[j, i]
    smoothed_data /= rank
    return smoothed_data * max_value


def smooth_image_based_svd(image, axis=0, lambda_threshold=0.5, rank=5,
                           ncore=None, prefer="processes"):
    """
    Apply SVD-based 1D smooth filter to an image.

    Parameters:
    ----------
    image : np.ndarray
        2D array.
    rank : int, optional (default=5)
        Number of singular values (components) to retain for reconstruction.
        Must be a positive integer less than the data length.
    lambda_threshold : float, optional (default=0.5)
        Threshold used for soft-thresholding the singular values.
        Must be non-negative.

    Returns:
    -------
    np.ndarray
        The smoothed signal, scaled to the original amplitude.
    """
    max_cores = mp.cpu_count()
    if ncore is None:
        ncore = max(1, max_cores - 1)
    else:
        ncore = max(1, min(ncore, max_cores))
    if axis != 0 and axis != 1:
        raise ValueError("Only 2 options: 0 or 1")
    if axis == 1:
        image = np.transpose(image)
    (height, width) = image.shape
    img_corr = Parallel(n_jobs=ncore, prefer=prefer)(delayed(
        smooth_filter_based_svd)(image[idx], lambda_threshold, rank) for idx in
                                                     range(height))
    img_corr = np.asarray(img_corr)
    if axis == 1:
        img_corr = np.transpose(img_corr)
    return img_corr


def locate_peaks_by_ranked_descent(list_data, sigma=0):
    """
    Implement a method for locating peaks of 1D signal based on
    sorting and index-coupling method (to be published in Ref [1]).
    The method locates peaks with their left and right shoulders.

    Parameters:
    ----------
    list_data : np.ndarray
        1D input signal.
    sigma : float, optional
        Standard deviation for Gaussian smoothing.

    Returns:
    -------
    tuple of ndarray
        (Peak locations, left-peak locations, right-peak locations)

    References
    ----------
    [1] Algorithms and software tools for energy-dispersive X-ray diffraction
        tomography, Nghia T. Vo et al.
    """
    if sigma > 0:
        list_data = ndi.gaussian_filter1d(list_data, sigma)
    indices = np.argsort(list_data)[::-1]
    npoint = len(list_data)
    peaks = []
    left_peaks = []
    right_peaks = []
    remained_ids = np.copy(indices)
    num_iter = 0
    while len(remained_ids) > 1 and num_iter < npoint:
        idx = remained_ids[0]
        val0 = list_data[idx]
        r_idx = idx
        r_val = val0
        while r_idx < npoint - 1:
            r_idx = r_idx + 1
            if list_data[r_idx] > r_val:
                r_idx = r_idx - 1
                break
            r_val = list_data[r_idx]
        l_idx = idx
        l_val = val0
        while l_idx > 0:
            l_idx = l_idx - 1
            if list_data[l_idx] > l_val:
                l_idx = l_idx + 1
                break
            l_val = list_data[l_idx]
        if r_idx > idx > l_idx:
            peaks.append(idx)
            left_peaks.append(l_idx)
            right_peaks.append(r_idx)
            mask = (remained_ids < l_idx) | (remained_ids > r_idx)
            remained_ids = remained_ids[mask]
        else:
            remained_ids = np.delete(remained_ids, 0)
            num_iter = num_iter + 1
    return np.asarray(peaks), np.asarray(left_peaks), np.asarray(right_peaks)


def get_peaks_by_ranked_descent(list_data, width=15, ratio=0.1, sigma=2,
                                abs_threshold=None):
    """
    Locating peaks of 1D signal based on sorting and index-coupling method
    (to be published in Ref [1]).

    Parameters:
    ----------
    list_data : np.ndarray
        1D input signal.
    width : int
        Minimum width of a peak.
    ratio : float
        To select peaks above the ratio. Ratio is the value of dividing
        of the distance of a peak and its fitted baseline vs the mean value of
        data around a peak.
    sigma : float, optional
        Standard deviation for Gaussian smoothing.
    abs_threshold : float, optional
        Use it to select peaks with distance (from a peak to its baseline)
        larger than the threshold. This is absolute value (e.g.
        intensity count)

    Returns:
    -------
    ndarray
        Peak locations

    References
    ----------
    [1] Algorithms and software tools for energy-dispersive X-ray diffraction
        tomography, Nghia T. Vo et al.
    """

    if sigma > 0:
        list_data = ndi.gaussian_filter1d(list_data, sigma)
    list_data = list_data - np.min(list_data)
    list_data = list_data + np.median(list_data)
    npoint = len(list_data)
    if npoint < 5:
        raise ValueError("Number of data points must be >= 5")
    if width < 5 or (width > npoint - 5):
        warnings.warn(f"Width must be in the range of (5, {npoint-5}))!!!",
                      UserWarning)
        width = np.clip(width, 5, npoint - 5)
    m_peaks, l_peaks, r_peaks = locate_peaks_by_ranked_descent(list_data)
    num_peak = len(m_peaks)
    list_width = r_peaks - l_peaks
    good_peaks = []
    for i in range(num_peak):
        if list_width[i] >= width:
            sub_list = list_data[l_peaks[i]:r_peaks[i] + 1]
            x_list = np.arange(l_peaks[i], r_peaks[i] + 1)
            (a, b) = np.polyfit(np.concatenate((x_list[:2], x_list[-2:])),
                                np.concatenate((sub_list[:2], sub_list[-2:])),
                                1)[:2]
            y_fit = a * x_list + b
            pos = m_peaks[i] - l_peaks[i]
            nmean = np.mean(sub_list)
            distance = sub_list[pos] - y_fit[pos]
            if nmean != 0.0:
                val = distance / nmean
                if val >= ratio:
                    if abs_threshold is None:
                        good_peaks.append(m_peaks[i])
                    else:
                        if distance >= abs_threshold:
                            good_peaks.append(m_peaks[i])
    return np.sort(np.array(good_peaks))
