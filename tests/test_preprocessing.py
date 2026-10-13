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
# ============================================================================
# Contributors:
# ============================================================================

"""
Tests for methods in preprocessing.py
"""

import unittest
import numpy as np
import matplotlib
import sys

# matplotlib.use("Agg")
import algodif.prep.preprocessing as prep
import matplotlib.pyplot as plt


def make_gaussian(x, center, sigma, amplitude=1.0):
    return amplitude * np.exp(-(x - center) ** 2 / (2.0 * sigma ** 2))


class PreprocessingMethods(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.npoint = 200
        self.x = np.arange(self.npoint, dtype=np.float64)
        self.centers = [50, 100, 150]
        # Clean signal with three well-separated peaks.
        self.peaks_clean = (make_gaussian(self.x, self.centers[0] , 4.0, 1.0)
                            + make_gaussian(self.x, self.centers[1], 4.0, 0.6)
                            + make_gaussian(self.x, self.centers[2], 4.0, 0.8))
        self.baseline = make_gaussian(self.x, 60, 100.0, 0.2)
        self.noise = 0.005 * rng.randn(self.npoint)
        self.signal = self.peaks_clean + self.baseline + self.noise
        self.signal_no_bgr = self.peaks_clean + self.noise
        self.image = np.tile(self.signal, (6, 1))
        self.image_no_bgr = np.tile(self.signal_no_bgr, (6, 1))

    def test_remove_background_based_als(self):
        corrected = prep.remove_background_based_als(self.signal)
        self.assertEqual(corrected.shape, self.signal.shape)
        self.assertTrue(np.sqrt(
            np.mean(np.abs(corrected - self.signal_no_bgr) ** 2)) < 0.1)
        self.assertRaises(ValueError, prep.remove_background_based_als,
                          np.ones((10, 10)))

    def test_remove_background_based_arpls(self):
        corrected = prep.remove_background_based_arpls(self.signal)
        self.assertEqual(corrected.shape, self.signal.shape)
        self.assertTrue(np.sqrt(
            np.mean(np.abs(corrected - self.signal_no_bgr) ** 2)) < 0.01)
        self.assertRaises(ValueError, prep.remove_background_based_arpls,
                          np.ones((10, 10)))

    def test_remove_image_background_based_als(self):
        corrected = prep.remove_image_background_based_als(
            self.image, axis=0, ncore=1, prefer="threads")
        self.assertEqual(corrected.shape, self.image.shape)
        self.assertTrue(np.sqrt(
            np.mean(np.abs(corrected - self.image_no_bgr) ** 2)) < 0.05)

        corrected_t = prep.remove_image_background_based_als(
            np.transpose(self.image), axis=1, ncore=2, prefer="threads")
        self.assertEqual(corrected_t.shape, self.image.T.shape)
        self.assertTrue(np.sqrt(
            np.mean(np.abs(
                corrected_t - np.transpose(self.image_no_bgr)) ** 2)) < 0.05)

        self.assertRaises(ValueError, prep.remove_image_background_based_als,
                          self.image, axis=2)
        self.assertRaises(ValueError, prep.remove_image_background_based_als,
                          np.zeros((4, 50)))

    def test_remove_image_background_based_arpls(self):
        corrected = prep.remove_image_background_based_arpls(
            self.image, axis=0, ncore=1, prefer="threads")
        self.assertEqual(corrected.shape, self.image.shape)
        self.assertTrue(np.sqrt(
            np.mean(np.abs(corrected - self.image_no_bgr) ** 2)) < 0.05)

        corrected_t = prep.remove_image_background_based_arpls(
            np.transpose(self.image), axis=1, ncore=1, prefer="threads")
        self.assertTrue(np.sqrt(
            np.mean(np.abs(
                corrected_t - np.transpose(self.image_no_bgr)) ** 2)) < 0.05)

        self.assertRaises(ValueError,
                          prep.remove_image_background_based_arpls,
                          self.image, axis=2)
        self.assertRaises(ValueError,
                          prep.remove_image_background_based_arpls,
                          np.zeros((4, 50)))

    def test_get_peaks(self):
        peaks = prep.get_peaks(self.peaks_clean, width=5, threshold=0.05)
        self.assertTrue(np.array_equal(peaks, self.centers))

        # Only peaks with a normalized prominence above the threshold.
        peaks = prep.get_peaks(self.peaks_clean, width=5, threshold=0.7)
        self.assertTrue(np.array_equal(peaks, [50, 150]))

        noisy = self.peaks_clean + 0.02 * np.random.RandomState(1).rand(
            self.npoint)
        peaks = prep.get_peaks(noisy, width=5, sigma=2, threshold=0.2)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(np.all(np.abs(peaks - self.centers) <= 1))

        peaks = prep.get_peaks(self.peaks_clean, width=5, norm=False,
                               threshold=0.7)
        self.assertTrue(np.array_equal(peaks, [50, 150]))

        self.assertRaises(ValueError, prep.get_peaks, np.ones((10, 10)))

    def test_locate_peak_subpixel(self):
        f_alias = prep._locate_peak_subpixel
        x = np.arange(9, dtype=np.float64)
        data = 20.0 - (x - 4.3) ** 2
        self.assertTrue(np.abs(f_alias(data, method="diff") - 4.3) < 1e-6)
        self.assertTrue(np.abs(f_alias(data, method="poly_fit") - 4.3)
                        < 1e-6)
        # Maximum at the border: no sub-pixel shift is possible.
        data = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
        self.assertEqual(f_alias(data, method="diff"), 0)

    def test_fit_gaussian_peak(self):
        gauss_function = getattr(prep, "__gauss_function")
        x = np.arange(21) - 10
        data = gauss_function(x, 1.0, 1.5, 0.7, 0.1)
        fit_data, peak_pos, can_fit = prep.fit_gaussian_peak(data)
        self.assertTrue(can_fit)
        self.assertTrue(np.abs(peak_pos - 10.7) < 0.05)
        self.assertTrue(np.allclose(fit_data, data, atol=0.02))

        results = prep.fit_gaussian_peak(data, return_para=True)
        self.assertEqual(len(results), 6)
        _, a, b, c, d, can_fit = results
        self.assertTrue(can_fit)
        self.assertTrue(np.abs(a - 1.0) < 0.05)
        self.assertTrue(np.abs(np.abs(b) - 1.5) < 0.05)
        self.assertTrue(np.abs(c - 0.7) < 0.05)
        self.assertTrue(np.abs(d - 0.1) < 0.05)

        fit_data, peak_pos, can_fit = prep.fit_gaussian_peak(data, sigma=1)
        self.assertTrue(np.abs(peak_pos - 10.7) < 0.3)

        # All-zero data cannot be fitted.
        results = prep.fit_gaussian_peak(np.zeros(21), return_para=True)
        self.assertEqual(len(results), 3)
        self.assertFalse(results[-1])

    def test_fit_skewed_gaussian_peak(self):
        skewed_function = getattr(prep, "__skewed_gauss_function")
        x = np.arange(21) - 10
        data = skewed_function(x, 7.5, 3.0, 0.5, 0.0, 0.0)
        fit_data, peak_pos, can_fit = prep.fit_skewed_gaussian_peak(data)
        self.assertTrue(can_fit)
        self.assertEqual(fit_data.shape, data.shape)
        self.assertTrue(np.abs(peak_pos - 10.5) < 0.5)

        results = prep.fit_skewed_gaussian_peak(data, return_para=True)
        self.assertEqual(len(results), 7)

        results = prep.fit_skewed_gaussian_peak(np.zeros(21),
                                                return_para=True)
        self.assertEqual(len(results), 3)
        self.assertFalse(results[-1])

    def test_select_gaussian_peaks(self):
        rng = np.random.RandomState(2)
        data = self.peaks_clean + 0.02 * rng.rand(self.npoint)
        peaks = prep.select_gaussian_peaks(data, self.centers, tol=0.2,
                                           width=15, sub_pixel=False,
                                           skew_fit=False)
        self.assertTrue(np.array_equal(peaks, self.centers))

        peaks = prep.select_gaussian_peaks(data, self.centers, tol=0.2,
                                           width=15, sub_pixel=True,
                                           skew_fit=False)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(np.all(np.abs(peaks - self.centers) < 0.5))

        # A candidate on pure noise is rejected.
        peaks = prep.select_gaussian_peaks(data, [20] + self.centers,
                                           tol=0.2, width=15,
                                           sub_pixel=False, skew_fit=False)
        self.assertTrue(np.array_equal(peaks, self.centers))

        peaks = prep.select_gaussian_peaks(data, [], skew_fit=False)
        self.assertEqual(len(peaks), 0)

    def test_locate_peaks_subpixel(self):
        centers = [50.3, 100.4, 149.8]
        data = sum(make_gaussian(self.x, c, 4.0) for c in centers)
        peaks = [int(round(c)) for c in centers]
        pos = prep.locate_peaks_subpixel(data, peaks, radius=11, sigma=1,
                                         method="diff")
        self.assertEqual(len(pos), 3)
        self.assertTrue(np.all(np.abs(pos - centers) < 0.1))

        pos = prep.locate_peaks_subpixel(data, peaks, radius=11, sigma=1,
                                         method="poly_fit")
        self.assertEqual(len(pos), 3)
        self.assertTrue(np.all(np.abs(pos - centers) < 0.25))

        # Peaks too close to the borders are skipped.
        pos = prep.locate_peaks_subpixel(data, [5, 195], radius=11)
        self.assertEqual(len(pos), 0)

    def test_locate_gaussian_peaks_subpixel(self):
        centers = [50.3, 100.4, 149.8]
        data = sum(make_gaussian(self.x, c, 4.0) for c in centers)
        peaks = [int(round(c)) for c in centers]
        pos = prep.locate_gaussian_peaks_subpixel(data, peaks, radius=11,
                                                  sigma=1, skew_fit=False)
        self.assertEqual(len(pos), 3)
        self.assertTrue(np.all(np.abs(pos - centers) < 0.3))

        pos = prep.locate_gaussian_peaks_subpixel(data, [], skew_fit=False)
        self.assertEqual(len(pos), 0)

    def test_smooth_filter_based_svd(self):
        rank = 5
        data = self.signal
        # Without thresholding, the interior is reconstructed exactly.
        smoothed = prep.smooth_filter_based_svd(data, lambda_threshold=0.0,
                                                rank=rank)
        self.assertEqual(smoothed.shape, data.shape)
        self.assertTrue(np.allclose(smoothed[rank:-rank], data[rank:-rank]))

        # Result scales with the input amplitude.
        smoothed1 = prep.smooth_filter_based_svd(data, 0.5, rank)
        smoothed2 = prep.smooth_filter_based_svd(2.0 * data, 0.5, rank)
        self.assertTrue(np.allclose(2.0 * smoothed1, smoothed2))

        # A threshold above all singular values removes everything.
        smoothed = prep.smooth_filter_based_svd(data, 1e6, rank)
        self.assertTrue(np.all(smoothed == 0.0))

        zeros = np.zeros(50)
        self.assertTrue(np.array_equal(
            prep.smooth_filter_based_svd(zeros), zeros))

        self.assertRaises(ValueError, prep.smooth_filter_based_svd,
                          np.ones(8), 0.5, 5)
        self.assertRaises(ValueError, prep.smooth_filter_based_svd,
                          self.signal, -0.1, 5)

    def test_smooth_image_based_svd(self):
        smoothed = prep.smooth_image_based_svd(
            self.image, axis=0, lambda_threshold=0.5, rank=5, ncore=1,
            prefer="threads")
        self.assertEqual(smoothed.shape, self.image.shape)
        expected = prep.smooth_filter_based_svd(self.signal, 0.5, 5)
        self.assertTrue(np.allclose(smoothed[0], expected))

        smoothed_t = prep.smooth_image_based_svd(
            np.transpose(self.image), axis=1, lambda_threshold=0.5, rank=5,
            ncore=2, prefer="threads")
        self.assertTrue(np.allclose(smoothed_t, smoothed.T))

        self.assertRaises(ValueError, prep.smooth_image_based_svd,
                          self.image, axis=2)

    def test_locate_peaks_by_ranked_descent(self):
        peaks, left_peaks, right_peaks = prep.locate_peaks_by_ranked_descent(
            self.peaks_clean)
        self.assertTrue(np.array_equal(np.sort(peaks), self.centers))
        self.assertEqual(len(left_peaks), 3)
        self.assertEqual(len(right_peaks), 3)
        self.assertTrue(np.all(left_peaks < peaks))
        self.assertTrue(np.all(peaks < right_peaks))

        # Peaks are ranked by decreasing height.
        self.assertTrue(np.array_equal(peaks, [50, 150, 100]))

        peaks, _, _ = prep.locate_peaks_by_ranked_descent(
            self.peaks_clean, sigma=2)
        self.assertTrue(np.all(np.abs(np.sort(peaks) - self.centers) <= 1))

        # A monotonic signal has no peaks.
        peaks, left_peaks, right_peaks = prep.locate_peaks_by_ranked_descent(
            np.arange(50, dtype=np.float64))
        self.assertEqual(len(peaks), 0)
        self.assertEqual(len(left_peaks), 0)
        self.assertEqual(len(right_peaks), 0)

    def test_get_peaks_by_ranked_descent(self):
        data = self.peaks_clean + 0.1
        peaks = prep.get_peaks_by_ranked_descent(data, width=15, ratio=0.1,
                                                 sigma=0)
        self.assertTrue(np.array_equal(peaks, self.centers))

        peaks = prep.get_peaks_by_ranked_descent(data)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(np.all(np.abs(peaks - self.centers) <= 1))

        peaks = prep.get_peaks_by_ranked_descent(data, sigma=0,
                                                 abs_threshold=0.7)
        self.assertTrue(np.array_equal(peaks, [50, 150]))

        with self.assertWarns(UserWarning):
            prep.get_peaks_by_ranked_descent(data, width=2, sigma=0)

        self.assertRaises(ValueError, prep.get_peaks_by_ranked_descent,
                          np.arange(4, dtype=np.float64))
