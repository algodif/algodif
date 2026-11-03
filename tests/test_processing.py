"""
Tests for methods in processing.py

Measured data are simulated by resampling theoretical spectra/patterns,
generated using utility.py, onto a pixel grid whose physical axis (energy or
q) is a known polynomial of the pixel index.
"""

import unittest
import numpy as np
import algodif.prep.preprocessing as prep
import algodif.proc.processing as proc
import algodif.util.utility as util

GET_PEAKS_PARA = {"width": 15, "ratio": 0.1, "sigma": 0,
                  "abs_threshold": None}
GET_PEAKS_PARA2 = {"width": 10, "ratio": 0.4, "sigma": 0,
                   "abs_threshold": None}
SUB_PEAKS_PARA = {"radius": 7, "sigma": 0, "skew_fit": False}
ENERGY_MATCH_PARA = {"tol": 0.4, "size": 15, "margin": 5,
                     "search_range": (-3, 3), "bounds": None,
                     "initial_guess": None}
Q_MATCH_PARA = {"tol": 0.4, "size": 9, "margin": 5,
                "search_range": (-0.15, 0.15), "bounds": None,
                "initial_guess": None}


def resample_data(list_data, poly_coefs):
    """
    Resample a 1d array on an equal-space grid to polynomially changed grid.
    """
    npoint = len(list_data)
    x = np.arange(npoint, dtype=np.float32)
    x_new = np.polyval(poly_coefs, x)
    return np.interp(x_new, x, list_data)


class ProcessingMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Simulated detector image: 32 strips x 1024 channels.
        cls.height, cls.npoint = 32, 1024
        cls.x = np.arange(cls.npoint)
        cls.energy_step = 0.12
        cls.poly_coefs_1_deg = [0.95, 20]
        cls.poly_coefs_2_deg = [0.0001, 0.95, 20]
        list_element = ["Sn", "Au", "Pb", "U"]
        fluo_lines = util.get_fluorescence_lines_from_database(list_element,
                                                               line_type="K",
                                                               threshold=0.03,
                                                               norm=True,
                                                               check=False)
        cls.theory_spectrum = util.make_fluorescence_spectrum(fluo_lines,
                                                              cls.energy_step,
                                                              cls.npoint,
                                                              sigma=3,
                                                              norm=True)
        ref_peaks = prep.get_peaks_by_ranked_descent(cls.theory_spectrum[:, 1],
                                                     width=5)
        ref_peaks_sub = prep.locate_gaussian_peaks_subpixel(
            cls.theory_spectrum[:, 1], ref_peaks, radius=2, sigma=1,
            skew_fit=False, show=False)
        cls.ref_energies = ref_peaks_sub * cls.energy_step
        cls.ref_spectrum = cls.theory_spectrum[:, 1]
        cls.mod_spectrum1 = resample_data(cls.ref_spectrum,
                                          cls.poly_coefs_1_deg)
        cls.mod_spectrum2 = resample_data(cls.ref_spectrum,
                                          cls.poly_coefs_2_deg)
        cls.cr2o3_d_spacing = np.asarray(
            [3.631, 2.665, 2.48, 2.266, 2.175, 2.048, 1.815, 1.673, 1.465,
             1.432, 1.296, 1.24])
        cls.cr2o3_q_vals = 2 * np.pi / cls.cr2o3_d_spacing

    def test_find_polynomial_warp(self):
        results1 = proc.find_polynomial_warp(self.ref_spectrum,
                                             self.mod_spectrum1)
        corrected1 = proc.apply_polynomial_warp(self.mod_spectrum1, results1)
        rmse = np.sqrt(np.mean((corrected1 - self.ref_spectrum) ** 2))
        self.assertTrue(rmse < 0.05)

        initial_guess = np.insert(results1, 0, 0.0)
        results2 = proc.find_polynomial_warp(self.ref_spectrum,
                                             self.mod_spectrum2,
                                             initial_guess=initial_guess,
                                             bounds=None)
        corrected2 = proc.apply_polynomial_warp(self.mod_spectrum2, results2)
        rmse = np.sqrt(np.mean((corrected2 - self.ref_spectrum) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_make_gaussian_peak_signal(self):
        vals = [1.0, 3.0, 4.5]
        min_val, max_val, step_val = 0.0, 6.0, 0.05
        x_list = np.arange(min_val, max_val + step_val, step_val)
        signal, idx, order = proc.make_gaussian_peak_signal(
            vals, min_val, max_val, step_val, size=9)
        self.assertEqual(signal.shape, x_list.shape)
        self.assertTrue(np.array_equal(idx, [20, 60, 90]))
        self.assertTrue(np.array_equal(order, [0, 1, 2]))
        self.assertTrue(np.isclose(np.max(signal), 1.0))
        self.assertTrue(np.allclose(signal[idx], 1.0))

        # Detected peaks sit at the grid points nearest to the input values.
        peaks = prep.get_peaks(signal, width=3, threshold=0.5)
        self.assertTrue(np.array_equal(peaks, np.sort(idx)))
        self.assertTrue(np.all(np.abs(x_list[peaks] - np.sort(vals))
                               <= step_val / 2))

        # Without smoothing, peaks are single-point spikes.
        signal, idx, _ = proc.make_gaussian_peak_signal(
            vals, min_val, max_val, step_val, size=0)
        self.assertEqual(np.count_nonzero(signal), len(vals))
        peaks = prep.get_peaks(signal, width=1, threshold=0.5)
        self.assertTrue(np.array_equal(peaks, np.sort(idx)))

        # A value outside the grid is clamped to the endpoint with a warning.
        with self.assertWarns(UserWarning):
            _, idx, _ = proc.make_gaussian_peak_signal(
                [3.0, 10.0], min_val, max_val, step_val, size=9)
        self.assertEqual(idx[-1], len(x_list) - 1)

    def test_estimate_scale_shift(self):
        results = proc.estimate_scale_shift(self.ref_spectrum,
                                            self.mod_spectrum1, -25, 25)
        corrected = proc.apply_polynomial_warp(self.mod_spectrum1, results)
        rmse = np.sqrt(np.mean((corrected - self.ref_spectrum) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_match_peaks(self):
        peaks = prep.get_peaks_by_ranked_descent(self.mod_spectrum2)
        selected_peaks, reference_ids, results = proc.match_peaks(
            peaks, self.ref_energies, self.energy_step, size=15, tol=0.3,
            return_check=True)
        measured_after = results["measured_after"]
        reference = results["reference"]
        rmse = np.sqrt(np.mean((measured_after - reference) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_calibrate_energy_1d(self):
        poly_coefs = proc.calibrate_energy_1d(self.mod_spectrum2,
                                              self.ref_energies,
                                              self.energy_step,
                                              get_peaks_para=GET_PEAKS_PARA,
                                              sub_peaks_para=SUB_PEAKS_PARA,
                                              match_peaks_para=ENERGY_MATCH_PARA)
        x_list = np.arange(self.npoint)
        list_energy = x_list * self.energy_step
        list_e_mea = np.sum(np.asarray(
            [coef * x_list ** i for i, coef in enumerate(poly_coefs)]),
            axis=0)
        corrected = np.interp(list_energy, list_e_mea, self.mod_spectrum2,
                              left=0, right=0)
        rmse = np.sqrt(np.mean((corrected - self.ref_spectrum) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_calibrate_energy_image(self):
        ref_e_image = np.tile(self.ref_spectrum, (self.height, 1))
        mod_e_image = np.zeros_like(ref_e_image)
        mod_poly_coef = np.zeros((self.height, 3), dtype=np.float32)
        np.random.seed(0)
        for i in range(self.height):
            mod_poly_coef[i, 0] = (np.random.rand() - 0.5) / 1e+5
            mod_poly_coef[i, 1] = 1.0 + 2.0 * (np.random.rand() - 0.5) / 1e+2
            mod_poly_coef[i, 2] = np.random.randint(-8, 8)
            mod_e_image[i] = resample_data(self.ref_spectrum,
                                           mod_poly_coef[i])
        results = proc.calibrate_energy_image(mod_e_image, self.ref_energies,
                                              self.energy_step,
                                              get_peaks_para=GET_PEAKS_PARA,
                                              sub_peaks_para=SUB_PEAKS_PARA,
                                              match_peaks_para=ENERGY_MATCH_PARA)
        x_list = np.arange(self.npoint)
        list_energy = x_list * self.energy_step
        corrected_image = np.zeros_like(mod_e_image)
        for i, poly_coefs in enumerate(results):
            list_e_mea = np.sum(np.asarray(
                [coef * x_list ** i for i, coef in enumerate(poly_coefs)]),
                axis=0)
            corrected_image[i] = np.interp(list_energy, list_e_mea,
                                           mod_e_image[i],
                                           left=0, right=0)
        rmse = np.sqrt(np.mean((corrected_image - ref_e_image) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_calibrate_q_germ_detector_1d(self):
        energy_step = 0.054
        width = 4096
        two_theta = 3.0
        q_step = (4.0 * np.pi * np.sin(
            np.deg2rad(two_theta / 2.0)) / 12.398) * energy_step
        # est_energy = 12.398 / (2 * self.cr2o3_d_spacing * np.sin(
        #     np.deg2rad(two_theta / 2.0)))
        intensities = np.ones_like(self.cr2o3_q_vals)
        ref_pattern = util.make_diffraction_pattern(self.cr2o3_q_vals,
                                                    intensities, q_step,
                                                    width, sigma=15,
                                                    norm=True)
        poly_coefs_2_deg = [0.00001, 0.95, 20]
        mod_pattern = resample_data(ref_pattern[:, 1], poly_coefs_2_deg)
        poly_coefs = proc.calibrate_q_germ_detector_1d(96, mod_pattern,
                                                       self.cr2o3_d_spacing,
                                                       two_theta, energy_step,
                                                       get_peaks_para=GET_PEAKS_PARA2,
                                                       sub_peaks_para=SUB_PEAKS_PARA,
                                                       match_peaks_para=Q_MATCH_PARA)
        x_list = np.arange(width)
        list_q = np.arange(width) * q_step
        list_q_mea = np.sum(np.asarray(
            [coef * x_list ** i for i, coef in enumerate(poly_coefs)]),
            axis=0)
        corrected = np.interp(list_q, list_q_mea, mod_pattern,
                              left=0, right=0)
        rmse = np.sqrt(np.mean((corrected - ref_pattern[:, 1]) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_calibrate_q_germ_detector_image(self):
        intensities = np.ones_like(self.cr2o3_q_vals)
        two_theta = 3.0
        energy_step = 0.054
        total_germ_strip = 192
        used_ids = range(96 - 10, 96 + 10, 1)
        start_idx = used_ids[0]
        height = len(used_ids)
        width = 4096
        strip_size = 0.25
        sample_slit = 250.0
        sample_detector = 2000.0
        two_theta_rad = np.deg2rad(two_theta)
        ref_q_image = np.zeros((height, width), dtype=np.float32)
        q_map = np.zeros((height, width), dtype=np.float32)
        for strip_idx in used_ids:
            num1 = (strip_idx - 0.5 * (total_germ_strip - 1)) * strip_size
            num2 = (sample_detector - sample_slit)
            two_theta_idx = (two_theta_rad - np.arctan(num1 / num2))
            q_step = ((4 * np.pi * np.sin(
                two_theta_idx / 2) * energy_step) / 12.398)
            results = util.make_diffraction_pattern(self.cr2o3_q_vals,
                                                    intensities, q_step,
                                                    width, sigma=11,
                                                    norm=True)
            q_map[strip_idx - start_idx] = results[:, 0]
            ref_q_image[strip_idx - start_idx] = results[:, 1]

        mod_q_image = np.zeros_like(ref_q_image)
        mod_poly_coef = np.zeros((height, 3), dtype=np.float32)
        np.random.seed(0)
        for i in range(height):
            mod_poly_coef[i, 0] = (np.random.rand() - 0.5) / 1e+5
            mod_poly_coef[i, 1] = 1.0 + 2.0 * (np.random.rand() - 0.5) / 1e+2
            mod_poly_coef[i, 2] = np.random.randint(-8, 8)
            mod_q_image[i] = resample_data(ref_q_image[i], mod_poly_coef[i])

        poly_map = proc.calibrate_q_germ_detector_image(mod_q_image,
                                                        self.cr2o3_d_spacing,
                                                        two_theta, energy_step,
                                                        strip_start=start_idx,
                                                        get_peaks_para=GET_PEAKS_PARA2,
                                                        sub_peaks_para=SUB_PEAKS_PARA,
                                                        match_peaks_para=Q_MATCH_PARA,
                                                        return_check=False)
        q_step0 = (4.0 * np.pi * np.sin(
            np.deg2rad(two_theta / 2.0)) / 12.398) * energy_step
        x_list = np.arange(width)
        list_q = np.arange(width) * q_step0
        corrected_image = np.zeros_like(mod_q_image)
        for k in range(height):
            list_q_mea = np.sum(np.asarray(
                [coef * x_list ** i for i, coef in enumerate(poly_map[k])]),
                axis=0)
            corrected_image[k] = np.interp(list_q, list_q_mea, mod_q_image[k],
                                           left=0, right=0)

        num = 0.0
        for i in range(1, height):
            num += np.mean(np.abs(corrected_image[i] - corrected_image[0]))
        self.assertTrue(num < 0.1)

    def test_calibrate_q_1d(self):
        energy_step = 0.054
        width = 4096
        two_theta = 3.0
        q_step = (4.0 * np.pi * np.sin(
            np.deg2rad(two_theta / 2.0)) / 12.398) * energy_step
        intensities = np.ones_like(self.cr2o3_q_vals)
        ref_pattern = util.make_diffraction_pattern(self.cr2o3_q_vals,
                                                    intensities, q_step,
                                                    width, sigma=15,
                                                    norm=True)
        poly_coefs_2_deg = [0.00001, 0.95, 20]
        mod_pattern = resample_data(ref_pattern[:, 1], poly_coefs_2_deg)
        poly_coefs = proc.calibrate_q_1d(mod_pattern, self.cr2o3_q_vals,
                                         q_step, fit_order=2,
                                         get_peaks_para=GET_PEAKS_PARA2,
                                         sub_peaks_para=SUB_PEAKS_PARA,
                                         match_peaks_para=Q_MATCH_PARA)
        x_list = np.arange(width)
        list_q = np.arange(width) * q_step
        list_q_mea = np.sum(np.asarray(
            [coef * x_list ** i for i, coef in enumerate(poly_coefs)]),
            axis=0)
        corrected = np.interp(list_q, list_q_mea, mod_pattern,
                              left=0, right=0)
        rmse = np.sqrt(np.mean((corrected - ref_pattern[:, 1]) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_calibrate_q_image(self):
        intensities = np.ones_like(self.cr2o3_q_vals)
        two_theta = 3.0
        energy_step = 0.054
        total_germ_strip = 192
        used_ids = range(96 - 10, 96 + 10, 1)
        start_idx = used_ids[0]
        height = len(used_ids)
        width = 4096
        strip_size = 0.25
        sample_slit = 250.0
        sample_detector = 2000.0
        two_theta_rad = np.deg2rad(two_theta)
        ref_q_image = np.zeros((height, width), dtype=np.float32)
        q_map = np.zeros((height, width), dtype=np.float32)
        q_step_list = []
        for strip_idx in used_ids:
            num1 = (strip_idx - 0.5 * (total_germ_strip - 1)) * strip_size
            num2 = (sample_detector - sample_slit)
            two_theta_idx = (two_theta_rad - np.arctan(num1 / num2))
            q_step = ((4 * np.pi * np.sin(
                two_theta_idx / 2) * energy_step) / 12.398)
            q_step_list.append(q_step)
            results = util.make_diffraction_pattern(self.cr2o3_q_vals,
                                                    intensities, q_step,
                                                    width, sigma=11,
                                                    norm=True)
            q_map[strip_idx - start_idx] = results[:, 0]
            ref_q_image[strip_idx - start_idx] = results[:, 1]

        mod_q_image = np.zeros_like(ref_q_image)
        mod_poly_coef = np.zeros((height, 3), dtype=np.float32)
        np.random.seed(0)
        for i in range(height):
            mod_poly_coef[i, 0] = (np.random.rand() - 0.5) / 1e+5
            mod_poly_coef[i, 1] = 1.0 + 2.0 * (np.random.rand() - 0.5) / 1e+2
            mod_poly_coef[i, 2] = np.random.randint(-8, 8)
            mod_q_image[i] = resample_data(ref_q_image[i], mod_poly_coef[i])

        q_step0 = (4.0 * np.pi * np.sin(
            np.deg2rad(two_theta / 2.0)) / 12.398) * energy_step
        poly_map = proc.calibrate_q_image(mod_q_image, self.cr2o3_q_vals,
                                          q_step0, fit_order=2,
                                          get_peaks_para=GET_PEAKS_PARA2,
                                          sub_peaks_para=SUB_PEAKS_PARA,
                                          match_peaks_para=Q_MATCH_PARA,
                                          return_check=False)
        x_list = np.arange(width)
        list_q = np.arange(width) * q_step0
        corrected_image = np.zeros_like(mod_q_image)
        for k in range(height):
            list_q_mea = np.sum(np.asarray(
                [coef * x_list ** i for i, coef in enumerate(poly_map[k])]),
                axis=0)
            corrected_image[k] = np.interp(list_q, list_q_mea, mod_q_image[k],
                                           left=0, right=0)
        num = 0.0
        for i in range(1, height):
            num += np.mean(np.abs(corrected_image[i] - corrected_image[0]))
        self.assertTrue(num < 0.1)

        # Input is a list of q steps
        poly_map = proc.calibrate_q_image(mod_q_image, self.cr2o3_q_vals,
                                          q_step_list, fit_order=2,
                                          get_peaks_para=GET_PEAKS_PARA2,
                                          sub_peaks_para=SUB_PEAKS_PARA,
                                          match_peaks_para=Q_MATCH_PARA,
                                          return_check=False)
        corrected_image = np.zeros_like(mod_q_image)
        for k in range(height):
            list_q_mea = np.sum(np.asarray(
                [coef * x_list ** i for i, coef in enumerate(poly_map[k])]),
                axis=0)
            corrected_image[k] = np.interp(list_q, list_q_mea, mod_q_image[k],
                                           left=0, right=0)
        num = 0.0
        for i in range(1, height):
            num += np.mean(np.abs(corrected_image[i] - corrected_image[0]))
        self.assertTrue(num < 0.1)
