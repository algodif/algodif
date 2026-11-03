"""
Tests for methods in postprocessing.py

Measured data are simulated by resampling theoretical spectra/patterns,
generated using utility.py, onto a pixel grid whose physical axis (energy or
q) is a known polynomial of the pixel index. Correcting with the same
polynomial must recover the theoretical data on a uniform grid.
"""

import unittest
import numpy as np
from numpy.polynomial import polynomial as poly
import algodif.prep.preprocessing as prep
import algodif.proc.processing as proc
import algodif.post.postprocessing as post
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


class PostprocessingMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Simulated detector image: 32 strips x 1024 channels.
        cls.height, cls.npoint = 8, 1024
        cls.x = np.arange(cls.npoint)
        cls.energy_step = 0.12
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
        cls.mod_spectrum = resample_data(cls.ref_spectrum,
                                         cls.poly_coefs_2_deg)
        cls.cr2o3_d_spacing = np.asarray(
            [3.631, 2.665, 2.48, 2.266, 2.175, 2.048, 1.815, 1.673, 1.465,
             1.432, 1.296, 1.24])
        cls.cr2o3_q_vals = 2 * np.pi / cls.cr2o3_d_spacing
        cls.mat_e_coefs = None
        cls.mat_q_coefs = None
        cls.size = 4

    def test_correct_energy_1d(self):
        poly_coefs = proc.calibrate_energy_1d(self.mod_spectrum,
                                              self.ref_energies,
                                              self.energy_step,
                                              get_peaks_para=GET_PEAKS_PARA,
                                              sub_peaks_para=SUB_PEAKS_PARA,
                                              match_peaks_para=ENERGY_MATCH_PARA)
        corrected = post.correct_energy_1d(self.mod_spectrum, poly_coefs,
                                           self.energy_step)
        self.assertEqual(corrected.shape, self.mod_spectrum.shape)
        rmse = np.sqrt(np.mean((corrected - self.ref_spectrum) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_correct_energy_image(self):
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
                                              match_peaks_para=ENERGY_MATCH_PARA,
                                              return_check=False)
        corrected_image = post.correct_energy_image(mod_e_image, results,
                                                    self.energy_step,
                                                    ncore=None,
                                                    prefer="processes")
        type(self).mat_e_coefs = results
        self.assertEqual(corrected_image.shape, mod_e_image.shape)
        rmse = np.sqrt(np.mean((corrected_image - ref_e_image) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_get_energy_map(self):
        if self.mat_e_coefs is not None:
            e_map = post.get_energy_map(self.height, self.npoint,
                                        self.mat_e_coefs)
            self.assertEqual(e_map.shape, (self.height, self.npoint))

            for i, coefs in enumerate(self.mat_e_coefs):
                self.assertTrue(np.allclose(e_map[i], poly.polyval(self.x, coefs)))
            self.assertRaises(ValueError, post.get_energy_map, self.height,
                              self.npoint, self.mat_e_coefs[:-1])

    def test_correct_q_1d(self):
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

        corrected = post.correct_q_1d(mod_pattern, poly_coefs, q_step)
        rmse = np.sqrt(np.mean((corrected - ref_pattern[:, 1]) ** 2))
        self.assertTrue(rmse < 0.05)

    def test_correct_q_image(self):
        intensities = np.ones_like(self.cr2o3_q_vals)
        two_theta = 3.0
        energy_step = 0.054
        total_germ_strip = 192
        used_ids = range(96 - self.size // 2,
                         96 - self.size // 2 + self.size, 1)
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

        corrected_image = post.correct_q_image(mod_q_image, poly_map, q_step0,
                                               ncore=None, prefer="processes")
        type(self).mat_q_coefs = poly_map
        num = 0.0
        for i in range(1, height):
            num += np.mean(np.abs(corrected_image[i] - corrected_image[0]))
        self.assertTrue(num < 0.1)

    def test_get_q_map(self):
        if self.mat_q_coefs is not None:
            q_map = post.get_q_map(self.size, self.npoint, self.mat_q_coefs)
            self.assertEqual(q_map.shape, (self.size, self.npoint))
            for i, coefs in enumerate(self.mat_q_coefs):
                self.assertTrue(np.allclose(q_map[i], poly.polyval(self.x,
                                                                   coefs)))
            self.assertRaises(ValueError, post.get_q_map, self.height,
                              self.npoint, self.mat_q_coefs[:-1])
