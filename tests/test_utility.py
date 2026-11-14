"""
Tests for methods in utility.py
"""

import io
import unittest
from contextlib import redirect_stdout
import numpy as np
import algodif.util.utility as util


class UtilityMethods(unittest.TestCase):

    def setUp(self):
        self.two_theta = 6.0
        self.total_strip = 192
        self.d_spacing = np.asarray([3.124, 2.706, 1.913, 1.632, 1.041])

    def test_get_fluorescence_lines_from_database(self):
        lines = util.get_fluorescence_lines_from_database(["Cu"])
        self.assertEqual(lines.ndim, 2)
        self.assertEqual(lines.shape[1], 2)
        self.assertTrue(np.all(lines[:, 1] > 0.01))
        # Cu K-alpha1 is the strongest K line, at 8.048 keV.
        self.assertTrue(abs(lines[np.argmax(lines[:, 1]), 0] - 8.048) < 0.01)

        lines_pb = util.get_fluorescence_lines_from_database(["Pb"])
        self.assertTrue(np.any(np.abs(lines_pb[:, 0] - 74.97) < 0.01))
        lines_both = util.get_fluorescence_lines_from_database(["Cu", "Pb"])
        self.assertEqual(len(lines_both), len(lines) + len(lines_pb))

        lines_high = util.get_fluorescence_lines_from_database(
            ["Cu", "Pb"], threshold=0.2)
        self.assertTrue(0 < len(lines_high) <= len(lines_both))
        self.assertTrue(np.all(lines_high[:, 1] > 0.2))

        lines_norm = util.get_fluorescence_lines_from_database(["Cu", "Pb"],
                                                               norm=True)
        self.assertTrue(np.isclose(np.max(lines_norm[:, 1]), 1.0))
        self.assertTrue(np.allclose(lines_norm[:, 0], lines_both[:, 0]))

        with redirect_stdout(io.StringIO()) as output:
            util.get_fluorescence_lines_from_database(["Cu"], check=True)
        self.assertTrue("Cu" in output.getvalue())

        self.assertRaises(ValueError, util.get_fluorescence_lines_from_database,
                          ["Cu"], threshold=10.0)
        self.assertRaises(ValueError, util.get_fluorescence_lines_from_database,
                          ["Xx"])

    def test_make_fluorescence_spectrum(self):
        lines = np.asarray([[10.03, 1.0], [20.0, 0.5]])
        spectrum = util.make_fluorescence_spectrum(lines, 0.1, 300, sigma=0,
                                                   norm=False)
        self.assertEqual(spectrum.shape, (300, 2))
        # Line energies replace the nearest grid values.
        self.assertEqual(spectrum[100, 0], 10.03)
        self.assertTrue(np.isclose(spectrum[200, 0], 20.0))
        self.assertTrue(np.isclose(spectrum[150, 0], 15.0))
        self.assertEqual(spectrum[100, 1], 1.0)
        self.assertEqual(spectrum[200, 1], 0.5)
        self.assertEqual(np.count_nonzero(spectrum[:, 1]), 2)

        spectrum = util.make_fluorescence_spectrum(lines, 0.1, 300, sigma=3,
                                                   norm=False)
        self.assertTrue(np.isclose(np.sum(spectrum[:, 1]), 1.5))
        spectrum = util.make_fluorescence_spectrum(lines, 0.1, 300, sigma=3)
        self.assertTrue(np.isclose(np.max(spectrum[:, 1]), 1.0))
        self.assertEqual(np.argmax(spectrum[:, 1]), 100)

        # Spectrum from database lines peaks at Cu K-alpha1.
        cu_lines = util.get_fluorescence_lines_from_database(["Cu"])
        spectrum = util.make_fluorescence_spectrum(cu_lines, 0.01, 1500,
                                                   sigma=2)
        self.assertTrue(
            abs(spectrum[np.argmax(spectrum[:, 1]), 0] - 8.048) < 0.02)

        spectrum = util.make_fluorescence_spectrum([10.03, 20.0], 0.1, 300,
                                                   sigma=0, norm=False)
        self.assertEqual(spectrum[100, 1], 1.0)
        self.assertEqual(spectrum[200, 1], 1.0)
        self.assertEqual(np.count_nonzero(spectrum[:, 1]), 2)

        self.assertRaises(ValueError, util.make_fluorescence_spectrum,
                          np.ones((2, 2, 2)), 0.1, 300)
        self.assertRaises(ValueError, util.make_fluorescence_spectrum,
                          np.ones((2, 3)), 0.1, 300)

    def test_estimate_energy_germ_detector(self):
        d = 2.0
        # With an odd number of strips, the middle strip sits at two_theta.
        energy = util.estimate_energy_germ_detector(95, d, self.two_theta,
                                                    total_germ_strip=191)
        expected = 12.398 / (2 * d * np.sin(np.deg2rad(self.two_theta) / 2))
        self.assertTrue(np.isclose(energy, expected))

        two_theta_list = util.get_two_theta_list_germ_detector(self.two_theta)
        for idx in [0, 50, 100, 191]:
            energy = util.estimate_energy_germ_detector(idx, d, self.two_theta)
            expected = 12.398 / (
                    2 * d * np.sin(np.deg2rad(two_theta_list[idx]) / 2))
            self.assertTrue(np.isclose(energy, expected))

        energies = [util.estimate_energy_germ_detector(i, d, self.two_theta)
                    for i in range(self.total_strip)]
        self.assertTrue(np.all(np.diff(energies) > 0.0))

    def test_estimate_energy_full_germ_detector(self):
        energies = util.estimate_energy_full_germ_detector(self.d_spacing,
                                                           self.two_theta)
        self.assertEqual(energies.shape,
                         (self.total_strip, len(self.d_spacing)))
        for idx in [0, 95, 191]:
            expected = [util.estimate_energy_germ_detector(
                idx, d, self.two_theta) for d in self.d_spacing]
            self.assertTrue(np.allclose(energies[idx], expected))

        energies = util.estimate_energy_full_germ_detector(
            self.d_spacing, self.two_theta, strip_size=0.5,
            total_germ_strip=64, sample_slit=300.0, sample_detector=1500.0)
        self.assertEqual(energies.shape, (64, len(self.d_spacing)))
        expected = [util.estimate_energy_germ_detector(
            10, d, self.two_theta, strip_size=0.5, total_germ_strip=64,
            sample_slit=300.0, sample_detector=1500.0) for d in self.d_spacing]
        self.assertTrue(np.allclose(energies[10], expected))

    def test_estimate_gauge_volume_germ_detector(self):
        z_size = util.estimate_gauge_volume_germ_detector(50, self.two_theta)
        self.assertTrue(np.isscalar(z_size) and z_size > 0.0)

        z_size, z_value = util.estimate_gauge_volume_germ_detector(
            50, self.two_theta, return_z=True)
        _, z_value_next = util.estimate_gauge_volume_germ_detector(
            51, self.two_theta, return_z=True)
        self.assertTrue(np.isclose(z_size, abs(z_value - z_value_next)))

        # Strip at the nominal angle sees the nominal sample position.
        _, z_value = util.estimate_gauge_volume_germ_detector(
            95, self.two_theta, total_germ_strip=191, return_z=True)
        self.assertTrue(abs(z_value) < 1e-9)

        # Small-angle estimate: dz ~ slit * d(2theta) / sin(2theta).
        z_size = util.estimate_gauge_volume_germ_detector(
            95, self.two_theta, total_germ_strip=191)
        expected = 250.0 * (0.25 / 1750.0) / np.sin(
            np.deg2rad(self.two_theta))
        self.assertTrue(abs(z_size - expected) / expected < 0.01)

    def test_estimate_gauge_volume_full_germ_detector(self):
        z_sizes = util.estimate_gauge_volume_full_germ_detector(
            self.two_theta)
        self.assertEqual(z_sizes.shape, (self.total_strip,))
        self.assertTrue(np.all(z_sizes > 0.0))

        z_sizes, z_values = util.estimate_gauge_volume_full_germ_detector(
            self.two_theta, return_z=True)
        self.assertEqual(z_values.shape, (self.total_strip,))
        for idx in [0, 95, 191]:
            z_size, z_value = util.estimate_gauge_volume_germ_detector(
                idx, self.two_theta, return_z=True)
            self.assertTrue(np.isclose(z_sizes[idx], z_size))
            self.assertTrue(np.isclose(z_values[idx], z_value))
        self.assertTrue(np.allclose(z_sizes[:-1],
                                    np.abs(np.diff(z_values))))

        z_sizes = util.estimate_gauge_volume_full_germ_detector(
            self.two_theta, total_germ_strip=64)
        self.assertEqual(z_sizes.shape, (64,))

    def test_estimate_q_step_germ_detector(self):
        # q = 2 * pi / d at the energy diffracted by d-spacing on that strip.
        for idx in [0, 95, 191]:
            for d in self.d_spacing:
                energy = util.estimate_energy_germ_detector(idx, d,
                                                            self.two_theta)
                q_val = util.estimate_q_step_germ_detector(idx, energy,
                                                           self.two_theta)
                self.assertTrue(np.isclose(q_val, 2 * np.pi / d))

        q_step1 = util.estimate_q_step_germ_detector(50, 0.05, self.two_theta)
        q_step2 = util.estimate_q_step_germ_detector(50, 0.10, self.two_theta)
        self.assertTrue(np.isclose(q_step2, 2 * q_step1))

    def test_get_two_theta_list_germ_detector(self):
        two_theta_list = util.get_two_theta_list_germ_detector(self.two_theta)
        self.assertEqual(two_theta_list.shape, (self.total_strip,))
        self.assertTrue(np.all(np.diff(two_theta_list) < 0.0))
        # Strips are symmetric around the nominal angle.
        self.assertTrue(np.isclose(np.mean(two_theta_list), self.two_theta))
        half_span = np.rad2deg(np.arctan(95.5 * 0.25 / 1750.0))
        self.assertTrue(np.isclose(two_theta_list[0],
                                   self.two_theta + half_span))
        self.assertTrue(np.isclose(two_theta_list[-1],
                                   self.two_theta - half_span))

        two_theta_list = util.get_two_theta_list_germ_detector(
            self.two_theta, total_germ_strip=191)
        self.assertTrue(np.isclose(two_theta_list[95], self.two_theta))

    def test_list_xrd_calibrants(self):
        names = util.list_xrd_calibrants()
        self.assertEqual(names, sorted(names))
        for name in ("Al2O3", "CeO2", "Cr2O3", "LaB6", "Si"):
            self.assertTrue(name in names)
        self.assertTrue(all(not name.endswith(".csv") for name in names))

    def test_get_calibrant_diffraction_pattern(self):
        f_alias = util.get_calibrant_diffraction_pattern
        # Cr2O3 reflections with intensity above 35 (strongest peak = 100).
        d_expected = np.asarray([3.631895, 2.665959, 2.480350, 1.815948,
                                 1.673162, 1.465227, 1.432031])
        i_expected = np.asarray([59.26184, 91.69583, 87.15836, 41.18903,
                                 100.0, 36.89734, 48.72557])
        results = f_alias("Cr2O3", threshold=35, check=False)
        self.assertTrue(np.allclose(results[:, 0], 2 * np.pi / d_expected))
        self.assertTrue(np.allclose(results[:, 1], i_expected / 100.0))
        self.assertTrue(np.all(np.diff(results[:, 0]) > 0))

        for name in ("cr2o3", "CR2O3", " Cr2O3 "):
            self.assertTrue(np.allclose(
                f_alias(name, threshold=35, check=False), results))

        results = f_alias("Cr2O3", threshold=35, norm=False, check=False)
        self.assertTrue(np.allclose(results[:, 1], i_expected))

        results = f_alias("Cr2O3", num_use=3, threshold=35, check=False)
        self.assertTrue(np.allclose(results[:, 0], 2 * np.pi / d_expected[:3]))
        self.assertTrue(np.isclose(np.max(results[:, 1]), 1.0))

        # Defaults: the 15 largest d-spacings with intensity above 10.
        d_default = np.asarray([3.631895, 2.665959, 2.480350, 2.175912,
                                1.815948, 1.673162, 1.465227, 1.432031,
                                1.296496, 1.240175, 1.148997, 1.124462,
                                1.087956, 1.042567, 0.896186])
        results = f_alias("Cr2O3", check=False)
        self.assertTrue(np.allclose(results[:, 0], 2 * np.pi / d_default))

        results = f_alias("Cr2O3", threshold=100, check=False)
        self.assertEqual(results.shape, (0, 2))

        results = f_alias("Cr2O3", num_use=None, threshold=0, norm=False,
                          check=False)
        self.assertEqual(len(np.unique(results, axis=0)), len(results))
        self.assertTrue(np.isclose(np.max(results[:, 1]), 100.0))

        for name in util.list_xrd_calibrants():
            results = f_alias(name, check=False)
            self.assertTrue(len(results) > 0)
            self.assertTrue(np.all(results[:, 0] > 0))
            self.assertTrue(np.isclose(np.max(results[:, 1]), 1.0))

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            f_alias("Cr2O3", threshold=35, check=True)
        lines = buffer.getvalue().strip().splitlines()
        self.assertEqual(len(lines), len(d_expected))
        self.assertTrue("q-value" in lines[0] and "d-spacing" in lines[0])

        self.assertRaises(ValueError, f_alias, "Unobtainium")

    def test_make_diffraction_pattern(self):
        q_step = 0.01
        npoint = 300
        pattern = util.make_diffraction_pattern([[1.003, 1.0], [2.0, 0.5]],
                                                q_step, npoint, sigma=0,
                                                norm=False)
        self.assertEqual(pattern.shape, (300, 2))
        # Unlike the fluorescence spectrum, the q grid stays uniform.
        self.assertTrue(np.allclose(pattern[:, 0], np.arange(npoint) * q_step))
        self.assertEqual(pattern[100, 1], 1.0)
        self.assertEqual(pattern[200, 1], 0.5)
        self.assertEqual(np.count_nonzero(pattern[:, 1]), 2)

        pattern = util.make_diffraction_pattern([[1.0, 1.0], [2.0, 0.5]],
                                                q_step, npoint, sigma=3)
        self.assertTrue(np.isclose(np.max(pattern[:, 1]), 1.0))
        self.assertEqual(np.argmax(pattern[:, 1]), 100)

        pattern = util.make_diffraction_pattern([1.0, 2.0], q_step, npoint,
                                                sigma=0, norm=False)
        self.assertEqual(pattern[100, 1], 1.0)
        self.assertEqual(pattern[200, 1], 1.0)
        self.assertEqual(np.count_nonzero(pattern[:, 1]), 2)

        self.assertRaises(ValueError, util.make_diffraction_pattern,
                          [[1.0, 1.0, 1.0]], q_step, npoint)
        self.assertRaises(ValueError, util.make_diffraction_pattern,
                          np.ones((2, 2, 2)), q_step, npoint)

    def test_q_to_two_theta(self):
        two_theta = util.q_to_two_theta(2.5)
        self.assertTrue(isinstance(two_theta, float))
        self.assertTrue(np.isclose(two_theta, 90.0))

        q = np.linspace(0.0, 4.0, 9)
        two_theta = util.q_to_two_theta(q, max_two_theta=60.0)
        self.assertEqual(two_theta.shape, q.shape)
        self.assertTrue(np.isclose(two_theta[0], 0.0))
        self.assertTrue(np.isclose(two_theta[-1], 60.0))
        self.assertTrue(np.all(np.diff(two_theta) > 0))
        # q / sin(theta) must stay constant for a single wavelength.
        ratio = q[1:] / np.sin(np.radians(two_theta[1:] / 2))
        self.assertTrue(np.allclose(ratio, ratio[0]))

        q_map = np.array([[1.0, 2.0], [3.0, 4.0]])
        two_theta, energy = util.q_to_two_theta(q_map, return_energy=True)
        self.assertEqual(two_theta.shape, (2, 2))
        self.assertTrue(np.isclose(two_theta[1, 1], 90.0))
        self.assertTrue(isinstance(energy, float))
        wavelength = 4 * np.pi * np.sin(np.radians(45.0)) / 4.0
        self.assertTrue(np.isclose(energy, 12.398 / wavelength))
        two_theta_row = util.q_to_two_theta(q_map[0])
        self.assertTrue(np.isclose(two_theta_row[-1], 90.0))
        self.assertFalse(np.isclose(two_theta[0, 1], 90.0))

        self.assertRaises(ValueError, util.q_to_two_theta, np.ones((2, 2, 2)))
        self.assertRaises(ValueError, util.q_to_two_theta, [])
        self.assertRaises(ValueError, util.q_to_two_theta, [1.0, -1.0])
        self.assertRaises(ValueError, util.q_to_two_theta, [1.0, np.nan])
        self.assertRaises(ValueError, util.q_to_two_theta, [0.0, 0.0])
        self.assertRaises(ValueError, util.q_to_two_theta, 1.0, 0.0)
        self.assertRaises(ValueError, util.q_to_two_theta, 1.0, 180.0)
        self.assertRaises(ValueError, util.q_to_two_theta, 1.0, np.inf)
