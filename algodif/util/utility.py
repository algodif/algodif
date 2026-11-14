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
# Description: Python module for utility methods.
# Contributors:
# ============================================================================


import warnings
from importlib.resources import files
import numpy as np
import scipy.ndimage as ndi
import xraydb


def get_fluorescence_lines_from_database(elements, line_type="K",
                                         threshold=0.01, norm=False,
                                         check=False):
    """
    Retrieve fluorescence line energies and intensities for a list of elements.

    Parameters
    ----------
    elements : list of str
        List of chemical element symbols, e.g., ["Sn", "Au", "Pb", "U"].
    line_type : str, optional
        Type of X-ray line to retrieve (e.g., 'K', 'L1',...).
    threshold : float, optional
        Minimum intensity threshold for including a line
    norm : bool, optional
        If True, normalize the intensities.
    check : bool, optional
        If True, print detailed information for each retrieved line.

    Returns
    -------
    array_like
        2D array of shape (N, 2), each row is [energy_keV, relative_intensity].
    """
    energy_intensity_list = []
    for element in elements:
        try:
            lines = xraydb.xray_lines(element, line_type)
        except Exception as error:
            raise ValueError(f"Could not retrieve for element '{element}' "
                             f"with line type '{line_type}'. Error {error}")
        for name, line in lines.items():
            energy = line.energy / 1000.0  # Convert from eV to keV
            intensity = line.intensity
            if intensity > threshold:
                energy_intensity_list.append([energy, intensity])
                if check:
                    print(f"Element: {element:<2}  Line: {name:<6}"
                          f" Energy: {energy:8.4f} keV  "
                          f"Intensity: {intensity:.4f}")
    if not energy_intensity_list:
        raise ValueError(f"No {line_type} lines with intensity > {threshold} "
                         f"found for the given elements {elements}")
    energy_intensity_list = np.asarray(energy_intensity_list)
    if norm:
        list_tmp = energy_intensity_list[:, 1]
        energy_intensity_list[:, 1] = list_tmp / np.max(list_tmp)
    return energy_intensity_list


def make_fluorescence_spectrum(fluorescent_lines, energy_step, data_point,
                               sigma=3, norm=True):
    """
    Generate a discrete fluorescence spectrum from theoretical lines.

    Parameters
    ----------
    fluorescent_lines : array_like
       Theoretical fluorescent lines. Either shape (N, 2) with rows of
       [energy (keV), intensity], e.g. the output of
       get_fluorescence_lines_from_database, or a 1D array of energies, in
       which case all intensities are 1.
    energy_step : float
       Step size between energy bins. Unit is keV.
    data_point : int
       Total number of points in the generated spectrum.
    sigma : float, optional
       Standard deviation for optional Gaussian smoothing.
    norm : bool, optional
       Normalize the maximum intensity to one.

    Returns
    -------
    array_like
       Shape of (data_point, 2). Simulated spectrum as an array of
       [energy, intensity] pairs.
    """
    fluorescent_lines = np.asarray(fluorescent_lines, dtype=float)
    if fluorescent_lines.ndim == 1:
        theo_energies = fluorescent_lines
        theo_intensities = np.ones_like(theo_energies)
    elif fluorescent_lines.ndim == 2 and fluorescent_lines.shape[1] == 2:
        theo_energies = fluorescent_lines[:, 0]
        theo_intensities = fluorescent_lines[:, 1]
    else:
        raise ValueError("Input must be a 1D array of energies or an (N, 2) "
                         "array of [energy, intensity]")
    inter_energies = np.arange(data_point) * energy_step
    max_inter_e = inter_energies[-1]
    max_theo_e = np.max(theo_energies)
    if max_inter_e < max_theo_e:
        warnings.warn(f"\n\nThe generated spectrum's maximum energy {max_inter_e:3f} keV"
                      f" is below the theoretical maximum {max_theo_e:3f} keV\n"
                      f"with energy step {energy_step:3f} and {data_point} data"
                      f" points. Use an energy step of at least "
                      f"{max_theo_e / (data_point - 1):.4f}.\n")

    inter_intensities = np.zeros_like(inter_energies)
    for i, energy in enumerate(theo_energies):
        diff = np.abs(energy - inter_energies)
        min_diff = np.min(diff)
        if min_diff <= energy_step:
            nearest_idx = np.argmin(diff)
            inter_energies[nearest_idx] = energy
            inter_intensities[nearest_idx] = theo_intensities[i]
    if sigma > 0:
        inter_intensities = ndi.gaussian_filter1d(inter_intensities, sigma)
    if norm:
        inter_intensities = inter_intensities / np.max(inter_intensities)
    return np.asarray(list(zip(inter_energies, inter_intensities)))


def estimate_energy_germ_detector(strip_idx, d_spacing, two_theta,
                                  strip_size=0.25, total_germ_strip=192,
                                  sample_slit=250.0, sample_detector=2000.0):
    """
    Estimating energy given the d-spacing of a standard sample and the geometry
    information of the GeRM detector.

    Parameters
    ----------
    strip_idx : int
        Zero-based strip index.
    d_spacing : float
        Lattice spacing in angstroms.
    two_theta : float
        Nominal scattering angle at the detector center, in degrees.
    strip_size : float, optional
        Strip pitch in mm.
    total_germ_strip : int, optional
        Total number of detector strips.
    sample_slit : float, optional
        Sample-to-slit distance in mm.
    sample_detector : float, optional
        Sample-to-detector distance in mm.

    Returns
    -------
    float
        Estimated energy in keV.
    """
    two_theta_rad = np.deg2rad(two_theta)
    num1 = (strip_idx - 0.5 * (total_germ_strip - 1)) * strip_size
    num2 = (sample_detector - sample_slit)
    two_theta_guess = (two_theta_rad - np.arctan(num1 / num2))
    est_energy = 12.398 / (2 * d_spacing * np.sin(two_theta_guess / 2.0))
    return est_energy


def estimate_energy_full_germ_detector(d_spacing_values, two_theta,
                                       strip_size=0.25, total_germ_strip=192,
                                       sample_slit=250.0,
                                       sample_detector=2000.0):
    """
    Estimating energy of each GeRM detector strip given the list of d-spacing
    of a standard sample.

    Parameters
    ----------
    d_spacing_values : array_like
        1d array of lattice spacings in angstroms.
    two_theta : float
        Nominal scattering angle at the detector center, in degrees.
    strip_size : float, optional
        Strip pitch in mm.
    total_germ_strip : int, optional
        Total number of detector strips.
    sample_slit : float, optional
        Sample-to-slit distance in mm.
    sample_detector : float, optional
        Sample-to-detector distance in mm.

    Returns
    -------
    numpy.ndarray
        Energies in keV, with shape (total_germ_strip, len(d_spacing_values)).
    """
    two_theta_rad = np.deg2rad(two_theta)
    est_energy_2d = []
    for strip_idx in range(total_germ_strip):
        num1 = (strip_idx - 0.5 * (total_germ_strip - 1)) * strip_size
        num2 = (sample_detector - sample_slit)
        two_theta_guess = (two_theta_rad - np.arctan(num1 / num2))
        list_est_energy = 12.398 / (2 * np.asarray(d_spacing_values) * np.sin(
            two_theta_guess / 2.0))
        est_energy_2d.append(list_est_energy)
    return np.asarray(est_energy_2d)


def estimate_gauge_volume_germ_detector(strip_idx, two_theta, strip_size=0.25,
                                        total_germ_strip=192,
                                        sample_slit=250.0,
                                        sample_detector=2000.0,
                                        return_z=False):
    """
    Estimating z-resolution (gauge volume) of the GeRM detector given the
    strip index and its geometry.

    Parameters
    ----------
    strip_idx : int
        Zero-based strip index.
    two_theta : float
        Nominal scattering angle at the detector center, in degrees.
    strip_size : float, optional
        Strip pitch in mm.
    total_germ_strip : int, optional
        Total number of detector strips.
    sample_slit : float, optional
        Sample-to-slit distance in mm.
    sample_detector : float, optional
        Sample-to-detector distance in mm.
    return_z : bool, optional
        Return the z-position associated with strip_idx if True.

    Returns
    -------
    z_size : float
        z-separation between rays for strip_idx and strip_idx + 1, in mm.
    z_value : float
        z-position for strip_idx, in mm relative to the nominal
        sample position. Returned if return_z is True.
    """
    two_theta_rad = np.deg2rad(two_theta)
    num0 = (sample_detector - sample_slit)
    num1 = (strip_idx - 0.5 * (total_germ_strip - 1)) * strip_size
    num2 = (strip_idx + 1 - 0.5 * (total_germ_strip - 1)) * strip_size
    two_theta1 = (two_theta_rad - np.arctan(num1 / num0))
    two_theta2 = (two_theta_rad - np.arctan(num2 / num0))
    length = sample_slit * np.sin(two_theta_rad)
    z_size = np.abs(length / np.tan(two_theta1) - length / np.tan(two_theta2))
    if return_z:
        z_value = sample_slit * np.cos(two_theta_rad) - (
                    sample_slit * np.sin(two_theta_rad)) / np.tan(two_theta1)
        return z_size, z_value
    else:
        return z_size


def estimate_gauge_volume_full_germ_detector(two_theta,
                                             strip_size=0.25,
                                             total_germ_strip=192,
                                             sample_slit=250.0,
                                             sample_detector=2000.0,
                                             return_z=False):
    """
    Estimate z-resolutions (gauge volume) for all GeRM detector strips.

    Parameters
    ----------
    two_theta : float
        Nominal scattering angle at the detector center, in degrees.
    strip_size : float, optional
        Strip pitch in mm.
    total_germ_strip : int, optional
        Total number of detector strips.
    sample_slit : float, optional
        Sample-to-slit distance in mm.
    sample_detector : float, optional
        Sample-to-detector distance in mm.
    return_z : bool, optional
        Return the z position associated with each strip if True.

    Returns
    -------
    z_sizes : numpy.ndarray
        z-resolutions in mm, with shape (total_germ_strip,).
    z_values : numpy.ndarray
        z-positions in mm relative to the nominal sample position,
        with shape (total_germ_strip,). Returned if return_z is True.
    """
    list_gauge_volume = []
    list_z_value = []
    for strip_idx in range(total_germ_strip):
        z_size, z_val = estimate_gauge_volume_germ_detector(
            strip_idx, two_theta, strip_size=strip_size,
            total_germ_strip=total_germ_strip, sample_slit=sample_slit,
            sample_detector=sample_detector, return_z=True)
        list_gauge_volume.append(z_size)
        list_z_value.append(z_val)
    if return_z:
        return np.asarray(list_gauge_volume), np.asarray(list_z_value)
    else:
        return np.asarray(list_gauge_volume)


def estimate_q_step_germ_detector(strip_idx, energy_step, two_theta,
                                  total_germ_strip=192,
                                  strip_size=0.25, sample_slit=250.0,
                                  sample_detector=2000.0):
    """
    Estimate the scattering-vector step for a GeRM detector strip.

    Parameters
    ----------
    strip_idx : int
        Zero-based strip index.
    energy_step : float
        Energy increment in keV.
    two_theta : float
        Nominal scattering angle at the detector center, in degrees.
    total_germ_strip : int, optional
        Total number of detector strips.
    strip_size : float, optional
        Strip pitch in mm.
    sample_slit : float, optional
        Sample-to-slit distance in mm.
    sample_detector : float, optional
        Sample-to-detector distance in mm.

    Returns
    -------
    float
        Scattering-vector increment in inverse angstroms.
    """
    two_theta_rad = np.deg2rad(two_theta)
    num1 = (strip_idx - 0.5 * (total_germ_strip - 1)) * strip_size
    num2 = (sample_detector - sample_slit)
    two_theta_guess = two_theta_rad - np.arctan(num1 / num2)
    q_step = 4 * np.pi * np.sin(two_theta_guess / 2) * energy_step / 12.398
    return q_step


def estimate_q_step_full_germ_detector(energy_step, two_theta,
                                       total_germ_strip=192,
                                       strip_size=0.25, sample_slit=250.0,
                                       sample_detector=2000.0):
    """
    Estimate the scattering-vector step for all GeRM detector strips.

    Parameters
    ----------
    energy_step : float
        Energy increment in keV.
    two_theta : float
        Nominal scattering angle at the detector center, in degrees.
    total_germ_strip : int, optional
        Total number of detector strips.
    strip_size : float, optional
        Strip pitch in mm.
    sample_slit : float, optional
        Sample-to-slit distance in mm.
    sample_detector : float, optional
        Sample-to-detector distance in mm.

    Returns
    -------
    float
        Scattering-vector increment in inverse angstroms.
    """
    list_q_step = []
    for i in range(total_germ_strip):
        q_step = estimate_q_step_germ_detector(
            i, energy_step, two_theta, total_germ_strip=total_germ_strip,
            strip_size=strip_size, sample_slit=sample_slit,
            sample_detector=sample_detector)
        list_q_step.append(q_step)
    return np.asarray(list_q_step)


def get_two_theta_list_germ_detector(two_theta, total_germ_strip=192,
                                     strip_size=0.25, sample_slit=250.0,
                                     sample_detector=2000.0):
    """
    Calculate scattering angles for all GeRM detector strips.

    Parameters
    ----------
    two_theta : float
        Nominal scattering angle at the detector center, in degrees.
    total_germ_strip : int, optional
        Total number of detector strips.
    strip_size : float, optional
        Strip pitch in mm.
    sample_slit : float, optional
        Sample-to-slit distance in mm.
    sample_detector : float, optional
        Sample-to-detector distance in mm.

    Returns
    -------
    numpy.ndarray
        Scattering angles in degrees, ordered by strip index.
    """
    two_theta_list = []
    slit_det_distance = sample_detector - sample_slit
    for idx in range(int(total_germ_strip)):
        dis = (idx - 0.5 * (total_germ_strip - 1)) * strip_size
        two_theta_list.append(
            two_theta - np.rad2deg(np.arctan(dis / slit_det_distance)))
    return np.asarray(two_theta_list)


def list_xrd_calibrants():
    """
    List the X-ray diffraction calibrants shipped with algodif.

    Returns
    -------
    list of str
        Sorted calibrant names, e.g. ["Al2O3", "CeO2", "Cr2O3", "LaB6", "Si"].
        Each name can be passed to get_calibrant_diffraction_pattern.
    """
    folder = files("algodif").joinpath("calibrants")
    return sorted(item.name[:-len(".csv")] for item in folder.iterdir()
                  if item.is_file() and item.name.endswith(".csv"))


def get_calibrant_diffraction_pattern(name, num_use=15, threshold=10,
                                      norm=True, check=True):
    """
    Load and select reflections from the reference diffraction pattern of a
    calibrant bundled with algodif.

    Parameters
    ----------
    name : str
        Calibrant name (case-insensitive), e.g. "Cr2O3" or "cr2o3". Use
        list_xrd_calibrants to see the available names.
    num_use : int, optional
        Maximum number of reflections to return (default: 15).
    threshold : float, optional
        Minimum intensity, on a scale where the strongest peak is 100, for
        peak selection (default: 10).
    norm : bool, optional
        Scale intensities so the maximum selected one is 1 if True, otherwise
        keep the 0-100 scale (default: True).
    check : bool, optional
        Print output if True (default: True).

    Returns
    -------
    numpy.ndarray
        2D array of shape (N, 2). Each row is [q (1/Angstrom), intensity],
        sorted by ascending q (i.e. decreasing d-spacing).
    """
    available = list_xrd_calibrants()
    lookup = {item.casefold(): item for item in available}
    file_name = lookup.get(str(name).strip().casefold())
    if file_name is None:
        raise ValueError(f"Unknown calibrant {name!r}. "
                         f"Available: {', '.join(available)}")
    resource = (files("algodif").joinpath("calibrants")
                .joinpath(f"{file_name}.csv"))
    with resource.open("r", encoding="utf-8-sig") as file:
        data = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(0, 1),
                          ndmin=2)
    data = np.unique(data[data[:, 1] > threshold], axis=0)[:num_use]
    q_values, intensity = data[:, 0], data[:, 1]
    if norm and intensity.size > 0:
        intensity = intensity / np.max(intensity)
    if check:
        d_spacing = 2 * np.pi / q_values
        for i, q in enumerate(q_values):
            print(f"{i + 1: 2}. q-value {q: 0.3f}. "
                  f"d-spacing: {d_spacing[i]: 0.3f}. "
                  f"Intensity {intensity[i]: 0.3f}")
    return np.column_stack((q_values, intensity))


def make_diffraction_pattern(diffraction_peaks, q_step, data_point, sigma=3,
                             norm=True):
    """
    Generate a diffraction pattern on a uniform q grid.

    Parameters
    ----------
    diffraction_peaks : array_like
        Theoretical peaks. Either shape (N, 2) with rows of
        [q (1/Angstrom), intensity], e.g. the output of
        get_calibrant_diffraction_pattern, or a 1D array of q values, in
        which case all intensities are 1.
    q_step : float
        Positive grid spacing, in the same units as the q values.
    data_point : int
        Number of grid points, starting at q = 0.
    sigma : float, optional
        Gaussian smoothing sigma in grid samples. No smoothing if <= 0.
    norm : bool, optional
        Normalize the maximum intensity to one.

    Returns
    -------
    numpy.ndarray
        Shape of (data_point, 2). Simulated pattern as [q, intensity] pairs.

    Notes
    -----
    Peaks are assigned to the nearest grid point; peaks more than q_step
    beyond the grid are dropped. If peaks share a grid point, the last
    intensity replaces earlier values. Normalization requires a nonzero
    maximum intensity.
    """
    diffraction_peaks = np.asarray(diffraction_peaks, dtype=float)
    if diffraction_peaks.ndim == 1:
        list_q_theory = diffraction_peaks
        list_intensity = np.ones_like(list_q_theory)
    elif diffraction_peaks.ndim == 2 and diffraction_peaks.shape[1] == 2:
        list_q_theory = diffraction_peaks[:, 0]
        list_intensity = diffraction_peaks[:, 1]
    else:
        raise ValueError("Input must be a 1D array of q values or an (N, 2) "
                         "array of [q, intensity]")
    q_list = np.arange(data_point) * q_step
    list_inten = np.zeros_like(q_list)
    max_inter_q = q_list[-1]
    max_theo_q = np.max(list_q_theory)
    if max_inter_q < max_theo_q:
        warnings.warn(f"\n\nThe generated pattern's maximum q {max_inter_q:3f} (1/A)"
                      f" is below the theoretical maximum {max_theo_q:3f} (1/A)\n"
                      f"with q-step {q_step:3f} and {data_point} data points. "
                      f"Use a q-step of at least "
                      f"{max_theo_q / (data_point - 1):.4f}.\n")
    for i, q in enumerate(list_q_theory):
        diff = np.abs(q - q_list)
        min_diff = np.min(diff)
        if min_diff <= q_step:
            pos = np.argmin(diff)
            list_inten[pos] = list_intensity[i]
    if sigma > 0:
        list_inten = ndi.gaussian_filter1d(list_inten, sigma)
    if norm:
        list_inten = list_inten / np.max(list_inten)
    return np.asarray(list(zip(q_list, list_inten)))


def q_to_two_theta(q, max_two_theta=90.0, return_energy=False):
    """Convert q to equivalent 2theta using a specified maximum angle.

    Parameters
    ----------
    q : float or array_like
        Finite, nonnegative q values (2*pi/d) in angstrom^-1.
        Accepts a scalar, 1D array, or 2D array; at least one value
        must be positive.
    max_two_theta : float, optional
        Angle assigned to the largest q, in degrees, strictly between
        0 and 180. Default is 90.
    return_energy : bool, optional
        Also return the reference energy in keV. Default is False.

    Returns
    -------
    two_theta : float or numpy.ndarray
        Equivalent angles in degrees, matching the input shape.
    energy : float, optional
        Reference energy in keV, returned if return_energy is True.

    Notes
    -----
    A scalar q maps to max_two_theta. Pass the full q-map to use
    one reference energy for all rows. Intensities are unchanged.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim > 2:
        raise ValueError("q must be a scalar, 1D array, or 2D array.")
    if q.size == 0 or np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("q must contain finite, nonnegative values.")

    max_two_theta = float(max_two_theta)
    if not np.isfinite(max_two_theta) or not 0 < max_two_theta < 180:
        raise ValueError("max_two_theta must be between 0 and 180 degrees.")

    q_max = np.max(q)
    if q_max == 0:
        raise ValueError("q must contain at least one positive value.")

    sin_theta_max = np.sin(np.radians(max_two_theta / 2.0))
    two_theta = np.degrees(2.0 * np.arcsin((q / q_max) * sin_theta_max))
    if q.ndim == 0:
        two_theta = float(two_theta)

    if return_energy:
        energy = 12.398 * q_max / (4.0 * np.pi * sin_theta_max)
        return two_theta, float(energy)
    return two_theta
