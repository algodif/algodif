import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig
from scipy import sparse
from scipy.linalg import cholesky
from scipy.sparse.linalg import spsolve
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
    fluorescent_lines : array_like,
       Shape of (N, 2). Array containing energy and intensity values for
       theoretical fluorescent lines. The first row must be energies, the
       second row must be corresponding intensities.
    energy_step : float
       Step size between energy bins. Unit is keV.
    data_point : int
       Total number of points in the generated spectrum.
    sigma : float, optional
       Standard deviation for optional Gaussian smoothing.

    Returns
    -------
    array_like
       Shape of (data_point, 2). Simulated spectrum as an array of
       [energy, intensity] pairs.
    """

    if len(fluorescent_lines.shape) != 2:
        raise ValueError("Wrong input! Check API document")
    if fluorescent_lines.shape[1] != 2:
        raise ValueError("Shape of the input must be (N, 2)")
    theo_energies = fluorescent_lines[:, 0]
    theo_intensities = fluorescent_lines[:, 1]
    inter_energies = np.arange(data_point) * energy_step
    inter_intensities = np.zeros_like(inter_energies)
    for i, energy in enumerate(theo_energies):
        nearest_idx = np.argmin(np.abs(energy - inter_energies))
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


def make_diffraction_pattern(list_q_theory, list_intensity, q_step, data_point,
                             sigma=3, norm=True):
    """
    Generate a diffraction pattern on a uniform q grid.

    Parameters
    ----------
    list_q_theory : array_like
        One-dimensional array of theoretical peak q positions.
    list_intensity : array_like
        Corresponding peak intensities.
    q_step : float
        Positive grid spacing, in the same units as list_q_theory.
    data_point : int
        Number of grid points, starting at q = 0.
    sigma : float, optional
        Gaussian smoothing sigma in grid samples. No smoothing if <= 0.
    norm : bool, optional
        Normalize the maximum intensity to one.

    Returns
    -------
    numpy.ndarray
        Array of shape (data_point, 2), containing q values and intensities.

    Notes
    -----
    Peaks are assigned to the nearest grid point. Peaks outside the grid
    are assigned to an endpoint. If peaks share a grid point, the last
    intensity replaces earlier values. Normalization requires a nonzero
    maximum intensity.
    """
    q_list = np.arange(data_point) * q_step
    list_inten = np.zeros_like(q_list)
    if len(list_q_theory) != len(list_intensity):
        raise ValueError("Number of q-values and intensities are not the same")
    for i, q in enumerate(list_q_theory):
        pos = np.argmin(np.abs(q - q_list))
        list_inten[pos] = list_intensity[i]
    if sigma > 0:
        list_inten = ndi.gaussian_filter1d(list_inten, sigma)
    if norm:
        list_inten = list_inten / np.max(list_inten)
    return np.asarray(list(zip(q_list, list_inten)))
