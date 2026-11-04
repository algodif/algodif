"""
Generate the figures used in the usage demonstrations (docs/source/usage).

The workflows are the same as the scripts in the "examples" folder, but
figures are saved to docs/source/usage/figs/demo_xx/ instead of being shown.

Usage:
    python docs/make_usage_figures.py            # all demos
    python docs/make_usage_figures.py 2 4        # selected demos
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import algodif.io.loadersaver as losa
import algodif.prep.preprocessing as prep
import algodif.proc.processing as proc
import algodif.post.postprocessing as post
import algodif.util.utility as util

DATA = ROOT / "data"
FIG_BASE = ROOT / "docs" / "source" / "usage" / "figs"
FIG_SIZE = (12, 6)
DPI = 80

ENERGY_FILE1 = DATA / "Pb_Au_Sn_calibration.tif"
ENERGY_FILE2 = DATA / "U_calibration.tif"
Q_FILE = DATA / "Cr2O3_6_degree_GeRM.tif"
PEARSON_FILE = DATA / "Pearson_Crystal_Data_Cr2O3.csv"
ELEMENTS = ["Sn", "Au", "Pb", "U"]
ENERGY_BIN = 0.0538357407779  # keV
TWO_THETA = 6.0  # Degree
GEOMETRY = {"total_germ_strip": 192, "strip_size": 0.2, "sample_slit": 200.0,
            "sample_detector": 2000.0}
SVD_PARA = {"rank": 5, "lambda_threshold": 0.5}


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------

def save_fig(fig, demo, name):
    file_path = FIG_BASE / demo / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    try:
        fig.savefig(file_path, dpi=DPI)
    except IOError:
        print(f"Couldn't write to file {file_path}")
        raise
    plt.close(fig)
    print(f"Saved: {file_path}")


def save_log_image(demo, name, mat, title, x_label="Pixel index",
                   extent=None):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    img = ax.imshow(np.log(mat + 1), aspect="auto", cmap="inferno",
                    extent=extent)
    fig.colorbar(img, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Strip index")
    save_fig(fig, demo, name)


def save_two_log_images(demo, name, mat1, mat2, titles, x_labels, extents):
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    for ax, mat, title, x_label, extent in zip(axes, (mat1, mat2), titles,
                                               x_labels, extents):
        ax.imshow(np.log(mat + 1), aspect="auto", cmap="inferno",
                  extent=extent)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Strip index")
    save_fig(fig, demo, name)


def save_peaks_plot(demo, name, x_list, y_list, peaks, title,
                    x_label="Pixel index", y_label="Intensity"):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(x_list, y_list, color="blue")
    ax.plot(x_list[peaks], y_list[peaks], "o", color="red")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.minorticks_on()
    ax.grid(True)
    save_fig(fig, demo, name)


def save_denoise_plot(demo, name, raw_data, denoised_data):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(raw_data, color="blue", label="Original")
    ax.plot(denoised_data, color="red", label="Denoised")
    ax.set_xlabel("Pixel index")
    ax.set_ylabel("Intensity")
    ax.legend()
    save_fig(fig, demo, name)


def save_1d_checks(demo, names, list_data, loc_peaks, loc_values, list_fact,
                   ref_x, ref_y, step, correct_func, strip_idx, unit):
    """Selected peaks, mapping, fitting residual, and calibrated results."""
    width = len(list_data)
    x_idx = np.arange(width)
    loc_peaks = np.asarray(loc_peaks)
    loc_values = np.asarray(loc_values)
    loc_int = np.int32(np.round(loc_peaks))
    save_peaks_plot(demo, names[0], x_idx, list_data, loc_int,
                    f"Strip idx {strip_idx}. Selected peaks corresponding "
                    f"to reference peaks")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    points = np.asarray([(p, v) for p in loc_peaks for v in loc_values])
    x_fit = np.linspace(loc_peaks[0], loc_peaks[-1], 100)
    ax.plot(points[:, 0], points[:, 1], ".", color="blue")
    ax.plot(loc_peaks, loc_values, "o", color="red")
    ax.plot(x_fit, poly.polyval(x_fit, list_fact), color="green")
    ax.set_xlim((0, width))
    ax.set_xlabel("Pixel index")
    ax.set_ylabel(unit)
    ax.set_title("Mapping between peak locations and reference values")
    save_fig(fig, demo, names[1])

    y_fit = poly.polyval(loc_peaks, list_fact)
    offset = 0.5 * (loc_values[0] + loc_values[-1])
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(loc_peaks, loc_values, "o", color="red", label="Measured")
    ax.plot(loc_peaks, y_fit, "-", color="green", label="Fitted")
    ax.plot(loc_peaks, offset + (loc_values - y_fit), "o", color="blue",
            label="Fitting residual")
    ax.plot(loc_peaks, offset * np.ones_like(loc_peaks), "-", color="red")
    ax.set_xlabel("Pixel index")
    ax.set_ylabel(unit)
    ax.set_title(f"Strip idx {strip_idx}. Polynomial mapping from pixels "
                 f"to {unit}")
    ax.legend()
    ax.grid(True)
    save_fig(fig, demo, names[2])

    correct_data = correct_func(list_data, list_fact, step)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(ref_x, ref_y / np.max(ref_y), color="blue", label="Reference")
    ax.plot(x_idx * step, correct_data / np.max(correct_data), color="red",
            label="Calibrated")
    ax.vlines(loc_values, 0.0, 1.0, colors="green", linestyles="solid",
              label="Reference lines")
    ax.set_xlabel(unit)
    ax.set_ylabel("Intensity")
    ax.set_title("Calibrated results")
    ax.minorticks_on()
    ax.grid(True)
    ax.legend()
    save_fig(fig, demo, names[3])


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------

def load_energy_image():
    return 0.5 * (losa.load_image(ENERGY_FILE1) + losa.load_image(ENERGY_FILE2))


def get_reference_energies(energy_bin, width):
    lines = util.get_fluorescence_lines_from_database(ELEMENTS, line_type="K",
                                                      threshold=0.01,
                                                      norm=False)
    spectrum = util.make_fluorescence_spectrum(lines, energy_bin, width,
                                               sigma=7)
    list_inten = spectrum[:, 1]
    peaks = prep.get_peaks_by_ranked_descent(list_inten, width=5, ratio=0.1,
                                             sigma=0)
    peaks_sub = prep.locate_gaussian_peaks_subpixel(list_inten, peaks,
                                                    radius=2, sigma=1,
                                                    skew_fit=False)
    return spectrum, peaks, peaks_sub * energy_bin


def get_reference_pattern(q_step, width, num_use, threshold):
    list_d_inten = losa.load_d_spacing_from_pearson_crystal_data_file(
        PEARSON_FILE, num_use=num_use, threshold=threshold, check=False)
    d_spacing_values = list_d_inten[:, 0]
    q_values_ref = 2 * np.pi / d_spacing_values
    pattern = util.make_diffraction_pattern(q_values_ref, list_d_inten[:, 1],
                                            q_step, width, sigma=7)
    ref_peaks = prep.get_peaks_by_ranked_descent(pattern[:, 1], width=5,
                                                 ratio=0.1, sigma=0)
    return d_spacing_values, q_values_ref, pattern, ref_peaks


# ----------------------------------------------------------------------------
# Demonstrations
# ----------------------------------------------------------------------------

def demo_01():
    """Estimate the energy bin and the energy resolution."""
    demo = "demo_01"
    image = load_energy_image()
    (height, width) = image.shape
    save_log_image(demo, "fig1.png", image, "Input data, log scale")

    lines = util.get_fluorescence_lines_from_database(ELEMENTS, line_type="K",
                                                      threshold=0.01,
                                                      norm=False)
    spectrum = util.make_fluorescence_spectrum(lines, 0.055, width, sigma=7)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(spectrum[:, 0], spectrum[:, 1], color="blue")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Theory spectrum of {ELEMENTS}")
    ax.grid(True)
    save_fig(fig, demo, "fig2.png")

    mid_channel = height // 2
    meas_spectrum = prep.smooth_filter_based_svd(image[mid_channel],
                                                 **SVD_PARA)
    save_denoise_plot(demo, "fig3.png", image[mid_channel], meas_spectrum)
    peaks = prep.get_peaks_by_ranked_descent(meas_spectrum, width=15,
                                             ratio=0.1, sigma=3)
    save_peaks_plot(demo, "fig4.png", np.arange(width), meas_spectrum, peaks,
                    "Located peaks")

    list_inten = np.sort(meas_spectrum[peaks])
    pos1 = np.where(meas_spectrum == list_inten[-1])[0][0]
    pos2 = np.where(meas_spectrum == list_inten[-2])[0][0]
    save_peaks_plot(demo, "fig5.png", np.arange(width), meas_spectrum,
                    [pos1, pos2], "Located 2 highest peaks")
    sub_peaks = prep.locate_gaussian_peaks_subpixel(image[mid_channel],
                                                    [pos1, pos2], radius=21,
                                                    sigma=1, skew_fit=True)
    energy_bin = 43.535 / abs(sub_peaks[0] - sub_peaks[1])
    print(f"Calculated energy bin: {energy_bin} keV")

    lines_u = util.get_fluorescence_lines_from_database(["U"], line_type="K",
                                                        threshold=0.02)
    spectrum_u = util.make_fluorescence_spectrum(lines_u, energy_bin, width,
                                                 sigma=7)
    meas_spectrum = prep.remove_background_based_arpls(meas_spectrum, ld=1e6,
                                                       weight=0.03,
                                                       num_iter=10)
    (min_e, max_e) = (92.0, 140.0)
    x_list = np.arange(width) * energy_bin
    mask = (x_list > min_e) & (x_list < max_e)
    x_list, y_list = x_list[mask], meas_spectrum[mask]
    x_list1 = spectrum_u[:, 0]
    mask1 = (x_list1 > min_e) & (x_list1 < max_e)
    x_list1, y_list1 = x_list1[mask1], spectrum_u[:, 1][mask1]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(x_list, y_list / np.max(y_list), color="red",
            label="Measured U peaks (not calibrated)")
    ax.plot(x_list1, y_list1 / np.max(y_list1), color="blue",
            label="Uranium theory peaks")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True)
    save_fig(fig, demo, "fig6.png")

    peaks = prep.get_peaks_by_ranked_descent(y_list, width=21, ratio=0.3,
                                             sigma=5)
    pos, radius = peaks[-1], 30
    list_data = y_list[pos - radius: pos + radius + 1]
    fit_data, a, b, c, d, can_fit = prep.fit_gaussian_peak(list_data,
                                                           return_para=True)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(list_data, "o", color="blue", label="Measured")
    ax.plot(fit_data, color="red", label="Gaussian fit")
    ax.set_xlabel("Pixel index")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True)
    save_fig(fig, demo, "fig7.png")
    energy_resolution = 100 * (2.355 * b * energy_bin) / 114.335
    print(f"Energy resolution at 114.335 keV: {energy_resolution} %")


def demo_02():
    """Energy calibration of a single detector strip."""
    demo = "demo_02"
    strip_idx = 60
    image = load_energy_image()
    (height, width) = image.shape
    save_log_image(demo, "fig1.png", image, "Input data, log scale")

    spectrum, theo_peaks, ref_energies = get_reference_energies(ENERGY_BIN,
                                                                width)
    save_peaks_plot(demo, "fig2.png", spectrum[:, 0], spectrum[:, 1],
                    theo_peaks, f"Theory spectrum of {ELEMENTS} and peaks "
                                f"used for calibration",
                    x_label="Energy (keV)")
    meas_spectrum = prep.smooth_filter_based_svd(image[strip_idx], **SVD_PARA)
    save_denoise_plot(demo, "fig3.png", image[strip_idx], meas_spectrum)
    peaks = prep.get_peaks_by_ranked_descent(meas_spectrum, width=15,
                                             ratio=0.1, sigma=3)
    save_peaks_plot(demo, "fig4.png", np.arange(width), meas_spectrum, peaks,
                    f"Strip idx {strip_idx}. Located peaks")

    list_fact, loc_peaks, energies = proc.calibrate_energy_1d(
        image[strip_idx], ref_energies, ENERGY_BIN, fit_order=2,
        zero_left=450, zero_right=-300, sigma=0, smooth_svd_para=SVD_PARA,
        get_peaks_para={"width": 15, "ratio": 0.1, "sigma": 3},
        sub_peaks_para={"radius": 15, "sigma": 1, "skew_fit": True},
        match_peaks_para={"tol": 0.4, "size": 31, "margin": 20,
                          "search_range": (-3, 3), "bounds": None,
                          "initial_guess": None},
        return_check=True)
    print(f"Polynomial coefficients: {list_fact}")
    save_1d_checks(demo, ["fig5.png", "fig6.png", "fig7.png", "fig8.png"],
                   meas_spectrum, loc_peaks, energies, list_fact,
                   spectrum[:, 0], spectrum[:, 1], ENERGY_BIN,
                   post.correct_energy_1d, strip_idx, "Energy (keV)")


def demo_03():
    """Energy calibration of all detector strips."""
    demo = "demo_03"
    image = load_energy_image()
    (height, width) = image.shape
    spectrum, _, ref_energies = get_reference_energies(ENERGY_BIN, width)
    extent = [0, ENERGY_BIN * (width - 1), height, 0]
    theory_image = np.tile(spectrum[:, 1], (height, 1)) * 2000
    save_two_log_images(demo, "fig1.png", image, theory_image,
                        ["Measured spectrum, log scale", "Theory spectrum"],
                        ["Pixel index", "Energy (keV)"], [None, extent])

    sub_peaks_para = {"radius": 21, "sigma": 0, "skew_fit": True}
    match_peaks_para = {"tol": 0.4, "size": 31, "margin": 20,
                        "search_range": (-3, 3), "bounds": None,
                        "initial_guess": None}
    results1 = proc.calibrate_energy_image(
        image[:32], ref_energies, ENERGY_BIN, fit_order=2, zero_left=200,
        zero_right=2000, sigma=4, smooth_svd_para=SVD_PARA,
        get_peaks_para={"width": 19, "ratio": 0.1, "sigma": 0},
        sub_peaks_para=sub_peaks_para, match_peaks_para=match_peaks_para)
    results2 = proc.calibrate_energy_image(
        image[32:], ref_energies, ENERGY_BIN, fit_order=2, zero_left=300,
        zero_right=2500, sigma=5, smooth_svd_para=SVD_PARA,
        get_peaks_para={"width": 17, "ratio": 0.1, "sigma": 0},
        sub_peaks_para=sub_peaks_para, match_peaks_para=match_peaks_para)
    mat_coef = np.concatenate((results1, results2), axis=0)
    mat_correct = post.correct_energy_image(image, mat_coef, ENERGY_BIN,
                                            ncore=1)
    save_two_log_images(demo, "fig2.png", mat_correct, theory_image,
                        ["Calibrated spectrum, log scale", "Theory spectrum"],
                        ["Energy (keV)", "Energy (keV)"], [extent, extent])


def demo_04():
    """q-calibration of a single strip using the GeRM geometry."""
    demo = "demo_04"
    strip_idx = 100
    image = losa.load_image(Q_FILE)
    (height, width) = image.shape
    save_log_image(demo, "fig1.png", image, "Input data, log scale")

    q_step = util.estimate_q_step_germ_detector(0, ENERGY_BIN, TWO_THETA,
                                                **GEOMETRY)
    d_spacing_values, q_values_ref, pattern, ref_peaks = \
        get_reference_pattern(q_step, width, num_use=16, threshold=35)
    save_peaks_plot(demo, "fig2.png", pattern[:, 0], pattern[:, 1],
                    ref_peaks, "Diffraction pattern of Cr2O3 and peaks used "
                               "for calibration", x_label="Q (1/A)")
    meas_pattern = prep.smooth_filter_based_svd(image[strip_idx], **SVD_PARA)
    save_denoise_plot(demo, "fig3.png", image[strip_idx], meas_pattern)
    peaks = prep.get_peaks_by_ranked_descent(meas_pattern, width=15,
                                             ratio=0.1, sigma=3)
    save_peaks_plot(demo, "fig4.png", np.arange(width), meas_pattern, peaks,
                    f"Strip idx {strip_idx}. Located peaks")

    list_fact, loc_peaks, loc_q_values = proc.calibrate_q_germ_detector_1d(
        strip_idx, image[strip_idx], d_spacing_values, TWO_THETA, ENERGY_BIN,
        fit_order=2, zero_left=450, zero_right=-300, sigma=0,
        smooth_svd_para=SVD_PARA,
        get_peaks_para={"width": 15, "ratio": 0.1, "sigma": 3,
                        "abs_threshold": None},
        sub_peaks_para={"radius": 15, "sigma": 1, "skew_fit": True},
        match_peaks_para={"tol": 0.4, "size": 31, "margin": 20,
                          "search_range": (-10, 10), "bounds": None,
                          "initial_guess": None},
        return_check=True)
    print(f"Polynomial coefficients: {list_fact}")
    save_1d_checks(demo, ["fig5.png", "fig6.png", "fig7.png", "fig8.png"],
                   meas_pattern, loc_peaks, loc_q_values, list_fact,
                   pattern[:, 0], pattern[:, 1], q_step, post.correct_q_1d,
                   strip_idx, "Q (1/A)")


def demo_05():
    """q-calibration of all strips using the GeRM geometry."""
    demo = "demo_05"
    image = losa.load_image(Q_FILE)
    (height, width) = image.shape
    q_step = util.estimate_q_step_germ_detector(0, ENERGY_BIN, TWO_THETA,
                                                **GEOMETRY)
    d_spacing_values, _, pattern, _ = get_reference_pattern(
        q_step, width, num_use=18, threshold=35)
    extent = [0, q_step * (width - 1), height, 0]
    ref_image = np.tile(pattern[:, 1], (height, 1)) * 2000
    save_two_log_images(demo, "fig1.png", image, ref_image,
                        ["Measured diffraction pattern, log scale",
                         "Reference diffraction pattern"],
                        ["Pixel index", "Q (1/A)"], [None, extent])

    mat_coef = proc.calibrate_q_germ_detector_image(
        image, d_spacing_values, TWO_THETA, ENERGY_BIN, fit_order=2,
        zero_left=450, zero_right=0, sigma=0, smooth_svd_para=SVD_PARA,
        get_peaks_para={"width": 31, "ratio": 0.05, "sigma": 3,
                        "abs_threshold": None},
        sub_peaks_para={"radius": 21, "sigma": 2, "skew_fit": True},
        match_peaks_para={"tol": 0.4, "size": 21, "margin": 20,
                          "search_range": (-10, 10), "bounds": None,
                          "initial_guess": None},
        ncore=None, prefer="processes")
    mat_correct = post.correct_q_image(image, mat_coef, q_step, ncore=None)
    save_two_log_images(demo, "fig2.png", mat_correct, ref_image,
                        ["Calibrated pattern, log scale",
                         "Reference diffraction pattern"],
                        ["Q (1/A)", "Q (1/A)"], [extent, extent])


def demo_06():
    """q-calibration of a single strip without the detector geometry."""
    demo = "demo_06"
    strip_idx = 100
    image = losa.load_image(Q_FILE)
    (height, width) = image.shape
    _, q_values_ref, pattern, ref_peaks = get_reference_pattern(
        0.0015, width, num_use=16, threshold=35)
    list_inten_ref = pattern[:, 1]

    meas_pattern = prep.smooth_filter_based_svd(image[strip_idx], **SVD_PARA)
    peaks = prep.get_peaks_by_ranked_descent(meas_pattern, width=15,
                                             ratio=0.1, sigma=3)
    sub_peaks_meas = prep.locate_gaussian_peaks_subpixel(meas_pattern, peaks,
                                                         radius=15, sigma=1,
                                                         skew_fit=True)
    meas_peaks_5_idx = np.argsort(meas_pattern[peaks])[-5:][::-1]
    top5_meas_peaks = peaks[meas_peaks_5_idx]
    ref_peaks_5_idx = np.argsort(list_inten_ref[ref_peaks])[-5:][::-1]
    top5_ref_peaks = ref_peaks[ref_peaks_5_idx]
    for name, meas_p, ref_p, label in (
            ("fig1.png", peaks, ref_peaks, "located peaks"),
            ("fig2.png", top5_meas_peaks, top5_ref_peaks, "top 5 peaks")):
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
        axes[0].plot(np.arange(width), meas_pattern, color="blue")
        axes[0].plot(np.arange(width)[meas_p], meas_pattern[meas_p], "o",
                     color="red")
        axes[0].set_title(f"Strip idx {strip_idx}. Measured pattern and "
                          f"{label}")
        axes[0].set_xlabel("Pixel index")
        axes[1].plot(pattern[:, 0], list_inten_ref, color="blue")
        axes[1].plot(pattern[:, 0][ref_p], list_inten_ref[ref_p], "o",
                     color="red")
        axes[1].set_title(f"Reference pattern of Cr2O3 and {label}")
        axes[1].set_xlabel("Q (1/A)")
        for ax in axes:
            ax.minorticks_on()
            ax.grid(True)
        save_fig(fig, demo, name)

    list_q_ref = pattern[:, 0][ref_peaks]
    pixel_distance = abs(sub_peaks_meas[meas_peaks_5_idx[1]]
                         - sub_peaks_meas[meas_peaks_5_idx[2]])
    q_distance = abs(list_q_ref[ref_peaks_5_idx][0]
                     - list_q_ref[ref_peaks_5_idx][3])
    q_step = q_distance / pixel_distance
    print(f"Measured q-step: {q_step}")

    list_fact, loc_peaks, loc_q_values = proc.calibrate_q_1d(
        image[strip_idx], q_values_ref, q_step, fit_order=2, zero_left=450,
        zero_right=-300, sigma=0, smooth_svd_para=SVD_PARA,
        get_peaks_para={"width": 15, "ratio": 0.1, "sigma": 3,
                        "abs_threshold": None},
        sub_peaks_para={"radius": 15, "sigma": 1, "skew_fit": True},
        match_peaks_para={"tol": 0.4, "size": 31, "margin": 20,
                          "search_range": (-0.5, 0.5), "bounds": None,
                          "initial_guess": None},
        return_check=True)
    print(f"Polynomial coefficients: {list_fact}")
    save_1d_checks(demo, ["fig3.png", "fig4.png", "fig5.png", "fig6.png"],
                   meas_pattern, loc_peaks, loc_q_values, list_fact,
                   pattern[:, 0], list_inten_ref, q_step, post.correct_q_1d,
                   strip_idx, "Q (1/A)")


def demo_07():
    """q-calibration of all strips using per-strip q-steps."""
    demo = "demo_07"
    image = losa.load_image(Q_FILE)
    (height, width) = image.shape
    q_step_list = [util.estimate_q_step_germ_detector(idx, ENERGY_BIN,
                                                      TWO_THETA, **GEOMETRY)
                   for idx in range(height)]
    q_step = np.max(q_step_list)
    _, q_values_ref, pattern, _ = get_reference_pattern(
        q_step, width, num_use=21, threshold=25)
    extent = [0, q_step * (width - 1), height, 0]
    ref_image = np.tile(pattern[:, 1], (height, 1)) * 2000
    save_two_log_images(demo, "fig1.png", image, ref_image,
                        ["Measured diffraction pattern, log scale",
                         "Reference diffraction pattern"],
                        ["Pixel index", "Q (1/A)"], [None, extent])

    mat_coef = proc.calibrate_q_image(
        image, q_values_ref, q_step_list, fit_order=2, zero_left=450,
        zero_right=-300, sigma=0, smooth_svd_para=SVD_PARA,
        get_peaks_para={"width": 15, "ratio": 0.1, "sigma": 3,
                        "abs_threshold": None},
        sub_peaks_para={"radius": 15, "sigma": 1, "skew_fit": True},
        match_peaks_para={"tol": 0.4, "size": 21, "margin": 20,
                          "search_range": (-0.5, 0.5), "bounds": None,
                          "initial_guess": None},
        ncore=None, prefer="processes")
    mat_correct = post.correct_q_image(image, mat_coef, q_step, ncore=None)
    save_two_log_images(demo, "fig2.png", mat_correct, ref_image,
                        ["Calibrated pattern, log scale",
                         "Reference diffraction pattern"],
                        ["Q (1/A)", "Q (1/A)"], [extent, extent])


DEMOS = {1: demo_01, 2: demo_02, 3: demo_03, 4: demo_04, 5: demo_05,
         6: demo_06, 7: demo_07}

if __name__ == "__main__":
    selected = [int(i) for i in sys.argv[1:]] or list(DEMOS)
    for idx in selected:
        t0 = time.time()
        print(f"\n===== Demo {idx:02d} =====")
        DEMOS[idx]()
        print(f"Time cost: {time.time() - t0:.1f} s")
