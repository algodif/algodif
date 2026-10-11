import os
import sys
import time
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import algodif.io.loadersaver as losa
import algodif.prep.preprocessing as prep
import algodif.proc.processing as proc
import algodif.util.utility as util
import algodif.post.postprocessing as post

file_path = "F:/EDXD_data/rawdata/cr2o3_6deg_10_06_2023.tif"
output_base = "F:/tmp/algodif/q_calibration_single/"

fit_order = 2
two_theta = 6.0  # Degree
fig_size = (16, 8)

q_step = 0.0015  # This can be pre-selected or extracted from measurement.
# Select the strip for calibration
strip_idx = 100
element = "Cr2O3"

image = losa.load_image(file_path)
(height, width) = image.shape
list_d_intensity_ref = losa.load_d_spacing_from_pearson_crystal_data_file(
    r"C:\Users\nvo\PycharmProjects\algodif\data\Pearson_Crystal_Data_Cr2O3.csv",
    num_use=16, threshold=35, check=True)

if strip_idx < 0 or strip_idx > height - 1:
    raise ValueError(f"Index is out of range (0, {height - 1})")

plt.figure(figsize=fig_size)
plt.imshow(np.log(image + 1), aspect="auto", cmap="inferno")
plt.colorbar()
plt.title("Input data, log scale")
plt.xlabel("Pixel index")
plt.ylabel("Strip index")
plt.tight_layout()
plt.show()

"""
# Manual input d-spacing
element = "Cr2O3"
if element == "Lab6":
    # #Lab6
    d_spacing_values = np.asarray(
        [4.157, 2.939, 2.4, 2.078, 1.859, 1.697, 1.47, 1.386, 1.314, 1.253,
         1.2, 1.153, 1.111, 1.008, 0.907])
    d_spacing_values = d_spacing_values[0:7]
elif element == "CeO2":
    # # CeO2
    d_spacing_values = np.asarray(
        [1.913, 1.632, 1.562, 1.353, 1.242, 1.21, 1.105, 1.041, 0.957, 0.915])
elif element == "Al2O3":
    # # Al2O3
    d_spacing_values = np.asarray([2.551, 2.379, 2.085, 1.740, 1.601, 1.374])
elif element == "Si":
    # # Si
    d_spacing_values = np.asarray(
        [3.136, 1.92, 1.638, 1.568, 1.358, 1.246, 1.109, 1.045, 0.96, 0.918,
         0.905, 0.859])
    d_spacing_values = d_spacing_values[0:4]
elif element == "Cr2O3":
    # # Cr2O3
    d_spacing_values = np.asarray(
        [3.631, 2.665, 2.48, 2.266, 2.175, 2.048, 1.815, 1.673, 1.465, 1.432,
         1.296, 1.24, 1.149, 1.088, 1.042, 1.024, 0.946, 0.866, 0.826])
    d_spacing_values = d_spacing_values[0:12]
else:
    raise ValueError("No such element {}".format(element))
intensities_ref = np.ones_like(d_spacing_values)
"""

d_spacing_values = list_d_intensity_ref[:, 0]
intensities_ref = list_d_intensity_ref[:, 1]
q_values_ref = 2 * np.pi / d_spacing_values


diff_pattern_ref = util.make_diffraction_pattern(q_values_ref, intensities_ref,
                                                 q_step, width, sigma=7)
list_inten_ref = diff_pattern_ref[:, 1]
ref_peaks = prep.get_peaks_by_ranked_descent(list_inten_ref, width=5,
                                             ratio=0.1, sigma=0)
ref_peaks_sub = prep.locate_gaussian_peaks_subpixel(list_inten_ref,
                                                    ref_peaks, radius=2,
                                                    sigma=0, skew_fit=False,
                                                    show=False)

# We can calculate the q-step (pixel size) by select two known q-values
# and the pixel distance between them.
meas_pattern = image[strip_idx]
# Clean data and remove background
plt.figure(figsize=fig_size)
plt.plot(meas_pattern, color="blue", label="Original")
# Denoising
meas_pattern = prep.smooth_filter_based_svd(meas_pattern,
                                            lambda_threshold=0.5, rank=5)
plt.plot(meas_pattern, color="green", label="Denoised")
# # Remove bck
# meas_pattern = prep.remove_background_based_arpls(meas_pattern, ld=1e5,
#                                                    weight=0.05, num_iter=10)
# meas_pattern[:400] = 0.0 # Remove non reliable data
# plt.plot(meas_pattern, color="red", label="Remove background")
plt.legend()
plt.tight_layout()
plt.show()

# Detect and select good peaks using water drain method
peaks = prep.get_peaks_by_ranked_descent(meas_pattern, width=15, ratio=0.1,
                                         sigma=3)
sub_peaks_meas = prep.locate_gaussian_peaks_subpixel(meas_pattern, peaks,
                                                     radius=15, sigma=1,
                                                     skew_fit=True)
fig, axes = plt.subplots(1, 2, figsize=fig_size)

ax = axes[0]
ax.plot(np.arange(width), meas_pattern, color="blue")
ax.plot(np.arange(width)[peaks], meas_pattern[peaks], "o", color="red")
ax.set_title(f"Strip idx {strip_idx} of measured patterns and located peaks")
ax.minorticks_on()
ax.grid(True)

ax = axes[1]
ax.plot(diff_pattern_ref[:, 0], diff_pattern_ref[:, 1], color="blue")
ax.plot(diff_pattern_ref[:, 0][ref_peaks],
        diff_pattern_ref[:, 1][ref_peaks],
        "o", color="red")
ax.set_xlabel("Q (1/A)")
ax.set_ylabel("Intensity")
ax.set_title(f"Reference pattern of {element} and major peaks")
ax.minorticks_on()
ax.grid(True)

plt.tight_layout()
plt.show()


# The corresponding peaks of top 5 highest peaks between the measured
# patterns and reference patterns are:
# peaks (2,1) <-> (0,3)  # Sorted intensity peaks

# Top 5 measured peaks
meas_peaks_5_idx = np.argsort(meas_pattern[peaks])[-5:][::-1]
top5_meas_peaks = peaks[meas_peaks_5_idx]
# Top 5 reference peaks
list_q_ref = diff_pattern_ref[:, 0][ref_peaks]
ref_peaks_5_idx = np.argsort(list_inten_ref[ref_peaks])[-5:][::-1]
top5_ref_peaks = ref_peaks[ref_peaks_5_idx]

pixel_distance = abs(sub_peaks_meas[meas_peaks_5_idx[1]] -
                     sub_peaks_meas[meas_peaks_5_idx[2]])
q_distance = abs(list_q_ref[ref_peaks_5_idx][0] - list_q_ref[ref_peaks_5_idx][3])
print("Top 5 peaks and corresponding q-values")
print(sub_peaks_meas[meas_peaks_5_idx])
print(list_q_ref[ref_peaks_5_idx])
q_step = q_distance / pixel_distance
print("======================================")
print(f"Pixel distance {pixel_distance}. Q distance: {q_distance}")
print(f"Measured Q-step: {q_step}")
print("======================================")


fig, axes = plt.subplots(1, 2, figsize=fig_size)
ax = axes[0]
ax.plot(np.arange(width), meas_pattern, color="blue")
ax.plot(np.arange(width)[top5_meas_peaks],
        meas_pattern[top5_meas_peaks],
        "o", color="red")
ax.set_title(f"Strip idx {strip_idx} of measured patterns and top 5 peaks")
ax.minorticks_on()
ax.grid(True)

ax = axes[1]
ax.plot(diff_pattern_ref[:, 0], diff_pattern_ref[:, 1], color="blue")
ax.plot(diff_pattern_ref[:, 0][top5_ref_peaks],
        diff_pattern_ref[:, 1][top5_ref_peaks],
        "o", color="red")
ax.set_xlabel("Q (1/A)")
ax.set_ylabel("Intensity")
ax.set_title(f"Reference pattern of {element} and top 5 peaks")
ax.minorticks_on()
ax.grid(True)

plt.tight_layout()
plt.show()


t0 = time.time()
list_data = image[strip_idx]
# Adjust search q-range if needs to
# "search_range": (-0.5, 0.5)
results = proc.calibrate_q_1d(list_data, q_values_ref, q_step,
                              fit_order=fit_order, zero_left=450,
                              zero_right=-300, sigma=0,
                              smooth_svd_para={"rank": 5,
                                               "lambda_threshold": 0.5},
                              get_peaks_para={"width": 15,
                                              "ratio": 0.1,
                                              "sigma": 3,
                                              "abs_threshold": None},
                              sub_peaks_para={"radius": 15,
                                              "sigma": 1,
                                              "skew_fit": True},
                              match_peaks_para={"tol": 0.4,
                                                "size": 31,
                                                "margin": 20,
                                                "search_range": (
                                                    -0.5, 0.5),
                                                "bounds": None,
                                                "initial_guess": None},
                              peaks=None, return_check=True)

list_fact, loc_peaks, loc_q_values = results
print("\nPolynomical coefficients (a0 + a1*x + a2*x^2):")
print(list_fact)
print("\n Peak locations (pixel):")
print(loc_peaks)
print("\n Reference q-values (keV):")
print(loc_q_values)

# To check
list_data = prep.smooth_filter_based_svd(list_data,
                                         lambda_threshold=0.5, rank=5)
plt.figure(figsize=fig_size)
plt.plot(np.arange(width), list_data, color="blue")
plt.plot(np.arange(width)[np.int32(np.round(loc_peaks))],
         list_data[np.int32(np.round(loc_peaks))], "o", color="red")
plt.title(
    f"Strip idx {strip_idx}. Selected peaks corresponding to reference peaks")
plt.tight_layout()
plt.show()

array_points = np.asarray(
    [(p, q) for p in list(loc_peaks) for q in loc_q_values])
plt.figure(figsize=fig_size)
plt.plot(array_points[:, 0], array_points[:, 1], ".", color="blue")
plt.plot(loc_peaks, loc_q_values, "o", color="red")
data_point = 100
x_list = np.linspace(loc_peaks[0], loc_peaks[-1], data_point)
list_data_corr = np.sum(
    np.asarray([factor * x_list ** i for i, factor in enumerate(list_fact)]),
    axis=0)
plt.xlim((0, width))
plt.xlabel("Index")
plt.ylabel("Q")
plt.title("The fitting is  to check how well the peak locations are mapped to "
          "the reference values.\nGood fitting is indicated by the green line "
          "which cross most of the blue dots (peak locations)")
plt.ylim((loc_q_values[0] - 0.1, loc_q_values[-1] + 0.1))
plt.plot(x_list, list_data_corr, color="green")
plt.tight_layout()
plt.show()

x_list = loc_peaks
y_list = loc_q_values
y_list_fit = np.sum(
    np.asarray([fact * x_list ** i for i, fact in enumerate(list_fact)]),
    axis=0)
offset = (y_list[-1] + y_list[0]) * 0.5
plt.figure(figsize=fig_size)
plt.plot(x_list, y_list, "o", color="red", label="Measured")
plt.plot(x_list, y_list_fit, "-", color="green", label="Fitted")
plt.plot(x_list, offset + (y_list - y_list_fit), "o", color="blue",
         label="Fitting residual")
plt.plot(x_list, offset * np.ones_like(x_list), "-", color="red")
plt.xlim((x_list[0] - 10, x_list[-1] + 10))
plt.ylim((q_values_ref[0] - 0.1, q_values_ref[-1] + 0.1))
plt.title(f"Strip idx {strip_idx}. "
          f"To check how good the polynomial mapping from pixels to q-values")
plt.xlabel("Pixel index")
plt.ylabel("Q")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

correct_data = post.correct_energy_1d(list_data, list_fact, q_step)
# correct_data = prep.remove_background_based_arpls(correct_data)

list_inten_ref = diff_pattern_ref[:, 1]
list_q_ref = diff_pattern_ref[:, 0]
fig, ax = plt.subplots(figsize=fig_size)
ax.plot(list_q_ref, list_inten_ref / np.max(list_inten_ref), color='blue',
        label="Database")
ax.plot(np.arange(width) * q_step, correct_data / np.max(correct_data),
        color='red', label="Calibrated")
vlines = ax.vlines(loc_q_values, 0.0, 1.0, colors="green",
                        linestyles="solid", label="Reference lines", picker=5)
ax.set_xlabel("Q")
ax.set_ylabel("Intensity")
ax.set_title("Calibrated results")
ax.minorticks_on()
ax.grid()
ax.legend()

annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def on_pick(event):
    if event.artist is not vlines:
        return
    ind = event.ind[0]
    q = loc_q_values[ind]
    annot.xy = (q, 1.0)
    annot.set_text(f"Q = {q:.3f} (1/A)")
    annot.set_visible(True)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("pick_event", on_pick)
plt.show()

t1 = time.time()
print("=========================================")
print("All done!!!. Time cost {}".format(t1 - t0))
print("=========================================")