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
# file_path = r"F:\EDXD_data\rawdata\cr2o4deg_20micron_3000s_07_17_2023.tif"
output_base = "F:/tmp/algodif/q_calibration_single/"
two_theta = 6.0  # Degree
energy_bin = 0.0538357407779  # keV Calculated using another script
fit_order = 2
fig_size = (16, 8)
element = "Cr2O3"

image = losa.load_image(file_path)
(height, width) = image.shape

q_step_list = [util.estimate_q_step_germ_detector(idx, energy_bin, two_theta,
                                  total_germ_strip=192,
                                  strip_size=0.2, sample_slit=200.0,
                                  sample_detector=2000.0) for idx in range(height)]
print(q_step_list)
q_step = np.max(np.asarray(q_step_list))
print("Q-step ", q_step)
list_d_intensity_ref = losa.load_d_spacing_from_pearson_crystal_data_file(
    r"C:\Users\nvo\PycharmProjects\algodif\data\Pearson_Crystal_Data_Cr2O3.csv",
    threshold=35, num_use=12, check=True)


plt.figure(figsize=fig_size)
plt.imshow(np.log(image + 1), aspect="auto", cmap="inferno")
plt.colorbar()
plt.title(f"Input data, log scale, two-theta: {two_theta} degree")
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
plt.figure(figsize=fig_size)
plt.plot(diff_pattern_ref[:, 0], diff_pattern_ref[:, 1], color='blue')
plt.plot(diff_pattern_ref[:, 0][ref_peaks], diff_pattern_ref[:, 1][ref_peaks],
         "o", color="red")
plt.xlabel("Q (1/A)")
plt.ylabel("Intensity")
plt.title(f"Diffraction pattern of {element} and peaks used for calibration")
plt.minorticks_on()
plt.tight_layout()
plt.grid()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# First image
im0 = axes[0].imshow(np.log(image + 1), aspect="auto", cmap="inferno")
axes[0].set_title(f"Measured diffraction pattern, log scale, two-theta: {two_theta}")
axes[0].set_xlabel("Pixel index")
axes[0].set_ylabel("Strip index")

image2 = np.tile(list_inten_ref, (height, 1))
# Second image
im1 = axes[1].imshow(np.log(image2 * 2000 + 1),
                     extent=[0, q_step * (width - 1), height, 0],
                     aspect="auto", cmap="inferno")
axes[1].set_title("Reference diffraction pattern")
axes[1].set_xlabel("Q (1/A)")
axes[1].set_ylabel("Strip index")

plt.tight_layout()
plt.show()

t0 = time.time()
print("\n---------------------")
print("Start calibration...\n")
results = proc.calibrate_q_image(image, q_values_ref, q_step_list,
                                 fit_order=fit_order,
                                 zero_left=450,
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
                                                   "size": 21,
                                                   "margin": 20,
                                                   "search_range": (-0.9, 0.9),
                                                   "bounds": None,
                                                   "initial_guess": None},
                                 return_check=False, ncore=None,
                                 prefer="processes")

mat_correct = post.correct_q_image(image, results, q_step)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
im0 = axes[0].imshow(np.log(mat_correct + 1),
                     extent=[0, q_step * (width - 1), height, 0],
                     aspect="auto", cmap="inferno")
axes[0].set_title("Calibrated spectrum, log scale")
axes[0].set_xlabel("Q (1/A)")
axes[0].set_ylabel("Strip index")

im1 = axes[1].imshow(np.log(np.tile(list_inten_ref, (height, 1)) * 2000 + 1),
                     extent=[0, q_step * (width - 1), height, 0],
                     aspect="auto", cmap="inferno")
axes[1].set_title("Reference pattern")
axes[1].set_xlabel("Q (1/A)")
axes[1].set_ylabel("Strip index")

plt.tight_layout()
plt.show()

t1 = time.time()
print("=========================================")
print("All done!!!. Time cost {}".format(t1 - t0))
print("=========================================")