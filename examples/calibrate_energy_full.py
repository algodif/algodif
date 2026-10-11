import sys
import time
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import algodif.io.loadersaver as losa
import algodif.prep.preprocessing as prep
import algodif.proc.processing as proc
import algodif.post.postprocessing as post
import algodif.util.utility as util


# Measurement of Au, Sn, Pb sample
file_path1 = "../data/Pb_Au_Sn_calibration.tif"
# Measurement of Uranium sample
file_path2 = "../data/U_calibration.tif"
# For saving output
output_base = "F:/tmp/algodif/calculate_energy_resolution/"

fit_order = 2
list_element = ["Sn", "Au", "Pb", "U"]
energy_bin = 0.0538357407779  # keV Size Calculated using another script.

fig_size = (12, 9)
image1 = losa.load_image(file_path1)
image2 = losa.load_image(file_path2)
image = 0.5 * (image1 + image2)
(height, width) = image.shape


fluorescent_lines = util.get_fluorescence_lines_from_database(list_element,
                                                              line_type="K",
                                                              threshold=0.01,
                                                              norm=False,
                                                              check=True)
# Show the theory spectrum
theory_spectrum = util.make_fluorescence_spectrum(fluorescent_lines,
                                                  energy_bin, width,
                                                  sigma=7)
list_theo_inten = theory_spectrum[:, 1]
theo_peaks = prep.get_peaks_by_ranked_descent(list_theo_inten, width=5,
                                              ratio=0.1, sigma=0)
theo_peaks_sub = prep.locate_gaussian_peaks_subpixel(list_theo_inten,
                                                     theo_peaks, radius=2,
                                                     sigma=1, skew_fit=False,
                                                     show=False)
reference_energies = theo_peaks_sub * energy_bin
# Note there's impact of Gaussian filter to simulate the spectrum.
plt.figure(figsize=fig_size)
plt.plot(theory_spectrum[:, 0], theory_spectrum[:, 1], color='blue')
plt.plot(theory_spectrum[:, 0][theo_peaks], theory_spectrum[:, 1][theo_peaks],
         "o", color="red")
plt.xlabel("Energy (keV)")
plt.ylabel("Intensity")
plt.title(f"Theory spectrum of {list_element} and peaks used for calibration")
plt.minorticks_on()
plt.grid()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# First image
im0 = axes[0].imshow(np.log(image + 1), aspect="auto", cmap="inferno")
axes[0].set_title("Measured spectrum, log scale")
axes[0].set_xlabel("Pixel index")
axes[0].set_ylabel("Strip index")
# Second image
im1 = axes[1].imshow(np.log(np.tile(list_theo_inten, (height, 1)) * 2000 + 1),
                     extent=[0, energy_bin * (width - 1), height, 0],
                     aspect="auto", cmap="inferno")
axes[1].set_title("Theory spectrum")
axes[1].set_xlabel("Energy (keV)")
axes[1].set_ylabel("Strip index")

plt.tight_layout()
plt.show()


t0 = time.time()
print("Start calibration...")
results1 = proc.calibrate_energy_image(image[:32], reference_energies,
                                       energy_bin, fit_order=2,
                                       zero_left=200, zero_right=2000,
                                       sigma=4,
                                       smooth_svd_para={"rank": 5,
                                                        "lambda_threshold": 0.5},
                                       get_peaks_para={"width": 19,
                                                       "ratio": 0.1,
                                                       "sigma": 0},
                                       sub_peaks_para={"radius": 21,
                                                       "sigma": 0,
                                                       "skew_fit": True},
                                       match_peaks_para={"tol": 0.4,
                                                         "size": 31,
                                                         "margin": 20,
                                                         "search_range": (-3, 3),
                                                         "bounds": None,
                                                         "initial_guess": None},
                                       peaks=None, return_check=False)
results2 = proc.calibrate_energy_image(image[32:], reference_energies,
                                       energy_bin, fit_order=2,
                                       zero_left=300, zero_right=2500,
                                       sigma=5,
                                       smooth_svd_para={"rank": 5,
                                                        "lambda_threshold": 0.5},
                                       get_peaks_para={"width": 17,
                                                       "ratio": 0.1,
                                                       "sigma": 0},
                                       sub_peaks_para={"radius": 21,
                                                       "sigma": 0,
                                                       "skew_fit": True},
                                       match_peaks_para={"tol": 0.4,
                                                         "size": 31,
                                                         "margin": 20,
                                                         "search_range": (-3, 3),
                                                         "bounds": None,
                                                         "initial_guess": None},
                                       peaks=None, return_check=False)

results = np.concatenate((results1, results2), axis=0)
t0 = time.time()
mat_correct = post.correct_energy_image(image, results, energy_bin, ncore=1)
t1 = time.time()
print(f"Time cost for correct image {t1- t0}")


fig, axes = plt.subplots(1, 2, figsize=(16, 8))
im0 = axes[0].imshow(np.log(mat_correct + 1),
                     extent=[0, energy_bin * (width - 1), height, 0],
                     aspect="auto", cmap="inferno")
axes[0].set_title("Calibrated spectrum, log scale")
axes[0].set_xlabel("Energy (keV)")
axes[0].set_ylabel("Strip index")

im1 = axes[1].imshow(np.log(np.tile(list_theo_inten, (height, 1)) * 2000 + 1),
                     extent=[0, energy_bin * (width - 1), height, 0],
                     aspect="auto", cmap="inferno")
axes[1].set_title("Theory spectrum")
axes[1].set_xlabel("Energy (keV)")
axes[1].set_ylabel("Strip index")

plt.tight_layout()
plt.show()

plt.figure(0, figsize=(16, 8))
line1 = np.log(mat_correct + 1)[90]
xlist = np.arange(width) * energy_bin
plt.plot(xlist, line1)
plt.show()

t1 = time.time()
print("All done!!!. Time cost {}".format(t1 - t0))
print("=========================================")