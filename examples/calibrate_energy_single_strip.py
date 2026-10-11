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
file_path1 = r"C:\Users\nvo\PycharmProjects\algodif\data\Pb_Au_Sn_calibration.tif"
# Measurement of Uranium sample
file_path2 = r"C:\Users\nvo\PycharmProjects\algodif\data\U_calibration.tif"
# For saving output
output_base = "F:/tmp/algodif/calculate_energy_resolution/"

fit_order = 2
list_element = ["Sn", "Au", "Pb", "U"]
energy_bin = 0.0538357407779  # keV Calculated using another script.

# Select the strip for calibration
strip_idx = 60

fig_size = (16, 8)
image1 = losa.load_image(file_path1)
image2 = losa.load_image(file_path2)
image = 0.5 * (image1 + image2)
(height, width) = image.shape

if strip_idx < 0 or strip_idx > height - 1:
    raise ValueError(f"Index is out of range (0, {height - 1})")

plt.figure(figsize=fig_size)
plt.imshow(np.log(image + 1), aspect="auto", cmap="inferno")
plt.colorbar()
plt.title("Input data, log scale")
plt.tight_layout()
plt.show()

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
plt.tight_layout()
plt.grid()
plt.show()

meas_spectrum = image[strip_idx]
# Clean data and remove background
plt.figure(figsize=fig_size)
plt.plot(meas_spectrum, color="blue", label="Original")
# Denoising
meas_spectrum = prep.smooth_filter_based_svd(meas_spectrum,
                                             lambda_threshold=0.5, rank=5)
plt.plot(meas_spectrum, color="red", label="Denoised")
# Remove bck
# meas_spectrum = prep.remove_background_based_arpls(meas_spectrum, ld=1e5,
#                                                    weight=0.05, num_iter=10)
# meas_spectrum[:400] = 0.0 # Remove non reliable data
# plt.plot(meas_spectrum, color="red", label="Remove background")
plt.legend()
plt.tight_layout()
plt.show()

# Detect and select good peaks using water drain method
peaks = prep.get_peaks_by_ranked_descent(meas_spectrum, width=15, ratio=0.1,
                                         sigma=3)
plt.figure(figsize=fig_size)
plt.plot(np.arange(width), meas_spectrum, color="blue")
plt.plot(np.arange(width)[peaks], meas_spectrum[peaks], "o", color="red")
plt.title(f"Strip idx {strip_idx}. Located peaks")
plt.minorticks_on()
plt.grid(True)
plt.tight_layout()
plt.show()

t0 = time.time()
list_data = image[strip_idx]
results = proc.calibrate_energy_1d(list_data, reference_energies,
                                   energy_bin, fit_order=2,
                                   zero_left=450, zero_right=-300, sigma=0,
                                   smooth_svd_para={"rank": 5,
                                                    "lambda_threshold": 0.5},
                                   get_peaks_para={"width": 15,
                                                   "ratio": 0.1,
                                                   "sigma": 3},
                                   sub_peaks_para={"radius": 15, "sigma": 1,
                                                   "skew_fit": True},
                                   match_peaks_para={"tol": 0.4,
                                                     "size": 31,
                                                     "margin": 20,
                                                     "search_range": (
                                                         -3, 3),
                                                     "bounds": None,
                                                     "initial_guess": None},
                                   peaks=None, return_check=True)

list_fact, loc_peaks, energies = results
print("\nPolynomical coefficients (a0 + a1*x + a2*x^2):")
print(list_fact)
print("\n Peak locations (pixel):")
print(loc_peaks)
print("\n Theory values (keV):")
print(energies)

# To check
list_data = prep.smooth_filter_based_svd(list_data,
                                         lambda_threshold=0.5, rank=5)
plt.figure(figsize=fig_size)
plt.plot(np.arange(width), list_data, color="blue")
plt.plot(np.arange(width)[np.int32(np.round(loc_peaks))],
         list_data[np.int32(np.round(loc_peaks))], "o", color="red")
plt.title(
    f"Strip idx {strip_idx}. Selected peaks corresponding to theory peaks")
plt.tight_layout()
plt.show()

array_points = np.asarray(
    [(p, e) for p in list(loc_peaks) for e in reference_energies])
plt.figure(figsize=fig_size)
plt.plot(array_points[:, 0], array_points[:, 1], ".", color="blue")
plt.plot(loc_peaks, energies, "o", color="red")
data_point = 100
x_list = np.linspace(loc_peaks[0], loc_peaks[-1], data_point)
list_data_corr = np.sum(
    np.asarray([factor * x_list ** i for i, factor in enumerate(list_fact)]),
    axis=0)
plt.xlim((0, width))
plt.xlabel("Index")
plt.ylabel("Energy (keV)")
plt.title("The fitting is  to check how well the peak locations are mapped to "
          "the theory values.\nGood fitting is indicated by the green line "
          "which cross most of the blue dots (peak locations)")
plt.ylim((reference_energies[0] - 3.0, reference_energies[-1] + 3.0))
plt.plot(x_list, list_data_corr, color="green")
plt.tight_layout()
plt.show()

x_list1 = loc_peaks
y_list1 = energies
offset = (y_list1[-1] + y_list1[0]) * 0.5
y_list1_fit = np.sum(
    np.asarray([fact * x_list1 ** i for i, fact in enumerate(list_fact)]),
    axis=0)
plt.figure(figsize=fig_size)
plt.xlim((x_list1[0] - 10, x_list1[-1] + 10))
plt.ylim((reference_energies[0] - 0.5, reference_energies[-1] + 0.5))
plt.plot(x_list1, y_list1, "o", color="red")
plt.plot(x_list1, y_list1_fit, "-", color="green")
plt.plot(x_list1, offset + (y_list1 - y_list1_fit), "o", color="blue",
         label="Fitting residual")
plt.plot(x_list1, offset * np.ones_like(x_list1), "-", color="red")
plt.title(f"Strip idx {strip_idx}. "
          f"To check how good the polynomial mapping from pixels to energies")
plt.xlabel("Pixel index")
plt.ylabel("keV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

correct_data = post.correct_energy_1d(list_data, list_fact, energy_bin)
# correct_data = prep.remove_background_based_arpls(correct_data)
theory_data = theory_spectrum[:, 1]
# plt.figure(figsize=fig_size)
# plt.plot(theory_spectrum[:, 0],  theory_data / np.max(theory_data), color='blue', label="Theory")
# plt.plot(np.arange(width) * energy_bin, correct_data/np.max(correct_data), color='red', label="Calibrated")
# plt.vlines(theo_peaks_sub * energy_bin, 0.0, 1.0, linestyles="solid", color="green", label="Theory lines")
# plt.xlabel("Energy (keV)")
# plt.ylabel("Intensity")
# plt.title(f"Calibrated results")
# plt.minorticks_on()
# plt.grid()
# plt.legend()
# plt.show()

fig, ax = plt.subplots(figsize=fig_size)
ax.plot(theory_spectrum[:, 0], theory_data / np.max(theory_data), color='blue',
        label="Theory")
ax.plot(np.arange(width) * energy_bin, correct_data / np.max(correct_data),
        color='red', label="Calibrated")
energies = theo_peaks_sub * energy_bin
plot_vlines = ax.vlines(energies, 0.0, 1.0, colors="green", linestyles="solid",
                        label="Theory lines", picker=5)
ax.set_xlabel("Energy (keV)")
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
    if event.artist is not plot_vlines:
        return
    ind = event.ind[0]
    energy = energies[ind]
    annot.xy = (energy, 1.0)
    annot.set_text(f"Energy = {energy:.3f} keV")
    annot.set_visible(True)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("pick_event", on_pick)
plt.show()

t1 = time.time()
print("=========================================")
print("All done!!!. Time cost {}".format(t1 - t0))
print("=========================================")
