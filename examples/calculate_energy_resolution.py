"""
The following code shows how to evaluate the energy resolution of
an EDXD (energy dispersive X ray diffraction) detector (GERM detector built at NSLS-II, BNL, US)
"""
import sys
import time
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import algodif.io.loadersaver as losa
import algodif.prep.preprocessing as prep
import algodif.proc.processing as proc
import algodif.util.utility as util


# Measurement of Au, Sn, Pb sample
file_path1 = "../data/Pb_Au_Sn_calibration.tif"
# Measurement of Uranium sample
file_path2 = "../data/U_calibration.tif"
# For saving output
output_base = "F:/tmp/algodif/calculate_energy_resolution/"

fit_order = 2
list_element = ["Sn", "Au", "Pb", "U"]
fig_size = (12, 9)

image1 = losa.load_image(file_path1)
image2 = losa.load_image(file_path2)
image = 0.5 * (image1 + image2)
(height, width) = image.shape

plt.imshow(np.log(image+1), aspect="auto", cmap="inferno")
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
energy_bin_guess = 0.055  # keV Size energy bin guess.
theory_spectrum = util.make_fluorescence_spectrum(fluorescent_lines,
                                                  energy_bin_guess, width,
                                                  sigma=7)
# Note there's impact of Gaussian filter to simulate the spectrum.
plt.figure(figsize=fig_size)
plt.plot(theory_spectrum[:, 0], theory_spectrum[:, 1], color='blue')
plt.xlabel("Energy (keV)")
plt.ylabel("Intensity")
plt.title(f"Theory spectrum of {list_element}")
plt.grid()
plt.show()

# From output of above command we got the values of 2 strong peaks

# Element: Sn  Line: Ka1     Energy:  25.2710 keV  Intensity: 0.5385
# Element: Au  Line: Ka1     Energy:  68.8060 keV  Intensity: 0.4929

# The energy gap between these 2 highest peaks are: 68.806 - 25.271 = 43.535 keV
energy_gap_2_peaks = 43.535

# In the first approach we can choose the measured spectrum from the middle
# channel of the EDXD detector, then locate 2 highest measured peaks (pixel unit),
# then the average energy resolution can be estimated by
# energy_bin = 43.535 / abs(peak_position_1 - peak_position2)

# The following showing how we can extract 2 strong measured peaks

# Select the middle channel
mid_channel = height // 2
meas_spectrum = image[mid_channel]
# Clean data and remove background
plt.figure(figsize=fig_size)
plt.plot(meas_spectrum, color="blue", label="Original")
# Denoising
meas_spectrum = prep.smooth_filter_based_svd(meas_spectrum,
                                             lambda_threshold=0.5, rank=5)
plt.plot(meas_spectrum, color="green", label="Denoised")
# Remove bck
# meas_spectrum = prep.remove_background_based_arpls(meas_spectrum, ld=1e5,
#                                                    weight=0.05, num_iter=10)
# meas_spectrum[:400] = 0.0 # Remove non reliable data
# plt.plot(meas_spectrum, color="red", label="Remove background")
plt.legend()
plt.show()

# Detect and select good peaks using water drain method
peaks = prep.get_peaks_by_ranked_descent(meas_spectrum, width=15, ratio=0.1,
                                         sigma=3)
plt.figure(figsize=fig_size)
plt.plot(np.arange(width), meas_spectrum, color="blue")
plt.plot(np.arange(width)[peaks], meas_spectrum[peaks], "o", color="red")
plt.title("Located peaks")
plt.minorticks_on()
plt.grid(True)
plt.show()

# meas_spectrum = prep.remove_background_based_arpls(meas_spectrum, ld=1e5,
#                                                    weight=0.05, num_iter=10)

# Get peaks of 2 highest peaks
list_inten = np.sort(meas_spectrum[peaks])
pos1 = np.where(meas_spectrum==list_inten[-1])[0][0]
pos2 = np.where(meas_spectrum==list_inten[-2])[0][0]
plt.figure(figsize=fig_size)
plt.plot(np.arange(width), meas_spectrum, color="blue")
plt.plot([pos1, pos2], [list_inten[-1], list_inten[-2]], "o", color="red")
plt.title("Located 2 highest peaks")
plt.minorticks_on()
plt.grid(True)
plt.show()

subpixel_peaks = prep.locate_gaussian_peaks_subpixel(image[mid_channel], [pos1, pos2],
                                                     radius=21, sigma=1,
                                                     skew_fit=True, show=True)
distance_pixel = abs(subpixel_peaks[0] - subpixel_peaks[1])

energy_bin = energy_gap_2_peaks / distance_pixel
print("\n***********************************************")
print(f"Calculated energy bin:  {energy_bin} keV")
print("***********************************************\n")

# The above routine can be repeat for each of the detector row (called strip in GERM detector)

# Now we can calculate the energy resolution at peaks at differrent energy ranges.

# Let use a high energy peak of Uranium

fluorescent_lines = util.get_fluorescence_lines_from_database(["U"],
                                                              line_type="K",
                                                              threshold=0.02,
                                                              check=True)
# Show the theory spectrum of U.
theory_spectrum = util.make_fluorescence_spectrum(fluorescent_lines,
                                                  energy_bin, width,
                                                  sigma=7)

plt.figure(0, figsize=fig_size)
plt.imshow(np.log(image+1), extent=[0, energy_bin * (width - 1), height, 0], aspect='auto', cmap="inferno")
plt.xlabel(" keV (estimated)")
plt.ylabel(" Strip index")
plt.colorbar()
plt.title("Input data, log scale")
plt.tight_layout()


# Element: U   Line: Ka2     Energy:  94.6580 keV  Intensity: 0.2935
# Element: U   Line: Ka1     Energy:  98.4400 keV  Intensity: 0.4731
# Element: U   Line: Kb1     Energy: 111.3030 keV  Intensity: 0.1103
# Element: U   Line: Kb2     Energy: 114.3350 keV  Intensity: 0.0582

meas_spectrum = prep.remove_background_based_arpls(meas_spectrum, ld=1e6,
                                                   weight=0.03, num_iter=10)

# We pick measured U peaks only
min_e = 92.0
max_e = 140.0
idx_list = np.arange(width)
x_list = np.arange(width) * energy_bin
idx_sel = idx_list[(x_list > min_e) & (x_list < max_e)]

x_list = x_list[idx_sel]
y_list = meas_spectrum[idx_sel]

plt.figure(1, figsize=fig_size)
plt.plot(x_list, y_list/ np.max(y_list), color="red",
         label="Measured U peaks (not calibrated)")

x_list1 = theory_spectrum[:, 0]
idx_sel1 = idx_list[(x_list1 > min_e) & (x_list1 < max_e)]
x_list1 = x_list1[idx_sel1]
y_list1 = theory_spectrum[:, 1][idx_sel1]
plt.plot(x_list1, y_list1 / np.max(y_list1), color='blue', label="Uranium theory peaks")
plt.xlabel("Energy (keV)")
plt.ylabel("Intensity")
plt.title(f"Measured spectrum (not calibrated) and U theory peaks")
plt.legend()
plt.grid()
plt.show()

# We can pick the 114.335 keV peak of U

# Element: U   Line: Ka2     Energy:  94.6580 keV  Intensity: 0.2935
# Element: U   Line: Ka1     Energy:  98.4400 keV  Intensity: 0.4731
# Element: U   Line: Kb1     Energy: 111.3030 keV  Intensity: 0.1103
# Element: U   Line: Kb2     Energy: 114.3350 keV  Intensity: 0.0582

peaks = prep.get_peaks_by_ranked_descent(y_list, width=21, ratio=0.3,
                                         sigma=5)

plt.figure(figsize=fig_size)
plt.plot(x_list, y_list, color="blue")
plt.plot(x_list[peaks], y_list[peaks],"o", color="red")
plt.show()

# We pick the peak at 114.335 keV and perform Gaussian fitting
pos = peaks[-1]
radius = 30
list_data = y_list[pos-radius: pos+ radius + 1]
results = prep.fit_gaussian_peak(list_data, sigma=0, return_para=True)
fit_data, a, b, c, d, can_fit = results
plt.plot(list_data, "o", color="blue")
plt.plot(fit_data, color="red")
plt.title(f"Gaussian fit")
plt.grid(True)
plt.show()

full_width_half_max_pixel = 2.354 * b

# Energy resolution at 114.335 keV
energy_resolution = 100 * (full_width_half_max_pixel * energy_bin) / 114.335

print("\n***********************************************")
print(f"Calculated energy resolution at 114.335 (delta_E / E %):  {energy_resolution}")
print("***********************************************\n")