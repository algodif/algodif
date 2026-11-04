.. _demo_01:

Estimate the energy bin and the energy resolution
=================================================

The following workflow shows how to use Algodif to estimate the energy bin
(keV per pixel) and the energy resolution of an EDX detector (the GeRM detector
at the HEX beamline, NSLS-II). The calibration data are fluorescence spectra of
a sample composed of thin foils of Sn, Pb, Au, and a small amount of U, which
are available in the `data <https://github.com/algodif/algodif/tree/main/data>`__
folder.

- Load the data:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        import algodif.io.loadersaver as losa
        import algodif.prep.preprocessing as prep
        import algodif.util.utility as util

        # Measurement of the Sn, Pb, Au sample
        file_path1 = "../data/Pb_Au_Sn_calibration.tif"
        # Measurement of the U sample
        file_path2 = "../data/U_calibration.tif"
        list_element = ["Sn", "Au", "Pb", "U"]
        fig_size = (12, 9)

        image1 = losa.load_image(file_path1)
        image2 = losa.load_image(file_path2)
        image = 0.5 * (image1 + image2)
        (height, width) = image.shape

        plt.imshow(np.log(image + 1), aspect="auto", cmap="inferno")
        plt.colorbar()
        plt.title("Input data, log scale")
        plt.show()

    .. figure:: figs/demo_01/fig1.png
        :name: fig_demo01_1
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Fluorescence spectra recorded by the detector strips (log scale).

- Generate the theoretical spectrum of the elements using the
  `xraydb <https://xraypy.github.io/XrayDB/>`__ database. An initial guess of
  the energy bin is used to plot the spectrum.

    .. code-block:: python

        fluorescent_lines = util.get_fluorescence_lines_from_database(
            list_element, line_type="K", threshold=0.01, norm=False, check=True)
        energy_bin_guess = 0.055  # keV
        theory_spectrum = util.make_fluorescence_spectrum(fluorescent_lines,
                                                          energy_bin_guess, width,
                                                          sigma=7)
        plt.figure(figsize=fig_size)
        plt.plot(theory_spectrum[:, 0], theory_spectrum[:, 1], color="blue")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Intensity")
        plt.show()

    .. figure:: figs/demo_01/fig2.png
        :name: fig_demo01_2
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Theoretical K-line spectrum of Sn, Au, Pb, and U.

- From the printed output of the above command, the two strongest peaks are
  Sn K\ :sub:`α1` (25.271 keV) and Au K\ :sub:`α1` (68.806 keV). The energy gap
  between them is 43.535 keV. Using the spectrum of the middle strip, we denoise
  the data and locate the peaks:

    .. code-block:: python

        mid_channel = height // 2
        meas_spectrum = prep.smooth_filter_based_svd(image[mid_channel],
                                                     lambda_threshold=0.5, rank=5)
        peaks = prep.get_peaks_by_ranked_descent(meas_spectrum, width=15,
                                                 ratio=0.1, sigma=3)

    .. figure:: figs/demo_01/fig3.png
        :name: fig_demo01_3
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Spectrum of the middle strip before and after SVD-based denoising.

    .. figure:: figs/demo_01/fig4.png
        :name: fig_demo01_4
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Peaks located using the ranked-descent method.

- Select the two highest peaks, locate them with sub-pixel accuracy, and
  calculate the energy bin:

    .. code-block:: python
        :emphasize-lines: 10

        energy_gap_2_peaks = 43.535  # keV
        list_inten = np.sort(meas_spectrum[peaks])
        pos1 = np.where(meas_spectrum == list_inten[-1])[0][0]
        pos2 = np.where(meas_spectrum == list_inten[-2])[0][0]
        subpixel_peaks = prep.locate_gaussian_peaks_subpixel(image[mid_channel],
                                                             [pos1, pos2],
                                                             radius=21, sigma=1,
                                                             skew_fit=True)
        distance_pixel = abs(subpixel_peaks[0] - subpixel_peaks[1])
        energy_bin = energy_gap_2_peaks / distance_pixel
        print(f"Calculated energy bin: {energy_bin} keV")

    .. figure:: figs/demo_01/fig5.png
        :name: fig_demo01_5
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        The two highest peaks used to estimate the energy bin.

  This routine can be repeated for each detector row (called a strip in the
  GeRM detector).

- Using the estimated energy bin, we can evaluate the energy resolution at
  different energy ranges. Here, the high-energy peaks of U are used. The
  background is removed, and the measured spectrum in the range of 92-140 keV
  is compared with the theoretical U peaks:

    .. code-block:: python

        fluorescent_lines = util.get_fluorescence_lines_from_database(
            ["U"], line_type="K", threshold=0.02, check=True)
        theory_spectrum = util.make_fluorescence_spectrum(fluorescent_lines,
                                                          energy_bin, width, sigma=7)
        meas_spectrum = prep.remove_background_based_arpls(meas_spectrum, ld=1e6,
                                                           weight=0.03, num_iter=10)
        (min_e, max_e) = (92.0, 140.0)
        x_list = np.arange(width) * energy_bin
        mask = (x_list > min_e) & (x_list < max_e)
        x_list, y_list = x_list[mask], meas_spectrum[mask]

        x_list1 = theory_spectrum[:, 0]
        mask1 = (x_list1 > min_e) & (x_list1 < max_e)
        x_list1, y_list1 = x_list1[mask1], theory_spectrum[:, 1][mask1]

        plt.figure(figsize=fig_size)
        plt.plot(x_list, y_list / np.max(y_list), color="red",
                 label="Measured U peaks (not calibrated)")
        plt.plot(x_list1, y_list1 / np.max(y_list1), color="blue",
                 label="Uranium theory peaks")
        plt.legend()
        plt.show()

    .. figure:: figs/demo_01/fig6.png
        :name: fig_demo01_6
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Measured U peaks (not calibrated) and the theoretical U peaks.

- Pick the U K\ :sub:`β2` peak (114.335 keV), fit a Gaussian to it, and
  calculate the energy resolution from the full width at half maximum (FWHM):

    .. code-block:: python
        :emphasize-lines: 9

        peaks = prep.get_peaks_by_ranked_descent(y_list, width=21, ratio=0.3,
                                                 sigma=5)
        pos = peaks[-1]
        radius = 30
        list_data = y_list[pos - radius: pos + radius + 1]
        fit_data, a, b, c, d, can_fit = prep.fit_gaussian_peak(list_data, sigma=0,
                                                               return_para=True)
        fwhm_pixel = 2.355 * b
        energy_resolution = 100 * (fwhm_pixel * energy_bin) / 114.335
        print(f"Energy resolution at 114.335 keV (dE/E %): {energy_resolution}")

    .. figure:: figs/demo_01/fig7.png
        :name: fig_demo01_7
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Gaussian fit to the U K\ :sub:`β2` peak.

Click `here <https://github.com/algodif/algodif/blob/main/examples/calculate_energy_resolution.py>`__
to download the Python codes.
