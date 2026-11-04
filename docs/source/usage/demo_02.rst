.. _demo_02:

Energy calibration of a single detector strip
=============================================

The following workflow shows how to calibrate the energy axis of a single
detector strip using fluorescence spectra of Sn, Pb, Au, and U. The energy bin
used here was estimated as shown in :ref:`demo_01`.

- Load the data and set parameters:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        import algodif.io.loadersaver as losa
        import algodif.prep.preprocessing as prep
        import algodif.proc.processing as proc
        import algodif.post.postprocessing as post
        import algodif.util.utility as util

        file_path1 = "../data/Pb_Au_Sn_calibration.tif"
        file_path2 = "../data/U_calibration.tif"
        fit_order = 2
        list_element = ["Sn", "Au", "Pb", "U"]
        energy_bin = 0.0538357407779  # keV
        strip_idx = 60  # Strip to be calibrated
        fig_size = (16, 8)

        image = 0.5 * (losa.load_image(file_path1) + losa.load_image(file_path2))
        (height, width) = image.shape

    .. figure:: figs/demo_02/fig1.png
        :name: fig_demo02_1
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Input data, log scale.

- Generate the theoretical spectrum and get the reference energies. Note that
  the peak positions are determined from the simulated spectrum, so the impact
  of the Gaussian smoothing used for simulation is included.

    .. code-block:: python

        fluorescent_lines = util.get_fluorescence_lines_from_database(
            list_element, line_type="K", threshold=0.01, norm=False, check=True)
        theory_spectrum = util.make_fluorescence_spectrum(fluorescent_lines,
                                                          energy_bin, width, sigma=7)
        list_theo_inten = theory_spectrum[:, 1]
        theo_peaks = prep.get_peaks_by_ranked_descent(list_theo_inten, width=5,
                                                      ratio=0.1, sigma=0)
        theo_peaks_sub = prep.locate_gaussian_peaks_subpixel(list_theo_inten,
                                                             theo_peaks, radius=2,
                                                             sigma=1, skew_fit=False)
        reference_energies = theo_peaks_sub * energy_bin

    .. figure:: figs/demo_02/fig2.png
        :name: fig_demo02_2
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Theoretical spectrum and the peaks used for calibration.

- Check the peak-detection step on the selected strip. This helps to choose
  parameters for the calibration method in the next step.

    .. code-block:: python

        meas_spectrum = prep.smooth_filter_based_svd(image[strip_idx],
                                                     lambda_threshold=0.5, rank=5)
        peaks = prep.get_peaks_by_ranked_descent(meas_spectrum, width=15,
                                                 ratio=0.1, sigma=3)

    .. figure:: figs/demo_02/fig3.png
        :name: fig_demo02_3
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Spectrum before and after SVD-based denoising.

    .. figure:: figs/demo_02/fig4.png
        :name: fig_demo02_4
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Located peaks.

- Calibrate the strip. The method detects peaks, refines them with sub-pixel
  accuracy, matches them to the reference energies, and fits a polynomial
  mapping from pixel index to energy.

    .. code-block:: python
        :emphasize-lines: 1-13

        results = proc.calibrate_energy_1d(
            image[strip_idx], reference_energies, energy_bin,
            fit_order=fit_order, zero_left=450, zero_right=-300, sigma=0,
            smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
            get_peaks_para={"width": 15, "ratio": 0.1, "sigma": 3},
            sub_peaks_para={"radius": 15, "sigma": 1, "skew_fit": True},
            match_peaks_para={"tol": 0.4, "size": 31, "margin": 20,
                              "search_range": (-3, 3), "bounds": None,
                              "initial_guess": None},
            peaks=None, return_check=True)
        list_fact, loc_peaks, energies = results
        print(f"Polynomial coefficients (a0 + a1*x + a2*x^2): {list_fact}")
        print(f"Peak locations (pixel): {loc_peaks}")
        print(f"Reference energies (keV): {energies}")

    .. figure:: figs/demo_02/fig5.png
        :name: fig_demo02_5
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Measured peaks matched to the reference peaks.

- Check the results. All possible pairs of (peak location, reference energy)
  are plotted in blue, the matched pairs in red. Good fitting is indicated by
  the green line crossing the matched pairs.

    .. code-block:: python

        array_points = np.asarray([(p, e) for p in loc_peaks for e in energies])
        x_list = np.linspace(loc_peaks[0], loc_peaks[-1], 100)
        list_e_fit = np.polynomial.polynomial.polyval(x_list, list_fact)
        plt.figure(figsize=fig_size)
        plt.plot(array_points[:, 0], array_points[:, 1], ".", color="blue")
        plt.plot(loc_peaks, energies, "o", color="red")
        plt.plot(x_list, list_e_fit, color="green")
        plt.show()

    .. figure:: figs/demo_02/fig6.png
        :name: fig_demo02_6
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Mapping between peak locations and reference energies.

    .. figure:: figs/demo_02/fig7.png
        :name: fig_demo02_7
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Polynomial fit and its residual (shifted for display).

- Apply the calibration to the spectrum and compare with the theory:

    .. code-block:: python

        list_data = prep.smooth_filter_based_svd(image[strip_idx],
                                                 lambda_threshold=0.5, rank=5)
        correct_data = post.correct_energy_1d(list_data, list_fact, energy_bin)
        plt.figure(figsize=fig_size)
        plt.plot(theory_spectrum[:, 0], list_theo_inten / np.max(list_theo_inten),
                 color="blue", label="Theory")
        plt.plot(np.arange(width) * energy_bin, correct_data / np.max(correct_data),
                 color="red", label="Calibrated")
        plt.vlines(energies, 0.0, 1.0, colors="green", label="Reference lines")
        plt.xlabel("Energy (keV)")
        plt.legend()
        plt.show()

    .. figure:: figs/demo_02/fig8.png
        :name: fig_demo02_8
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Calibrated spectrum compared with the theoretical spectrum.

Click `here <https://github.com/algodif/algodif/blob/main/examples/calibrate_energy_single_strip.py>`__
to download the Python codes.
