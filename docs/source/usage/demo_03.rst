.. _demo_03:

Energy calibration of the full detector
=======================================

After finding good parameters for a single strip (:ref:`demo_02`), the energy
calibration can be applied to all strips of the detector. Each row of the image
is treated as an independent spectrum and processed in parallel.

- Load the data and get the reference energies as shown in :ref:`demo_02`:

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
        list_element = ["Sn", "Au", "Pb", "U"]
        energy_bin = 0.0538357407779  # keV

        image = 0.5 * (losa.load_image(file_path1) + losa.load_image(file_path2))
        (height, width) = image.shape

        fluorescent_lines = util.get_fluorescence_lines_from_database(
            list_element, line_type="K", threshold=0.01, norm=False)
        theory_spectrum = util.make_fluorescence_spectrum(fluorescent_lines,
                                                          energy_bin, width, sigma=7)
        list_theo_inten = theory_spectrum[:, 1]
        theo_peaks = prep.get_peaks_by_ranked_descent(list_theo_inten, width=5,
                                                      ratio=0.1, sigma=0)
        theo_peaks_sub = prep.locate_gaussian_peaks_subpixel(list_theo_inten,
                                                             theo_peaks, radius=2,
                                                             sigma=1, skew_fit=False)
        reference_energies = theo_peaks_sub * energy_bin

    .. figure:: figs/demo_03/fig1.png
        :name: fig_demo03_1
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Measured spectra (left) and theoretical spectrum (right).

- Calibrate all strips. Because the signal-to-noise ratio and the usable
  energy range vary across the detector, the strips are split into two groups
  with different parameters:

    .. code-block:: python

        sub_peaks_para = {"radius": 21, "sigma": 0, "skew_fit": True}
        match_peaks_para = {"tol": 0.4, "size": 31, "margin": 20,
                            "search_range": (-3, 3), "bounds": None,
                            "initial_guess": None}
        results1 = proc.calibrate_energy_image(
            image[:32], reference_energies, energy_bin, fit_order=2,
            zero_left=200, zero_right=2000, sigma=4,
            smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
            get_peaks_para={"width": 19, "ratio": 0.1, "sigma": 0},
            sub_peaks_para=sub_peaks_para, match_peaks_para=match_peaks_para)
        results2 = proc.calibrate_energy_image(
            image[32:], reference_energies, energy_bin, fit_order=2,
            zero_left=300, zero_right=2500, sigma=5,
            smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
            get_peaks_para={"width": 17, "ratio": 0.1, "sigma": 0},
            sub_peaks_para=sub_peaks_para, match_peaks_para=match_peaks_para)
        mat_coef = np.concatenate((results1, results2), axis=0)
        # Save the coefficients for later use
        losa.save_csv("F:/tmp/algodif/energy_calibration_coefs.csv", mat_coef)

- Apply the calibration to the image:

    .. code-block:: python

        mat_correct = post.correct_energy_image(image, mat_coef, energy_bin)

        extent = [0, energy_bin * (width - 1), height, 0]
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(np.log(mat_correct + 1), extent=extent, aspect="auto",
                       cmap="inferno")
        axes[0].set_title("Calibrated spectrum, log scale")
        axes[1].imshow(np.log(np.tile(list_theo_inten, (height, 1)) * 2000 + 1),
                       extent=extent, aspect="auto", cmap="inferno")
        axes[1].set_title("Theory spectrum")
        plt.show()

    .. figure:: figs/demo_03/fig2.png
        :name: fig_demo03_2
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Calibrated spectra (left) and theoretical spectrum (right).

Click `here <https://github.com/algodif/algodif/blob/main/examples/calibrate_energy_full.py>`__
to download the Python codes.
