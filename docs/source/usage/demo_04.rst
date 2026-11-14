.. _demo_04:

q-calibration of a single strip using the GeRM geometry
=======================================================

The following workflow shows how to calibrate the q-axis (reciprocal space) of
a single strip of the GeRM detector using the diffraction pattern of
Cr\ :sub:`2`\ O\ :sub:`3`. The detector geometry is used to estimate the energy
of each reference peak, which provides a good initial guess for peak matching.

- Load the data and the reference diffraction peaks. The q-values and
  intensities of Cr\ :sub:`2`\ O\ :sub:`3` are loaded from the calibrants
  bundled with algodif (see ``util.list_xrd_calibrants()``):

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        import algodif.io.loadersaver as losa
        import algodif.prep.preprocessing as prep
        import algodif.proc.processing as proc
        import algodif.post.postprocessing as post
        import algodif.util.utility as util

        file_path = "../data/Cr2O3_6_degree_GeRM.tif"
        calibrant = "Cr2O3"
        fit_order = 2
        two_theta = 6.0  # Degree
        energy_bin = 0.0538357407779  # keV
        strip_idx = 100
        fig_size = (16, 8)

        image = losa.load_image(file_path)
        (height, width) = image.shape
        calibrant_peaks = util.get_calibrant_diffraction_pattern(
            calibrant, num_use=16, threshold=3.5, check=True)
        q_values_ref = calibrant_peaks[:, 0]
        d_spacing_values = 2 * np.pi / q_values_ref

    .. figure:: figs/demo_04/fig1.png
        :name: fig_demo04_1
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Input data, log scale. The tilt of the diffraction lines reflects the
        variation of the scattering angle across the detector strips.

- Estimate the q-step from the detector geometry and generate the reference
  pattern:

    .. code-block:: python

        q_step = util.estimate_q_step_germ_detector(0, energy_bin, two_theta,
                                                    total_germ_strip=192,
                                                    strip_size=0.2,
                                                    sample_slit=200.0,
                                                    sample_detector=2000.0)
        diff_pattern_ref = util.make_diffraction_pattern(calibrant_peaks,
                                                         q_step, width,
                                                         sigma=7)
        ref_peaks = prep.get_peaks_by_ranked_descent(diff_pattern_ref[:, 1],
                                                     width=5, ratio=0.1, sigma=0)

    .. figure:: figs/demo_04/fig2.png
        :name: fig_demo04_2
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Reference diffraction pattern of Cr\ :sub:`2`\ O\ :sub:`3` and the
        peaks used for calibration.

- Check the peak-detection step on the selected strip:

    .. code-block:: python

        meas_pattern = prep.smooth_filter_based_svd(image[strip_idx],
                                                    lambda_threshold=0.5, rank=5)
        peaks = prep.get_peaks_by_ranked_descent(meas_pattern, width=15,
                                                 ratio=0.1, sigma=3)

    .. figure:: figs/demo_04/fig3.png
        :name: fig_demo04_3
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Diffraction pattern before and after SVD-based denoising.

    .. figure:: figs/demo_04/fig4.png
        :name: fig_demo04_4
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Located peaks.

- Calibrate the strip:

    .. code-block:: python
        :emphasize-lines: 1-14

        results = proc.calibrate_q_germ_detector_1d(
            strip_idx, image[strip_idx], d_spacing_values, two_theta, energy_bin,
            energy_guess=None, fit_order=fit_order, zero_left=450,
            zero_right=-300, sigma=0,
            smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
            get_peaks_para={"width": 15, "ratio": 0.1, "sigma": 3,
                            "abs_threshold": None},
            sub_peaks_para={"radius": 15, "sigma": 1, "skew_fit": True},
            match_peaks_para={"tol": 0.4, "size": 31, "margin": 20,
                              "search_range": (-10, 10), "bounds": None,
                              "initial_guess": None},
            peaks=None, return_check=True)
        list_fact, loc_peaks, loc_q_values = results
        print(f"Polynomial coefficients (a0 + a1*x + a2*x^2): {list_fact}")

    .. figure:: figs/demo_04/fig5.png
        :name: fig_demo04_5
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Measured peaks matched to the reference peaks.

    .. figure:: figs/demo_04/fig6.png
        :name: fig_demo04_6
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Mapping between peak locations and reference q-values.

    .. figure:: figs/demo_04/fig7.png
        :name: fig_demo04_7
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Polynomial fit and its residual (shifted for display).

- Apply the calibration and compare with the reference pattern:

    .. code-block:: python

        correct_data = post.correct_q_1d(meas_pattern, list_fact, q_step)
        list_inten_ref = diff_pattern_ref[:, 1]
        plt.figure(figsize=fig_size)
        plt.plot(diff_pattern_ref[:, 0], list_inten_ref / np.max(list_inten_ref),
                 color="blue", label="Database")
        plt.plot(np.arange(width) * q_step, correct_data / np.max(correct_data),
                 color="red", label="Calibrated")
        plt.vlines(loc_q_values, 0.0, 1.0, colors="green", label="Reference lines")
        plt.xlabel("Q (1/A)")
        plt.legend()
        plt.show()

    .. figure:: figs/demo_04/fig8.png
        :name: fig_demo04_8
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Calibrated pattern compared with the reference pattern.

Click `here <https://github.com/algodif/algodif/blob/main/examples/q_calibration_single_strip.py>`__
to download the Python codes.
