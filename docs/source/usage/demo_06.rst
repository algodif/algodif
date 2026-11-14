.. _demo_06:

Direct q-calibration of a single strip
======================================

The methods in :ref:`demo_04` rely on the GeRM detector geometry to estimate
initial guesses. For other instruments, the q-calibration can be performed
directly in q-space without the detector geometry. The only required input is
an estimate of the q-step (q per pixel), which can be extracted from the
measured data as shown below.

- Load the data and the reference diffraction peaks. An initial q-step is only
  used to generate the reference pattern for plotting.

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
        strip_idx = 100
        q_step = 0.0015  # Initial guess
        fig_size = (16, 8)

        image = losa.load_image(file_path)
        (height, width) = image.shape
        calibrant_peaks = util.get_calibrant_diffraction_pattern(
            calibrant, num_use=20, threshold=3.5, check=True)
        q_values_ref = calibrant_peaks[:, 0]
        d_spacing_values = 2 * np.pi / q_values_ref

        diff_pattern_ref = util.make_diffraction_pattern(calibrant_peaks,
                                                         q_step, width,
                                                         sigma=7)
        list_inten_ref = diff_pattern_ref[:, 1]
        ref_peaks = prep.get_peaks_by_ranked_descent(list_inten_ref, width=5,
                                                     ratio=0.1, sigma=0)

- Locate peaks of the measured pattern with sub-pixel accuracy:

    .. code-block:: python

        meas_pattern = prep.smooth_filter_based_svd(image[strip_idx],
                                                    lambda_threshold=0.5, rank=5)
        peaks = prep.get_peaks_by_ranked_descent(meas_pattern, width=15,
                                                 ratio=0.1, sigma=3)
        sub_peaks_meas = prep.locate_gaussian_peaks_subpixel(meas_pattern, peaks,
                                                             radius=15, sigma=1,
                                                             skew_fit=True)

    .. figure:: figs/demo_06/fig1.png
        :name: fig_demo06_1
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Measured pattern and located peaks (left). Reference pattern and its
        major peaks (right).

- Estimate the q-step from the top 3 highest peaks of the measured pattern and
  of the reference pattern. The q-step is the q-distance between the outermost
  reference peaks divided by the pixel distance between the outermost measured
  peaks:

    .. code-block:: python
        :emphasize-lines: 11

        # Top 3 measured peaks
        meas_peaks_3_idx = np.argsort(meas_pattern[peaks])[-3:][::-1]
        top3_meas_peaks = peaks[meas_peaks_3_idx]
        # Top 3 reference peaks
        list_q_ref = diff_pattern_ref[:, 0][ref_peaks]
        ref_peaks_3_idx = np.argsort(list_inten_ref[ref_peaks])[-3:][::-1]
        top3_ref_peaks = ref_peaks[ref_peaks_3_idx]

        pixel_distance = np.ptp(sub_peaks_meas[meas_peaks_3_idx])
        q_distance = np.ptp(list_q_ref[ref_peaks_3_idx])
        q_step = q_distance / pixel_distance
        print(f"Measured q-step: {q_step}")

    .. figure:: figs/demo_06/fig2.png
        :name: fig_demo06_2
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Top 3 peaks of the measured pattern (left) and of the reference
        pattern (right).

- Calibrate the strip directly in q-space. Note that the search range is now
  given in q-unit:

    .. code-block:: python
        :emphasize-lines: 1-12

        results = proc.calibrate_q_1d(
            image[strip_idx], q_values_ref, q_step, fit_order=fit_order,
            zero_left=450, zero_right=-300, sigma=0,
            smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
            get_peaks_para={"width": 15, "ratio": 0.1, "sigma": 3,
                            "abs_threshold": None},
            sub_peaks_para={"radius": 15, "sigma": 1, "skew_fit": True},
            match_peaks_para={"tol": 0.4, "size": 31, "margin": 20,
                              "search_range": (-0.5, 0.5), "bounds": None,
                              "initial_guess": None},
            peaks=None, return_check=True)
        list_fact, loc_peaks, loc_q_values = results
        print(f"Polynomial coefficients (a0 + a1*x + a2*x^2): {list_fact}")

    .. figure:: figs/demo_06/fig3.png
        :name: fig_demo06_3
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Measured peaks matched to the reference peaks.

    .. figure:: figs/demo_06/fig4.png
        :name: fig_demo06_4
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Mapping between peak locations and reference q-values.

    .. figure:: figs/demo_06/fig5.png
        :name: fig_demo06_5
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Polynomial fit and its residual (shifted for display).

- Apply the calibration:

    .. code-block:: python

        correct_data = post.correct_q_1d(meas_pattern, list_fact, q_step)

    .. figure:: figs/demo_06/fig6.png
        :name: fig_demo06_6
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Calibrated pattern compared with the reference pattern.

Click `here <https://github.com/algodif/algodif/blob/main/examples/q_calibration_single_strip_direct.py>`__
to download the Python codes.
