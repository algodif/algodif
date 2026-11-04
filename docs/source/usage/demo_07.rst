.. _demo_07:

Direct q-calibration of the full detector
=========================================

This workflow applies the direct q-calibration (:ref:`demo_06`) to all strips.
The q-step can be a single value for all strips or a list of values, one per
strip. Here, the q-step of each strip is estimated from the GeRM geometry, but
it can also be measured from the data of each strip as shown in
:ref:`demo_06`.

- Load the data and estimate the q-step of each strip:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        import algodif.io.loadersaver as losa
        import algodif.proc.processing as proc
        import algodif.post.postprocessing as post
        import algodif.util.utility as util

        file_path = "../data/Cr2O3_6_degree_GeRM.tif"
        ref_path = "../data/Pearson_Crystal_Data_Cr2O3.csv"
        two_theta = 6.0  # Degree
        energy_bin = 0.0538357407779  # keV
        fit_order = 2

        image = losa.load_image(file_path)
        (height, width) = image.shape
        q_step_list = [util.estimate_q_step_germ_detector(
            idx, energy_bin, two_theta, total_germ_strip=192, strip_size=0.2,
            sample_slit=200.0, sample_detector=2000.0) for idx in range(height)]
        q_step = np.max(q_step_list)

        list_d_intensity_ref = losa.load_d_spacing_from_pearson_crystal_data_file(
            ref_path, num_use=21, threshold=25, check=True)
        d_spacing_values = list_d_intensity_ref[:, 0]
        intensities_ref = list_d_intensity_ref[:, 1]
        q_values_ref = 2 * np.pi / d_spacing_values
        diff_pattern_ref = util.make_diffraction_pattern(q_values_ref,
                                                         intensities_ref, q_step,
                                                         width, sigma=7)

    .. figure:: figs/demo_07/fig1.png
        :name: fig_demo07_1
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Measured diffraction patterns (left) and the reference pattern (right).

- Calibrate all strips and apply the calibration:

    .. code-block:: python
        :emphasize-lines: 1-12

        mat_coef = proc.calibrate_q_image(
            image, q_values_ref, q_step_list, fit_order=fit_order, zero_left=450,
            zero_right=-300, sigma=0,
            smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
            get_peaks_para={"width": 15, "ratio": 0.1, "sigma": 3,
                            "abs_threshold": None},
            sub_peaks_para={"radius": 15, "sigma": 1, "skew_fit": True},
            match_peaks_para={"tol": 0.4, "size": 21, "margin": 20,
                              "search_range": (-0.5, 0.5), "bounds": None,
                              "initial_guess": None},
            return_check=False, ncore=None, prefer="processes")
        mat_correct = post.correct_q_image(image, mat_coef, q_step, ncore=None)

        extent = [0, q_step * (width - 1), height, 0]
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(np.log(mat_correct + 1), extent=extent, aspect="auto",
                       cmap="inferno")
        axes[0].set_title("Calibrated pattern, log scale")
        axes[1].imshow(np.log(np.tile(diff_pattern_ref[:, 1], (height, 1)) * 2000 + 1),
                       extent=extent, aspect="auto", cmap="inferno")
        axes[1].set_title("Reference pattern")
        plt.show()

    .. figure:: figs/demo_07/fig2.png
        :name: fig_demo07_2
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Calibrated diffraction patterns (left) and the reference pattern
        (right).

Click `here <https://github.com/algodif/algodif/blob/main/examples/q_calibration_full_direct.py>`__
to download the Python codes.
