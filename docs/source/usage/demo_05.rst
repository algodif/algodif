.. _demo_05:

q-calibration of the full GeRM detector
=======================================

After finding good parameters for a single strip (:ref:`demo_04`), the
q-calibration can be applied to all strips of the GeRM detector. For each strip,
the energies of the reference peaks are estimated from the detector geometry.

- Load the data, the reference d-spacing values, and estimate the q-step:

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        import algodif.io.loadersaver as losa
        import algodif.proc.processing as proc
        import algodif.post.postprocessing as post
        import algodif.util.utility as util

        file_path = "../data/Cr2O3_6_degree_GeRM.tif"
        ref_path = "../data/Pearson_Crystal_Data_Cr2O3.csv"
        output_base = "F:/tmp/algodif/q_calibration_full/"
        fit_order = 2
        two_theta = 6.0  # Degree
        energy_bin = 0.0538357407779  # keV

        image = losa.load_image(file_path)
        (height, width) = image.shape
        list_d_intensity_ref = losa.load_d_spacing_from_pearson_crystal_data_file(
            ref_path, num_use=18, threshold=35, check=True)
        d_spacing_values = list_d_intensity_ref[:, 0]
        intensities_ref = list_d_intensity_ref[:, 1]
        q_values_ref = 2 * np.pi / d_spacing_values

        q_step = util.estimate_q_step_germ_detector(0, energy_bin, two_theta,
                                                    total_germ_strip=192,
                                                    strip_size=0.2,
                                                    sample_slit=200.0,
                                                    sample_detector=2000.0)
        diff_pattern_ref = util.make_diffraction_pattern(q_values_ref,
                                                         intensities_ref, q_step,
                                                         width, sigma=7)

    .. figure:: figs/demo_05/fig1.png
        :name: fig_demo05_1
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Measured diffraction patterns (left) and the reference pattern (right).

- Calibrate all strips in parallel and save the coefficients:

    .. code-block:: python
        :emphasize-lines: 1-13

        mat_coef = proc.calibrate_q_germ_detector_image(
            image, d_spacing_values, two_theta, energy_bin, energy_guess=None,
            fit_order=fit_order, zero_left=450, zero_right=0, sigma=0,
            smooth_svd_para={"rank": 5, "lambda_threshold": 0.5},
            get_peaks_para={"width": 31, "ratio": 0.05, "sigma": 3,
                            "abs_threshold": None},
            sub_peaks_para={"radius": 21, "sigma": 2, "skew_fit": True},
            match_peaks_para={"tol": 0.4, "size": 21, "margin": 20,
                              "search_range": (-10, 10), "bounds": None,
                              "initial_guess": None},
            return_check=False, ncore=None, prefer="processes")
        losa.save_csv(output_base + "/calibration_coefs.csv", mat_coef)

- Apply the calibration to the image:

    .. code-block:: python

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

    .. figure:: figs/demo_05/fig2.png
        :name: fig_demo05_2
        :figwidth: 100 %
        :align: center
        :figclass: align-center

        Calibrated diffraction patterns (left) and the reference pattern
        (right). The diffraction lines are straight after calibration.

Click `here <https://github.com/algodif/algodif/blob/main/examples/q_calibration_full.py>`__
to download the Python codes.
