#!/nsls2/conda/envs/2023-2.0-py310/bin/python

# ============================
# CLI for processing EDDT data
# ============================

import os
import glob
import shutil
import sys
import argparse
import numpy as np
import scipy.ndimage as ndi
import algodif.io.loadersaver as losa
import algodif.prep.preprocessing as prep
import algodif.post.postprocessing as post


proposal = "pass-320871"
cycle = "2026-2"
beamline_folder = "/nsls2/data/hex/proposals/"
technique = "edxd"

parser = argparse.ArgumentParser(description="Process data acquired by the Germ strip detector")
parser.add_argument("-s", dest="scan_number", help="Scan number (integer)", type=int, required=True)
parser.add_argument("-e", dest="scan_number2", help="End of scan number (to process multiple datasets). Default is None", type=int, 
                    required=False, default=None)
parser.add_argument("-p", dest="calib_path", help="Path to the calibration csv file",
                    type=str, required=False, default="/nsls2/data/hex/proposals/2026-2/pass-320871/edxd/calibration/scan_00001/results/scan_00001_3_0deg_q_calib.csv")
parser.add_argument("-r", dest="remove_background", help="To enable or disable (1 or 0) background removal. Default is 1.",
                    type=int, required=False, default=1)
parser.add_argument("--qstep", dest="q_step", help="Select one of options for calculating q-step: 'mean', 'median' or 'max'. Default is 'mean'",
                    type=str, required=False, default="mean")
parser.add_argument("--qmax", dest="q_max", help="Select q-max of processed data. If set, this will overwrite the q-step option. Default is None",
                    type=float, required=False, default=None)

args = parser.parse_args()
scan_number = args.scan_number
scan_number2 = args.scan_number2
if scan_number2 is not None:
    scan1 = min(scan_number, scan_number2)
    scan2 = max(scan_number, scan_number2)
    list_scan = range(scan1, scan2 + 1)
else:
    list_scan = [scan_number]
qstep_opt = args.q_step
q_max = args.q_max
remove_background = args.remove_background
calib_path = args.calib_path

input_base = "raw_data"
output_base = "processed"

list_input = []
list_scan1 = []
for scan in list_scan:
    name = "scan_" + ("0000" + str(scan))[-5:]
    list_input.append(name)
    list_scan1.append(scan)

print("\n*********************************************************")
print("            Process Germ strip detector raw data          ")
print("*********************************************************")
print("\n...\n")


# Check if the proposal folder exists
proposal_base = beamline_folder + "/" + cycle + "/" + proposal
if not os.path.exists(proposal_base):
    raise ValueError(f"There's no such folder: {proposal_base}")

for ii, scan in enumerate(list_input):
    input_folder = proposal_base + "/" + technique + "/" + input_base + "/" + scan + "/"
    output_folder = proposal_base + "/" + technique + "/" + output_base + "/" + scan + "/"
    list_files = losa.find_file(input_folder + "/tifs/*.tif*")
    if len(list_files) == 0:
        raise ValueError("Can't find tif files under folder {0}".format(input_folder + "/tifs/"))
    print("=========================================================")
    print("Process the scan: {0}. Total image: {1}".format(scan, len(list_files)))
    print("=========================================================")    

    metadata = losa.load_json(losa.find_file(input_folder + "/*json")[0])
    # theta_measured = metadata["Metadata"]["mcf5"]["EDXD_Axis_Rx"]
    q_calib = losa.load_csv(calib_path)

    num_file = len(list_files)
    (height, width) = np.shape(losa.load_image(list_files[0]))    
    q_mat = post.get_q_map(height, width, q_calib)
    if q_max is None:
        if qstep_opt == "max":
            q_step = np.max(np.diff(q_mat, axis=1))
        elif qstep_opt == "median":
            q_step = np.median(np.diff(q_mat, axis=1))
        else:
            q_step = np.mean(np.diff(q_mat, axis=1))
    else:
        q_step = q_max / (width - 1.0)
    losa.save_csv(q_calib, output_folder + "/q_calibration_coefficients.csv")
    losa.save_image(output_folder + "/q_map.tif", q_mat)
    metadata["q_step"] = float(q_step)
    losa.save_json(output_folder + "/metadata.json", metadata)

    for i, file in enumerate(list_files):
        mat = losa.load_image(file)
        mat_corrected0 = post.correct_q_image(mat, q_calib, q_step)
        file_name = os.path.splitext(os.path.basename(file))[0]
        losa.save_image(output_folder + "/tifs_with_bgr/" + file_name + ".tif", mat_corrected0)
        if remove_background != 0:
            mat_corrected = prep.remove_image_background_ranked_descent(mat_corrected0, width=15,
                                                                        ratio=0.1, sigma=5)
            mat_corrected = np.clip(mat_corrected, 0, None)
            losa.save_image(output_folder + "/tifs_without_bgr/" + file_name + ".tif", mat_corrected)
        if i % 5 == 0 and i > 0:
            print("     ----> Done up to: {}".format(i))

    print("  =======================================================")
    print("  Done scan: {} ".format(scan))
    print("  Data saved at: \n   {}".format(output_folder))
    print("  =======================================================\n")
