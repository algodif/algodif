#!/nsls2/conda/envs/2023-3.2-py310/bin/python

# ============================
# GUI for viewing 1D-scan EDDT
# ============================

import os
import re
import sys
import glob
import time
import signal
import timeit
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkFont
import matplotlib
matplotlib.use("TkAgg")

import algodif.io.loadersaver as losa
import algodif.post.postprocessing as post


# Constants for GUI Font
GUI_FONT_NAME = ""
FONT_SIZE = 12
FONT_WEIGHT = "bold"
FIG_DPI = 100
PLOT_FIG_SIZE = (24, 9)
WINDOW_SCREEN_FRACTION = (0.9, 0.85)  # initial window size relative to the screen
PLOT_FRAME = {"bottom" : 0.065, "left":0.035, "right": 0.99, "top":0.97}

proposal = "pass-318635"
cycle = "2025-2"
beamline_folder = "/nsls2/data/hex/proposals/"
technique = "edxd"

parser = argparse.ArgumentParser(description="Process data and display q-slice image!!!")
parser.add_argument("-s", dest="scan_number", help="Scan number (integer)", type=int, required=True)
parser.add_argument("-p", dest="calib_path", help="Path to the calibration csv file",
                    type=str, required=False, default="/nsls2/data/hex/proposals/2025-2/pass-318635/edxd/calibration/scan_00001/results/scan_00001_3_0deg_q_calib.csv")
parser.add_argument("-g", dest="background", help="To choose image with or without (1 or 0) background. Default is 1.",
                    type=int, required=False, default=1)
args = parser.parse_args()

scan_number = args.scan_number
with_bgr = args.background
calib_path = args.calib_path

q_calib = losa.load_csv(calib_path)
# Check if the proposal folder exists
proposal_base = beamline_folder + "/" + cycle + "/" + proposal
if not os.path.exists(proposal_base):
    raise ValueError(f"There's no such folder: {proposal_base}")

scan_name = "scan_" + ("0000" + str(scan_number))[-5:]
input_folder = proposal_base + "/" + technique + "/processed/" + scan_name + "/"
input_raw = proposal_base + "/" + technique + "/raw_data/" + scan_name + "/"
output_folder = proposal_base + "/" + technique + "/tmp/" + scan_name + "/"

metadata = losa.load_json(losa.find_file(input_folder + "/*json")[0])
key = losa.find_key_of_list_json(metadata)
if len(key) == 0:
    raise ValueError("There's no scanning motor metadata!!!")
else:
    key = key[0]

motor_positions = losa.get_list_by_key_json(metadata, key)
if not isinstance(motor_positions, list):
    print("\n!!!There's only one image in this scan: {}\n".format(scan_name))
    motor_positions = [motor_positions]

# theta_measured = metadata["Metadata"]["Metadata"]["mcf5"]["EDXD_Axis_Rx"]
q_step = metadata["Metadata"]["q_step"]

if with_bgr == 0:
    list_file = losa.find_file(input_folder + "/tifs_without_bgr/*.tif*")
    if len(list_file) == 0:
        print("!!!! There's no image with background removal. Load image with background instead!!!")
        list_file = losa.find_file(input_folder + "/tifs_with_bgr/*.tif*")
else:
    list_file = losa.find_file(input_folder + "/tifs_with_bgr/*.tif*")
list_raw = losa.find_file(input_raw + "/tifs/*.tif*")

num_file = len(list_file)
data = np.asarray([losa.load_image(file) for file in list_file])
(depth, height, width) = data.shape

print("=========================================================\n")
print(f"Showing the scan: {scan_name}. Total image: {num_file}")
# print(f"Theta measured: {theta_measured:3f}")
print("=========================================================\n")


class EDXDViewerApp:
    def __init__(self, root, data, radius=11):
        self.root = root
        self.root.title(f"EDDT Viewer: {scan_name}")
        self.z_resolution = 1.0 # gauge volume, change to suit experiment

        # Apply theme and global font for Tkinter
        self.style = ttk.Style()
        self.style.theme_use("clam")

        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.config(size=FONT_SIZE, weight=FONT_WEIGHT)

        self.style.configure(".", font=default_font)
        self.root.option_add("*Font", default_font)

        # Sync Matplotlib font with GUI Constants
        mpl_size = (FONT_SIZE - 2) * float(self.root.tk.call("tk", "scaling")) * 72.0 / FIG_DPI
        plt.rcParams.update({"font.family": GUI_FONT_NAME, "font.size": mpl_size,
                             "axes.titlesize": mpl_size, "axes.labelsize": mpl_size,
                             "xtick.labelsize": mpl_size, "ytick.labelsize": mpl_size})

        self.data = data
        self.radius = radius
        self.depth, self.height, self.width = data.shape
        self.q_max = (self.width - 1) * q_step
        self.q_list = np.arange(self.width) * q_step
        self.clicked_row = None
        self.selected_q = 0.0
        self.ave_image = None
        self.selection_line = None
        self.selection_span = None
        self.ax1_hline = None
        self.ax1_vline = None
        self.radius_var = tk.StringVar(value=str(radius))
        self.z_resolution_var = tk.StringVar(value=str(self.z_resolution))

        self.setup_ui()
        self.update_axis0()
        self.update_axis2()

        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        signal.signal(signal.SIGINT, lambda signum, frame: self.exit_app())
        self._poll_signals()

    def _poll_signals(self):
        self.root.after(200, self._poll_signals)

    def exit_app(self):
        print("================")
        print("Exit EDDT Viewer")
        print("================")
        self.root.destroy()

    def rescale_data(self, img):
        img_res = np.log(img - np.min(img) + 1.0)
        return img_res

    def setup_ui(self):
        # Create Main Frame
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib Figure with minimized margins
        self.fig = Figure(figsize=PLOT_FIG_SIZE, dpi=FIG_DPI)
        self.ax = self.fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.2})
        # self.fig.tight_layout(pad=1.0)
        self.fig.subplots_adjust(bottom=PLOT_FRAME["bottom"], left=PLOT_FRAME["left"], 
                                    right=PLOT_FRAME["right"], top=PLOT_FRAME["top"])


        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.mpl_connect("button_press_event", self.plot_intensity_along_row)

        # Controls and toolbar are packed before the canvas so they are never clipped on a small screen
        controls_frame = ttk.Frame(main_frame, padding="0")
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
        controls_frame.columnconfigure(1, weight=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(2, 0))

        # Same window size relative to any screen; the figure follows the canvas size
        self.root.geometry("{0}x{1}".format(int(WINDOW_SCREEN_FRACTION[0] * self.root.winfo_screenwidth()),
                                            int(WINDOW_SCREEN_FRACTION[1] * self.root.winfo_screenheight())))
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.idx0_var = tk.StringVar(value="0")
        self.idx1_var = tk.StringVar(value="0")
        int_vcmd = (self.root.register(self._validate_int_input), "%P")

        # Row 0: Axis 0
        ttk.Label(controls_frame, text="Axis 0 (motor-position):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.slider0 = ttk.Scale(controls_frame, from_=0, to=max(0, self.depth - 1), orient=tk.HORIZONTAL,
                                 command=lambda x: self.update_axis0())
        self.slider0.grid(row=0, column=1, sticky="we", padx=5, pady=2)
        self.idx0_entry = ttk.Entry(controls_frame, textvariable=self.idx0_var, width=8, justify=tk.CENTER,
                                    validate="key", validatecommand=int_vcmd)
        self.idx0_entry.grid(row=0, column=2, padx=5)
        self.idx0_entry.bind("<Return>", lambda e: self._on_index_entry(self.slider0, self.idx0_var, self.update_axis0))
        self.idx0_entry.bind("<FocusOut>", lambda e: self._on_index_entry(self.slider0, self.idx0_var, self.update_axis0))

        # Z-resolution label + entry grouped in a single grid cell (column 3)
        z_res_frame = ttk.Frame(controls_frame)
        z_res_frame.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(z_res_frame, text="Z-resolution:", width=13).pack(side=tk.LEFT)
        float_vcmd = (self.root.register(self._validate_float_input), "%P")
        self.z_resolution_entry = ttk.Entry(z_res_frame, textvariable=self.z_resolution_var, width=7,
                                            validate="key", validatecommand=float_vcmd)
        self.z_resolution_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.z_resolution_entry.bind("<FocusOut>", self._on_z_resolution_changed)
        self.z_resolution_entry.bind("<Return>", self._on_z_resolution_changed)

        self.btn_save_img = ttk.Button(controls_frame, text="Save Image", command=self.save_image)
        self.btn_save_img.grid(row=0, column=4, padx=5, pady=2, sticky=tk.EW)

        # Row 1: Axis 2
        ttk.Label(controls_frame, text="Axis 1 (q-values):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.slider2 = ttk.Scale(controls_frame, from_=0, to=max(0, self.width - 1), orient=tk.HORIZONTAL,
                                 command=lambda x: self.update_axis2())
        self.slider2.grid(row=1, column=1, sticky="we", padx=5, pady=2)
        self.idx1_entry = ttk.Entry(controls_frame, textvariable=self.idx1_var, width=8, justify=tk.CENTER,
                                    validate="key", validatecommand=int_vcmd)
        self.idx1_entry.grid(row=1, column=2, padx=5)
        self.idx1_entry.bind("<Return>", lambda e: self._on_index_entry(self.slider2, self.idx1_var, self.update_axis2))
        self.idx1_entry.bind("<FocusOut>", lambda e: self._on_index_entry(self.slider2, self.idx1_var, self.update_axis2))

        # Radius label + dropdown grouped in a single grid cell (column 3)
        radius_frame = ttk.Frame(controls_frame)
        radius_frame.grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(radius_frame, text="Radius:", width=13).pack(side=tk.LEFT)
        self.radius_combo = ttk.Combobox(radius_frame, textvariable=self.radius_var, state="readonly", width=5,
                                         values=[str(r) for r in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)])
        self.radius_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.radius_combo.bind("<<ComboboxSelected>>", lambda e: self.update_axis2())

        self.btn_save_csv = ttk.Button(controls_frame, text="Save Pattern", command=self.save_data_csv)
        self.btn_save_csv.grid(row=1, column=4, padx=5, pady=2, sticky=tk.EW)

        # Initialize plots
        self.ax[0].set_xlabel("q-value")
        self.ax[0].set_ylabel("Z (index)")
        self.slice0 = self.ax[0].imshow(self.rescale_data(self.data[0, :, :]), cmap="inferno", aspect="auto", 
                                         extent=[0.0, self.q_max, self.height, 0])
        self.ax[0].set_xlim(0.0, self.q_max)
        self.slice_ave = self.ax[1].imshow(10.0 * np.ones_like(self.data[:, :, 0]), cmap="inferno", aspect="auto")

    def _validate_int_input(self, value):
        return re.match(r"^-?\d*$", value) is not None

    def _on_index_entry(self, slider, var, update):
        """Move the slider to the typed index; values outside the slider range go to the nearest end."""
        low, high = int(float(slider.cget("from"))), int(float(slider.cget("to")))
        try:
            index = int(np.clip(int(var.get()), low, high))
        except ValueError:
            index = int(round(float(slider.get())))  # empty or just "-": keep the current index
        slider.set(index)
        update()  # also refreshes the entry text with the clamped index

    def _validate_float_input(self, value):
        if value in ("", "."):
            return True
        return re.match(r"^\d*\.?\d*$", value) is not None

    def _on_z_resolution_changed(self, event=None):
        try:
            value = max(0.05, float(self.z_resolution_var.get()))
        except ValueError:
            value = 0.05
        self.z_resolution = value
        self.z_resolution_var.set(str(value))

    def update_axis0(self, val=None):
        index = int(float(self.slider0.get()))
        self.idx0_var.set(str(index))
        if self.depth > 1:
            self.ax[0].set_title(f"{key}: {motor_positions[index]:2.3f} mm | Image index {index}")
        else:
            self.ax[0].set_title(f"{key}: {motor_positions[0]:2.3f}")
        
        img = self.rescale_data(self.data[index, :, :])
        self.slice0.set_data(img)
        self.slice0.set_clim(vmin=np.min(img), vmax=np.max(img))
        self.update_crosshairs()
        self.canvas.draw_idle()

    def update_axis2(self, val=None):
        index = int(float(self.slider2.get()))
        self.idx1_var.set(str(index))
        try:
            radius = int(self.radius_var.get())
        except (ValueError, tk.TclError):
            radius = self.radius
        x_q = index * q_step
        self.selected_q = x_q

        # Update yellow span around the selected q-position
        if self.selection_span:
            self.selection_span.remove()
        span_half_width = radius * q_step
        self.selection_span = self.ax[0].axvspan(x_q - span_half_width, x_q + span_half_width,
                                                 color="yellow", alpha=0.2)

        start = np.clip(index - radius, 0, None)
        stop = np.clip(index + radius + 1, None, self.width)
        self.ave_image = self.rescale_data(np.mean(self.data[:, :, start:stop], axis=2))
        self.ax[1].clear()
        self.ax[1].imshow(self.ave_image, cmap="inferno", aspect="auto")
        self.ax[1].set_title(f"Mean of q-slices around q={x_q:2.2f}")
        self.ax[1].set_xlabel("Z (index)")
        self.ax[1].set_ylabel(key + " (index)")

        self.ax1_hline = None
        self.ax1_vline = None
        self.update_crosshairs()
        self.canvas.draw_idle()

    def update_crosshairs(self):
        """Keep the ax[0]/ax[1] crosshair markers in sync at the shared Z-row and motor index."""
        if self.clicked_row is None:
            return
        y = self.clicked_row
        motor_index = int(float(self.slider0.get()))

        if self.selection_line:
            self.selection_line.remove()
        self.selection_line = self.ax[0].axhline(y, color="lime", linewidth=1)

        if self.ax1_hline:
            self.ax1_hline.remove()
        if self.ax1_vline:
            self.ax1_vline.remove()
        self.ax1_hline = self.ax[1].axhline(motor_index, color="lime", linewidth=1)
        self.ax1_vline = self.ax[1].axvline(y, color="lime", linewidth=1)

    def plot_intensity_along_row(self, event):
        if event.button != 1:
            return
        if event.inaxes == self.ax[0]:
            y = int(np.clip(event.ydata, 0, self.height - 1))
            self.clicked_row = y

            # Move axis 2 slider to the clicked column (q-position)
            x_index = int(np.clip(round(event.xdata / q_step), 0, self.width - 1))
            self.slider2.set(x_index)
            self.update_axis2()

            self.update_diffraction_profile(y)
            self.update_crosshairs()
            self.canvas.draw_idle()
        elif event.inaxes == self.ax[1]:
            self.handle_ax1_click(event)

    def handle_ax1_click(self, event):
        z_index = int(np.clip(round(event.xdata), 0, self.height - 1))
        motor_index = int(np.clip(round(event.ydata), 0, self.depth - 1))
        self.clicked_row = z_index

        # Move axis 0 slider to the corresponding motor position
        self.slider0.set(motor_index)
        self.update_axis0()

        self.update_diffraction_profile(z_index)
        self.canvas.draw_idle()

    def update_diffraction_profile(self, y):
        index0 = int(self.slider0.get())
        intensity_profile = self.data[index0, y, :]
        self.ax[2].clear()
        self.ax[2].plot(self.q_list, intensity_profile, color="blue", linewidth=0.5)
        self.ax[2].set_title("Diffraction pattern at Z-index: {0}".format(y))
        self.ax[2].set_xlabel("q-value")
        self.ax[2].set_ylabel("Intensity")

    def save_data_csv(self):
        if self.clicked_row is not None:
            y = self.clicked_row
            nfile = int(self.slider0.get())            
            mat = losa.load_image(list_raw[nfile])
            (hei, wid) = mat.shape
            q_mat = post.get_q_map(hei, wid, q_calib)
            diff_pattern = np.asarray(list(zip(q_mat[y], mat[y])))
            y_name = ("000" + str(y))[-3:]
            img_name = ("0000" + str(nfile))[-5:]
            out_file = output_folder + f"/diffraction_pattern/{key}_img_" + img_name + "_strip_idx_" + y_name + ".csv"
            losa.save_csv(out_file, diff_pattern)
            messagebox.showinfo("Saved", "Diffraction pattern saved to:\n{0}".format(out_file))
        else:
            messagebox.showwarning("Warning", "Please click on the first image to select a Z-index.")

    def save_image(self):
        if self.ave_image is not None:
            y_height = abs(motor_positions[-1] - motor_positions[0])
            z_width = self.ave_image.shape[1] * self.z_resolution
            # factor = (1, z_width/y_height)
            figsize = (8.0, 8.0)
            q_name = "q_{0:2.2f}".format(self.selected_q)
            out_file_tif = output_folder + "/q_slice/tomo_slice_" + q_name + ".tif"
            # mat_zoom = ndi.zoom(self.ave_image, factor)
            losa.save_image(out_file_tif, self.ave_image)
            mat_png = np.log(1 + np.clip(self.ave_image, 0, None))
            out_file_png = output_folder + "/q_slice/tomo_slice_" + q_name + ".png"
            losa.save_image_plt(out_file_png,  mat_png, "z", "y",
                                x_max=z_width, y_max=y_height, cmap="inferno", aspect="auto",
                                color_bar=False, figsize=figsize)
            messagebox.showinfo("Saved", "Image saved to:\n{0}\n{1}".format(out_file_tif, out_file_png))
        else:
            messagebox.showwarning("Warning", "Please move the axis-1 slider to generate a q-slice.")


if __name__ == "__main__":
    root = tk.Tk()
    app = EDXDViewerApp(root, data)
    root.mainloop()
