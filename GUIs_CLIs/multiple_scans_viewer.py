#!/nsls2/conda/envs/2023-3.2-py310/bin/python

# ==========================
# Author: Nghia T. Vo
# E-mail: nvo@bnl.gov 
# Time: 2023
# ==========================

import os
import re
import sys
import signal
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
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
PLOT_FRAME = {"bottom" : 0.05, "left":0.03, "right": 0.99, "top":0.96}


proposal = "pass-320871"
cycle = "2026-2"
beamline_folder = "/nsls2/data/hex/proposals/"
technique = "edxd"


parser = argparse.ArgumentParser(description="Process data and display q-slice image!!!")
parser.add_argument("-s", dest="start_scan", help="Starting scan (integer)", type=int, required=True)
parser.add_argument("-e", dest="end_scan", help="Ending scan (integer)", type=int, required=True)
parser.add_argument("--gap", dest="gap", help="Gap betweeing scans", type=int, required=False, default=1)
parser.add_argument("-p", dest="calib_path", help="Path to the calibration csv file",
                    type=str, required=False, default="/nsls2/data/hex/proposals/2026-2/pass-320871/edxd/calibration/scan_00001/results/scan_00001_3_0deg_q_calib.csv")
parser.add_argument("-g", dest="background", help="To choose image with or without (1 or 0) background. Default is 1.",
                    type=int, required=False, default=1)
args = parser.parse_args()

start_scan = args.start_scan
end_scan = args.end_scan
gap = args.gap
with_bgr = args.background
calib_path = args.calib_path

q_calib = losa.load_csv(calib_path)

list_scans = range(start_scan, end_scan + 1, gap)
if len(list_scans) < 1:
    raise ValueError("Error !!! Please check the scan number !!!")

list_scan_name = []
for scan_number in list_scans:
    name = "scan_" + ("0000" + str(scan_number))[-5:]
    list_scan_name.append(name)

list_motor_positions = []
list_3D_data = []

# Global key variable to be used in plotting
key = None
for scan_name in list_scan_name:
    # Check if the proposal folder exists
    proposal_base = beamline_folder + "/" + cycle + "/" + proposal
    if not os.path.exists(proposal_base):
        raise ValueError(f"There's no such folder: {proposal_base}")
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

    # count_time = losa.get_metadata_value(metadata, "count_time")
    q_step = metadata["Metadata"]["q_step"]

    if with_bgr == 0:
        list_file = losa.find_file(input_folder + "/tifs_without_bgr/*.tif*")
        if len(list_file) == 0:
            print("!!!! There's no image with background removal. Load image with background instead!!!")
            list_file = losa.find_file(input_folder + "/tifs_with_bgr/*.tif*")
    else:
        list_file = losa.find_file(input_folder + "/tifs_with_bgr/*.tif*")

    num_file = len(list_file)
    data = np.asarray([losa.load_image(file) for file in list_file])

    list_3D_data.append(data)
    list_motor_positions.append(motor_positions)

    print("=========================================================\n")
    print(f"Load scan: {scan_name}. Total image: {num_file}")
    print("=========================================================\n")

# Combined folder used to save data gathered across all loaded scans
combined_output_folder = proposal_base + "/" + technique + "/tmp/scans_" + list_scan_name[0] + "_to_" + list_scan_name[-1] + "/"


class MultipleScansViewerApp:
    def __init__(self, root, list_data, list_names, list_pos, radius=9):
        self.root = root
        self.root.title(f"Multiple-Scans Viewer: {list_names[0]} - {list_names[-1]}")

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

        self.list_data = list_data
        self.list_names = list_names
        self.list_pos = list_pos
        self.num_datasets = len(list_data)
        self.depth, self.height, self.width = list_data[0].shape
        self.q_max = (self.width - 1) * q_step
        self.q_list = np.arange(self.width) * q_step
        self.clicked_point = None  # (row, q-value) of the last click on ax[0] / ax[2]
        self.ave_image = None
        self.crosshair = []
        self.radius = radius
        self.radius_var = tk.StringVar(value=str(radius))
        self.z_resolution = 1.0
        self.z_resolution_var = tk.StringVar(value=str(self.z_resolution))
        self._suppress = False  # ignore slider callbacks while a slider is set from code

        self.setup_ui()
        self.update_display()

        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        signal.signal(signal.SIGINT, lambda signum, frame: self.exit_app())
        self._poll_signals()

    def _poll_signals(self):
        self.root.after(200, self._poll_signals)

    def exit_app(self):
        print("==============================")
        print("Exit Multiple-Scans Viewer")
        print("==============================")
        self.root.destroy()

    def rescale_data(self, img):
        img_res = np.log(img - np.min(img) + 1.0)
        return img_res

    def setup_ui(self):
        # Create Main Frame
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib Figure with 2D / 3D / 2D subplots
        self.fig = Figure(figsize=PLOT_FIG_SIZE, dpi=FIG_DPI)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.2)
        self.ax = []
        self.ax.append(self.fig.add_subplot(gs[0, 0]))  # ax[0]: 2D image
        self.ax.append(self.fig.add_subplot(gs[0, 1], projection="3d"))  # ax[1]: 3D waterfall across datasets
        self.ax.append(self.fig.add_subplot(gs[0, 2]))  # ax[2]: mean q-slice of the displayed scan
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

        self.img_idx_var = tk.StringVar(value="0")
        self.ds_idx_var = tk.StringVar(value="0")
        int_vcmd = (self.root.register(self._validate_int_input), "%P")

        # Row 0: Image index slider
        ttk.Label(controls_frame, text="Image Index:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.slider_img = ttk.Scale(controls_frame, from_=0, to=max(0, self.depth - 1), orient=tk.HORIZONTAL,
                                    command=self.on_slider)
        self.slider_img.grid(row=0, column=1, sticky="we", padx=5, pady=2)
        self.img_idx_entry = ttk.Entry(controls_frame, textvariable=self.img_idx_var, width=8, justify=tk.CENTER,
                                       validate="key", validatecommand=int_vcmd)
        self.img_idx_entry.grid(row=0, column=2, padx=5)
        for sequence in ("<Return>", "<FocusOut>"):
            self.img_idx_entry.bind(sequence, self._on_index_entry)

        z_res_frame = ttk.Frame(controls_frame)
        z_res_frame.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(z_res_frame, text="Z-resolution:", width=13).pack(side=tk.LEFT)
        float_vcmd = (self.root.register(self._validate_float_input), "%P")
        self.z_resolution_entry = ttk.Entry(z_res_frame, textvariable=self.z_resolution_var, width=7,
                                            validate="key", validatecommand=float_vcmd)
        self.z_resolution_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.z_resolution_entry.bind("<FocusOut>", self._on_z_resolution_changed)
        self.z_resolution_entry.bind("<Return>", self._on_z_resolution_changed)

        self.btn_save_csv = ttk.Button(controls_frame, text="Save CSV", command=self.save_data_csv)
        self.btn_save_csv.grid(row=0, column=4, padx=5, pady=2, sticky=tk.EW)

        # Row 1: Dataset index slider
        ttk.Label(controls_frame, text="Dataset Index:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.slider_dataset = ttk.Scale(controls_frame, from_=0, to=max(0, self.num_datasets - 1), orient=tk.HORIZONTAL,
                                        command=self.on_slider)
        self.slider_dataset.grid(row=1, column=1, sticky="we", padx=5, pady=2)
        self.ds_idx_entry = ttk.Entry(controls_frame, textvariable=self.ds_idx_var, width=8, justify=tk.CENTER,
                                      validate="key", validatecommand=int_vcmd)
        self.ds_idx_entry.grid(row=1, column=2, padx=5)
        for sequence in ("<Return>", "<FocusOut>"):
            self.ds_idx_entry.bind(sequence, self._on_index_entry)

        radius_frame = ttk.Frame(controls_frame)
        radius_frame.grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(radius_frame, text="Radius:", width=13).pack(side=tk.LEFT)
        radius_values = sorted({1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, self.radius})
        self.radius_combo = ttk.Combobox(radius_frame, textvariable=self.radius_var, state="readonly", width=5,
                                         values=[str(r) for r in radius_values])
        self.radius_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.radius_combo.bind("<<ComboboxSelected>>", self._on_radius_changed)

        self.btn_save_img = ttk.Button(controls_frame, text="Save Image", command=self.save_image)
        self.btn_save_img.grid(row=1, column=4, padx=5, pady=2, sticky=tk.EW)

        # Row 2: Message/status bar
        self.message_var = tk.StringVar(value="Click on the left image to plot line profiles")
        # Wrap long messages (e.g. saved paths) so they cannot widen the window
        message_label = ttk.Label(controls_frame, textvariable=self.message_var, wraplength=800)
        message_label.grid(row=2, column=0, columnspan=5, sticky=tk.W, padx=5, pady=(2, 0))
        controls_frame.bind("<Configure>", lambda e: message_label.configure(wraplength=max(200, e.width - 20)))

        # Initialize plot
        self.ax[0].set_xlabel("q-value")
        self.ax[0].set_ylabel("Z (index)")
        self.slice0 = self.ax[0].imshow(self.rescale_data(self.list_data[0][0, :, :]), cmap="inferno", aspect="auto",
                                         extent=[0.0, self.q_max, self.height, 0])
        self.ax[2].set_xlabel("Z (index)")
        self.ax[2].set_ylabel(key + " (index)")
        self.ax[2].imshow(10.0 * np.ones((self.depth, self.height)), cmap="inferno", aspect="auto")

    def _validate_int_input(self, value):
        return re.match(r"^-?\d*$", value) is not None

    def _on_index_entry(self, event=None):
        """Move the sliders to the typed indices; values outside the range go to the nearest end."""
        indices = []
        for var, slider, n in ((self.img_idx_var, self.slider_img, self.depth),
                               (self.ds_idx_var, self.slider_dataset, self.num_datasets)):
            try:
                index = int(np.clip(int(var.get()), 0, max(0, n - 1)))
            except ValueError:
                index = int(round(float(slider.get())))  # empty or just "-": keep the current index
            indices.append(index)
        self._suppress = True
        self.slider_img.set(indices[0])
        self.slider_dataset.set(indices[1])
        self._suppress = False
        self.update_display()  # also refreshes the entry texts with the clamped indices

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

    def _on_radius_changed(self, event=None):
        try:
            self.radius = int(self.radius_var.get())
        except (ValueError, tk.TclError):
            self.radius = 10
        if self.clicked_point is not None:
            self.update_intensity_plot()

    def on_slider(self, _val):
        if not self._suppress:
            self.update_display()

    def update_display(self, val=None):
        img_idx = int(float(self.slider_img.get()))
        ds_idx = int(float(self.slider_dataset.get()))
        self.img_idx_var.set(str(img_idx))
        self.ds_idx_var.set(str(ds_idx))

        current_data = self.list_data[ds_idx]
        current_name = self.list_names[ds_idx]
        current_pos = self.list_pos[ds_idx]

        # Datasets may have a different depth, so clip to a valid index
        actual_depth = current_data.shape[0]
        safe_img_idx = min(img_idx, actual_depth - 1)

        if actual_depth > 1:
            self.ax[0].set_title(f"Scan: {current_name}\n{key}: {current_pos[safe_img_idx]:2.3f} | Image index: {safe_img_idx}")
        else:
            self.ax[0].set_title(f"Scan: {current_name}\n{key}: {current_pos[0]:2.3f} | Image index: 0")

        img = self.rescale_data(current_data[safe_img_idx, :, :])
        self.slice0.set_data(img)
        self.slice0.set_clim(vmin=np.min(img), vmax=np.max(img))
        if self.clicked_point is not None:
            self.update_intensity_plot()
        self.canvas.draw_idle()

    def update_crosshair(self):
        """ax[0] marker: green row line and a yellow band covering the averaged q-range."""
        if self.clicked_point is None:
            return
        for artist in self.crosshair:
            artist.remove()
        y, x = self.clicked_point
        x_q = int(x / q_step) * q_step
        half_width = self.radius * q_step
        self.crosshair = [
            self.ax[0].axhline(y, color="lime", linewidth=1),
            self.ax[0].axvline(x_q, color="yellow", linewidth=2, alpha=0.5),
            self.ax[0].axvspan(max(0.0, x_q - half_width), min(self.q_max, x_q + half_width),
                               color="yellow", alpha=0.5, linewidth=0)]

    def plot_intensity_along_row(self, event):
        if event.button != 1 or self.toolbar.mode or event.xdata is None:
            return
        if event.inaxes == self.ax[0]:
            y = int(np.clip(event.ydata, 0, self.height - 1))
            x = float(np.clip(event.xdata, 0.0, self.q_max))
            self.clicked_point = (y, x)
            self.update_intensity_plot()
        elif event.inaxes == self.ax[2]:
            self.handle_ax2_click(event)

    def handle_ax2_click(self, event):
        if self.clicked_point is None:
            return
        ds_idx = int(float(self.slider_dataset.get()))
        z = int(np.clip(round(event.xdata), 0, self.height - 1))
        img_idx = int(np.clip(round(event.ydata), 0, self.list_data[ds_idx].shape[0] - 1))
        self.clicked_point = (z, self.clicked_point[1])

        # Move the image slider; the update refreshes ax[0], ax[1] and ax[2]
        self._suppress = True
        self.slider_img.set(img_idx)
        self._suppress = False
        self.update_display()

    def update_intensity_plot(self):
        if self.clicked_point is None:
            return
        y, x = self.clicked_point
        ds_idx = int(float(self.slider_dataset.get()))
        img_idx = int(float(self.slider_img.get()))

        self.ax[1].clear()
        # X=q_value, Y=Dataset Index, Z=Intensity
        for i in range(self.num_datasets):
            curr_data = self.list_data[i]
            if img_idx < curr_data.shape[0]:
                intensity_profile = curr_data[img_idx, y, :]
                self.ax[1].plot(self.q_list, intensity_profile, zs=i, zdir="y", color="blue", linewidth=0.5, alpha=0.7)

        self.ax[1].set_title(f"Line profiles at row: {y} | Image index: {img_idx}")
        self.ax[1].set_xlabel("q-value (X)")
        self.ax[1].set_ylabel("Dataset Index (Y)")
        self.ax[1].set_zlabel("Intensity (Z)")

        # Mean of q-slices around the clicked q for every image of the displayed scan
        data = self.list_data[ds_idx]
        x_pos = int(x / q_step)
        start = int(np.clip(x_pos - self.radius, 0, None))
        stop = int(np.clip(x_pos + self.radius + 1, None, self.width))
        self.ave_image = self.rescale_data(np.mean(data[:, :, start:stop], axis=2))
        self.ax[2].clear()
        self.ax[2].imshow(self.ave_image, cmap="inferno", aspect="auto")
        self.ax[2].axhline(min(img_idx, data.shape[0] - 1), color="lime", linewidth=1)
        self.ax[2].axvline(y, color="lime", linewidth=1)
        self.ax[2].set_title(f"Scan: {self.list_names[ds_idx]}\nMean of q-slices around q={x:2.2f}")
        self.ax[2].set_xlabel("Z (index)")
        self.ax[2].set_ylabel(key + " (index)")

        self.update_crosshair()
        self.canvas.draw_idle()

    def save_data_csv(self):
        if self.clicked_point is None:
            messagebox.showwarning("Warning", "Please click on the left image to select a row.")
            return
        y, _ = self.clicked_point
        img_idx = int(float(self.slider_img.get()))

        headers = ["q_value"]
        columns = [self.q_list]
        for i in range(self.num_datasets):
            curr_data = self.list_data[i]
            if img_idx < curr_data.shape[0]:
                headers.append(self.list_names[i])
                columns.append(curr_data[img_idx, y, :])
        data_rows = list(zip(*columns))

        y_name = ("0000" + str(y))[-5:]
        img_name = ("0000" + str(img_idx))[-5:]
        out_file = combined_output_folder + "/line_profiles/img_" + img_name + "_row_" + y_name + ".csv"
        losa.save_csv(out_file, [headers] + data_rows)
        self.message_var.set("Data saved at: {}".format(out_file))

    def save_image(self):
        if self.ave_image is None or self.clicked_point is None:
            messagebox.showwarning("Warning", "Please click on the left image to generate a q-slice.")
            return
        self._on_z_resolution_changed()
        _, x = self.clicked_point
        ds_idx = int(float(self.slider_dataset.get()))
        positions = self.list_pos[ds_idx]
        y_height = abs(positions[-1] - positions[0])
        z_width = self.ave_image.shape[1] * self.z_resolution
        q_name = "q_{0:2.2f}".format(x).replace(".", "_")
        out_base = (proposal_base + "/" + technique + "/tmp/" + self.list_names[ds_idx]
                    + "/q_slice/tomo_slice_" + q_name)
        losa.save_image(out_base + ".tif", self.ave_image)
        mat_png = np.log(1 + np.clip(self.ave_image, 0, None))
        losa.save_image_plt(out_base + ".png", mat_png, "z", "y",
                            x_max=z_width, y_max=y_height, cmap="inferno", aspect="auto",
                            color_bar=False, figsize=(8.0, 8.0))
        self.message_var.set("Data saved at: {}".format(out_base))


if __name__ == "__main__":
    root = tk.Tk()
    app = MultipleScansViewerApp(root, list_3D_data, list_scan_name, list_motor_positions)
    root.mainloop()
