#!/nsls2/conda/envs/2023-3.2-py310/bin/python

# =============================
# GUI for viewing 2D-scan EDDT
# =============================

import os
import re
import sys
import signal
import argparse
import textwrap
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkFont

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import algodif.io.loadersaver as losa
import algodif.post.postprocessing as post


# Constants for GUI Font
GUI_FONT_NAME = ""
FONT_SIZE = 12
FONT_WEIGHT = "bold"

# Number of full images kept in memory at once
IMAGE_CACHE_SIZE = 20
FIG_DPI = 100
PLOT_FIG_SIZE = (24, 9)
WINDOW_SCREEN_FRACTION = (0.9, 0.85)  # initial window size relative to the screen
PLOT_FRAME = {"bottom" : 0.05, "left":0.04, "right": 0.98, "top":0.94}


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
keys = losa.find_key_of_list_json(metadata)
if len(keys) !=2:
    raise ValueError("Check metadata, it's not 2D-scan!!!")
else:
    key2, key1 = keys[:2]

motor_positions1 = losa.get_list_by_key_json(metadata, key1)
if not isinstance(motor_positions1, list):
    print("\n!!!There's only one image in this scan: {}\n".format(scan_name))
    motor_positions1 = [motor_positions1]

motor_positions2 = losa.get_list_by_key_json(metadata, key2)
if not isinstance(motor_positions2, list):
    print("\n!!!There's only one image in this scan: {}\n".format(scan_name))
    motor_positions2 = [motor_positions2]


# count_time = losa.get_metadata_value(metadata, "count_time")
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


def structure_filenames(file_list):
    temp = {}
    max_row = 0
    max_col = 0

    for full_path in file_list:
        fname = os.path.basename(full_path)  # e.g. 'scan_00013_00024.tif'
        name = fname.replace(".tif", "")
        parts = name.split("_")
        if len(parts) < 3:
            continue  # skip malformed names
        try:
            row = int(parts[-2])
            col = int(parts[-1])
        except ValueError:
            continue  # skip files with invalid indices

        temp.setdefault(row, {})[col] = full_path
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    # Build a rectangular 2D list with None for missing entries
    file_path = []
    for row in range(max_row + 1):
        file_path.append([temp.get(row, {}).get(col, None) for col in range(max_col + 1)])

    return file_path, max_row, max_col


list_file_2d, f_nrow, f_ncol = structure_filenames(list_file)
list_raw_2d = structure_filenames(list_raw)[0]
f_nrow += 1
f_ncol += 1

# Only the first image is read here; all others are loaded on demand.
height, width = np.shape(losa.load_image(list_file[0]))

print("=========================================================\n")
print(f"Showing the scan: {scan_name}. Total image: {num_file}")
# print(f"Theta measured: {theta_measured:3f}")
print("=========================================================\n")


class LazyScanData:
    """Load tiff files on demand and keep only a few recent images in memory."""

    def __init__(self, file_2d, nrow, ncol, height, width, cache_size=IMAGE_CACHE_SIZE):
        self.file_2d = file_2d
        self.nrow = nrow
        self.ncol = ncol
        self.height = height
        self.width = width
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def _read(self, i, j):
        try:
            path = self.file_2d[i][j]
        except IndexError:
            path = None
        if path is None:
            return np.zeros((self.height, self.width), dtype=np.float32)
        return np.asarray(losa.load_image(path), dtype=np.float32)

    def get_image(self, i, j):
        key = (i, j)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        img = self._read(i, j)
        self._cache[key] = img
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return img

    def get_q_slices_mean(self, axis, fixed, start, stop):
        """Mean over q-columns [start, stop) along one scan axis with the other index fixed.
        axis=0: images (i, fixed) for all i; axis=1: images (fixed, j) for all j.
        Returns shape (n, height) without keeping the images in memory."""
        n = self.nrow if axis == 0 else self.ncol
        out = np.zeros((n, self.height), dtype=np.float32)
        for k in range(n):
            key = (k, fixed) if axis == 0 else (fixed, k)
            img = self._cache.get(key)
            if img is None:
                img = self._read(*key)
            out[k] = np.mean(img[:, start:stop], axis=1)
        return out


class Display2DScanApp:
    def __init__(self, root, scan_data, radius=5):
        self.root = root
        self.root.title(f"2D-Scan Viewer: {scan_name}")
        self.z_resolution = 1.0

        # Apply theme and global font for Tkinter
        self.style = ttk.Style()
        self.style.theme_use("clam")

        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.config(size=FONT_SIZE, weight=FONT_WEIGHT)

        self.style.configure(".", font=default_font)
        self.root.option_add("*Font", default_font)

        # Sync Matplotlib font with GUI Constants
        # Tk sizes are in points at the Tk scaling (px/pt); convert to matplotlib points at FIG_DPI
        mpl_size = (FONT_SIZE - 2) * float(self.root.tk.call("tk", "scaling")) * 72.0 / FIG_DPI
        plt.rcParams.update({"font.family": GUI_FONT_NAME, "font.size": mpl_size,
                             "axes.titlesize": mpl_size, "axes.labelsize": mpl_size,
                             "xtick.labelsize": mpl_size, "ytick.labelsize": mpl_size})

        self.scan_data = scan_data
        self.radius = radius
        self.radius_var = tk.StringVar(value=str(radius))
        self.z_resolution_var = tk.StringVar(value=str(self.z_resolution))
        self.q_max = (width - 1) * q_step
        self.q_list = np.arange(width) * q_step
        self.clicked_point = None
        self.ave_image = None
        self.idx0 = 0
        self.idx1 = 0
        self.active_panel = None  # 0 or 1: the image (ax[0] / ax[1]) the user clicked last
        self.panel_idx = [(0, 0), (0, 0)]  # (row-index, column-index) of the file shown in each panel
        self.crosshair = {0: [], 1: [], 3: []}
        self._qslice_key = None
        self._qslice_img = None
        self._suppress = False  # ignore slider callbacks while the slider is set from code
        self._titles = {}  # unwrapped title text per axes, re-wrapped on resize

        self.setup_ui()
        self.update_axis0()
        self.update_axis2()

        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        signal.signal(signal.SIGINT, lambda signum, frame: self.exit_app())
        self._poll_signals()

    def _poll_signals(self):
        self.root.after(200, self._poll_signals)

    def exit_app(self):
        print("==============================")
        print("Exit 2D-Scan Viewer")
        print("==============================")
        self.root.destroy()

    def rescale_data(self, img):
        return np.log(img - np.min(img) + 1)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=PLOT_FIG_SIZE, dpi=FIG_DPI)
        gs = self.fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.25)
        self.ax = [self.fig.add_subplot(gs[0, i]) for i in range(4)]
        self.fig.subplots_adjust(bottom=PLOT_FRAME["bottom"], left=PLOT_FRAME["left"],
                                         right=PLOT_FRAME["right"], top=PLOT_FRAME["top"])

        blank = np.ones((height, width))
        extent = [0.0, self.q_max, height, 0]
        self.slice0 = self.ax[0].imshow(blank, cmap="inferno", aspect="auto", extent=extent)
        self.slice2 = self.ax[1].imshow(blank, cmap="inferno", aspect="auto", extent=extent)
        for a in (self.ax[0], self.ax[1]):
            a.set_xlabel("q-value")
            a.set_ylabel("Z (index)")
        self.ax[3].imshow(10.0 * np.ones((f_nrow, height)), cmap="inferno", aspect="auto")
        self.ax[3].set_xlabel("Z (index)")
        self.ax[3].set_ylabel(key1 + " (index)")
        self.ax[2].set_xlabel("q-value")
        self.ax[2].set_ylabel("Intensity")

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.mpl_connect("button_press_event", self.plot_intensity_along_row)
        self.canvas.mpl_connect("resize_event", self._rewrap_titles)

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

        # Row 0: first-motor slider
        ttk.Label(controls_frame, text="Axis 0 (1st-motor-position):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.slider0 = ttk.Scale(controls_frame, from_=0, to=max(0, f_nrow - 1), orient=tk.HORIZONTAL,
                                 command=self.on_slider0)
        self.slider0.grid(row=0, column=1, sticky="we", padx=5, pady=2)
        self.slider0.bind("<ButtonRelease-1>", self.on_slider_release)
        self.idx0_entry = ttk.Entry(controls_frame, textvariable=self.idx0_var, width=8, justify=tk.CENTER,
                                    validate="key", validatecommand=int_vcmd)
        self.idx0_entry.grid(row=0, column=2, padx=5)
        self.idx0_entry.bind("<Return>", self.on_index_entry)
        self.idx0_entry.bind("<FocusOut>", self.on_index_entry)

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

        # Row 1: second-motor slider
        ttk.Label(controls_frame, text="Axis 1 (2nd-motor-position):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.slider1 = ttk.Scale(controls_frame, from_=0, to=max(0, f_ncol - 1), orient=tk.HORIZONTAL,
                                 command=self.on_slider1)
        self.slider1.grid(row=1, column=1, sticky="we", padx=5, pady=2)
        self.slider1.bind("<ButtonRelease-1>", self.on_slider_release)
        self.idx1_entry = ttk.Entry(controls_frame, textvariable=self.idx1_var, width=8, justify=tk.CENTER,
                                    validate="key", validatecommand=int_vcmd)
        self.idx1_entry.grid(row=1, column=2, padx=5)
        self.idx1_entry.bind("<Return>", self.on_index_entry)
        self.idx1_entry.bind("<FocusOut>", self.on_index_entry)

        radius_frame = ttk.Frame(controls_frame)
        radius_frame.grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(radius_frame, text="Radius:", width=13).pack(side=tk.LEFT)
        radius_values = sorted({1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, self.radius})
        self.radius_combo = ttk.Combobox(radius_frame, textvariable=self.radius_var, state="readonly", width=5,
                                         values=[str(r) for r in radius_values])
        self.radius_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.radius_combo.bind("<<ComboboxSelected>>", self.on_radius_changed)

        self.btn_save_img = ttk.Button(controls_frame, text="Save Image", command=self.save_image)
        self.btn_save_img.grid(row=1, column=4, padx=5, pady=2, sticky=tk.EW)

        # Row 2: message bar
        self.message_var = tk.StringVar(value="Display of scan: {}".format(scan_name))
        # Wrap long messages (e.g. saved paths) so they cannot widen the window
        message_label = ttk.Label(controls_frame, textvariable=self.message_var, wraplength=800)
        message_label.grid(row=2, column=0, columnspan=5, sticky=tk.W, padx=5, pady=(2, 0))
        controls_frame.bind("<Configure>", lambda e: message_label.configure(wraplength=max(200, e.width - 20)))

    def _validate_int_input(self, value):
        return re.match(r"^-?\d*$", value) is not None

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

    def _radius(self):
        try:
            return int(self.radius_var.get())
        except (ValueError, tk.TclError):
            return self.radius

    def on_radius_changed(self, _event=None):
        if self.clicked_point is not None:
            self.update_intensity_plot()
            self.canvas.draw_idle()

    def _indices(self):
        return int(round(float(self.slider0.get()))), int(round(float(self.slider1.get())))

    def on_slider0(self, _val):
        if self._suppress:
            return
        idx0 = self._indices()[0]
        if self.active_panel == 1:
            # Slider 0 is the fixed index of the clicked ax[1] image: wait for the mouse release
            self.idx0_var.set(str(idx0))
        elif idx0 != self.idx0:
            self.update_axis0()

    def on_slider1(self, _val):
        if self._suppress:
            return
        idx1 = self._indices()[1]
        if self.active_panel == 0:
            # Slider 1 is the fixed index of the clicked ax[0] image: wait for the mouse release
            self.idx1_var.set(str(idx1))
        elif idx1 != self.idx1:
            self.update_axis2()

    def on_slider_release(self, _event=None):
        idx0, idx1 = self._indices()
        # Snap the sliders to whole indices
        self._suppress = True
        self.slider0.set(idx0)
        self.slider1.set(idx1)
        self._suppress = False
        self._apply_indices(idx0, idx1)

    def on_index_entry(self, _event=None):
        """Move the sliders to the typed indices; values outside the range are clamped to the nearest end."""
        typed = []
        for var, current, n in ((self.idx0_var, self.idx0, f_nrow), (self.idx1_var, self.idx1, f_ncol)):
            try:
                value = int(var.get())
            except ValueError:
                value = current  # empty or just "-": keep the current index
            typed.append(int(np.clip(value, 0, max(0, n - 1))))
        self._suppress = True
        self.slider0.set(typed[0])
        self.slider1.set(typed[1])
        self._suppress = False
        self._apply_indices(*typed)

    def _apply_indices(self, idx0, idx1):
        """Load the data for new slider indices not yet shown; refresh the entries to the shown indices."""
        # Load the q-slices and refresh both images, ax[2] and ax[3]; the crosshair stays on the clicked image
        if self.active_panel == 1 and idx0 != self.idx0:
            self.update_axis0()
            self.update_axis2()
        elif self.active_panel == 0 and idx1 != self.idx1:
            self.update_axis2()
            self.update_axis0()
        else:
            if idx0 != self.idx0:
                self.update_axis0()
            if idx1 != self.idx1:
                self.update_axis2()
        self.idx0_var.set(str(self.idx0))
        self.idx1_var.set(str(self.idx1))

    def _sync_sliders(self, idx0, idx1):
        """Set both sliders (and entries) from code without triggering the slider callbacks."""
        self._suppress = True
        self.slider0.set(idx0)
        self.slider1.set(idx1)
        self._suppress = False
        self.idx0, self.idx1 = idx0, idx1
        self.idx0_var.set(str(idx0))
        self.idx1_var.set(str(idx1))

    @staticmethod
    def _pos(positions, idx):
        return positions[min(idx, len(positions) - 1)]

    def _set_title(self, ax, text):
        """Set a title wrapped to the current pixel width of the axes."""
        self._titles[ax] = text
        width_px = ax.get_window_extent().width
        char_px = ax.title.get_fontsize() * self.fig.dpi / 72.0 * 0.6  # approx. average glyph width
        n_chars = max(10, int(width_px / char_px))
        lines = [wrapped for line in text.split("\n")
                 for wrapped in (textwrap.wrap(line, n_chars, break_long_words=False) or [line])]
        ax.set_title("\n".join(lines))

    def _rewrap_titles(self, _event=None):
        for ax, text in list(self._titles.items()):
            self._set_title(ax, text)

    def _show(self, panel, image_artist, title_axis, key, positions, pos_idx):
        idx0, idx1 = self._indices()
        self.idx0, self.idx1 = idx0, idx1
        self.panel_idx[panel] = (idx0, idx1)
        self.idx0_var.set(str(idx0))
        self.idx1_var.set(str(idx1))
        self._set_title(title_axis, "{0}: {1:2.3f}\n{2}: {3:2.3f}".format(
            key1, self._pos(motor_positions1, idx0), key2, self._pos(motor_positions2, idx1)))

        img = self.rescale_data(self.scan_data.get_image(idx0, idx1))
        image_artist.set_data(img)
        image_artist.set_clim(vmin=np.min(img), vmax=np.max(img))
        if self.active_panel == panel:
            self.update_intensity_plot()
        self.canvas.draw_idle()

    def update_axis0(self):
        self._show(0, self.slice0, self.ax[0], key1, motor_positions1, self._indices()[0])

    def update_axis2(self):
        self._show(1, self.slice2, self.ax[1], key2, motor_positions2, self._indices()[1])

    def _clear_crosshair(self, key):
        for line in self.crosshair[key]:
            line.remove()
        self.crosshair[key] = []

    def _draw_crosshair(self, key, x, y, half_width=None):
        """Green horizontal line; vertical is green, or yellow line + span of half_width in q."""
        self._clear_crosshair(key)
        ax = self.ax[key]
        artists = [ax.axhline(y, color="lime", linewidth=1)]
        if half_width is None:
            artists.append(ax.axvline(x, color="lime", linewidth=1))
        else:
            artists.append(ax.axvline(x, color="yellow", linewidth=2, alpha=0.5))
            artists.append(ax.axvspan(max(0.0, x - half_width), min(self.q_max, x + half_width),
                                      color="yellow", alpha=0.5, linewidth=0))
        self.crosshair[key] = artists

    def update_crosshairs(self):
        """Show the crosshair only on the clicked image; the other image is cleared."""
        panel = self.active_panel
        y, x = self.clicked_point
        self._clear_crosshair(1 - panel)
        x_q = int(x / q_step) * q_step
        self._draw_crosshair(panel, x_q, y, half_width=self._radius() * q_step)
        # ax[3]: x = Z (row) index, y = index of the motor scanned by the q-slice
        self._draw_crosshair(3, y, self.panel_idx[panel][panel])

    def plot_intensity_along_row(self, event):
        if event.button != 1 or self.toolbar.mode or event.xdata is None:
            return
        if event.inaxes == self.ax[3]:
            self.handle_qslice_click(event)
            return
        if event.inaxes == self.ax[0]:
            panel = 0
        elif event.inaxes == self.ax[1]:
            panel = 1
        else:
            return
        y = int(np.clip(event.ydata, 0, height - 1))
        x = float(np.clip(event.xdata, 0.0, self.q_max))
        self.active_panel = panel
        self.clicked_point = (y, x)
        # The sliders follow the clicked image, which may differ from the other image's indices
        self._sync_sliders(*self.panel_idx[panel])
        self.update_intensity_plot()
        self.canvas.draw_idle()

    def handle_qslice_click(self, event):
        panel = self.active_panel
        if panel is None or self.clicked_point is None:
            self.message_var.set("Click on one of the two images first to select a row")
            return
        n = f_nrow if panel == 0 else f_ncol
        z = int(np.clip(round(event.xdata), 0, height - 1))
        motor_index = int(np.clip(round(event.ydata), 0, n - 1))
        self.clicked_point = (z, self.clicked_point[1])

        # Move the slider of the scanned motor; the update refreshes the clicked image and the plots
        slider = self.slider0 if panel == 0 else self.slider1
        self._suppress = True
        slider.set(motor_index)
        self._suppress = False
        if panel == 0:
            self.update_axis0()
        else:
            self.update_axis2()
        self.canvas.draw_idle()

    def update_intensity_plot(self):
        if self.clicked_point is None:
            return
        y, x = self.clicked_point
        panel = self.active_panel
        idx0, idx1 = self.panel_idx[panel]
        pos1 = "{0}: {1:2.3f}".format(key1, self._pos(motor_positions1, idx0))
        pos2 = "{0}: {1:2.3f}".format(key2, self._pos(motor_positions2, idx1))
        motor_label = pos1 + "\n" + pos2

        intensity_profile = self.scan_data.get_image(idx0, idx1)[y, :]
        self.ax[2].clear()
        self.ax[2].plot(self.q_list, intensity_profile, color="blue", linewidth=0.5)
        self._set_title(self.ax[2], "Diffraction pattern at Z-index: {0}\n{1}".format(y, motor_label))
        self.ax[2].set_xlabel("q-value")
        self.ax[2].set_ylabel("Intensity")

        # Mean around the clicked q-value, read from disk image by image
        x_pos = int(x / q_step)
        radius = self._radius()
        start = int(np.clip(x_pos - radius, 0, None))
        stop = int(np.clip(x_pos + radius + 1, None, width))
        # ax[0] click scans motor 1 at fixed motor 2; ax[1] click scans motor 2 at fixed motor 1
        fixed = idx1 if panel == 0 else idx0
        scan_key, fixed_label = (key1, pos2) if panel == 0 else (key2, pos1)
        qslice_key = (panel, fixed, start, stop)
        if qslice_key != self._qslice_key:
            self.message_var.set("Loading q-slices along {0} ...".format(scan_key))
            self.root.update_idletasks()
            self._qslice_img = self.rescale_data(self.scan_data.get_q_slices_mean(panel, fixed, start, stop))
            self._qslice_key = qslice_key
            self.message_var.set("Display of scan {}".format(scan_name))
        self.ave_image = self._qslice_img

        self.crosshair[3] = []  # lines are removed together with the axes content
        self.ax[3].clear()
        self.ax[3].imshow(self.ave_image, cmap="inferno", aspect="auto")
        self._set_title(self.ax[3], "Mean of q-slices around q={0:2.2f}\n{1}".format(x, fixed_label))
        self.ax[3].set_xlabel("Z (index)")
        self.ax[3].set_ylabel(scan_key + " (index)")
        self.update_crosshairs()
        self.canvas.draw_idle()

    def _title_filename(self, ax):
        """File name from the plot title: '.' becomes '_', ',' is dropped, other symbols become '_'."""
        name = self._titles[ax].replace(".", "_").replace(",", "")
        return re.sub(r"[^\w\-]+", "_", name).strip("_")

    def save_data_csv(self):
        if self.clicked_point is None:
            messagebox.showwarning("Warning", "Please click on the left image to select a row.")
            return
        y, _ = self.clicked_point
        nfile, nfile2 = self.panel_idx[self.active_panel]
        mat = losa.load_image(list_raw_2d[nfile][nfile2])
        (hei, wid) = mat.shape
        q_mat = post.get_q_map(hei, wid, q_calib)
        diff_pattern = np.asarray(list(zip(q_mat[y], mat[y])))
        out_file = output_folder + "/diffraction_pattern/" + self._title_filename(self.ax[2]) + ".csv"
        losa.save_csv(out_file, diff_pattern)
        self.message_var.set("Data saved at: {}".format(out_file))

    def save_image(self):
        if self.ave_image is None or self.clicked_point is None:
            messagebox.showwarning("Warning", "Please click on the left image to select a row.")
            return
        self._on_z_resolution_changed()
        if self.active_panel == 0:
            positions = motor_positions1
            y_label = "x"
        else:
            positions = motor_positions2
            y_label = "y"
        y_axis_length = abs(positions[-1] - positions[0])
        z_width = self.ave_image.shape[1] * self.z_resolution
        # factor = (z_width / y_height, 1)
        figsize = (12.0, 12.0)
        out_file_path = output_folder + "/q_slice/" + self._title_filename(self.ax[3])
        losa.save_image(out_file_path + ".tif", self.ave_image)
        mat_png = np.log(1 + np.clip(self.ave_image, 0, None))
        losa.save_image_plt(out_file_path + ".png", mat_png, "z", y_label,
                            x_max=z_width, y_max=y_axis_length, cmap="inferno", aspect="equal",
                            color_bar=False, figsize=figsize)
        self.message_var.set(f"Data saved at: {out_file_path}")


if __name__ == "__main__":
    scan_data = LazyScanData(list_file_2d, f_nrow, f_ncol, height, width)
    root = tk.Tk()
    app = Display2DScanApp(root, scan_data)
    root.mainloop()
