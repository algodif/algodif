"""
Interactive GUI for q (scattering vector) calibration of an EDXD detector.

Workflow (based on the q_calibration_*.py scripts):
    1. Load data (image, HDF, or GeRM dat file; multiple files are averaged).
    2. Reference pattern: select a calibrant bundled with algodif and/or
       input d-spacings manually. Choose how the q-step (q per pixel) is
       obtained:
         - Constant q-step: same for all strips (can be estimated from two
           peaks in the Processing tab).
         - GeRM geometry, q-step per strip: estimated from the detector
           geometry, two-theta and the energy step. The reference pattern
           then varies between strips.
         - GeRM geometry, energy matching: peaks are matched in energy
           space using energies estimated from the geometry.
    3. Single-strip calibration: preprocess, detect/edit peaks, locate them
       with sub-pixel accuracy, match to reference q-values, and fit.
    4. Multiple-strip calibration: calibrate groups of strips with their own
       parameters, check and correct the image to a uniform q-grid.
    5. Data correction: correct new data with the calibrated coefficients.

Run:
    python q_calibration_gui.py
"""

import atexit
import functools
import multiprocessing
import signal
import textwrap
import threading
import traceback
import warnings
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import h5py
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk, filedialog, messagebox
from joblib import Parallel, delayed, cpu_count
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from numpy.polynomial import polynomial as poly

import algodif.io.loadersaver as losa
import algodif.prep.preprocessing as prep
import algodif.proc.processing as proc
import algodif.post.postprocessing as post
import algodif.util.utility as util

# GUI appearance
# ttk theme: "" (system default), "clam", "alt", "default", "classic",
# or OS-specific: "vista", "winnative", "xpnative" (Windows), "aqua" (macOS)
THEME = ""
GUI_FONT_NAME = ""  # Empty string: use the system default font
FONT_SIZE = 10
FONT_WEIGHT = "normal"  # "normal" or "bold"
WINDOW_SCREEN_FRACTION = (0.9, 0.85)  # Initial window size relative to screen
WINDOW_MAX_SIZE = (int(0.9 * 1920), int(0.9 * 1080))  # For wide/multi screens
WINDOW_MIN_SIZE = (1100, 700)
LEFT_PANEL_CHARS = 36  # Width of parameter panels, in characters
ENTRY_WIDTH = 10
REF_ENTRY_WIDTH = 4  # Small input fields in the Reference pattern frame
FIG_DPI = 100
# "nearest" is fastest for large images; "antialiased" is smoother
IMAGE_INTERPOLATION = "nearest"
IMAGE_CMAPS = ("inferno", "viridis", "magma", "gray", "jet")
IMAGE_SCALES = ("log", "linear")
IMAGE_PERCENTILES = (0, 100)  # Default contrast: full range
# Rapid strip changes (e.g. holding '>') are merged into one update
STRIP_UPDATE_DELAY_MS = 30
TOOLTIP_DELAY_MS = 500
TOOLTIP_WRAP_LENGTH = 350  # Pixels
# Spacing (pixels)
PAD_OUTER = 5  # Between the window border and its content
PAD_FRAME = 5  # Between sibling frames
PAD_X = 3  # Between widgets inside a frame
PAD_Y = 3
FRAME_PADDING = (2, 2)  # Inner padding of labelled frames

IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
HDF_EXT = (".hdf", ".h5", ".hdf5", ".nxs")
DAT_EXT = (".dat",)
Q_UNIT = "1/\u00c5"
ANGSTROM = "\u00c5"

# Q-step and reference pattern
Q_STEP = "0.003"
Q_MODES = {"Constant": "constant",
           "GeRM: q-step per strip": "germ_q",
           "GeRM: energy matching": "germ_energy"}
DEFAULT_Q_MODE = "GeRM: energy matching"
# Default search ranges of the shift for q-space and energy-space matching
Q_SEARCH = (-0.5, 0.5)
E_SEARCH = (-10, 10)
# Q-step of corrected data: statistic of the pixel spacing of the q-map
# computed from the coefficients, or from a q-max or q-step value
OUTPUT_Q_OPTIONS = ("mean", "median", "max", "q-max", "q-step")
# (key, label, default, tooltip) of the GeRM detector geometry
GEOMETRY_FIELDS = [
    ("two_theta", "Two-theta (degree)", "6.0",
     "Nominal scattering angle at the detector center."),
    ("energy_step", "Energy step (keV)", "0.0538357",
     "Energy per pixel (keV), e.g. from the energy calibration."),
    ("total_strips", "Total strips", "192",
     "Total number of detector strips."),
    ("strip_size", "Strip size (mm)", "0.25", "Strip pitch in mm."),
    ("sample_slit", "Sample-slit (mm)", "250.0",
     "Sample-to-slit distance in mm."),
    ("sample_detector", "Sample-detector (mm)", "2000.0",
     "Sample-to-detector distance in mm."),
    ("strip_offset", "First strip index", "0",
     "Detector strip index of the first row of the image.")]

# Multiple-strip calibration
WORKER_POLL_MS = 200
STATUS_COLORS = {"ok": "blue", "failed": "red"}
# (key, label, default, tooltip). Booleans are check boxes.
PARAM_SECTIONS = [
    ("Zeroing and smoothing", [
        ("zero_left", "Zero left", 450,
         "Number of points on the left side to be zeroed."),
        ("zero_right", "Zero right", -1,
         "Index from which data are zeroed. Negative: counted from the end."),
        ("sigma", "Gaussian sigma", 0,
         "Gaussian smoothing before peak detection. 0: no smoothing."),
        ("use_svd", "SVD-based smoothing", True, None),
        ("svd_rank", "SVD rank", 5, "Window size of the SVD-based filter."),
        ("svd_lambda", "SVD lambda", 0.5,
         "Soft-threshold for singular values."),
    ]),
    ("Peak detection", [
        ("peak_width", "Min width", 15, "Minimum width of a peak (pixel)."),
        ("peak_ratio", "Ratio", 0.1,
         "Minimum peak height above its baseline relative to the local "
         "mean."),
        ("peak_sigma", "Sigma", 3, "Gaussian smoothing for peak detection."),
        ("peak_abs", "Abs threshold", "None",
         "Minimum absolute peak height (counts). 'None' to disable."),
    ]),
    ("Sub-pixel location", [
        ("use_sub", "Sub-pixel location", True, None),
        ("sub_radius", "Radius", 15, "Window radius for Gaussian fitting."),
        ("sub_sigma", "Sigma", 1, "Gaussian smoothing before fitting."),
        ("skew_fit", "Skewed Gaussian", True, None),
    ]),
    ("Peak matching and fitting", [
        ("fit_order", "Fit order", 2, "Polynomial order: pixel -> q."),
        ("match_tol", "Tolerance", 0.4,
         "Tolerance for accepting a matched peak, in [0, 1]."),
        ("match_size", "Size", 21, "Gaussian size used to represent peaks."),
        ("match_margin", "Margin", 20, "Margin (pixel) of the matching grid."),
        ("search_min", "Search min", E_SEARCH[0],
         f"Lower bound of the shift: {Q_UNIT} for q matching, keV for "
         f"energy matching."),
        ("search_max", "Search max", E_SEARCH[1],
         f"Upper bound of the shift: {Q_UNIT} for q matching, keV for "
         f"energy matching."),
    ]),
]


def parse_strips(text, height):
    """Parse e.g. 'all', '5', '0-31', '0-31, 40, 50-60' into strip indices."""
    indices = set()
    for token in text.replace(";", ",").split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            indices.update(range(height))
            continue
        try:
            if "-" in token[1:]:
                start, stop = (int(v) for v in token.split("-", 1))
            else:
                start = stop = int(token)
        except ValueError:
            raise ValueError(f"Invalid strip range: '{token}'. Use e.g. "
                             f"'all', '5', '0-31', '0-31, 40'")
        if start > stop:
            start, stop = stop, start
        if start < 0 or stop >= height:
            raise ValueError(f"Strip range '{token}' is outside "
                             f"[0, {height - 1}]")
        indices.update(range(start, stop + 1))
    if not indices:
        raise ValueError("No strips given!")
    return np.asarray(sorted(indices), dtype=int)


def to_number(text, name, dtype=float, allow_none=False):
    text = str(text).strip()
    if allow_none and text.lower() in ("", "none"):
        return None
    try:
        return int(float(text)) if dtype is int else dtype(text)
    except ValueError:
        raise ValueError(f"Invalid value for '{name}': '{text}'")


def default_group_params():
    params = {}
    for _, specs in PARAM_SECTIONS:
        for key, _, default, _ in specs:
            params[key] = default if isinstance(default, bool) \
                else str(default)
    return params


def build_calibration_kwargs(params):
    """Convert group parameters to arguments of the calibration methods."""
    num = to_number
    kwargs = {
        "fit_order": num(params["fit_order"], "Fit order", int),
        "zero_left": num(params["zero_left"], "Zero left", int),
        "zero_right": num(params["zero_right"], "Zero right", int),
        "sigma": num(params["sigma"], "Gaussian sigma"),
        "smooth_svd_para": None,
        "get_peaks_para": {
            "width": num(params["peak_width"], "Min width", int),
            "ratio": num(params["peak_ratio"], "Ratio"),
            "sigma": num(params["peak_sigma"], "Peak sigma"),
            "abs_threshold": num(params["peak_abs"], "Abs threshold",
                                 allow_none=True)},
        "sub_peaks_para": None,
        "match_peaks_para": {
            "tol": num(params["match_tol"], "Tolerance"),
            "size": num(params["match_size"], "Size", int),
            "margin": num(params["match_margin"], "Margin", int),
            "search_range": (num(params["search_min"], "Search min"),
                             num(params["search_max"], "Search max")),
            "bounds": None, "initial_guess": None}}
    if params["use_svd"]:
        kwargs["smooth_svd_para"] = {
            "rank": num(params["svd_rank"], "SVD rank", int),
            "lambda_threshold": num(params["svd_lambda"], "SVD lambda")}
    if params["use_sub"]:
        kwargs["sub_peaks_para"] = {
            "radius": num(params["sub_radius"], "Radius", int),
            "sigma": num(params["sub_sigma"], "Sub-pixel sigma"),
            "skew_fit": bool(params["skew_fit"])}
    return kwargs


def compute_rms(coefs, peaks, values):
    """Root-mean-square residual of the pixel -> q fit."""
    if len(values) == 0:
        return np.nan
    residual = np.asarray(values) - poly.polyval(np.asarray(peaks), coefs)
    return float(np.sqrt(np.mean(residual ** 2)))


def run_in_worker(func, *args, **kwargs):
    """Run func; worker processes ignore Ctrl+C, the GUI process handles it."""
    if multiprocessing.parent_process() is not None:
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except ValueError:  # Not the main thread of the process
            pass
    return func(*args, **kwargs)


def safe_call(func):
    """Show errors in a message box and display a busy cursor."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            self.config(cursor="watch")
            self.update_idletasks()
            return func(self, *args, **kwargs)
        except Exception as error:
            self.set_status(f"Error: {error}")
            messagebox.showerror("Error", str(error))
        finally:
            self.config(cursor="")
    return wrapper


class ToolTip:
    """Show a text box when the mouse hovers over a widget."""

    def __init__(self, widget, text, font=None):
        self.widget = widget
        self.text = text
        self.font = font
        self.tip_window = None
        self.after_id = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, event=None):
        self.cancel()
        self.after_id = self.widget.after(TOOLTIP_DELAY_MS, self.show)

    def cancel(self):
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def show(self):
        self.after_id = None
        if self.tip_window is not None or not self.text:
            return
        x_pos = self.widget.winfo_rootx() + 15
        y_pos = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x_pos}+{y_pos}")
        tk.Label(self.tip_window, text=self.text, justify="left",
                 background="#ffffe0", relief="solid", borderwidth=1,
                 font=self.font, wraplength=TOOLTIP_WRAP_LENGTH, padx=4,
                 pady=2).pack()

    def hide(self, event=None):
        self.cancel()
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class QCalibrationGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Algodif - Q calibration")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(int(WINDOW_SCREEN_FRACTION[0] * screen_width),
                    WINDOW_MAX_SIZE[0])
        height = min(int(WINDOW_SCREEN_FRACTION[1] * screen_height),
                     WINDOW_MAX_SIZE[1])
        # Screen size/origin refer to the main screen, also with multi-screens
        x_pos = max(0, (screen_width - width) // 2)
        y_pos = max(0, (screen_height - height) // 2)
        self.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
        self.minsize(min(WINDOW_MIN_SIZE[0], width),
                     min(WINDOW_MIN_SIZE[1], height))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.closing = False
        self.install_signal_handlers()
        self.setup_fonts()

        self.image = None
        self.raw_data = None
        self.proc_data = None
        # Reference: d-spacings, intensities, q-values (sorted by q)
        self.d_spacings = None
        self.ref_inten = None
        self.ref_q = None
        self.ref_cache = {}
        # Reference pattern of the current strip, for display
        self.ref_pattern = None
        self.ref_pattern_peaks = None
        self.peaks = []
        self.sub_peaks = {}
        self.selected_peak = None
        self.current_result = None
        self.results = {}
        self.results_dict = {}
        self.pick_vlines = None
        self.pick_values = None
        self.scroll_canvases = []
        self.pre_auto_limits = {}
        self.pending_draws = {}
        self.log_cache = {}
        self.display_cache = {}
        self.image_artists = {}
        self.img_percentiles = tuple(float(v) for v in IMAGE_PERCENTILES)
        self.pre_img_src = None
        self.pre_ref_src = None
        self.pre_marker = None
        self.pre_ref_lines = None
        self.pre_lines = None
        self.pre_params = None
        self.fit_info = None
        self.strip_after_id = None
        self.geom_dialog = None
        # Each top-level tab keeps its own selected strip
        self.main_tab = "single"
        self.tab_strips = {}
        self.tab_shown = {}

        # A read-only Text widget so the status can be selected and copied
        self.status_text = tk.Text(
            self, height=1, wrap="word", relief="sunken", borderwidth=1,
            highlightthickness=0, padx=PAD_X, pady=1,
            font=tkFont.nametofont("TkDefaultFont"),
            background=ttk.Style(self).lookup("TFrame", "background"))
        self.status_text.pack(fill="x", side="bottom", padx=PAD_OUTER,
                              pady=(0, PAD_OUTER))
        self.status_text.bind("<Key>", self.on_status_key)
        for seq in ("<<Paste>>", "<<Cut>>", "<<Clear>>",
                    "<<PasteSelection>>"):
            self.status_text.bind(seq, lambda e: "break")
        self.status_text.bind("<Configure>",
                              lambda e: self.fit_status_height())
        self.set_status("Load data to start.")
        self.build_top_panel()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=PAD_OUTER,
                           pady=(0, PAD_FRAME))
        self.build_tabs()
        self.update_q_step_state()
        # Plots in hidden tabs are drawn when their tab is shown
        self.bind_all("<<NotebookTabChanged>>",
                      lambda e: self.after_idle(self.flush_draws), add="+")
        self.notebook.bind("<<NotebookTabChanged>>",
                           self.on_main_tab_changed, add="+")
        self.bind("<Map>", lambda e: self.after_idle(self.flush_draws),
                  add="+")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, self.on_mouse_wheel, add="+")

    def build_tabs(self):
        style = ttk.Style(self)
        self.tab_font = tkFont.nametofont("TkDefaultFont").copy()
        self.tab_font.config(weight="bold")
        for name in ("Main.TNotebook", "Sub.TNotebook"):
            style.configure(name, tabposition="n")
            style.configure(f"{name}.Tab", font=self.tab_font,
                            anchor="center")
        style.configure("Sub.TNotebook", tabposition="nw")
        style.configure("Sub.TNotebook.Tab",
                        font=tkFont.nametofont("TkDefaultFont"), anchor="w")
        self.notebook.configure(style="Main.TNotebook")
        tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(tab, text=" Single-strip calibration ")
        self.single_tab = tab
        self.single_notebook = self.make_sub_notebook(tab)
        self.build_preprocessing_tab()
        self.build_processing_tab()
        self.build_postprocessing_tab()
        self.build_multi_strip_tab()
        self.build_apply_tab()

    @staticmethod
    def make_sub_notebook(parent):
        sub_notebook = ttk.Notebook(parent, style="Sub.TNotebook")
        sub_notebook.pack(fill="both", expand=True, padx=0, pady=0)
        return sub_notebook

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------

    def setup_fonts(self):
        """Apply the GUI font to Tk widgets and a matching size to Matplotlib."""
        style = ttk.Style(self)
        if THEME:
            if THEME in style.theme_names():
                style.theme_use(THEME)
            else:
                print(f"Theme '{THEME}' is not available. Choose one of: "
                      f"{', '.join(style.theme_names())}")
        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont",
                     "TkMenuFont", "TkHeadingFont"):
            font = tkFont.nametofont(name)
            if GUI_FONT_NAME:
                font.config(family=GUI_FONT_NAME)
            font.config(size=FONT_SIZE, weight=FONT_WEIGHT)
        default_font = tkFont.nametofont("TkDefaultFont")
        style.configure(".", font=default_font)
        self.option_add("*Font", default_font)
        self.tooltip_font = default_font.copy()
        self.tooltip_font.config(size=max(FONT_SIZE - 1, 6), weight="normal")
        self.left_panel_width = default_font.measure("0") * LEFT_PANEL_CHARS
        # Convert the GUI font size (points on screen) to Matplotlib points
        mpl_size = (FONT_SIZE * float(self.tk.call("tk", "scaling"))
                    * 72.0 / FIG_DPI)
        params = {"font.size": mpl_size, "axes.titlesize": mpl_size,
                  "axes.labelsize": mpl_size, "xtick.labelsize": mpl_size,
                  "ytick.labelsize": mpl_size, "legend.fontsize": mpl_size}
        if GUI_FONT_NAME:
            params["font.family"] = GUI_FONT_NAME
        matplotlib.rcParams.update(params)

    def add_entry(self, parent, label, default, row, width=ENTRY_WIDTH,
                  tooltip=None):
        var = tk.StringVar(value=str(default))
        parent.columnconfigure(1, weight=1)
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=PAD_X,
                          pady=PAD_Y)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=PAD_X, pady=PAD_Y)
        if tooltip:
            self.add_tooltip(label_widget, tooltip)
            self.add_tooltip(entry, tooltip)
        return var

    @staticmethod
    def add_check(parent, label, default, row):
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(parent, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=PAD_X,
            pady=PAD_Y)
        return var

    @staticmethod
    def add_buttons(parent, row, buttons):
        """Add a row of equally sized buttons spanning the two columns."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew",
                   pady=PAD_Y + 1)
        widgets = []
        for col, (text, command) in enumerate(buttons):
            frame.columnconfigure(col, weight=1, uniform="buttons")
            button = ttk.Button(frame, text=text, command=command)
            button.grid(row=0, column=col, sticky="ew", padx=PAD_X)
            widgets.append(button)
        return widgets

    def add_tooltip(self, widget, text):
        return ToolTip(widget, text, font=self.tooltip_font)

    def make_left_panel(self, tab):
        """Fixed-width, vertically scrollable panel for parameters."""
        container = ttk.Frame(tab, width=self.left_panel_width + 20)
        container.pack(side="left", fill="y", padx=(PAD_FRAME, 0),
                       pady=PAD_FRAME)
        container.pack_propagate(False)
        canvas = tk.Canvas(container, highlightthickness=0)
        scroll = ttk.Scrollbar(container, orient="vertical",
                               command=canvas.yview)
        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(
            window, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.scroll_canvases.append(canvas)
        return inner

    def on_mouse_wheel(self, event):
        """Scroll the parameter panel under the mouse pointer."""
        if getattr(self, "ms_dialog", None) is not None:
            return
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            return
        if widget is None or isinstance(widget, (tk.Listbox, tk.Text)):
            return
        while widget is not None and widget not in self.scroll_canvases:
            widget = widget.master
        if widget is None:
            return
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        widget.yview_scroll(step, "units")

    @staticmethod
    def make_param_frame(parent, text):
        frame = ttk.LabelFrame(parent, text=text, padding=FRAME_PADDING)
        frame.pack(fill="x", padx=(0, PAD_FRAME), pady=(0, PAD_FRAME))
        frame.columnconfigure(1, weight=1)
        return frame

    @staticmethod
    def get_value(var, name, dtype=float, allow_none=False):
        text = var.get().strip()
        if allow_none and text.lower() in ("", "none"):
            return None
        try:
            if dtype is int:
                return int(float(text))
            return dtype(text)
        except ValueError:
            raise ValueError(f"Invalid value for '{name}': '{text}'")

    def make_canvas(self, parent, fig):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        widget = canvas.get_tk_widget()
        widget.pack(side="top", fill="both", expand=True)
        widget.bind("<Configure>", lambda e: self.flush_draws(), add="+")
        return canvas, toolbar

    def request_draw(self, canvas, layout=True):
        """Draw now if the canvas is visible, otherwise when it is shown."""
        layout = layout or self.pending_draws.get(canvas, False)
        if canvas.get_tk_widget().winfo_viewable():
            self.pending_draws.pop(canvas, None)
            if layout:
                canvas.figure.tight_layout()
            canvas.draw_idle()
        else:
            self.pending_draws[canvas] = layout

    def flush_draws(self):
        for canvas, layout in list(self.pending_draws.items()):
            if canvas.get_tk_widget().winfo_viewable():
                del self.pending_draws[canvas]
                if layout:
                    canvas.figure.tight_layout()
                canvas.draw_idle()

    def log_image(self, key, mat):
        """Log-scaled image for display, recomputed only for new data."""
        entry = self.log_cache.get(key)
        if entry is None or entry[0] is not mat:
            entry = (mat, np.log(np.clip(mat, 0, None) + 1))
            self.log_cache[key] = entry
        return entry[1]

    def display_data(self, key, mat):
        """Image data and color limits for the current display settings."""
        scale = self.scale_var.get()
        pmin, pmax = self.img_percentiles
        entry = self.display_cache.get(key)
        if (entry is not None and entry[0] is mat
                and entry[1:4] == (scale, pmin, pmax)):
            return entry[4:]
        data = self.log_image(key, mat) if scale == "log" else mat
        if pmin <= 0 and pmax >= 100:
            vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        else:
            vmin, vmax = (float(v) for v in
                          np.nanpercentile(data, (pmin, pmax)))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self.display_cache[key] = (mat, scale, pmin, pmax, data, vmin, vmax)
        return data, vmin, vmax

    def show_image(self, ax, key, mat, title, **kwargs):
        """imshow with the settings of the Image display frame.

        'title' may contain '{scale}', e.g. 'Input data ({scale})'.
        """
        data, vmin, vmax = self.display_data(key, mat)
        image = ax.imshow(data, cmap=self.cmap_var.get(), vmin=vmin,
                          vmax=vmax, aspect="auto",
                          interpolation=IMAGE_INTERPOLATION, **kwargs)
        ax.set_title(title.format(scale=f"{self.scale_var.get()} scale"))
        self.image_artists[ax] = (image, key, mat, title)
        return image

    @safe_call
    def apply_display_settings(self):
        pmin = self.get_value(self.pmin_var, "Min percentile")
        pmax = self.get_value(self.pmax_var, "Max percentile")
        if not 0 <= pmin < pmax <= 100:
            raise ValueError("Percentiles must satisfy 0 <= Min < Max <= "
                             "100!")
        self.img_percentiles = (pmin, pmax)
        scale_text = f"{self.scale_var.get()} scale"
        canvases = []
        for ax, (image, key, mat, title) in list(self.image_artists.items()):
            if image not in ax.get_images():  # Axes cleared since
                del self.image_artists[ax]
                continue
            data, vmin, vmax = self.display_data(key, mat)
            image.set_data(data)
            image.set_cmap(self.cmap_var.get())
            image.set_clim(vmin, vmax)
            ax.set_title(title.format(scale=scale_text))
            if ax.figure.canvas not in canvases:
                canvases.append(ax.figure.canvas)
        for canvas in canvases:
            self.request_draw(canvas, layout=False)

    def set_status(self, text):
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert("1.0", " ".join(str(text).split()))
        self.fit_status_height()

    def fit_status_height(self):
        """Grow/shrink the status bar to show all wrapped lines."""
        num = self.status_text.count("1.0", "end-1c", "displaylines")
        if isinstance(num, tuple):
            num = num[0]
        height = (num or 0) + 1
        if int(self.status_text.cget("height")) != height:
            self.status_text.configure(height=height)

    def on_status_key(self, event):
        """Allow selecting, copying and moving the cursor, but no edits."""
        if event.state & 0x4:  # Control key
            if event.keysym.lower() == "a":
                self.status_text.tag_add("sel", "1.0", "end-1c")
                return "break"
            if event.keysym.lower() in ("c", "insert"):
                return None
        if event.keysym in ("Left", "Right", "Up", "Down", "Home", "End",
                            "Prior", "Next", "Shift_L", "Shift_R",
                            "Control_L", "Control_R"):
            return None
        return "break"

    def on_close(self):
        worker = getattr(self, "ms_worker", None)
        if worker is not None and worker.is_alive():
            if not messagebox.askyesno("Quit", "A calculation is running. "
                                               "Quit anyway?"):
                return
        self.shutdown()

    def shutdown(self):
        if self.closing:
            return
        self.closing = True
        cancel_event = getattr(self, "ms_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        try:
            self.quit()
            self.destroy()
        except tk.TclError:
            pass

    def install_signal_handlers(self):
        def handler(signum, frame):
            self.shutdown()

        for name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
        self.signal_heartbeat()

    def signal_heartbeat(self):
        # Tk's mainloop blocks Python; waking up periodically lets signal
        # handlers run.
        if not self.closing:
            self.after(WORKER_POLL_MS, self.signal_heartbeat)

    # ------------------------------------------------------------------
    # Top panel: input data, reference pattern, q-step, strip, display
    # ------------------------------------------------------------------

    def build_top_panel(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=PAD_OUTER, pady=PAD_OUTER)
        ttk.Style(self).configure("Center.TButton", justify="center",
                                  anchor="center")

        frame_file = ttk.LabelFrame(top, text="Input data",
                                    padding=FRAME_PADDING)
        frame_file.pack(side="left", fill="both", expand=True,
                        padx=(0, PAD_FRAME))
        frame_file.columnconfigure(1, weight=1)
        self.file_var = tk.StringVar()
        self.hdf_key_var = tk.StringVar()
        self.slice_var = tk.StringVar(value="0")
        self.axis_var = tk.StringVar(value="0")
        ttk.Label(frame_file, text="File(s)").grid(row=0, column=0,
                                                  sticky="w", padx=PAD_X,
                                                  pady=PAD_Y)
        file_entry = ttk.Entry(frame_file, textvariable=self.file_var,
                               width=24)
        file_entry.grid(row=0, column=1, columnspan=5, sticky="ew",
                        padx=PAD_X, pady=PAD_Y)
        browse_button = ttk.Button(frame_file, text="Browse...",
                                   command=self.browse_files)
        browse_button.grid(row=0, column=6, sticky="ew", padx=PAD_X,
                           pady=PAD_Y)
        file_tip = ("Image (tif, png), HDF (hdf, h5, nxs) or GeRM dat file. "
                    "Multiple files (separated by ';') are averaged.")
        self.add_tooltip(file_entry, file_tip)
        self.add_tooltip(browse_button, file_tip)
        ttk.Label(frame_file, text="HDF key").grid(row=1, column=0,
                                                   sticky="w", padx=PAD_X,
                                                   pady=PAD_Y)
        self.hdf_key_combo = ttk.Combobox(frame_file,
                                          textvariable=self.hdf_key_var,
                                          width=12)
        self.hdf_key_combo.grid(row=1, column=1, sticky="ew", padx=PAD_X,
                                pady=PAD_Y)
        self.add_tooltip(self.hdf_key_combo,
                         "Path to the dataset in the HDF file. Datasets with "
                         "2 or more dimensions are listed after browsing.")
        slice_tip = ("For 3D datasets: index of the 2D slice along the "
                     "selected axis. Not used if the dataset has a "
                     "dimension of size 1 (it is squeezed).")
        axis_tip = "For 3D datasets: axis (0, 1, or 2) to slice along."
        for col, (text, var, width, tip) in enumerate(
                [("Slice", self.slice_var, 5, slice_tip),
                 ("Axis", self.axis_var, 3, axis_tip)]):
            label = ttk.Label(frame_file, text=text)
            label.grid(row=1, column=2 + 2 * col, sticky="e", padx=PAD_X,
                       pady=PAD_Y)
            entry = ttk.Entry(frame_file, textvariable=var, width=width)
            entry.grid(row=1, column=3 + 2 * col, sticky="w", padx=PAD_X,
                       pady=PAD_Y)
            self.add_tooltip(label, tip)
            self.add_tooltip(entry, tip)
        load_button = ttk.Button(frame_file, text="Load",
                                 command=self.load_data)
        load_button.grid(row=1, column=6, sticky="ew", padx=PAD_X,
                         pady=PAD_Y)
        self.add_tooltip(load_button, "Reload data using the current file "
                                      "path(s), HDF key, slice and axis.")

        self.build_reference_frame(top)
        self.build_q_step_frame(top)

        frame_strip = ttk.LabelFrame(top, text="Strip", padding=FRAME_PADDING)
        frame_strip.pack(side="left", fill="y", padx=(0, PAD_FRAME))
        self.strip_var = tk.StringVar(value="100")
        ttk.Button(frame_strip, text="<", width=3,
                   command=lambda: self.shift_strip(-1)).grid(
            row=0, column=0, padx=PAD_X, pady=PAD_Y)
        self.strip_spin = ttk.Spinbox(frame_strip, from_=0, to=0, width=6,
                                      textvariable=self.strip_var,
                                      command=self.schedule_strip_change)
        self.strip_spin.grid(row=0, column=1, padx=PAD_X, pady=PAD_Y)
        self.strip_spin.bind("<Return>",
                             lambda e: self.schedule_strip_change())
        self.add_tooltip(self.strip_spin, "Strip (row) to calibrate. Press "
                                          "Enter to apply, or click on the "
                                          "image in the Preprocessing tab.")
        ttk.Button(frame_strip, text=">", width=3,
                   command=lambda: self.shift_strip(1)).grid(
            row=0, column=2, padx=PAD_X, pady=PAD_Y)
        self.calib_count_var = tk.StringVar(value="Calibrated strips: 0")
        ttk.Label(frame_strip, textvariable=self.calib_count_var).grid(
            row=1, column=0, columnspan=3, padx=PAD_X, pady=PAD_Y)

        self.build_display_frame(top)

    def build_reference_frame(self, top):
        frame_ref = ttk.LabelFrame(top, text="Reference pattern",
                                   padding=FRAME_PADDING)
        frame_ref.pack(side="left", fill="y", padx=(0, PAD_FRAME))
        self.calibrant_var = tk.StringVar(value="Cr2O3")
        self.num_use_var = tk.StringVar(value="16")
        self.d_threshold_var = tk.StringVar(value="3.5")
        self.manual_d_var = tk.StringVar(value="")
        self.ref_sigma_var = tk.StringVar(value="7")
        file_tip = ("Calibrant bundled with algodif. Select 'None' to use "
                    "only manual d-spacings.")
        manual_tip = (f"Extra d-spacings in {ANGSTROM}, separated by ',' e.g. "
                      f"'4.157, 2.939, 2.4' for LaB6. Combined with the "
                      f"calibrant, with the maximum intensity.")
        widgets = [(ttk.Label(frame_ref, text="Calibrant"), 0, 0, 1,
                    file_tip),
                   (ttk.Combobox(frame_ref, textvariable=self.calibrant_var,
                                 values=["None"] + util.list_xrd_calibrants(),
                                 state="readonly", width=10), 0, 1, 2,
                    file_tip),
                   (ttk.Label(frame_ref, text=f"Manual d ({ANGSTROM})"), 1, 0,
                    1, manual_tip),
                   (ttk.Entry(frame_ref, textvariable=self.manual_d_var,
                              width=10), 1, 1, 2, manual_tip)]
        for widget, row, col, span, tip in widgets:
            widget.grid(row=row, column=col, columnspan=span,
                        sticky="w" if isinstance(widget, ttk.Label)
                        else "ew", padx=PAD_X, pady=PAD_Y)
            self.add_tooltip(widget, tip)
        frame_ref.columnconfigure(1, weight=1)
        entries = [
            ("Num use", self.num_use_var, 0, 3,
             "Maximum number of reflections (largest d-spacings) used from "
             "the calibrant."),
            ("Threshold", self.d_threshold_var, 0, 5,
             "Minimum intensity of reflections used from the calibrant "
             "(strongest peak = 100)."),
            ("Peak sigma", self.ref_sigma_var, 1, 3,
             "Gaussian sigma (pixel) to broaden the simulated peaks of the "
             "reference pattern (display only).")]
        for label, var, row, col, tip in entries:
            label_widget = ttk.Label(frame_ref, text=label)
            label_widget.grid(row=row, column=col, sticky="w", padx=PAD_X,
                              pady=PAD_Y)
            entry = ttk.Entry(frame_ref, textvariable=var,
                              width=REF_ENTRY_WIDTH)
            entry.grid(row=row, column=col + 1, sticky="ew", padx=PAD_X,
                       pady=PAD_Y)
            self.add_tooltip(label_widget, tip)
            self.add_tooltip(entry, tip)
        ttk.Button(frame_ref, text="Build reference",
                   command=self.build_reference).grid(row=1, column=5,
                                                      columnspan=2,
                                                      sticky="ew",
                                                      padx=PAD_X,
                                                      pady=PAD_Y)

    def build_q_step_frame(self, top):
        frame_q = ttk.LabelFrame(top, text="Q-step", padding=FRAME_PADDING)
        frame_q.pack(side="left", fill="y", padx=(0, PAD_FRAME))
        self.q_mode_var = tk.StringVar(value=DEFAULT_Q_MODE)
        self.q_step_var = tk.StringVar(value=Q_STEP)
        self.geom_vars = {key: tk.StringVar(value=default)
                          for key, _, default, _ in GEOMETRY_FIELDS}
        mode_tip = ("How the q-step (q per pixel) is obtained:\n"
                    "- Constant: same q-step for all strips (can be "
                    "estimated from two peaks in the Processing tab).\n"
                    "- GeRM, q-step per strip: estimated from the "
                    "detector geometry; the reference pattern varies "
                    "between strips.\n"
                    "- GeRM, energy matching: peaks are matched in "
                    "energy space (search range in keV).")
        label = ttk.Label(frame_q, text="Mode")
        label.grid(row=0, column=0, sticky="w", padx=PAD_X, pady=PAD_Y)
        combo = ttk.Combobox(frame_q, textvariable=self.q_mode_var,
                             values=list(Q_MODES), state="readonly",
                             width=19)
        combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=PAD_X,
                   pady=PAD_Y)
        combo.bind("<<ComboboxSelected>>", lambda e: self.on_q_mode_changed())
        self.add_tooltip(label, mode_tip)
        self.add_tooltip(combo, mode_tip)
        step_tip = (f"Constant q-step ({Q_UNIT} per pixel). Press Enter to "
                    f"update the reference pattern.")
        label = ttk.Label(frame_q, text=f"Q-step ({Q_UNIT})")
        label.grid(row=1, column=0, sticky="w", padx=PAD_X, pady=PAD_Y)
        self.q_step_entry = ttk.Entry(frame_q, textvariable=self.q_step_var,
                                      width=7)
        self.q_step_entry.grid(row=1, column=1, sticky="ew", padx=PAD_X,
                               pady=PAD_Y)
        self.q_step_entry.bind("<Return>",
                               lambda e: self.on_q_settings_changed())
        self.add_tooltip(label, step_tip)
        self.add_tooltip(self.q_step_entry, step_tip)
        button = ttk.Button(frame_q, text="Geometry...",
                            command=self.open_geometry_dialog)
        button.grid(row=1, column=2, sticky="ew", padx=PAD_X, pady=PAD_Y)
        self.add_tooltip(button, "Two-theta, energy step and geometry of the "
                                 "GeRM detector, for the GeRM modes.")

    def build_display_frame(self, top):
        frame_disp = ttk.LabelFrame(top, text="Image display",
                                    padding=FRAME_PADDING)
        frame_disp.pack(side="left", fill="y")
        self.cmap_var = tk.StringVar(value=IMAGE_CMAPS[0])
        self.scale_var = tk.StringVar(value=IMAGE_SCALES[0])
        self.pmin_var = tk.StringVar(value=str(IMAGE_PERCENTILES[0]))
        self.pmax_var = tk.StringVar(value=str(IMAGE_PERCENTILES[1]))
        for col in (1, 3):
            frame_disp.columnconfigure(col, uniform="disp_entry")
        combos = [("Colormap", self.cmap_var, IMAGE_CMAPS, 0, 0,
                   "Colormap of the images."),
                  ("Scale", self.scale_var, IMAGE_SCALES, 0, 2,
                   "Intensity scale of the images.")]
        for label, var, values, row, col, tip in combos:
            label_widget = ttk.Label(frame_disp, text=label)
            label_widget.grid(row=row, column=col, sticky="w", padx=PAD_X,
                              pady=PAD_Y)
            combo = ttk.Combobox(frame_disp, textvariable=var, values=values,
                                 state="readonly", width=7)
            combo.grid(row=row, column=col + 1, sticky="ew", padx=PAD_X,
                       pady=PAD_Y)
            combo.bind("<<ComboboxSelected>>",
                       lambda e: self.apply_display_settings())
            self.add_tooltip(label_widget, tip)
            self.add_tooltip(combo, tip)
        entries = [("Min %", self.pmin_var, 1, 0,
                    "Lower percentile of the intensity for the contrast. "
                    "Press Enter to apply."),
                   ("Max %", self.pmax_var, 1, 2,
                    "Upper percentile of the intensity for the contrast. "
                    "Press Enter to apply.")]
        for label, var, row, col, tip in entries:
            label_widget = ttk.Label(frame_disp, text=label)
            label_widget.grid(row=row, column=col, sticky="w", padx=PAD_X,
                              pady=PAD_Y)
            entry = ttk.Entry(frame_disp, textvariable=var, width=7)
            entry.grid(row=row, column=col + 1, sticky="ew", padx=PAD_X,
                       pady=PAD_Y)
            entry.bind("<Return>", lambda e: self.apply_display_settings())
            self.add_tooltip(label_widget, tip)
            self.add_tooltip(entry, tip)
        ttk.Button(frame_disp, text="Apply", style="Center.TButton",
                   command=self.apply_display_settings).grid(
            row=0, column=4, rowspan=2, sticky="nsew", padx=PAD_X,
            pady=PAD_Y)

    # ------------------------------------------------------------------
    # Q-step modes and detector geometry
    # ------------------------------------------------------------------

    def q_mode(self):
        return Q_MODES[self.q_mode_var.get()]

    def update_q_step_state(self):
        state = "normal" if self.q_mode() == "constant" else "disabled"
        self.q_step_entry.configure(state=state)

    def on_q_mode_changed(self):
        """Switch default search ranges between q and energy matching."""
        if self.q_mode() == "germ_energy":
            new, old = E_SEARCH, Q_SEARCH
        else:
            new, old = Q_SEARCH, E_SEARCH
        pairs = [(self.search_min_var, self.search_max_var),
                 (self.ms_param_vars["search_min"],
                  self.ms_param_vars["search_max"])]
        for var_min, var_max in pairs:
            try:
                current = (float(var_min.get()), float(var_max.get()))
            except ValueError:
                continue
            if current == (float(old[0]), float(old[1])):
                var_min.set(str(new[0]))
                var_max.set(str(new[1]))
        self.update_q_step_state()
        self.on_q_settings_changed()

    @safe_call
    def on_q_settings_changed(self):
        """Rebuild the reference pattern of the current strip."""
        self.ref_cache = {}
        if self.image is None or self.ref_q is None:
            return
        self.set_strip_reference(self.single_strip())
        self.plot_reference_lines()
        self.plot_preprocessing()
        self.set_status(f"Reference pattern updated. {self.q_step_info()}")

    def get_geometry(self):
        geom = {}
        for key, label, _, _ in GEOMETRY_FIELDS:
            dtype = int if key in ("total_strips", "strip_offset") else float
            geom[key] = self.get_value(self.geom_vars[key], label, dtype)
        if geom["sample_detector"] <= geom["sample_slit"]:
            raise ValueError("Sample-detector distance must be larger than "
                             "the sample-slit distance!")
        return geom

    def get_q_steps(self, rows):
        """Q-step (q per pixel) of image rows for the current mode."""
        rows = np.asarray(rows, dtype=int)
        if self.q_mode() == "constant":
            q_step = self.get_value(self.q_step_var, "Q-step")
            if q_step <= 0:
                raise ValueError("Q-step must be positive!")
            return np.full(len(rows), q_step)
        geom = self.get_geometry()
        return np.asarray([util.estimate_q_step_germ_detector(
            row + geom["strip_offset"], geom["energy_step"],
            geom["two_theta"], total_germ_strip=geom["total_strips"],
            strip_size=geom["strip_size"], sample_slit=geom["sample_slit"],
            sample_detector=geom["sample_detector"]) for row in rows])

    def get_q_step(self, row):
        return float(self.get_q_steps([row])[0])

    def energy_guess(self, row, geom):
        """Energies of the reference d-spacings estimated for a strip."""
        return np.asarray(util.estimate_energy_germ_detector(
            row + geom["strip_offset"], self.d_spacings, geom["two_theta"],
            strip_size=geom["strip_size"],
            total_germ_strip=geom["total_strips"],
            sample_slit=geom["sample_slit"],
            sample_detector=geom["sample_detector"]))

    def get_output_q_step(self, method_var, value_var, mat, width):
        """Q-step of the uniform q-grid of corrected data, and its source."""
        method = method_var.get()
        if method in ("q-max", "q-step"):
            value = self.get_value(value_var, method)
            if value <= 0:
                raise ValueError(f"The {method} value must be positive!")
            if method == "q-max":
                return value / (width - 1.0), f"from q-max {value:.4f}"
            return value, "given q-step"
        valid = np.any(mat != 0, axis=1)
        if not np.any(valid):
            raise ValueError("No calibrated strips in the coefficients!")
        q_map = post.get_q_map(int(np.sum(valid)), width, mat[valid])
        func = {"mean": np.mean, "median": np.median, "max": np.max}[method]
        return float(func(np.diff(q_map, axis=1))), \
            f"{method} pixel spacing of the q-map"

    @staticmethod
    def q_out_text(q_step, source, width):
        return (f"Output q-step: {q_step:.6f} {Q_UNIT} ({source}); q-max: "
                f"{q_step * (width - 1):.4f} {Q_UNIT}.")

    def add_q_out_widgets(self, frame, row, on_change=None):
        """Option and value of the output q-step; returns their variables."""
        method_var = tk.StringVar(value="max")
        value_var = tk.StringVar(value="")
        tip = (f"Q-step of the uniform q-grid of the corrected data:\n"
               f"- mean/median/max: of the pixel spacing of the q-map "
               f"computed from the coefficients.\n"
               f"- q-max: q-step = q-max / (width - 1), q-max in {Q_UNIT}.\n"
               f"- q-step: the given value in {Q_UNIT}.")
        label = ttk.Label(frame, text="Output q-step")
        label.grid(row=row, column=0, sticky="w", padx=PAD_X, pady=PAD_Y)
        combo = ttk.Combobox(frame, textvariable=method_var,
                             values=OUTPUT_Q_OPTIONS, state="readonly",
                             width=ENTRY_WIDTH)
        combo.grid(row=row, column=1, sticky="ew", padx=PAD_X, pady=PAD_Y)
        value_label = ttk.Label(frame, text=f"Value ({Q_UNIT})")
        value_label.grid(row=row + 1, column=0, sticky="w", padx=PAD_X,
                         pady=PAD_Y)
        entry = ttk.Entry(frame, textvariable=value_var, width=ENTRY_WIDTH)
        entry.grid(row=row + 1, column=1, sticky="ew", padx=PAD_X,
                   pady=PAD_Y)
        for widget in (label, combo, value_label, entry):
            self.add_tooltip(widget, tip)

        def update_state(event=None):
            state = "normal" if method_var.get() in ("q-max", "q-step") \
                else "disabled"
            entry.configure(state=state)
            if event is not None and on_change is not None:
                on_change()

        combo.bind("<<ComboboxSelected>>", update_state)
        if on_change is not None:
            entry.bind("<Return>", lambda e: on_change())
        update_state()
        return method_var, value_var

    def q_step_info(self):
        if self.image is None:
            return ""
        try:
            q_steps = self.get_q_steps(np.arange(self.image.shape[0]))
        except ValueError as error:
            return str(error)
        if np.allclose(q_steps, q_steps[0]):
            return f"Q-step: {q_steps[0]:.6f} {Q_UNIT}."
        return (f"Q-step: {np.min(q_steps):.6f} - {np.max(q_steps):.6f} "
                f"{Q_UNIT} (strip {int(np.argmin(q_steps))} - "
                f"{int(np.argmax(q_steps))}).")

    def open_geometry_dialog(self):
        if self.geom_dialog is not None and self.geom_dialog.winfo_exists():
            self.geom_dialog.lift()
            return
        dialog = tk.Toplevel(self)
        dialog.title("GeRM detector geometry")
        dialog.transient(self)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        for row, (key, label, _, tip) in enumerate(GEOMETRY_FIELDS):
            label_widget = ttk.Label(frame, text=label)
            label_widget.grid(row=row, column=0, sticky="w", padx=PAD_X,
                              pady=PAD_Y)
            entry = ttk.Entry(frame, textvariable=self.geom_vars[key],
                              width=ENTRY_WIDTH)
            entry.grid(row=row, column=1, sticky="ew", padx=PAD_X,
                       pady=PAD_Y)
            self.add_tooltip(label_widget, tip)
            self.add_tooltip(entry, tip)
        row = len(GEOMETRY_FIELDS)
        self.geom_info_var = tk.StringVar(value=self.q_step_info())
        ttk.Label(frame, textvariable=self.geom_info_var, wraplength=300,
                  justify="left").grid(row=row, column=0, columnspan=2,
                                       sticky="w", padx=PAD_X, pady=PAD_Y)
        buttons = ttk.Frame(frame)
        buttons.grid(row=row + 1, column=0, columnspan=2, sticky="ew",
                     pady=(PAD_Y + 4, 0))
        for col, (text, command) in enumerate(
                [("Apply", self.apply_geometry),
                 ("Close", dialog.destroy)]):
            buttons.columnconfigure(col, weight=1, uniform="buttons")
            ttk.Button(buttons, text=text, command=command).grid(
                row=0, column=col, sticky="ew", padx=PAD_X)
        dialog.update_idletasks()
        x_pos = self.winfo_rootx() + (self.winfo_width()
                                      - dialog.winfo_width()) // 2
        y_pos = self.winfo_rooty() + 80
        dialog.geometry(f"+{max(x_pos, 0)}+{max(y_pos, 0)}")
        self.geom_dialog = dialog

    @safe_call
    def apply_geometry(self):
        self.get_geometry()  # Validate
        self.geom_info_var.set(self.q_step_info())
        self.on_q_settings_changed()

    # ------------------------------------------------------------------
    # Input data and reference pattern
    # ------------------------------------------------------------------

    def get_file_paths(self):
        return [p.strip() for p in self.file_var.get().split(";")
                if p.strip()]

    @safe_call
    def browse_files(self):
        supported = " ".join("*" + ext for ext in IMAGE_EXT + HDF_EXT
                             + DAT_EXT)
        file_paths = filedialog.askopenfilenames(
            title="Select file(s)",
            filetypes=[("Supported files", supported),
                       ("Image files", " ".join("*" + e for e in IMAGE_EXT)),
                       ("HDF files", " ".join("*" + e for e in HDF_EXT)),
                       ("DAT files", "*.dat"), ("All files", "*.*")])
        if not file_paths:
            return
        self.file_var.set("; ".join(file_paths))
        hdf_files = [p for p in file_paths
                     if Path(p).suffix.lower() in HDF_EXT]
        if hdf_files:
            self.update_hdf_keys(hdf_files[0])
            if not self.hdf_key_var.get().strip():
                self.set_status("No 2D/3D dataset found. Input the HDF key "
                                "then click Load.")
                return
        self.load_data()

    def update_hdf_keys(self, file_path):
        """List datasets with 2 or more dimensions in the HDF key box."""
        keys, shapes, _ = losa.get_hdf_information(file_path)
        candidates = [k for k, s in zip(keys, shapes)
                      if s is not None and len(s) >= 2]
        self.hdf_key_combo.config(values=candidates)
        if candidates and self.hdf_key_var.get().strip() not in candidates:
            self.hdf_key_var.set(candidates[0])

    def load_file(self, file_path):
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name
        if ext in HDF_EXT:
            key = self.hdf_key_var.get().strip()
            if not key:
                raise ValueError("Please input the HDF key (path to the "
                                 "dataset)!")
            dataset, hdf_object = losa.load_hdf(file_path, key,
                                                return_file_obj=True)
            try:
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(f"'{key}' is not a dataset!")
                shape = dataset.shape
                if len(shape) == 3 and 1 not in shape:
                    axis = self.get_value(self.axis_var, "Axis", int)
                    slice_idx = self.get_value(self.slice_var, "Slice", int)
                    if axis not in (0, 1, 2):
                        raise ValueError("Axis must be 0, 1, or 2!")
                    if not 0 <= slice_idx < shape[axis]:
                        raise ValueError(f"Slice index is out of range "
                                         f"[0, {shape[axis] - 1}] for axis "
                                         f"{axis} of shape {shape}")
                    index = [slice(None)] * 3
                    index[axis] = slice_idx
                    mat = dataset[tuple(index)]
                else:
                    mat = np.squeeze(dataset[()])
            finally:
                hdf_object.close()
        elif ext in DAT_EXT:
            mat = losa.load_dat(file_path)
        else:
            mat = losa.load_image(file_path)
        mat = np.asarray(mat, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        if mat.ndim != 2:
            raise ValueError(f"Data from '{name}' must be 2D. Got shape "
                             f"{mat.shape}")
        return mat

    @safe_call
    def load_data(self):
        file_paths = self.get_file_paths()
        if not file_paths:
            raise ValueError("Please select file(s) to load!")
        images = [self.load_file(path) for path in file_paths]
        shapes = {img.shape for img in images}
        if len(shapes) != 1:
            raise ValueError(f"Data have different shapes: {shapes}")
        self.reset_state()
        self.image = np.mean(np.asarray(images), axis=0)
        (height, width) = self.image.shape
        # Results of single-strip calibration, saved as JSON
        self.results_dict = {
            "input": {"file_paths": file_paths,
                      "hdf_key": self.hdf_key_var.get().strip(),
                      "slice": self.slice_var.get().strip(),
                      "axis": self.axis_var.get().strip(),
                      "data_shape": [height, width]},
            "reference": None,
            "strips": {}}
        self.strip_spin.config(to=height - 1)
        try:
            strip_idx = int(self.strip_var.get())
        except ValueError:
            strip_idx = height // 2
        if not 0 <= strip_idx < height:
            self.strip_var.set(height // 2)
        self.notebook.select(0)
        self.change_strip()
        self.after_load()
        self.set_status(f"Loaded {len(file_paths)} file(s). Data shape: "
                        f"{self.image.shape}. Build the reference pattern "
                        f"to continue.")

    def after_load(self):
        """Update the multiple-strip calibration tab for new data."""
        self.single_notebook.select(0)
        self.ms_refresh_group_table()
        self.ms_plot_correction()

    def reset_state(self):
        """Clear data, reference pattern, peaks, results and plots."""
        self.image = None
        self.raw_data = None
        self.proc_data = None
        self.d_spacings = None
        self.ref_inten = None
        self.ref_q = None
        self.ref_cache = {}
        self.ref_pattern = None
        self.ref_pattern_peaks = None
        self.peaks = []
        self.sub_peaks = {}
        self.selected_peak = None
        self.current_result = None
        self.results = {}
        self.results_dict = {}
        self.pick_vlines = None
        self.pick_values = None
        self.annot = None
        self.pre_auto_limits = {}
        self.log_cache = {}
        self.display_cache = {}
        self.tab_strips = {}
        self.tab_shown = {}
        self.peak_listbox.delete(0, tk.END)
        self.update_calib_count()
        for toolbar in (self.toolbar_pre, self.toolbar_proc,
                        self.toolbar_post):
            toolbar.update()  # Clear the zoom/pan history
        self.plot_preprocessing()
        self.plot_processing()
        self.plot_postprocessing()
        self.ms_reset()

    @safe_call
    def build_reference(self):
        if self.image is None:
            raise ValueError("Please load an image first!")
        calibrant = self.calibrant_var.get().strip()
        if calibrant == "None":
            calibrant = ""
        text = self.manual_d_var.get().replace(";", ",")
        try:
            manual_d = [float(v) for v in text.replace(",", " ").split()]
        except ValueError:
            raise ValueError(f"Invalid manual d-spacings: '{text}'. Input "
                             f"values in {ANGSTROM} separated by ','")
        if not calibrant and not manual_d:
            raise ValueError("Please select a calibrant and/or input "
                             "manual d-spacings!")
        if any(d <= 0 for d in manual_d):
            raise ValueError("D-spacings must be positive!")
        num_use = self.get_value(self.num_use_var, "Num use", int)
        threshold = self.get_value(self.d_threshold_var, "Threshold")
        sigma = self.get_value(self.ref_sigma_var, "Peak sigma")
        if calibrant:
            data = util.get_calibrant_diffraction_pattern(
                calibrant, num_use=num_use, threshold=threshold, norm=True,
                check=False)
            d_list = list(2 * np.pi / data[:, 0])
            inten_list = list(data[:, 1])
        else:
            d_list, inten_list = [], []
        if manual_d:
            inten = max(inten_list) if inten_list else 1.0
            d_list += manual_d
            inten_list += [inten] * len(manual_d)
        if not d_list:
            raise ValueError("No reflections! Lower the threshold or "
                             "increase 'Num use'.")
        d_spacings, idx = np.unique(np.asarray(d_list), return_index=True)
        order = np.argsort(d_spacings)[::-1]  # Increasing q
        self.d_spacings = d_spacings[order]
        self.ref_inten = np.asarray(inten_list)[idx][order]
        self.ref_q = 2 * np.pi / self.d_spacings
        self.ref_cache = {}
        self.results_dict["reference"] = {
            "calibrant": calibrant,
            "num_use": num_use,
            "threshold": threshold,
            "manual_d_spacings": manual_d,
            "peak_sigma": sigma,
            "d_spacings": self.d_spacings,
            "intensities": self.ref_inten,
            "q_values": self.ref_q}
        q_text = [f"{q:.4f}" for q in self.ref_q]
        self.est_qa_combo.config(values=q_text)
        self.est_qb_combo.config(values=q_text)
        self.set_strip_reference(self.single_strip())
        self.plot_reference_lines()
        self.plot_preprocessing()
        self.set_status(f"Reference built with {len(self.ref_q)} "
                        f"reflections. Q-values ({Q_UNIT}): "
                        f"{', '.join(q_text)}. {self.q_step_info()}")

    def reference_pattern(self, row):
        """Reference pattern on the q-grid of a strip, and its peaks."""
        q_step = self.get_q_step(row)
        sigma = self.get_value(self.ref_sigma_var, "Peak sigma")
        key = (round(q_step, 12), sigma)
        if key not in self.ref_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pattern = util.make_diffraction_pattern(
                    np.column_stack((self.ref_q, self.ref_inten)), q_step,
                    self.image.shape[1], sigma=sigma, norm=False)
            max_val = np.max(pattern[:, 1])
            if max_val > 0:
                pattern[:, 1] = pattern[:, 1] / max_val
                peaks = prep.get_peaks_by_ranked_descent(
                    pattern[:, 1], width=5, ratio=0.1, sigma=0)
            else:  # All reflections are outside the q-range of the strip
                peaks = np.zeros(0, dtype=int)
            self.ref_cache[key] = (pattern, peaks, q_step)
        return self.ref_cache[key]

    def set_strip_reference(self, row):
        if self.ref_q is None or self.image is None:
            self.ref_pattern = None
            self.ref_pattern_peaks = None
            return
        try:
            self.ref_pattern, self.ref_pattern_peaks, _ = \
                self.reference_pattern(row)
        except ValueError as error:
            self.ref_pattern = None
            self.ref_pattern_peaks = None
            self.set_status(f"Can't build the reference pattern: {error}")

    def single_strip(self):
        """Strip shown in the Single-strip calibration tab."""
        if self.raw_data is not None and "single" in self.tab_shown:
            return self.tab_shown["single"]
        try:
            idx = int(self.strip_var.get())
        except ValueError:
            idx = 0
        if self.image is not None:
            idx = int(np.clip(idx, 0, self.image.shape[0] - 1))
        return idx

    # ------------------------------------------------------------------
    # Strip selection
    # ------------------------------------------------------------------

    def shift_strip(self, step):
        if self.image is None:
            return
        idx = int(self.strip_var.get()) + step
        self.strip_var.set(int(np.clip(idx, 0, self.image.shape[0] - 1)))
        self.schedule_strip_change()

    def schedule_strip_change(self):
        """Update for the latest strip once pending GUI events are handled."""
        if self.strip_after_id is None:
            self.strip_after_id = self.after(STRIP_UPDATE_DELAY_MS,
                                             self.run_strip_change)

    def run_strip_change(self):
        self.strip_after_id = None
        if self.ms_worker is not None and self.ms_worker.is_alive():
            self.schedule_strip_change()
            return
        self.change_strip()

    def current_main_tab(self):
        """'single', 'multi', or None for tabs not using the Strip frame."""
        current = self.notebook.index("current")
        if current == self.notebook.index(self.single_tab):
            return "single"
        if current == self.notebook.index(self.ms_tab):
            return "multi"
        return None

    def on_main_tab_changed(self, event=None):
        """Restore the strip last selected in the newly shown tab."""
        tab = self.current_main_tab()
        if self.main_tab is not None:
            self.tab_strips[self.main_tab] = self.strip_var.get()
        self.main_tab = tab
        if tab is None or self.image is None:
            return
        saved = self.tab_strips.get(tab)
        if saved is not None:
            self.strip_var.set(saved)
        try:
            idx = int(self.strip_var.get())
        except ValueError:
            return
        if self.tab_shown.get(tab) != idx:
            self.change_strip()

    @safe_call
    def change_strip(self):
        """Update the plots of the active tab for the selected strip."""
        if self.image is None:
            return
        height = self.image.shape[0]
        idx = self.get_value(self.strip_var, "Strip", int)
        if not 0 <= idx < height:
            raise ValueError(f"Strip index is out of range [0, {height - 1}]")
        tab = self.current_main_tab()
        if tab is not None:
            self.tab_shown[tab] = idx
        if tab == "multi":
            self.ms_update_cursor()
            self.ms_plot_correction()
        elif tab == "single":
            self.update_single_strip(idx)

    def update_single_strip(self, idx):
        self.raw_data = self.image[idx]
        self.peaks = []
        self.sub_peaks = {}
        self.selected_peak = None
        self.current_result = self.results.get(idx)
        self.set_strip_reference(idx)
        self.apply_preprocessing()
        self.plot_postprocessing()

    def update_calib_count(self):
        self.calib_count_var.set(f"Calibrated strips: {len(self.results)}")

    # ------------------------------------------------------------------
    # Preprocessing tab
    # ------------------------------------------------------------------

    def build_preprocessing_tab(self):
        tab = ttk.Frame(self.single_notebook)
        self.single_notebook.add(tab, text="Preprocessing")
        left = self.make_left_panel(tab)

        frame = self.make_param_frame(left, "Zeroing unreliable data")
        self.zero_left_var = self.add_entry(
            frame, "Zero left", 450, 0,
            tooltip="Number of points on the left side to be zeroed.")
        self.zero_right_var = self.add_entry(
            frame, "Zero right", -300, 1,
            tooltip="Index from which data are zeroed. Negative: counted "
                    "from the end.")

        frame = self.make_param_frame(left, "Smoothing")
        self.pre_sigma_var = self.add_entry(frame, "Gaussian sigma", 0, 0)
        self.use_svd_var = self.add_check(frame, "SVD-based smoothing", True,
                                          1)
        self.svd_rank_var = self.add_entry(frame, "Rank", 5, 2)
        self.svd_lambda_var = self.add_entry(frame, "Lambda threshold", 0.5, 3)

        frame = self.make_param_frame(left, "Background removal (arPLS)")
        self.use_bck_var = self.add_check(frame, "Remove background", False, 0)
        self.bck_ld_var = self.add_entry(frame, "Smoothness (ld)", 1e5, 1)
        self.bck_weight_var = self.add_entry(frame, "Weight", 0.05, 2)
        self.bck_iter_var = self.add_entry(frame, "Iterations", 10, 3)

        ttk.Button(left, text="Apply preprocessing",
                   command=self.apply_preprocessing).pack(
            fill="x", padx=(0, PAD_FRAME), pady=(0, PAD_FRAME))

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=PAD_FRAME,
                   pady=PAD_FRAME)
        self.fig_pre = Figure(figsize=(10, 8), dpi=FIG_DPI)
        grid = self.fig_pre.add_gridspec(3, 1, height_ratios=[1.2, 1, 1])
        self.ax_img = self.fig_pre.add_subplot(grid[0])
        self.ax_strip = self.fig_pre.add_subplot(grid[1])
        self.ax_ref = self.fig_pre.add_subplot(grid[2])
        self.canvas_pre, self.toolbar_pre = self.make_canvas(right,
                                                             self.fig_pre)
        self.image_press_xy = None
        self.canvas_pre.mpl_connect("button_press_event", self.on_press_image)
        self.canvas_pre.mpl_connect("button_release_event",
                                    self.on_release_image)

    @safe_call
    def apply_preprocessing(self):
        if self.raw_data is None:
            raise ValueError("Please load an image first!")
        raw_data = self.raw_data
        npoint = len(raw_data)
        zero_left = self.get_value(self.zero_left_var, "Zero left", int)
        zero_right = self.get_value(self.zero_right_var, "Zero right", int)
        zero_left = int(np.clip(zero_left, 0, npoint - 1))
        if zero_right == 0:
            zero_right = npoint - 1
        sigma = self.get_value(self.pre_sigma_var, "Gaussian sigma")
        svd_para = None
        if self.use_svd_var.get():
            svd_para = {
                "lambda_threshold": self.get_value(self.svd_lambda_var,
                                                   "Lambda threshold"),
                "rank": self.get_value(self.svd_rank_var, "Rank", int)}
        bck_para = None
        if self.use_bck_var.get():
            bck_para = {
                "ld": self.get_value(self.bck_ld_var, "Smoothness"),
                "weight": self.get_value(self.bck_weight_var, "Weight"),
                "num_iter": self.get_value(self.bck_iter_var, "Iterations",
                                           int)}
        params = {"zero_left": zero_left, "zero_right": zero_right,
                  "gaussian_sigma": sigma, "smooth_svd": svd_para,
                  "remove_background_arpls": bck_para}

        def compute():
            data = np.copy(raw_data)
            if zero_left > 0:
                data[:zero_left] = 0.0
            data[zero_right:] = 0.0
            if sigma > 0:
                data = ndi.gaussian_filter1d(data, sigma)
            if svd_para is not None:
                data = prep.smooth_filter_based_svd(data, **svd_para)
            if bck_para is not None:
                data = prep.remove_background_based_arpls(data, **bck_para)
            return data

        if bck_para is None:
            self.finish_preprocessing(raw_data, compute(), params)
            return
        if self.ms_worker is not None and self.ms_worker.is_alive():
            raise ValueError("A calculation is running!")
        self.ms_run_worker(
            lambda cancel_event: compute(),
            lambda data: self.finish_preprocessing(raw_data, data, params),
            "Preprocessing in progress",
            f"Removing background of strip {self.strip_var.get()}...",
            cancellable=False)

    def finish_preprocessing(self, raw_data, data, params):
        if raw_data is not self.raw_data:  # Strip changed meanwhile
            return
        self.proc_data = data
        self.pre_params = params
        self.peaks = []
        self.sub_peaks = {}
        self.selected_peak = None
        self.plot_preprocessing()
        self.plot_processing()
        self.set_status(f"Strip {self.strip_var.get()} preprocessed.")

    def plot_preprocessing(self):
        axes = (self.ax_img, self.ax_strip, self.ax_ref)
        # Views that differ from the last automatic limits were zoomed/panned
        zoomed = {}
        for ax in axes:
            view = (ax.get_xlim(), ax.get_ylim())
            auto_view = self.pre_auto_limits.get(ax)
            if auto_view is not None and ax.has_data() and view != auto_view:
                zoomed[ax] = view
        # Axes whose content is unchanged are kept; only markers/lines move
        kept = []
        relayout = False
        idx = self.single_strip()
        if self.image is None or self.image is not self.pre_img_src:
            self.ax_img.clear()
            self.pre_marker = None
            self.pre_ref_lines = None
            if self.image is not None:
                self.show_image(self.ax_img, "image", self.image,
                                "Input data ({scale}). Click to select a "
                                "strip")
                self.pre_marker = self.ax_img.axhline(idx, color="cyan",
                                                      linewidth=1.0)
                self.ax_img.set_ylabel("Strip index")
                self.plot_reference_lines()
            self.pre_img_src = self.image
            relayout = True
        else:
            self.pre_marker.set_ydata([idx, idx])
            kept.append(self.ax_img)
        if (self.pre_lines is not None and self.raw_data is not None
                and self.proc_data is not None):
            raw_line, proc_line = self.pre_lines
            raw_line.set_data(np.arange(len(self.raw_data)), self.raw_data)
            proc_line.set_data(np.arange(len(self.proc_data)),
                               self.proc_data)
            self.ax_strip.set_title(f"Strip {idx}")
            self.ax_strip.relim()
            self.ax_strip.set_autoscale_on(True)
            self.ax_strip.autoscale_view()
        else:
            self.ax_strip.clear()
            self.pre_lines = None
            if self.raw_data is not None:
                raw_line, = self.ax_strip.plot(self.raw_data, color="blue",
                                               label="Original")
                if self.proc_data is not None:
                    proc_line, = self.ax_strip.plot(self.proc_data,
                                                    color="red",
                                                    label="Preprocessed")
                    self.pre_lines = (raw_line, proc_line)
                self.ax_strip.set_title(f"Strip {idx}")
                self.ax_strip.set_xlabel("Pixel index")
                self.ax_strip.legend(loc="upper right")
                self.ax_strip.grid(True)
            relayout = True
        if self.ref_pattern is None or self.ref_pattern is not \
                self.pre_ref_src:
            self.ax_ref.clear()
            if self.ref_pattern is not None:
                x_list = self.ref_pattern[:, 0]
                y_list = self.ref_pattern[:, 1]
                peaks = self.ref_pattern_peaks
                q_step = x_list[1] - x_list[0]
                self.ax_ref.plot(x_list, y_list, color="blue")
                self.ax_ref.plot(x_list[peaks], y_list[peaks], "o",
                                 color="red")
                self.ax_ref.set_title(f"Reference pattern (q-step "
                                      f"{q_step:.6f}) and its peaks")
                self.ax_ref.set_xlabel(f"Q ({Q_UNIT})")
                self.ax_ref.grid(True)
            self.pre_ref_src = self.ref_pattern
            relayout = True
        else:
            kept.append(self.ax_ref)
        for ax in axes:
            if ax in kept:
                # Show the full view while rebuilding the 'Home' history
                if ax in zoomed and ax in self.pre_auto_limits:
                    ax.set_xlim(self.pre_auto_limits[ax][0])
                    ax.set_ylim(self.pre_auto_limits[ax][1])
            elif ax.has_data():
                self.pre_auto_limits[ax] = (ax.get_xlim(), ax.get_ylim())
            else:
                self.pre_auto_limits.pop(ax, None)
        # Rebuild the toolbar history so 'Home' shows the new full views
        self.toolbar_pre.update()
        self.toolbar_pre.push_current()
        for ax, (xlim, ylim) in zoomed.items():
            if ax.has_data():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        if zoomed:
            self.toolbar_pre.push_current()
        self.request_draw(self.canvas_pre, layout=relayout)

    def plot_reference_lines(self):
        """Reference q-values on the input image at pixel = q / q-step."""
        if self.pre_ref_lines is not None:
            self.pre_ref_lines.remove()
            self.pre_ref_lines = None
        if self.image is None or self.ref_q is None:
            return
        height = self.image.shape[0]
        try:
            q_steps = self.get_q_steps(np.arange(height))
        except ValueError:
            return
        if np.allclose(q_steps, q_steps[0]):
            segments = [[(q / q_steps[0], -0.5), (q / q_steps[0],
                                                  height - 0.5)]
                        for q in self.ref_q]
        else:  # The q-step varies between strips: lines become curves
            rows = np.arange(height)
            segments = [np.column_stack((q / q_steps, rows))
                        for q in self.ref_q]
        self.pre_ref_lines = self.ax_img.add_collection(
            LineCollection(segments, colors="lime", linewidths=1.0),
            autolim=False)

    def on_press_image(self, event):
        if event.inaxes is self.ax_img and event.button == 1:
            self.image_press_xy = (event.x, event.y)
        else:
            self.image_press_xy = None

    def on_release_image(self, event):
        """Select a strip by clicking the image, also in zoom/pan mode."""
        press_xy = self.image_press_xy
        self.image_press_xy = None
        if press_xy is None or self.image is None:
            return
        if event.inaxes is not self.ax_img or event.ydata is None:
            return
        # A drag is a zoom/pan action, not a click
        if abs(event.x - press_xy[0]) > 5 or abs(event.y - press_xy[1]) > 5:
            return
        idx = int(np.clip(round(event.ydata), 0, self.image.shape[0] - 1))
        self.strip_var.set(idx)
        # Let the toolbar finish handling the release before redrawing
        self.schedule_strip_change()

    # ------------------------------------------------------------------
    # Processing tab
    # ------------------------------------------------------------------

    def build_processing_tab(self):
        tab = ttk.Frame(self.single_notebook)
        self.single_notebook.add(tab, text="Processing")
        left = self.make_left_panel(tab)

        frame = self.make_param_frame(left, "Peak detection (ranked descent)")
        self.peak_width_var = self.add_entry(frame, "Min width", 15, 0)
        self.peak_ratio_var = self.add_entry(frame, "Ratio", 0.1, 1)
        self.peak_sigma_var = self.add_entry(frame, "Sigma", 3, 2)
        self.peak_abs_var = self.add_entry(frame, "Abs threshold", "None", 3)
        self.add_buttons(frame, 4, [("Detect peaks", self.detect_peaks),
                                    ("Clear peaks", self.clear_peaks)])

        frame = self.make_param_frame(left, "Peaks")
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=0, column=0, columnspan=2, sticky="ew",
                        padx=PAD_X, pady=PAD_Y)
        self.peak_listbox = tk.Listbox(list_frame, height=6,
                                       exportselection=False)
        self.peak_listbox.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_frame, command=self.peak_listbox.yview)
        scroll.pack(side="left", fill="y")
        self.peak_listbox.config(yscrollcommand=scroll.set)
        self.peak_listbox.bind("<<ListboxSelect>>", self.on_select_listbox)
        self.peak_listbox.bind("<Delete>", self.remove_selected)
        self.add_tooltip(self.peak_listbox,
                         "Peak (pixel) -> sub-pixel position. On the plot: "
                         "left-click to add/select a peak; right-click, "
                         "Shift-click or Delete to remove.")
        remove_button = self.add_buttons(
            frame, 1, [("Remove selected peak", self.remove_selected)])[0]
        self.add_tooltip(remove_button, "Remove the selected peak (or press "
                                        "Delete).")

        frame = self.make_param_frame(left, "Sub-pixel location (Gaussian fit)")
        self.sub_radius_var = self.add_entry(frame, "Radius", 15, 0)
        self.sub_sigma_var = self.add_entry(frame, "Sigma", 1, 1)
        self.sub_skew_var = self.add_check(frame, "Skewed Gaussian", True, 2)
        self.add_buttons(frame, 3, [("Locate selected", self.locate_selected),
                                    ("Locate all", self.locate_all)])

        frame = self.make_param_frame(left, "Q-step from two peaks")
        tip_peak = ("Peak position (pixel). Select a peak and click "
                    "'Selected -> A/B'; the sub-pixel position is used if "
                    "located.")
        tip_q = (f"Q-value ({Q_UNIT}) of the reference reflection of the "
                 f"peak. Choose from the reference or type a value.")
        self.est_pa_var = self.add_entry(frame, "Peak A (pixel)", "", 0,
                                         tooltip=tip_peak)
        self.est_pb_var = self.add_entry(frame, "Peak B (pixel)", "", 1,
                                         tooltip=tip_peak)
        self.est_qa_var = tk.StringVar()
        self.est_qb_var = tk.StringVar()
        combos = []
        for row, (label, var) in enumerate(
                [(f"q A ({Q_UNIT})", self.est_qa_var),
                 (f"q B ({Q_UNIT})", self.est_qb_var)], start=2):
            label_widget = ttk.Label(frame, text=label)
            label_widget.grid(row=row, column=0, sticky="w", padx=PAD_X,
                              pady=PAD_Y)
            combo = ttk.Combobox(frame, textvariable=var, width=ENTRY_WIDTH)
            combo.grid(row=row, column=1, sticky="ew", padx=PAD_X,
                       pady=PAD_Y)
            self.add_tooltip(label_widget, tip_q)
            self.add_tooltip(combo, tip_q)
            combos.append(combo)
        self.est_qa_combo, self.est_qb_combo = combos
        self.add_buttons(frame, 4, [
            ("Selected -> A", lambda: self.use_selected_peak(self.est_pa_var)),
            ("Selected -> B",
             lambda: self.use_selected_peak(self.est_pb_var))])
        button = self.add_buttons(frame, 5, [("Estimate q-step",
                                              self.estimate_q_step)])[0]
        self.add_tooltip(button, "q-step = |qA - qB| / |pA - pB|. Sets the "
                                 "constant q-step mode.")

        frame = self.make_param_frame(left, "Peak matching and fitting")
        self.fit_order_var = self.add_entry(frame, "Fit order", 2, 0)
        self.match_tol_var = self.add_entry(frame, "Tolerance", 0.4, 1)
        self.match_size_var = self.add_entry(frame, "Size", 31, 2)
        self.match_margin_var = self.add_entry(frame, "Margin", 20, 3)
        search_tip = (f"Shift range: {Q_UNIT} for q matching, keV for "
                      f"energy matching.")
        self.search_min_var = self.add_entry(frame, "Search min",
                                             E_SEARCH[0], 4,
                                             tooltip=search_tip)
        self.search_max_var = self.add_entry(frame, "Search max",
                                             E_SEARCH[1], 5,
                                             tooltip=search_tip)
        self.use_sub_var = self.add_check(frame, "Use sub-pixel peaks", True,
                                          6)
        self.add_buttons(frame, 7, [("Calibrate strip",
                                     self.calibrate_strip)])

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=PAD_FRAME,
                   pady=PAD_FRAME)
        self.fig_proc = Figure(figsize=(10, 8), dpi=FIG_DPI)
        grid = self.fig_proc.add_gridspec(2, 1, height_ratios=[1.6, 1])
        self.ax_peaks = self.fig_proc.add_subplot(grid[0])
        self.ax_fit = self.fig_proc.add_subplot(grid[1])
        self.canvas_proc, self.toolbar_proc = self.make_canvas(right,
                                                               self.fig_proc)
        self.canvas_proc.mpl_connect("button_press_event", self.on_click_peaks)
        self.canvas_proc.mpl_connect("key_press_event", self.on_key_peaks)

    def on_key_peaks(self, event):
        if event.key in ("delete", "backspace"):
            self.remove_selected()

    @safe_call
    def detect_peaks(self):
        if self.proc_data is None:
            raise ValueError("Please load an image first!")
        peaks = prep.get_peaks_by_ranked_descent(
            self.proc_data,
            width=self.get_value(self.peak_width_var, "Min width", int),
            ratio=self.get_value(self.peak_ratio_var, "Ratio"),
            sigma=self.get_value(self.peak_sigma_var, "Sigma"),
            abs_threshold=self.get_value(self.peak_abs_var, "Abs threshold",
                                         allow_none=True))
        self.peaks = sorted(int(p) for p in peaks)
        self.sub_peaks = {}
        self.selected_peak = None
        self.plot_processing()
        self.set_status(f"Detected {len(self.peaks)} peaks.")

    def clear_peaks(self):
        self.peaks = []
        self.sub_peaks = {}
        self.selected_peak = None
        self.plot_processing()

    def on_click_peaks(self, event):
        # Errors in Matplotlib callbacks are otherwise printed silently
        try:
            self.handle_click_peaks(event)
        except Exception as error:
            traceback.print_exc()
            self.set_status(f"Error when editing peaks: {error}")

    def handle_click_peaks(self, event):
        if self.proc_data is None:
            self.set_status("Please load data first!")
            return
        if event.inaxes is not self.ax_peaks or event.xdata is None:
            return
        if self.toolbar_proc.mode != "":
            self.set_status("Zoom/Pan mode is active. Click the Zoom/Pan "
                            "button again to turn it off and edit peaks.")
            return
        self.canvas_proc.get_tk_widget().focus_set()
        npoint = len(self.proc_data)
        x_pos = int(np.clip(round(event.xdata), 0, npoint - 1))
        button = int(event.button) if event.button is not None else 0
        # Right button is 3 on Windows/Linux but 2 on macOS (Tk)
        remove = button in (2, 3) or (button == 1 and event.key in (
            "shift", "control", "ctrl+shift", "shift+control"))
        nearest = self.find_nearest_peak(event)
        if remove:
            if nearest is None:
                self.set_status("No peak near the clicked position to "
                                "remove.")
                return
            self.remove_peak(nearest)
            return
        if button != 1:
            return
        if nearest is not None:
            self.selected_peak = nearest
        else:
            try:
                radius = self.get_value(self.sub_radius_var, "Radius", int)
            except ValueError:
                radius = 15
            start = max(0, x_pos - radius)
            stop = min(npoint, x_pos + radius + 1)
            new_peak = start + int(np.argmax(self.proc_data[start:stop]))
            if new_peak not in self.peaks:
                self.peaks = sorted(self.peaks + [new_peak])
                self.set_status(f"Added peak at {new_peak}.")
            self.selected_peak = new_peak
        self.plot_processing()
        if self.fit_info:
            self.set_status(self.fit_info)

    def find_nearest_peak(self, event, tol_display=15):
        """Nearest peak to the mouse, measured horizontally on screen."""
        if not self.peaks:
            return None
        peaks = np.asarray(self.peaks, dtype=float)
        points = np.column_stack((peaks, np.zeros_like(peaks)))
        x_display = self.ax_peaks.transData.transform(points)[:, 0]
        dist = np.abs(x_display - event.x)
        idx = int(np.argmin(dist))
        return self.peaks[idx] if dist[idx] <= tol_display else None

    def remove_peak(self, peak):
        self.peaks = [p for p in self.peaks if p != peak]
        self.sub_peaks.pop(peak, None)
        if self.selected_peak == peak:
            self.selected_peak = None
        self.plot_processing()
        self.set_status(f"Removed peak at {peak}. "
                        f"Remaining peaks: {len(self.peaks)}")

    def remove_selected(self, event=None):
        if self.selected_peak is None:
            self.set_status("No peak selected to remove.")
            return
        self.remove_peak(self.selected_peak)

    def on_select_listbox(self, event):
        selection = self.peak_listbox.curselection()
        if selection:
            self.selected_peak = self.peaks[selection[0]]
            self.plot_processing(update_list=False)
            if self.fit_info:
                self.set_status(self.fit_info)

    def get_sub_pixel_para(self):
        return (self.get_value(self.sub_radius_var, "Radius", int),
                self.get_value(self.sub_sigma_var, "Sigma"),
                self.sub_skew_var.get())

    def locate_peak(self, peak, radius, sigma, skew_fit):
        sub_pos = self.subpixel_position(self.proc_data, peak, radius, sigma,
                                         skew_fit)
        if sub_pos is not None:
            self.sub_peaks[peak] = sub_pos

    @staticmethod
    def subpixel_position(data, peak, radius, sigma, skew_fit):
        result = prep.locate_gaussian_peaks_subpixel(
            data, [peak], radius=radius, sigma=sigma, skew_fit=skew_fit)
        return float(result[0]) if len(result) == 1 else None

    @safe_call
    def locate_selected(self):
        if self.selected_peak is None:
            raise ValueError("Please select a peak (left-click on the plot "
                             "or on the list)!")
        self.locate_peak(self.selected_peak, *self.get_sub_pixel_para())
        self.plot_processing()

    @safe_call
    def locate_all(self):
        if not self.peaks:
            raise ValueError("No peaks! Please detect or add peaks first.")
        para = self.get_sub_pixel_para()
        for peak in self.peaks:
            self.locate_peak(peak, *para)
        self.plot_processing()
        self.set_status(f"Sub-pixel location done for {len(self.sub_peaks)}"
                        f"/{len(self.peaks)} peaks.")

    @safe_call
    def use_selected_peak(self, var):
        if self.selected_peak is None:
            raise ValueError("Please select a peak (left-click on the plot "
                             "or on the list)!")
        pos = self.sub_peaks.get(self.selected_peak,
                                 float(self.selected_peak))
        var.set(f"{pos:.4f}")

    @safe_call
    def estimate_q_step(self):
        p_a = self.get_value(self.est_pa_var, "Peak A")
        p_b = self.get_value(self.est_pb_var, "Peak B")
        q_a = self.get_value(self.est_qa_var, "q A")
        q_b = self.get_value(self.est_qb_var, "q B")
        if p_a == p_b or q_a == q_b:
            raise ValueError("Peaks A and B and their q-values must be "
                             "different!")
        q_step = abs(q_a - q_b) / abs(p_a - p_b)
        self.q_step_var.set(f"{q_step:.8f}")
        if self.q_mode() != "constant":
            self.q_mode_var.set(list(Q_MODES)[0])
            self.on_q_mode_changed()
        else:
            self.on_q_settings_changed()
        self.set_status(f"Estimated q-step: {q_step:.8f} {Q_UNIT} (pixel "
                        f"distance {abs(p_a - p_b):.3f}, q distance "
                        f"{abs(q_a - q_b):.4f} {Q_UNIT}). Mode set to "
                        f"constant q-step.")

    def plot_subpixel_fit(self):
        self.ax_fit.clear()
        self.fit_info = None
        if self.selected_peak is None or self.proc_data is None:
            self.ax_fit.set_title("Select a peak to show its Gaussian fit")
            return
        try:
            radius, sigma, skew_fit = self.get_sub_pixel_para()
        except ValueError:
            return
        data = self.proc_data
        if sigma > 0:
            data = ndi.gaussian_filter1d(data, sigma)
        peak = self.selected_peak
        start = max(0, peak - radius)
        stop = min(len(data), peak + radius + 1)
        list_sub = data[start:stop]
        std = np.std(list_sub)
        if (stop - start) <= 3 or std == 0.0:
            self.ax_fit.set_title(f"Peak {peak}: can't fit")
            self.fit_info = f"Peak {peak}: can't fit (window too small or " \
                            f"flat data)."
            return
        list_norm = (list_sub - np.min(list_sub)) / std
        if skew_fit:
            fit_data, sub_pos, check = prep.fit_skewed_gaussian_peak(list_norm)
        else:
            fit_data, sub_pos, check = prep.fit_gaussian_peak(list_norm)
        if check:
            # Parameters refer to the normalized data in the fitting window
            fit_func = prep.fit_skewed_gaussian_peak if skew_fit \
                else prep.fit_gaussian_peak
            paras = fit_func(list_norm, return_para=True)[1:-1]
            names = ("a", "b", "c", "d", "skew") if skew_fit \
                else ("a", "b", "c", "d")
            para_text = ", ".join(f"{n}={v:.4f}"
                                  for n, v in zip(names, paras))
            model = "Skewed Gaussian" if skew_fit else "Gaussian"
            self.fit_info = (f"Peak {peak}: sub-pixel position "
                             f"{start + sub_pos:.4f}. {model} fit in window "
                             f"[{start}, {stop - 1}] (a: amplitude, b: "
                             f"width, c: center offset, d: baseline): "
                             f"{para_text} | FWHM: {2.355 * paras[1]:.4f}")
        else:
            self.fit_info = f"Peak {peak}: Gaussian fitting failed."
        x_list = np.arange(start, stop)
        self.ax_fit.plot(x_list, list_norm, "-o", color="blue",
                         label="Normalized data")
        self.ax_fit.plot(x_list, fit_data, color="red", label="Fit")
        self.ax_fit.axvline(start + sub_pos, color="green",
                            label=f"Sub-pixel: {start + sub_pos:.3f}")
        self.ax_fit.set_title(f"Peak {peak}. Fit successful: {check}")
        self.ax_fit.set_xlabel("Pixel index")
        self.ax_fit.legend(loc="upper right")
        self.ax_fit.grid(True)

    def plot_processing(self, update_list=True):
        self.ax_peaks.clear()
        if self.proc_data is not None:
            x_list = np.arange(len(self.proc_data))
            self.ax_peaks.plot(x_list, self.proc_data, color="blue")
            if self.peaks:
                peaks = np.asarray(self.peaks)
                self.ax_peaks.plot(peaks, self.proc_data[peaks], "o",
                                   color="red", label="Peaks")
            if self.sub_peaks:
                self.ax_peaks.vlines(list(self.sub_peaks.values()), 0,
                                     np.max(self.proc_data), colors="orange",
                                     linestyles="dashed", linewidth=0.8,
                                     label="Sub-pixel")
            if self.selected_peak is not None:
                self.ax_peaks.plot([self.selected_peak],
                                   [self.proc_data[self.selected_peak]], "o",
                                   markersize=12, markerfacecolor="none",
                                   markeredgecolor="green", markeredgewidth=2,
                                   label="Selected")
            self.ax_peaks.set_title(f"Strip {self.single_strip()}. "
                                    f"Left-click: add/select peak. "
                                    f"Right/Shift-click or Delete: remove")
            self.ax_peaks.set_xlabel("Pixel index")
            if self.peaks:
                self.ax_peaks.legend(loc="upper right")
            self.ax_peaks.grid(True)
        self.plot_subpixel_fit()
        if update_list:
            self.peak_listbox.delete(0, tk.END)
            for peak in self.peaks:
                sub = self.sub_peaks.get(peak)
                text = f"{peak}" if sub is None else f"{peak}  ->  {sub:.3f}"
                self.peak_listbox.insert(tk.END, text)
            if self.selected_peak in self.peaks:
                idx = self.peaks.index(self.selected_peak)
                self.peak_listbox.selection_set(idx)
                self.peak_listbox.see(idx)
        self.request_draw(self.canvas_proc)

    @safe_call
    def calibrate_strip(self):
        if self.proc_data is None:
            raise ValueError("Please load an image first!")
        if self.ref_q is None:
            raise ValueError("Please build the reference pattern first!")
        if self.ms_worker is not None and self.ms_worker.is_alive():
            raise ValueError("A calculation is running!")
        fit_order = self.get_value(self.fit_order_var, "Fit order", int)
        if len(self.peaks) <= fit_order + 1:
            raise ValueError(f"Need more than {fit_order + 1} peaks for "
                             f"fitting order {fit_order}!")
        sub_para = self.get_sub_pixel_para() if self.use_sub_var.get() \
            else None
        match_peaks_para = {
            "tol": self.get_value(self.match_tol_var, "Tolerance"),
            "size": self.get_value(self.match_size_var, "Size", int),
            "margin": self.get_value(self.match_margin_var, "Margin", int),
            "search_range": (self.get_value(self.search_min_var, "Search min"),
                             self.get_value(self.search_max_var, "Search max")),
            "bounds": None, "initial_guess": None}
        strip_idx = self.single_strip()
        mode = self.q_mode()
        q_step = self.get_q_step(strip_idx)
        geom = self.get_geometry() if mode != "constant" else None
        energy_guess = self.energy_guess(strip_idx, geom) \
            if mode == "germ_energy" else None
        proc_data = self.proc_data
        ref_q = self.ref_q
        d_spacings = self.d_spacings
        peak_list = list(self.peaks)
        known_sub = dict(self.sub_peaks)
        sub_names = ("radius", "sigma", "skew_fit")
        parameters = {
            "preprocessing": self.pre_params,
            "processing": {
                "peak_detection": {
                    "width": self.peak_width_var.get().strip(),
                    "ratio": self.peak_ratio_var.get().strip(),
                    "sigma": self.peak_sigma_var.get().strip(),
                    "abs_threshold": self.peak_abs_var.get().strip()},
                "peaks": peak_list,
                "sub_pixel": (dict(zip(sub_names, sub_para))
                              if sub_para is not None else None),
                "match_peaks": match_peaks_para,
                "fit_order": fit_order},
            "q_step_mode": self.q_mode_var.get(),
            "geometry": geom}

        def task(cancel_event):
            new_sub = {}
            if sub_para is not None:
                for peak in peak_list:
                    if peak not in known_sub:
                        sub_pos = self.subpixel_position(proc_data, peak,
                                                         *sub_para)
                        if sub_pos is not None:
                            new_sub[peak] = sub_pos
                all_sub = {**known_sub, **new_sub}
                peaks = np.asarray([all_sub.get(p, float(p))
                                    for p in peak_list])
            else:
                peaks = np.asarray(peak_list, dtype=float)
            common = dict(fit_order=fit_order, zero_left=0,
                          zero_right=len(proc_data), sigma=0,
                          smooth_svd_para=None, get_peaks_para=None,
                          sub_peaks_para=None,
                          match_peaks_para=match_peaks_para,
                          peaks=np.sort(peaks), return_check=True)
            if mode == "germ_energy":
                result = proc.calibrate_q_germ_detector_1d(
                    strip_idx + geom["strip_offset"], proc_data, d_spacings,
                    geom["two_theta"], geom["energy_step"],
                    energy_guess=energy_guess, **common)
            else:
                result = proc.calibrate_q_1d(proc_data, ref_q, q_step,
                                             **common)
            return result, new_sub

        def done(output):
            (coefs, loc_peaks, loc_q), new_sub = output
            if proc_data is not self.proc_data:  # Strip changed meanwhile
                return
            if new_sub:
                self.sub_peaks.update(new_sub)
                self.plot_processing()
            self.finish_calibration(strip_idx, coefs, loc_peaks, loc_q,
                                    q_step, parameters)

        self.ms_run_worker(task, done, "Calibration in progress",
                           f"Calibrating strip {strip_idx}...",
                           cancellable=False)

    def finish_calibration(self, strip_idx, coefs, loc_peaks, loc_q, q_step,
                           parameters):
        if len(loc_q) == 0:
            self.set_status("Calibration failed: not enough matched peaks. "
                            "Adjust peaks, the q-step or matching "
                            "parameters.")
            messagebox.showwarning("Calibration failed",
                                   "Not enough matched peaks. Adjust the "
                                   "peaks, the q-step or the matching "
                                   "parameters.")
            return
        self.current_result = {"strip_idx": strip_idx,
                               "coefficients": np.asarray(coefs),
                               "peaks": np.asarray(loc_peaks),
                               "q_values": np.asarray(loc_q),
                               "q_step": q_step,
                               "parameters": parameters,
                               "proc_data": np.copy(self.proc_data)}
        self.plot_postprocessing()
        self.notebook.select(0)
        self.single_notebook.select(2)
        self.set_status(f"Strip {strip_idx} calibrated. Coefficients "
                        f"(a0, a1, ...): {coefs}. Click 'Add current strip "
                        f"result' to add it to the results of all strips.")

    # ------------------------------------------------------------------
    # Post-processing tab
    # ------------------------------------------------------------------

    def build_postprocessing_tab(self):
        tab = ttk.Frame(self.single_notebook)
        self.single_notebook.add(tab, text="Post-processing")
        left = self.make_left_panel(tab)

        frame = self.make_param_frame(left, "Results")
        self.result_text = tk.Text(frame, width=10, height=22, wrap="word",
                                   font=tkFont.nametofont("TkFixedFont"))
        self.result_text.grid(row=0, column=0, columnspan=2, sticky="ew",
                              padx=0, pady=0)

        frame = self.make_param_frame(left, "Save")
        for row, (text, command, tip) in enumerate([
                ("Add current strip result", self.add_current_result,
                 "Add the result and parameters of the current strip to "
                 "the results of all strips."),
                ("Remove current strip result", self.remove_current,
                 "Remove the current strip from the results of all "
                 "strips."),
                ("Save current results (JSON)", self.save_current,
                 "Input file(s), reference pattern, and for each added "
                 "strip: coefficients, matched peaks, q-step and the "
                 "preprocessing and processing parameters."),
                ("Save calibrated coefficients (CSV)",
                 self.save_calibrated_strip,
                 "Matrix of coefficients (a0, a1, ...), one row per strip, "
                 "as in the Multiple-strip calibration tab. Strips without "
                 "an added result are zeros.")]):
            button = ttk.Button(frame, text=text, command=command)
            button.grid(row=row, column=0, columnspan=2, sticky="ew",
                        padx=PAD_X, pady=PAD_Y)
            self.add_tooltip(button, tip)

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=PAD_FRAME,
                   pady=PAD_FRAME)
        self.fig_post = Figure(figsize=(10, 8), dpi=FIG_DPI)
        grid = self.fig_post.add_gridspec(2, 2)
        self.ax_map = self.fig_post.add_subplot(grid[0, 0])
        self.ax_res = self.fig_post.add_subplot(grid[0, 1])
        self.ax_cal = self.fig_post.add_subplot(grid[1, :])
        self.canvas_post, self.toolbar_post = self.make_canvas(right,
                                                               self.fig_post)
        self.canvas_post.mpl_connect("pick_event", self.on_pick_line)
        self.annot = None

    def plot_postprocessing(self):
        for ax in (self.ax_map, self.ax_res, self.ax_cal):
            ax.clear()
        self.result_text.delete("1.0", tk.END)
        self.pick_vlines = None
        result = self.current_result
        if result is None:
            self.ax_cal.set_title("No calibration result for this strip")
            self.request_draw(self.canvas_post)
            return
        coefs = result["coefficients"]
        loc_peaks = result["peaks"]
        q_values = result["q_values"]
        q_step = result["q_step"]
        list_data = result["proc_data"]
        width = len(list_data)
        if self.ref_q is not None:
            points = np.asarray([(p, q) for p in loc_peaks
                                 for q in self.ref_q])
            self.ax_map.plot(points[:, 0], points[:, 1], ".", color="blue",
                             label="All pairs")
        self.ax_map.plot(loc_peaks, q_values, "o", color="red",
                         label="Matched")
        x_fit = np.linspace(0, loc_peaks[-1] + 15, 200)
        self.ax_map.plot(x_fit, poly.polyval(x_fit, coefs), color="green",
                         label="Fitted")
        margin = 0.1 * (np.max(q_values) - np.min(q_values)) + 0.05
        self.ax_map.set_ylim((np.min(q_values) - margin,
                              np.max(q_values) + margin))
        self.ax_map.set_xlim((loc_peaks[0] - 15, loc_peaks[-1] + 15))
        self.ax_map.set_xlabel("Pixel index")
        self.ax_map.set_ylabel(f"Q ({Q_UNIT})")
        self.ax_map.set_title("Mapping: pixel -> q")
        self.ax_map.legend(loc="upper left")
        self.ax_map.grid(True)

        residual = q_values - poly.polyval(loc_peaks, coefs)
        self.ax_res.plot(loc_peaks, residual, "o-", color="blue")
        self.ax_res.axhline(0.0, color="red")
        self.ax_res.set_xlabel("Pixel index")
        self.ax_res.set_ylabel(f"Residual ({Q_UNIT})")
        self.ax_res.set_title("Fitting residual")
        self.ax_res.grid(True)

        correct_data = post.correct_q_1d(list_data, coefs, q_step)
        if self.ref_pattern is not None:
            self.ax_cal.plot(self.ref_pattern[:, 0], self.ref_pattern[:, 1],
                             color="blue", label="Reference")
        max_val = np.max(correct_data)
        norm = correct_data / max_val if max_val > 0 else correct_data
        self.ax_cal.plot(np.arange(int(width)) * q_step, norm,
                         color="red", label="Calibrated")
        self.pick_values = q_values
        self.pick_vlines = self.ax_cal.vlines(q_values, 0.0, 1.0,
                                              colors="green",
                                              linestyles="solid",
                                              label="Matched reference lines",
                                              picker=5)
        self.ax_cal.set_xlabel(f"Q ({Q_UNIT})")
        self.ax_cal.set_ylabel("Intensity")
        self.ax_cal.set_title(f"Strip {result['strip_idx']}. Calibrated "
                              f"results. Click a green line to show its "
                              f"q-value")
        self.ax_cal.minorticks_on()
        self.ax_cal.grid(True)
        self.ax_cal.legend(loc="upper right")
        self.annot = self.ax_cal.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.request_draw(self.canvas_post)

        rms = np.sqrt(np.mean(residual ** 2))
        lines = [f"Strip: {result['strip_idx']}",
                 f"Q-step: {q_step:.6f} {Q_UNIT}",
                 "Coefficients (a0 + a1*x + a2*x^2 + ...):",
                 "  " + ", ".join(f"{c:.6e}" for c in coefs),
                 f"Number of matched peaks: {len(loc_peaks)}",
                 f"RMS residual: {rms:.5f} {Q_UNIT}", "",
                 f"Peak (pixel)  ->  q ({Q_UNIT})"]
        lines += [f"  {p:10.3f}  ->  {q:8.4f}"
                  for p, q in zip(loc_peaks, q_values)]
        self.result_text.insert(tk.END, "\n".join(lines))

    def on_pick_line(self, event):
        if self.pick_vlines is None or event.artist is not self.pick_vlines:
            return
        q_value = self.pick_values[event.ind[0]]
        self.annot.xy = (q_value, 1.0)
        self.annot.set_text(f"q = {q_value:.4f} {Q_UNIT}\n"
                            f"d = {2 * np.pi / q_value:.4f} {ANGSTROM}")
        self.annot.set_visible(True)
        self.canvas_post.draw_idle()

    @safe_call
    def save_current(self):
        if not self.results_dict.get("strips"):
            raise ValueError("No results! Calibrate a strip and click 'Add "
                             "current strip result' first.")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="q_calib_results.json")
        if not file_path:
            return
        file_path = losa.save_json(file_path, self.results_dict)
        self.set_status(f"Saved results of "
                        f"{len(self.results_dict['strips'])} strip(s) to "
                        f"{file_path}")

    @safe_call
    def add_current_result(self):
        if self.current_result is None:
            raise ValueError("No calibration result for the current strip!")
        idx = self.current_result["strip_idx"]
        self.results[idx] = self.current_result
        strips = self.results_dict["strips"]
        strips[str(idx)] = {key: val for key, val in
                            self.current_result.items() if key != "proc_data"}
        self.results_dict["strips"] = dict(
            sorted(strips.items(), key=lambda item: int(item[0])))
        self.update_calib_count()
        self.set_status(f"Added the result of strip {idx}. Strips in the "
                        f"results: {len(self.results)}")

    @safe_call
    def save_calibrated_strip(self):
        if not self.results:
            raise ValueError("No results! Calibrate a strip and click 'Add "
                             "current strip result' first.")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="q_calib_coefficients.csv")
        if not file_path:
            return
        num_coef = max(len(r["coefficients"]) for r in self.results.values())
        # Same format as the multiple-strip calibration: one row per strip
        mat = np.zeros((self.image.shape[0], num_coef))
        for idx, result in self.results.items():
            coefs = result["coefficients"]
            mat[idx, :len(coefs)] = coefs
        losa.save_csv(file_path, mat)
        self.set_status(f"Saved coefficients of {len(self.results)} "
                        f"calibrated strip(s) ({mat.shape[0]} rows) to "
                        f"{file_path}")

    def remove_current(self):
        idx = self.single_strip()
        if idx in self.results:
            del self.results[idx]
            self.results_dict.get("strips", {}).pop(str(idx), None)
            self.update_calib_count()
            self.set_status(f"Removed strip {idx} from the results. Strips "
                            f"in the results: {len(self.results)}")
        else:
            self.set_status(f"Strip {idx} is not in the results.")

    # ------------------------------------------------------------------
    # Multiple-strip calibration tab
    # ------------------------------------------------------------------

    def build_multi_strip_tab(self):
        self.ms_groups = []
        self.ms_selected_group = None
        self.ms_worker = None
        self.ms_worker_result = None
        self.ms_worker_error = None
        self.ms_progress = ""
        self.ms_run_buttons = []
        self.ms_failed = []
        self.ms_auto_limits = {}
        self.ms_xlim_cids = []
        self.ms_img_key = None
        self.ms_markers = []
        self.ms_cal_cursors = []
        self.ms_q_out = None
        self.ms_clear_results()

        tab = ttk.Frame(self.notebook, padding=0)
        self.ms_tab = tab
        self.notebook.add(tab, text=" Multiple-strip calibration ")
        sub_notebook = self.make_sub_notebook(tab)
        for text, builder in ((" Parameters ", self.build_ms_parameters_tab),
                              (" Calibrating ", self.build_ms_calibrating_tab),
                              (" Correcting", self.build_ms_correction_tab)):
            sub_tab = ttk.Frame(sub_notebook)
            sub_notebook.add(sub_tab, text=text)
            builder(sub_tab)
        self.ms_refresh_group_table()
        self.ms_plot_calibration()
        self.ms_plot_correction()

    def ms_clear_results(self):
        self.ms_coefs = {}
        self.ms_peaks = {}
        self.ms_q_values = {}
        self.ms_status = {}
        self.ms_correct = None

    def ms_reset(self):
        """Clear results of the multiple-strip calibration; keep groups."""
        self.ms_clear_results()
        self.ms_auto_limits = {}
        for toolbar in (self.toolbar_ms_cal, self.toolbar_ms_cor):
            toolbar.update()
        self.ms_refresh_group_table()
        self.ms_refresh_info()
        self.ms_plot_calibration()
        self.ms_plot_correction()

    # Parameters sub-tab

    def build_ms_parameters_tab(self, tab):
        left = self.make_left_panel(tab)
        frame = self.make_param_frame(left, "Parameter group")
        self.ms_strips_var = self.add_entry(
            frame, "Strips", "all", 0,
            tooltip="Strips using the parameters below: 'all', a single "
                    "strip '45', a chunk '0-31', or a list '0-31, 40, "
                    "50-60'. Later groups override earlier ones.")
        buttons = self.add_buttons(frame, 1, [("Add", self.ms_add_group),
                                              ("Remove",
                                               self.ms_remove_group)])
        self.add_tooltip(buttons[0], "Add the strips and parameters below as "
                                     "a new group.")
        self.add_tooltip(buttons[1], "Remove the group selected in the "
                                     "table.")
        self.ms_param_vars = {}
        for title, specs in PARAM_SECTIONS:
            frame = self.make_param_frame(left, title)
            for row, (key, label, default, tip) in enumerate(specs):
                if isinstance(default, bool):
                    self.ms_param_vars[key] = self.add_check(frame, label,
                                                             default, row)
                else:
                    self.ms_param_vars[key] = self.add_entry(
                        frame, label, default, row, tooltip=tip)

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=PAD_FRAME,
                   pady=PAD_FRAME)
        columns = ("name", "strips", "count", "summary")
        ttk.Style(self).configure("MS.Treeview")
        self.ms_group_table = ttk.Treeview(right, columns=columns,
                                           show="headings",
                                           selectmode="browse",
                                           style="MS.Treeview")
        for col, text, width in zip(columns, ("Group", "Strips", "Effective",
                                              "Parameters"),
                                    (90, 160, 80, 600)):
            self.ms_group_table.heading(col, text=text)
            self.ms_group_table.column(col, width=width, anchor="nw",
                                       stretch=col == "summary")
        scroll = ttk.Scrollbar(right, command=self.ms_group_table.yview)
        self.ms_group_table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.ms_group_table.pack(side="left", fill="both", expand=True)
        # Load parameters on user clicks only, not on programmatic selection
        self.ms_group_table.bind("<ButtonRelease-1>", self.ms_on_select_group)
        self.ms_group_table.bind("<Configure>",
                                 lambda e: self.ms_refresh_group_table())
        self.add_tooltip(self.ms_group_table,
                         "Later groups override earlier ones. 'Effective' is "
                         "the number of strips actually using the group. "
                         "Select a group to show its parameters.")

    def ms_get_editor_params(self):
        params = {}
        for key, var in self.ms_param_vars.items():
            params[key] = bool(var.get()) if isinstance(var, tk.BooleanVar) \
                else var.get().strip()
        build_calibration_kwargs(params)  # Validate
        return params

    def ms_get_assignment(self):
        """Group index of each strip; later groups override earlier ones."""
        if self.image is None:
            raise ValueError("Please load data first!")
        height = self.image.shape[0]
        assignment = np.full(height, -1, dtype=int)
        for i, group in enumerate(self.ms_groups):
            assignment[parse_strips(group["strips"], height)] = i
        return assignment

    @safe_call
    def ms_add_group(self):
        strips = self.ms_strips_var.get().strip() or "all"
        if self.image is not None:
            parse_strips(strips, self.image.shape[0])
        self.ms_groups.append({"strips": strips,
                               "params": self.ms_get_editor_params()})
        self.ms_selected_group = len(self.ms_groups) - 1
        self.ms_refresh_group_table()
        self.set_status(f"Added Group {len(self.ms_groups)} for strips: "
                        f"{strips}")

    @safe_call
    def ms_remove_group(self):
        if self.ms_selected_group is None or not self.ms_groups:
            raise ValueError("Please select a group in the table!")
        self.ms_groups.pop(self.ms_selected_group)
        self.ms_selected_group = (min(self.ms_selected_group,
                                      len(self.ms_groups) - 1)
                                  if self.ms_groups else None)
        self.ms_refresh_group_table()
        self.set_status(f"Group removed. Remaining groups: "
                        f"{len(self.ms_groups)}")

    def ms_on_select_group(self, event=None):
        row = self.ms_group_table.identify_row(event.y) if event else ""
        if not row:
            return
        self.ms_selected_group = self.ms_group_table.index(row)
        group = self.ms_groups[self.ms_selected_group]
        self.ms_strips_var.set(group["strips"])
        defaults = default_group_params()
        for key, var in self.ms_param_vars.items():
            var.set(group["params"].get(key, defaults[key]))

    @staticmethod
    def ms_summarize(params, width_chars):
        """All parameters of a group, wrapped to the column width."""
        items = []
        for _, specs in PARAM_SECTIONS:
            for key, label, _, _ in specs:
                value = params.get(key, "")
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                items.append(f"{label}: {value}")
        return textwrap.fill("; ".join(items), width=max(width_chars, 30))

    def ms_refresh_group_table(self):
        try:
            assignment = self.ms_get_assignment()
        except ValueError:
            assignment = None
        font = tkFont.nametofont("TkDefaultFont")
        width_chars = (int(self.ms_group_table.column("summary", "width"))
                       // max(font.measure("0"), 1)) - 2
        self.ms_group_table.delete(*self.ms_group_table.get_children())
        max_lines = 1
        for i, group in enumerate(self.ms_groups):
            count = "-" if assignment is None \
                else int(np.sum(assignment == i))
            summary = self.ms_summarize(group["params"], width_chars)
            max_lines = max(max_lines, summary.count("\n") + 1)
            self.ms_group_table.insert("", "end", iid=str(i), values=(
                f"Group {i + 1}", group["strips"], count, summary))
        ttk.Style(self).configure(
            "MS.Treeview", rowheight=font.metrics("linespace") * max_lines + 6)
        if self.ms_selected_group is not None and self.ms_groups:
            self.ms_group_table.selection_set(str(self.ms_selected_group))

    # Calibrating sub-tab

    def build_ms_calibrating_tab(self, tab):
        left = self.make_left_panel(tab)
        frame = self.make_param_frame(left, "Parallel computing")
        self.ms_ncore_var = self.add_entry(
            frame, "CPU cores", "None", 0,
            tooltip="Number of CPU cores for groups with more than one "
                    "strip. 'None': all cores minus one.")
        ttk.Label(frame, text="Backend").grid(row=1, column=0, sticky="w",
                                              padx=PAD_X, pady=PAD_Y)
        self.ms_prefer_var = tk.StringVar(value="processes")
        ttk.Combobox(frame, textvariable=self.ms_prefer_var, state="readonly",
                     values=("processes", "threads"), width=10).grid(
            row=1, column=1, sticky="ew", padx=PAD_X, pady=PAD_Y)
        self.ms_run_buttons += self.add_buttons(frame, 2, [
            ("Run calibration", self.ms_run_calibration)])

        frame = self.make_param_frame(left, "Information")
        self.ms_info_var = tk.StringVar(value="No calibration results yet.")
        ttk.Label(frame, textvariable=self.ms_info_var, justify="left",
                  wraplength=self.left_panel_width - 20).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=PAD_X,
            pady=PAD_Y)

        frame = self.make_param_frame(left, "Failed strips")
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=0, column=0, columnspan=2, sticky="ew",
                        padx=0, pady=0)
        self.ms_failed_listbox = tk.Listbox(list_frame, height=10,
                                            exportselection=False)
        self.ms_failed_listbox.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_frame,
                               command=self.ms_failed_listbox.yview)
        scroll.pack(side="left", fill="y")
        self.ms_failed_listbox.config(yscrollcommand=scroll.set)
        self.ms_failed_listbox.bind("<<ListboxSelect>>",
                                    self.ms_on_select_failed)
        self.add_tooltip(self.ms_failed_listbox,
                         "Strips where the calibration failed. Click to mark "
                         "a strip on the plots. Add a group with new "
                         "parameters for these strips and run again.")

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=PAD_FRAME,
                   pady=PAD_FRAME)
        self.fig_ms_cal = Figure(figsize=(10, 8), dpi=FIG_DPI)
        axes = self.fig_ms_cal.subplots(2, 2, sharex=True)
        (self.ax_ms_a0, self.ax_ms_a1), (self.ax_ms_npeak,
                                         self.ax_ms_rms) = axes
        self.canvas_ms_cal, self.toolbar_ms_cal = self.make_canvas(
            right, self.fig_ms_cal)
        self.connect_click(self.canvas_ms_cal, self.ms_on_click_calibration)

    def ms_get_parallel_para(self):
        ncore = self.get_value(self.ms_ncore_var, "CPU cores", int,
                               allow_none=True)
        return ncore, self.ms_prefer_var.get()

    def ms_get_reference(self):
        if self.ref_q is None:
            raise ValueError("Please build the reference pattern first!")
        return np.asarray(self.ref_q), np.asarray(self.d_spacings)

    @safe_call
    def ms_run_calibration(self):
        if self.ms_worker is not None and self.ms_worker.is_alive():
            raise ValueError("A calculation is running!")
        if not self.ms_groups:
            raise ValueError("Please add at least one parameter group!")
        ref_q, d_spacings = self.ms_get_reference()
        assignment = self.ms_get_assignment()
        ncore, prefer = self.ms_get_parallel_para()
        max_cores = cpu_count()
        n_jobs = max(1, max_cores - 1) if ncore is None \
            else max(1, min(ncore, max_cores))
        mode = self.q_mode()
        height = self.image.shape[0]
        q_steps = self.get_q_steps(np.arange(height))
        geom = self.get_geometry() if mode == "germ_energy" else None
        tasks = []
        for i, group in enumerate(self.ms_groups):
            rows = np.where(assignment == i)[0]
            if len(rows) > 0:
                tasks.append((i + 1, rows,
                              build_calibration_kwargs(group["params"])))
        guesses = {}
        if mode == "germ_energy":
            guesses = {int(r): self.energy_guess(int(r), geom)
                       for _, rows, _ in tasks for r in rows}
        image = self.image
        total = sum(len(rows) for _, rows, _ in tasks)

        def make_job(row, kwargs):
            if mode == "germ_energy":
                return delayed(run_in_worker)(
                    proc.calibrate_q_germ_detector_1d,
                    int(row) + geom["strip_offset"], image[row], d_spacings,
                    geom["two_theta"], geom["energy_step"],
                    energy_guess=guesses[int(row)], peaks=None,
                    return_check=True, **kwargs)
            return delayed(run_in_worker)(
                proc.calibrate_q_1d, image[row], ref_q, q_steps[row],
                peaks=None, return_check=True, **kwargs)

        def task(cancel_event):
            results = {}
            for k, (number, rows, kwargs) in enumerate(tasks):
                if cancel_event.is_set():
                    return results, True
                self.ms_progress = (f"Group {number} ({k + 1}/{len(tasks)}): "
                                    f"{len(rows)} strip(s)")
                jobs = [make_job(row, kwargs) for row in rows]
                if len(rows) > 1 and n_jobs > 1:
                    outputs = Parallel(n_jobs=n_jobs, prefer=prefer)(jobs)
                else:
                    outputs = [func(*args, **kw) for func, args, kw in jobs]
                for row, (coefs, peaks, q_values) in zip(rows, outputs):
                    results[int(row)] = (np.asarray(coefs),
                                         np.asarray(peaks),
                                         np.asarray(q_values))
            return results, False

        self.ms_run_worker(task, self.ms_finish_calibration,
                           "Calibration in progress",
                           f"Calibrating {total} strip(s)...")

    def ms_finish_calibration(self, result):
        results, cancelled = result
        self.ms_clear_results()
        for idx, (coefs, peaks, q_values) in results.items():
            self.ms_coefs[idx] = coefs
            self.ms_peaks[idx] = peaks
            self.ms_q_values[idx] = q_values
            self.ms_status[idx] = "ok" if len(q_values) > 0 else "failed"
        self.ms_refresh_group_table()
        self.ms_refresh_info()
        self.ms_plot_calibration()
        self.ms_plot_correction()
        prefix = "Calibration cancelled. " if cancelled else ""
        self.set_status(f"{prefix}Calibrated {len(results)} strip(s). "
                        f"Failed: {len(self.ms_failed)}.")

    def ms_run_worker(self, task, on_done, title, message, cancellable=True):
        """Run task(cancel_event) in a thread behind a modal dialog."""
        self.ms_worker_result = None
        self.ms_worker_error = None
        self.ms_progress = message
        self.ms_cancel_event = threading.Event()
        self.ms_show_progress_dialog(title, message, cancellable)

        def target():
            try:
                self.ms_worker_result = task(self.ms_cancel_event)
            except Exception as error:
                self.ms_worker_error = error

        self.ms_worker = threading.Thread(target=target, daemon=True)
        self.ms_worker.start()
        self.ms_poll_worker(on_done)

    def ms_show_progress_dialog(self, title, message, cancellable=True):
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.resizable(False, False)
        # A single computation can't be interrupted, so closing is disabled
        dialog.protocol("WM_DELETE_WINDOW", self.ms_cancel_worker
                        if cancellable else lambda: None)
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill="both", expand=True)
        bold_font = tkFont.nametofont("TkDefaultFont").copy()
        bold_font.config(weight="bold")
        dialog.bold_font = bold_font  # Keep a reference
        ttk.Label(frame, text=f"{title}. Please wait...",
                  font=bold_font).pack(anchor="w", pady=PAD_Y)
        self.ms_dialog_text = tk.StringVar(value=message)
        ttk.Label(frame, textvariable=self.ms_dialog_text, width=50,
                  wraplength=400).pack(anchor="w", pady=PAD_Y)
        self.ms_cancel_button = None
        if cancellable:
            self.ms_cancel_button = ttk.Button(frame, text="Cancel",
                                               command=self.ms_cancel_worker)
            self.ms_cancel_button.pack(pady=(PAD_Y + 4, 0))
        dialog.update_idletasks()
        x_pos = self.winfo_rootx() + (self.winfo_width()
                                      - dialog.winfo_width()) // 2
        y_pos = self.winfo_rooty() + (self.winfo_height()
                                      - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(x_pos, 0)}+{max(y_pos, 0)}")
        # Block interaction with the main window until the task ends
        dialog.grab_set()
        dialog.focus_set()
        self.ms_dialog = dialog

    def ms_cancel_worker(self):
        if self.ms_worker is None or not self.ms_worker.is_alive():
            return
        self.ms_cancel_event.set()
        self.ms_cancel_button.state(["disabled"])
        self.ms_progress = ("Cancelling... The group being calibrated must "
                            "finish first.")

    def ms_poll_worker(self, on_done):
        if self.ms_worker.is_alive():
            self.ms_dialog_text.set(self.ms_progress)
            self.after(WORKER_POLL_MS, lambda: self.ms_poll_worker(on_done))
            return
        self.ms_dialog.grab_release()
        self.ms_dialog.destroy()
        self.ms_dialog = None
        if self.ms_worker_error is not None:
            self.set_status(f"Error: {self.ms_worker_error}")
            messagebox.showerror("Error", str(self.ms_worker_error))
            return
        on_done(self.ms_worker_result)

    def ms_refresh_info(self):
        self.ms_failed = sorted(idx for idx, s in self.ms_status.items()
                                if s == "failed")
        self.ms_failed_listbox.delete(0, tk.END)
        for idx in self.ms_failed:
            self.ms_failed_listbox.insert(tk.END, f"Strip {idx}")
        if not self.ms_status:
            self.ms_info_var.set("No calibration results yet.")
            return
        _, npeak, rms = self.ms_get_coef_table()
        num_ok = len(self.ms_status) - len(self.ms_failed)
        lines = [f"Calibrated strips: {len(self.ms_status)}",
                 f"Successful: {num_ok}", f"Failed: {len(self.ms_failed)}"]
        if num_ok > 0:
            lines += [f"Matched peaks: {int(np.nanmin(npeak))} - "
                      f"{int(np.nanmax(npeak))}",
                      f"RMS residual: {np.nanmin(rms):.5f} - "
                      f"{np.nanmax(rms):.5f} {Q_UNIT}",
                      f"Median RMS: {np.nanmedian(rms):.5f} {Q_UNIT}"]
        self.ms_info_var.set("\n".join(lines))

    def ms_on_select_failed(self, event=None):
        selection = self.ms_failed_listbox.curselection()
        if selection:
            self.ms_select_strip(self.ms_failed[selection[0]])

    def ms_get_coef_table(self):
        """Coefficients (zero-padded), matched peaks and RMS of all strips."""
        height = self.image.shape[0]
        num_coef = max([len(c) for c in self.ms_coefs.values()] + [1])
        mat = np.zeros((height, num_coef))
        npeak = np.full(height, np.nan)
        rms = np.full(height, np.nan)
        for idx, coefs in self.ms_coefs.items():
            if self.ms_status[idx] != "ok":
                continue
            mat[idx, :len(coefs)] = coefs
            npeak[idx] = len(self.ms_q_values[idx])
            rms[idx] = compute_rms(coefs, self.ms_peaks[idx],
                                   self.ms_q_values[idx])
        return mat, npeak, rms

    def ms_plot_calibration(self):
        axes = (self.ax_ms_a0, self.ax_ms_a1, self.ax_ms_npeak,
                self.ax_ms_rms)
        for ax in axes:
            ax.clear()
        self.ms_cal_cursors = []
        if self.image is not None and self.ms_status:
            mat, npeak, rms = self.ms_get_coef_table()
            gain = mat[:, 1] if mat.shape[1] > 1 else np.zeros(len(mat))
            data = [(self.ax_ms_a0, mat[:, 0], f"Offset a0 ({Q_UNIT})"),
                    (self.ax_ms_a1, gain, f"Gain a1 ({Q_UNIT}/pixel)"),
                    (self.ax_ms_npeak, npeak, "Number of matched peaks"),
                    (self.ax_ms_rms, rms, f"RMS residual ({Q_UNIT})")]
            ok = np.asarray([i for i, s in self.ms_status.items()
                             if s == "ok"], dtype=int)
            failed_segs = [[(i, 0.0), (i, 1.0)] for i in self.ms_failed]
            for ax, values, title in data:
                if len(ok) > 0:
                    ax.plot(ok, values[ok], ".", color=STATUS_COLORS["ok"],
                            label="ok")
                if failed_segs:
                    # One collection, spanning the axes height, for speed
                    ax.add_collection(LineCollection(
                        failed_segs, colors=STATUS_COLORS["failed"],
                        alpha=0.3, linewidths=1.0,
                        transform=ax.get_xaxis_transform()), autolim=False)
                self.ms_cal_cursors.append(ax.axvline(
                    self.ms_current_strip(), color="cyan", linewidth=1.0))
                ax.set_title(title)
                ax.grid(True)
            # The x-axis is shared; show all strips
            self.ax_ms_a0.set_xlim(-0.5, self.image.shape[0] - 0.5)
            if self.ms_failed:
                self.ax_ms_a0.plot([], [], color=STATUS_COLORS["failed"],
                                   label="failed")
            self.ax_ms_a0.legend(loc="best")
            self.ax_ms_npeak.set_xlabel("Strip index")
            self.ax_ms_rms.set_xlabel("Strip index")
        else:
            self.ax_ms_a0.set_title("No calibration results yet")
        self.request_draw(self.canvas_ms_cal)

    def ms_update_cursor(self):
        """Move the strip cursor on the calibration plots."""
        if not self.ms_cal_cursors:
            return
        idx = self.ms_current_strip()
        for line in self.ms_cal_cursors:
            line.set_xdata([idx, idx])
        self.request_draw(self.canvas_ms_cal, layout=False)

    def ms_on_click_calibration(self, event):
        if self.image is None or event.inaxes is None:
            return
        if event.xdata is None:
            return
        self.ms_select_strip(int(np.clip(round(event.xdata), 0,
                                         self.image.shape[0] - 1)))

    def ms_current_strip(self):
        try:
            idx = int(self.strip_var.get())
        except (ValueError, tk.TclError):
            idx = 0
        if self.image is not None:
            idx = int(np.clip(idx, 0, self.image.shape[0] - 1))
        return idx

    def ms_select_strip(self, idx):
        """Select a strip, shared with the Strip frame and other tabs."""
        self.strip_var.set(int(idx))
        self.schedule_strip_change()

    def connect_click(self, canvas, callback):
        """Call callback on a left click (not a drag), also in zoom/pan."""
        state = {}

        def on_press(event):
            state["xy"] = (event.x, event.y) if event.button == 1 else None

        def on_release(event):
            press_xy = state.pop("xy", None)
            if press_xy is None or abs(event.x - press_xy[0]) > 5 \
                    or abs(event.y - press_xy[1]) > 5:
                return
            # Let the toolbar finish handling the release before redrawing
            self.after_idle(lambda: callback(event))

        canvas.mpl_connect("button_press_event", on_press)
        canvas.mpl_connect("button_release_event", on_release)

    # Correction sub-tab

    def build_ms_correction_tab(self, tab):
        left = self.make_left_panel(tab)
        frame = self.make_param_frame(left, "Correction")
        self.ms_q_method_var, self.ms_q_value_var = self.add_q_out_widgets(
            frame, 0)
        self.ms_run_buttons += self.add_buttons(frame, 2, [
            ("Apply correction", self.ms_apply_correction)])

        frame = self.make_param_frame(left, "Selected strip")
        self.ms_strip_text = tk.Text(frame, width=10, height=16, wrap="word",
                                     font=tkFont.nametofont("TkFixedFont"))
        self.ms_strip_text.grid(row=0, column=0, columnspan=2, sticky="ew",
                                padx=0, pady=0)

        frame = self.make_param_frame(left, "Save")
        for row, (text, command, tip) in enumerate([
                ("Save coefficients (CSV)", self.ms_save_coefficients,
                 "Matrix of coefficients (a0, a1, ...), one row per strip. "
                 "Uncalibrated/failed strips are zeros."),
                ("Save corrected image (TIFF)", self.ms_save_corrected_image,
                 "Image corrected to a uniform q-grid."),
                ("Save q-corrected map (TIFF)",
                 self.ms_save_q_corrected_map,
                 f"Q-value ({Q_UNIT}) of each pixel of the corrected image: "
                 f"output q-step x pixel index, the same for all strips. "
                 f"Apply the correction first."),
                ("Save q-map (TIFF)", self.ms_save_q_image,
                 f"Q-value ({Q_UNIT}) of each pixel of the original "
                 f"(uncorrected) image, computed from the coefficients. It "
                 f"differs between strips. Uncalibrated/failed strips are "
                 f"zeros."),
                ("Save original image (TIFF)", self.ms_save_original_image,
                 "The loaded image before correction.")]):
            button = ttk.Button(frame, text=text, command=command)
            button.grid(row=row, column=0, columnspan=2, sticky="ew",
                        padx=PAD_X, pady=PAD_Y)
            self.add_tooltip(button, tip)

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=PAD_FRAME,
                   pady=PAD_FRAME)
        self.fig_ms_cor = Figure(figsize=(10, 8), dpi=FIG_DPI)
        grid = self.fig_ms_cor.add_gridspec(2, 2, height_ratios=[1.2, 1])
        self.ax_ms_before = self.fig_ms_cor.add_subplot(grid[0, 0])
        self.ax_ms_after = self.fig_ms_cor.add_subplot(
            grid[0, 1], sharey=self.ax_ms_before)
        self.ax_ms_strip = self.fig_ms_cor.add_subplot(grid[1, :])
        self.canvas_ms_cor, self.toolbar_ms_cor = self.make_canvas(
            right, self.fig_ms_cor)
        self.connect_click(self.canvas_ms_cor, self.ms_on_click_correction)

    def ms_valid_strips(self):
        return np.asarray(sorted(idx for idx, s in self.ms_status.items()
                                 if s == "ok"), dtype=int)

    @safe_call
    def ms_apply_correction(self):
        if self.ms_worker is not None and self.ms_worker.is_alive():
            raise ValueError("A calculation is running!")
        rows = self.ms_valid_strips()
        if len(rows) == 0:
            raise ValueError("No calibrated strips! Please run the "
                             "calibration first.")
        mat, _, _ = self.ms_get_coef_table()
        width = self.image.shape[1]
        q_out, source = self.get_output_q_step(
            self.ms_q_method_var, self.ms_q_value_var, mat, width)
        ncore, prefer = self.ms_get_parallel_para()
        self.set_status(f"Correcting {len(rows)} strip(s)...")
        self.update_idletasks()
        mat_correct = np.zeros_like(self.image, dtype=np.float32)
        mat_correct[rows] = post.correct_q_image(
            self.image[rows], mat[rows], q_out, ncore=ncore, prefer=prefer)
        self.ms_q_out = q_out
        self.ms_correct = mat_correct
        self.ms_plot_correction()
        self.set_status(f"Correction applied to {len(rows)} strip(s). "
                        f"{self.q_out_text(q_out, source, width)}")

    def ms_plot_correction(self):
        axes = (self.ax_ms_before, self.ax_ms_after, self.ax_ms_strip)
        zoomed = self.capture_zoom(axes, self.ms_auto_limits)
        self.ms_strip_text.delete("1.0", tk.END)
        if self.image is None:
            for ax in axes:
                ax.clear()
            self.ms_img_key = None
            self.ms_markers = []
            self.ms_auto_limits = {}
            self.ax_ms_before.set_title("Load data to start")
            self.request_draw(self.canvas_ms_cor)
            return
        height, width = self.image.shape
        idx = self.ms_current_strip()
        q_out = self.ms_q_out if self.ms_correct is not None else None
        old_key = self.ms_img_key
        kept = ()
        if (old_key is not None and old_key[0] is self.image
                and old_key[1] is self.ms_correct
                and old_key[2] == q_out):
            for line in self.ms_markers:
                line.set_ydata([idx, idx])
            kept = (self.ax_ms_before, self.ax_ms_after)
        else:
            self.ax_ms_before.clear()
            self.ax_ms_after.clear()
            self.show_image(self.ax_ms_before, "image", self.image,
                            "Before correction ({scale})",
                            extent=[0, width - 1, height - 0.5, -0.5])
            self.ax_ms_before.set_xlabel("Pixel index")
            self.ax_ms_before.set_ylabel("Strip index")
            if q_out is not None:
                self.show_image(self.ax_ms_after, "ms_correct",
                                self.ms_correct, "After correction ({scale})",
                                extent=[0, q_out * (width - 1),
                                        height - 0.5, -0.5])
                self.ax_ms_after.set_xlabel(f"Q ({Q_UNIT})")
            else:
                self.ax_ms_after.set_title("Click 'Apply correction'")
            self.ms_markers = [ax.axhline(idx, color="cyan", linewidth=1.0)
                               for ax in (self.ax_ms_before,
                                          self.ax_ms_after)]
            self.link_x_axes(self.ax_ms_before, self.ax_ms_after, q_out,
                             self.ms_xlim_cids)
            self.ms_img_key = (self.image, self.ms_correct, q_out)
        self.ax_ms_strip.clear()

        status = self.ms_status.get(idx, "not calibrated")
        try:
            assignment = self.ms_get_assignment()
            group = f"Group {assignment[idx] + 1}" if assignment[idx] >= 0 \
                else "-"
        except ValueError:
            group = "-"
        lines = [f"Strip: {idx}", f"Group: {group}", f"Status: {status}"]
        try:
            strip_q_step = self.get_q_step(idx)
            lines.append(f"Q-step: {strip_q_step:.6f} {Q_UNIT}")
        except ValueError:
            strip_q_step = None
        ax = self.ax_ms_strip
        if status == "ok" and strip_q_step is not None:
            coefs = self.ms_coefs[idx]
            peaks = self.ms_peaks[idx]
            q_values = self.ms_q_values[idx]
            lines += ["Coefficients (a0, a1, ...):",
                      "  " + ", ".join(f"{c:.6e}" for c in coefs),
                      f"Matched peaks: {len(q_values)}",
                      f"RMS residual: "
                      f"{compute_rms(coefs, peaks, q_values):.5f} {Q_UNIT}",
                      "", f"Peak (pixel) -> q ({Q_UNIT})"]
            lines += [f"  {p:9.3f} -> {q:8.4f}"
                      for p, q in zip(peaks, q_values)]
            q_grid = q_out if q_out is not None else strip_q_step
            corrected = post.correct_q_1d(self.image[idx], coefs, q_grid)
            max_val = np.max(corrected)
            if max_val > 0:
                corrected = corrected / max_val
            if self.ref_q is not None:
                try:
                    pattern = self.reference_pattern(idx)[0]
                    ax.plot(pattern[:, 0], pattern[:, 1], color="blue",
                            label="Reference")
                except ValueError:
                    pass
            ax.plot(np.arange(width) * q_grid, corrected, color="red",
                    label="Calibrated")
            ax.vlines(q_values, 0.0, 1.0, colors="green",
                      label="Matched reference lines")
            ax.set_xlabel(f"Q ({Q_UNIT})")
            ax.legend(loc="upper right")
        else:
            ax.plot(self.image[idx], color="blue")
            ax.set_xlabel("Pixel index")
        ax.set_title(f"Strip {idx} ({status})")
        ax.grid(True)
        self.ms_strip_text.insert(tk.END, "\n".join(lines))
        self.restore_zoom(axes, zoomed, self.ms_auto_limits,
                          self.toolbar_ms_cor, kept)
        self.request_draw(self.canvas_ms_cor, layout=not kept)

    @staticmethod
    def capture_zoom(axes, auto_limits):
        """Views that differ from the last automatic limits (zoomed/panned)."""
        # Read all views before any clearing: clearing resets shared limits
        views = {ax: (ax.get_xlabel(), ax.get_xlim(), ax.get_ylim())
                 for ax in axes}
        zoomed = {}
        for ax in axes:
            auto_view = auto_limits.get(ax)
            if auto_view is not None and ax.has_data() \
                    and views[ax] != auto_view:
                zoomed[ax] = views[ax]
        return zoomed

    @staticmethod
    def restore_zoom(axes, zoomed, auto_limits, toolbar, kept=()):
        """Keep zoomed views if the x-unit is unchanged; reset 'Home'.

        Axes in 'kept' were not redrawn, so their full views are unchanged.
        """
        for ax in axes:
            if ax in kept:
                # Show the full view while rebuilding the 'Home' history
                if ax in zoomed and ax in auto_limits:
                    ax.set_xlim(auto_limits[ax][1])
                    ax.set_ylim(auto_limits[ax][2])
            elif ax.has_data():
                auto_limits[ax] = (ax.get_xlabel(), ax.get_xlim(),
                                   ax.get_ylim())
            else:
                auto_limits.pop(ax, None)
        toolbar.update()
        toolbar.push_current()
        restored = False
        for ax, (x_label, xlim, ylim) in zoomed.items():
            if ax.has_data() and ax.get_xlabel() == x_label:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                restored = True
        if restored:
            toolbar.push_current()

    @staticmethod
    def link_x_axes(ax_pixel, ax_q, q_step, cids):
        """Sync x-ranges of a pixel-axis image and a q-axis image."""
        for ax, cid in cids:
            ax.callbacks.disconnect(cid)
        cids.clear()
        if not q_step:
            return
        syncing = {"active": False}

        def make_sync(target, factor):
            def on_change(ax):
                if syncing["active"]:
                    return
                syncing["active"] = True
                try:
                    x_min, x_max = ax.get_xlim()
                    target.set_xlim(x_min * factor, x_max * factor)
                finally:
                    syncing["active"] = False
            return on_change

        for source, target, factor in (
                (ax_pixel, ax_q, q_step),
                (ax_q, ax_pixel, 1.0 / q_step)):
            cid = source.callbacks.connect("xlim_changed",
                                           make_sync(target, factor))
            cids.append((source, cid))

    def ms_on_click_correction(self, event):
        if self.image is None or event.ydata is None:
            return
        if event.inaxes not in (self.ax_ms_before, self.ax_ms_after):
            return
        self.ms_select_strip(int(np.clip(round(event.ydata), 0,
                                         self.image.shape[0] - 1)))

    @safe_call
    def ms_save_coefficients(self):
        if not self.ms_status:
            raise ValueError("No calibration results!")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="q_calib_coefficients.csv")
        if not file_path:
            return
        mat, _, _ = self.ms_get_coef_table()
        losa.save_csv(file_path, mat)
        self.set_status(f"Saved coefficients of {mat.shape[0]} strips to "
                        f"{file_path}")

    @safe_call
    def ms_save_q_image(self):
        if not self.ms_status:
            raise ValueError("No calibration results!")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif", filetypes=[("TIFF", "*.tif")],
            initialfile="q_map.tif")
        if not file_path:
            return
        height, width = self.image.shape
        mat, _, _ = self.ms_get_coef_table()
        q_map = post.get_q_map(height, width, mat)
        file_path = losa.save_image(file_path, np.float32(q_map))
        self.set_status(f"Saved: {file_path}")

    @safe_call
    def ms_save_q_corrected_map(self):
        if self.ms_correct is None:
            raise ValueError("Please apply the correction first!")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif", filetypes=[("TIFF", "*.tif")],
            initialfile="q_corrected_map.tif")
        if not file_path:
            return
        height, width = self.ms_correct.shape
        q_list = self.ms_q_out * np.arange(width)
        q_map = np.tile(np.float32(q_list), (height, 1))
        file_path = losa.save_image(file_path, q_map)
        self.set_status(f"Saved: {file_path}. Q-step: {self.ms_q_out:.6f} "
                        f"{Q_UNIT}, q-max: {q_list[-1]:.4f} {Q_UNIT}")

    @safe_call
    def ms_save_original_image(self):
        if self.image is None:
            raise ValueError("Please load data first!")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif", filetypes=[("TIFF", "*.tif")],
            initialfile="original_image.tif")
        if not file_path:
            return
        file_path = losa.save_image(file_path, np.float32(self.image))
        self.set_status(f"Saved: {file_path}")

    @safe_call
    def ms_save_corrected_image(self):
        if self.ms_correct is None:
            raise ValueError("Please apply the correction first!")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif", filetypes=[("TIFF", "*.tif")],
            initialfile="q_corrected_image.tif")
        if not file_path:
            return
        file_path = losa.save_image(file_path, np.float32(self.ms_correct))
        self.set_status(f"Saved: {file_path}. Q-step of the image: "
                        f"{self.ms_q_out:.6f} {Q_UNIT}")

    # ------------------------------------------------------------------
    # Data correction tab: correct new data with calibrated coefficients
    # ------------------------------------------------------------------

    def build_apply_tab(self):
        self.ac_items = []
        self.ac_coefs = None
        self.ac_image = None
        self.ac_correct = None
        self.ac_q_out = None
        self.ac_strip = 0
        self.ac_auto_limits = {}
        self.ac_xlim_cids = []
        self.ac_img_key = None
        self.ac_markers = []

        tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(tab, text=" Data correction ")
        left = self.make_left_panel(tab)

        frame = self.make_param_frame(left, "Input data")
        self.ac_input_var = self.add_entry(
            frame, "Input", "", 0,
            tooltip="File(s) (tif, dat, hdf; separated by ';') or a folder "
                    "of tif files. Each 2D image, or each slice of a 3D "
                    "HDF dataset, is corrected and saved separately.")
        self.add_buttons(frame, 1, [("Browse files...", self.ac_browse_files),
                                    ("Browse folder...",
                                     self.ac_browse_folder)])
        ttk.Label(frame, text="HDF key").grid(row=2, column=0, sticky="w",
                                              padx=PAD_X, pady=PAD_Y)
        self.ac_hdf_key_var = tk.StringVar()
        self.ac_hdf_key_combo = ttk.Combobox(
            frame, textvariable=self.ac_hdf_key_var, width=ENTRY_WIDTH)
        self.ac_hdf_key_combo.grid(row=2, column=1, sticky="ew", padx=PAD_X,
                                   pady=PAD_Y)
        self.add_tooltip(self.ac_hdf_key_combo,
                         "Path to the dataset in HDF files.")
        self.ac_axis_var = self.add_entry(
            frame, "Axis (3D HDF)", 0, 3,
            tooltip="For 3D HDF datasets: axis along which the images are "
                    "stacked. Each slice is corrected separately.")
        self.add_buttons(frame, 4, [("Load", self.ac_load_input)])
        self.ac_input_info_var = tk.StringVar(value="No input loaded.")
        ttk.Label(frame, textvariable=self.ac_input_info_var,
                  wraplength=self.left_panel_width - 20).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=PAD_X,
            pady=PAD_Y)

        frame = self.make_param_frame(left, "Coefficients")
        self.ac_coef_var = tk.StringVar(value="None")
        ttk.Entry(frame, textvariable=self.ac_coef_var,
                  state="readonly").grid(row=0, column=0, columnspan=2,
                                         sticky="ew", padx=PAD_X,
                                         pady=PAD_Y)
        buttons = self.add_buttons(frame, 1, [
            ("Load from file...", self.ac_load_coefficients),
            ("Clear", self.ac_clear_coefficients)])
        tip = ("CSV file of coefficients (a0, a1, ...), one row per strip. "
               "'None': use the coefficients from the Multiple-strip "
               "calibration tab.")
        self.add_tooltip(buttons[0], tip)
        self.add_tooltip(buttons[1], "Use the coefficients from the "
                                     "Multiple-strip calibration tab.")
        self.ac_q_method_var, self.ac_q_value_var = self.add_q_out_widgets(
            frame, 2, on_change=self.ac_update_preview_safe)
        self.add_buttons(frame, 4, [("Update preview",
                                     self.ac_update_preview_safe)])

        frame = self.make_param_frame(left, "Output")
        self.ac_output_var = self.add_entry(
            frame, "Folder", "", 0,
            tooltip="Output folder. Corrected images are saved as "
                    "'corrected_<input name>.tif'.")
        buttons = self.add_buttons(frame, 1, [
            ("Browse...", self.ac_browse_output),
            ("Make folder", self.ac_make_folder)])
        self.add_tooltip(buttons[1], "Create the folder typed above.")
        self.add_buttons(frame, 2, [("Correct and save all",
                                     self.ac_run)])

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=PAD_FRAME,
                   pady=PAD_FRAME)
        self.fig_ac = Figure(figsize=(10, 8), dpi=FIG_DPI)
        grid = self.fig_ac.add_gridspec(2, 2, height_ratios=[1.2, 1])
        self.ax_ac_before = self.fig_ac.add_subplot(grid[0, 0])
        self.ax_ac_after = self.fig_ac.add_subplot(
            grid[0, 1], sharey=self.ax_ac_before)
        self.ax_ac_strip = self.fig_ac.add_subplot(grid[1, :])
        self.canvas_ac, self.toolbar_ac = self.make_canvas(right,
                                                           self.fig_ac)
        self.connect_click(self.canvas_ac, self.ac_on_click)
        self.ac_plot()

    @safe_call
    def ac_browse_files(self):
        supported = " ".join("*" + ext for ext in IMAGE_EXT + HDF_EXT
                             + DAT_EXT)
        file_paths = filedialog.askopenfilenames(
            title="Select file(s) to correct",
            filetypes=[("Supported files", supported),
                       ("All files", "*.*")])
        if not file_paths:
            return
        self.ac_input_var.set("; ".join(file_paths))
        hdf_files = [p for p in file_paths
                     if Path(p).suffix.lower() in HDF_EXT]
        if hdf_files:
            keys, shapes, _ = losa.get_hdf_information(hdf_files[0])
            candidates = [k for k, s in zip(keys, shapes)
                          if s is not None and len(s) >= 2]
            self.ac_hdf_key_combo.config(values=candidates)
            if candidates and self.ac_hdf_key_var.get() not in candidates:
                self.ac_hdf_key_var.set(candidates[0])
        self.ac_load_input()

    @safe_call
    def ac_browse_folder(self):
        folder = filedialog.askdirectory(title="Select a folder of tif "
                                               "files")
        if folder:
            self.ac_input_var.set(folder)
            self.ac_load_input()

    def ac_get_items(self):
        """List of images to correct: (file, index along axis or None)."""
        text = self.ac_input_var.get().strip()
        if not text:
            raise ValueError("Please select input file(s) or a folder!")
        if Path(text).is_dir():
            paths = losa.find_file(str(Path(text) / "*.tif*"))
        else:
            paths = [p.strip() for p in text.split(";") if p.strip()]
        items = []
        for path in paths:
            if not Path(path).is_file():
                raise ValueError(f"No such file: {path}")
            if Path(path).suffix.lower() not in HDF_EXT:
                items.append((path, None))
                continue
            key = self.ac_hdf_key_var.get().strip()
            if not key:
                raise ValueError("Please input the HDF key!")
            dataset, hdf_object = losa.load_hdf(path, key,
                                                return_file_obj=True)
            shape = dataset.shape
            hdf_object.close()
            if len(shape) == 3 and 1 not in shape:
                axis = self.get_value(self.ac_axis_var, "Axis", int)
                if axis not in (0, 1, 2):
                    raise ValueError("Axis must be 0, 1, or 2!")
                items += [(path, i) for i in range(shape[axis])]
            else:
                items.append((path, None))
        return items

    def ac_load_item(self, item, axis):
        path, index = item
        if index is None or Path(path).suffix.lower() not in HDF_EXT:
            if Path(path).suffix.lower() in HDF_EXT:
                dataset, hdf_object = losa.load_hdf(
                    path, self.ac_hdf_key_var.get().strip(),
                    return_file_obj=True)
                try:
                    mat = np.squeeze(dataset[()])
                finally:
                    hdf_object.close()
            elif Path(path).suffix.lower() in DAT_EXT:
                mat = losa.load_dat(path)
            else:
                mat = losa.load_image(path)
        else:
            dataset, hdf_object = losa.load_hdf(
                path, self.ac_hdf_key_var.get().strip(),
                return_file_obj=True)
            try:
                slices = [slice(None)] * 3
                slices[axis] = index
                mat = dataset[tuple(slices)]
            finally:
                hdf_object.close()
        mat = np.asarray(mat, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"Data from '{Path(path).name}' must be 2D. "
                             f"Got shape {mat.shape}")
        return mat

    @staticmethod
    def ac_output_name(item):
        path, index = item
        stem = Path(path).stem
        if index is not None:
            return f"corrected_{stem}_{index:05d}.tif"
        return f"corrected_{stem}.tif"

    @safe_call
    def ac_load_input(self):
        self.ac_items = self.ac_get_items()
        axis = self.get_value(self.ac_axis_var, "Axis", int)
        self.ac_image = self.ac_load_item(self.ac_items[0], axis)
        self.ac_auto_limits = {}  # New data: show the full view
        self.ac_strip = int(np.clip(self.ac_strip, 0,
                                    self.ac_image.shape[0] - 1))
        self.ac_input_info_var.set(f"{len(self.ac_items)} image(s). "
                                   f"Shape: {self.ac_image.shape}. Showing "
                                   f"the first image.")
        self.ac_update_preview()

    @safe_call
    def ac_load_coefficients(self):
        file_path = filedialog.askopenfilename(
            title="Select coefficients file",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not file_path:
            return
        coefs = np.asarray(losa.load_csv(file_path), dtype=np.float64)
        if coefs.ndim == 1:
            coefs = coefs[np.newaxis, :]
        if coefs.ndim != 2 or np.any(np.isnan(coefs)):
            raise ValueError("The file must contain a numeric matrix: one "
                             "row of coefficients (a0, a1, ...) per strip.")
        self.ac_coefs = coefs
        self.ac_coef_var.set(file_path)
        self.set_status(f"Loaded coefficients of shape {coefs.shape}.")
        self.ac_update_preview()

    def ac_clear_coefficients(self):
        self.ac_coefs = None
        self.ac_coef_var.set("None")
        self.ac_update_preview()

    def ac_get_coefficients(self, height):
        if self.ac_coefs is not None:
            mat = self.ac_coefs
        elif self.ms_status:
            mat = self.ms_get_coef_table()[0]
        else:
            raise ValueError("No coefficients! Load a coefficients file or "
                             "run the multiple-strip calibration.")
        if mat.shape[0] != height:
            raise ValueError(f"Number of coefficient rows ({mat.shape[0]}) "
                             f"differs from the image height ({height})!")
        return mat

    @staticmethod
    def ac_correct_image(image, mat, q_step):
        """Correct strips with non-zero coefficients; others are zeros."""
        valid = np.any(mat != 0, axis=1)
        mat_correct = np.zeros_like(image, dtype=np.float32)
        if np.any(valid):
            mat_correct[valid] = post.correct_q_image(
                image[valid], mat[valid], q_step, ncore=1)
        return mat_correct

    @safe_call
    def ac_update_preview_safe(self):
        self.ac_update_preview()

    def ac_update_preview(self):
        self.ac_correct = None
        self.ac_q_out = None
        if self.ac_image is not None:
            try:
                height, width = self.ac_image.shape
                mat = self.ac_get_coefficients(height)
                q_out, source = self.get_output_q_step(
                    self.ac_q_method_var, self.ac_q_value_var, mat, width)
                self.ac_correct = self.ac_correct_image(self.ac_image, mat,
                                                        q_out)
                self.ac_q_out = q_out
                self.set_status(f"Preview updated. "
                                f"{self.q_out_text(q_out, source, width)}")
            except ValueError as error:
                self.set_status(f"Preview without correction: {error}")
        self.ac_plot()

    @safe_call
    def ac_browse_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.ac_output_var.set(folder)

    @safe_call
    def ac_make_folder(self):
        folder = self.ac_output_var.get().strip()
        if not folder:
            raise ValueError("Please type the path of the folder to create!")
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.set_status(f"Folder ready: {folder}")

    @safe_call
    def ac_run(self):
        if self.ms_worker is not None and self.ms_worker.is_alive():
            raise ValueError("A calculation is running!")
        if not self.ac_items:
            raise ValueError("Please load the input data first!")
        folder = self.ac_output_var.get().strip()
        if not folder:
            raise ValueError("Please select the output folder!")
        if not Path(folder).is_dir():
            raise ValueError(f"Output folder does not exist: {folder}. "
                             f"Click 'Make folder' to create it.")
        height, width = self.ac_image.shape
        axis = self.get_value(self.ac_axis_var, "Axis", int)
        mat = self.ac_get_coefficients(height)
        q_out, source = self.get_output_q_step(
            self.ac_q_method_var, self.ac_q_value_var, mat, width)
        q_text = self.q_out_text(q_out, source, width)
        items = list(self.ac_items)

        def task(cancel_event):
            q_map_path = losa.save_image(
                str(Path(folder) / "q_map" / "q_map_corrected.tif"),
                np.tile(np.float32(q_out * np.arange(width)), (height, 1)))
            num_done = 0
            for i, item in enumerate(items):
                if cancel_event.is_set():
                    return num_done, q_map_path, True
                name = self.ac_output_name(item)
                self.ms_progress = f"Image {i + 1}/{len(items)}: {name}"
                image = self.ac_load_item(item, axis)
                if image.shape[0] != mat.shape[0]:
                    raise ValueError(f"Image {Path(item[0]).name} has "
                                     f"{image.shape[0]} strips, but there "
                                     f"are {mat.shape[0]} coefficient "
                                     f"rows!")
                losa.save_image(str(Path(folder) / name),
                                self.ac_correct_image(image, mat, q_out))
                num_done += 1
            return num_done, q_map_path, False

        def done(result):
            num_done, q_map_path, cancelled = result
            prefix = "Cancelled. " if cancelled else ""
            self.set_status(f"{prefix}Saved {num_done} corrected image(s) "
                            f"to {folder}. Q-values of their pixels: "
                            f"{q_map_path}. {q_text}")

        self.ms_run_worker(task, done, "Correction in progress",
                           f"Correcting {len(items)} image(s)...")

    def ac_plot(self):
        axes = (self.ax_ac_before, self.ax_ac_after, self.ax_ac_strip)
        zoomed = self.capture_zoom(axes, self.ac_auto_limits)
        if self.ac_image is None:
            for ax in axes:
                ax.clear()
            self.ac_img_key = None
            self.ac_markers = []
            self.ac_auto_limits = {}
            self.ax_ac_before.set_title("Load input data to preview")
            self.request_draw(self.canvas_ac)
            return
        height, width = self.ac_image.shape
        idx = int(np.clip(self.ac_strip, 0, height - 1))
        q_out = self.ac_q_out
        has_correct = self.ac_correct is not None and q_out is not None
        old_key = self.ac_img_key
        kept = ()
        if (old_key is not None and old_key[0] is self.ac_image
                and old_key[1] is self.ac_correct
                and old_key[2] == q_out):
            for line in self.ac_markers:
                line.set_ydata([idx, idx])
            kept = (self.ax_ac_before, self.ax_ac_after)
        else:
            self.ax_ac_before.clear()
            self.ax_ac_after.clear()
            self.show_image(self.ax_ac_before, "ac_image", self.ac_image,
                            "First image before correction ({scale})",
                            extent=[0, width - 1, height - 0.5, -0.5])
            self.ax_ac_before.set_xlabel("Pixel index")
            self.ax_ac_before.set_ylabel("Strip index")
            if has_correct:
                self.show_image(self.ax_ac_after, "ac_correct",
                                self.ac_correct, "After correction ({scale})",
                                extent=[0, q_out * (width - 1),
                                        height - 0.5, -0.5])
                self.ax_ac_after.set_xlabel(f"Q ({Q_UNIT})")
            else:
                self.ax_ac_after.set_title("No coefficients for correction")
            self.ac_markers = [ax.axhline(idx, color="cyan", linewidth=1.0)
                               for ax in (self.ax_ac_before,
                                          self.ax_ac_after)]
            self.link_x_axes(self.ax_ac_before, self.ax_ac_after, q_out,
                             self.ac_xlim_cids)
            self.ac_img_key = (self.ac_image, self.ac_correct, q_out)
        ax = self.ax_ac_strip
        ax.clear()
        if has_correct:
            x_q = np.arange(width) * q_out
            # Before: nominal q-axis (pixel x output q-step), no correction
            ax.plot(x_q, self.ac_image[idx], color="blue",
                    label="Before correction")
            ax.plot(x_q, self.ac_correct[idx], color="red",
                    label="After correction")
            ax.set_xlabel(f"Q ({Q_UNIT})")
            ax.legend(loc="upper right")
        else:
            ax.plot(self.ac_image[idx], color="blue")
            ax.set_xlabel("Pixel index")
        ax.set_title(f"Strip {idx}")
        ax.grid(True)
        self.restore_zoom(axes, zoomed, self.ac_auto_limits, self.toolbar_ac,
                          kept)
        self.request_draw(self.canvas_ac, layout=not kept)

    def ac_on_click(self, event):
        if self.ac_image is None or event.ydata is None:
            return
        if event.inaxes not in (self.ax_ac_before, self.ax_ac_after):
            return
        self.ac_strip = int(np.clip(round(event.ydata), 0,
                                    self.ac_image.shape[0] - 1))
        self.ac_plot()


EXIT_MESSAGE_PRINTED = False


def print_exit_message():
    global EXIT_MESSAGE_PRINTED
    if EXIT_MESSAGE_PRINTED:
        return
    EXIT_MESSAGE_PRINTED = True
    print("\n====================================")
    print("  Exit the q-calibration app!")
    print("====================================\n", flush=True)


def main():
    atexit.register(print_exit_message)
    # Workers already stopped by Ctrl+C can't be killed again at exit
    warnings.filterwarnings("ignore", message="Failed ?to kill subprocesses")
    app = None
    try:
        app = QCalibrationGUI()
        app.mainloop()
    except KeyboardInterrupt:
        if app is not None:
            app.shutdown()
    finally:
        print_exit_message()


if __name__ == "__main__":
    main()
