import sys
import h5py
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
# Set global matplotlib font
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.stretch'] = 'condensed'


file_path = "/nsls2/data/hex/proposals/2025-2/pass-318635/edxd/converted/scan_00004/scan_00004_dsp.h5"
bin = 2
q_step = 0.00177540630102 * bin
row_pixel_size = 0.1 * bin # mm
col_pixel_size = 0.1 * bin # mm
height_pixel_size = 0.5691 * bin # mm


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, h5_path, dataset_path="data", parent=None):
        super().__init__(parent)
        self.setWindowTitle("4D Data Visualizer")

        # Set application font
        app_font = QtGui.QFont("Arial Narrow")
        self.setFont(app_font)

        # Data handling
        self.h5_path = h5_path
        self.dataset_path = dataset_path
        self.h5f = h5py.File(h5_path, "r")
        self.dset = self.h5f[dataset_path]

        if self.dset.ndim != 4:
            raise ValueError(
                f"Expected a 4D dataset, got shape {self.dset.shape}")

        self.nq, self.nz, self.ny, self.nx = self.dset.shape
        self.cache = {}
        self.cache_size = 5

        # State
        self.current_q = 0
        self.current_row = 0
        self.current_col = 0
        self.current_clicked_z = 0
        self.current_cmap = "viridis"
        self.clim_min_percentile = 0
        self.clim_max_percentile = 100
        self.clip_min_percentile = 0
        self.clip_max_percentile = 100
        self.vol_actor = None
        self.slice_axis = "nrow"  # "nrow" for Height vs Col, "ncol" for Height vs Row

        # Set up UI
        self.init_ui()

        # Initial render
        self.update_volume()
        self.update_slice()
        self.update_middle_left_plot()
        self.update_middle_right_plot()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QGridLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # --- Top Left: 3D Volume Rendering ---
        self.plotter = QtInteractor(self)
        self.plotter.add_axes(xlabel="Z", ylabel="Y", zlabel="X")
        self.plotter.show_bounds(
            xtitle="Z",
            ytitle="Y",
            ztitle="X",
            grid="back",
            location="outer",
            all_edges=False,
            font_family="arial",
            font_size=12,
        )
        layout.addWidget(self.plotter.interactor, 0, 0)

        # --- Top Right: 2D Slice Plot (nq, nz) ---
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas, 0, 1)

        # Connect events
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)

        # --- Middle Left: New 2D Slice Plot ---
        self.fig_ml = Figure(figsize=(5, 4))
        self.canvas_ml = FigureCanvas(self.fig_ml)
        self.ax_ml = self.fig_ml.add_subplot(111)
        layout.addWidget(self.canvas_ml, 1, 0)

        # --- Middle Right: New 1D Profile Plot ---
        self.fig_mr = Figure(figsize=(5, 4))
        self.canvas_mr = FigureCanvas(self.fig_mr)
        self.ax_mr = self.fig_mr.add_subplot(111)
        layout.addWidget(self.canvas_mr, 1, 1)

        # --- Bottom Control Panels ---
        input_width = 75
        label_width = 110

        # --- Left Panel ---
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QGridLayout(left_widget)
        left_layout.setContentsMargins(5, 0, 5, 0)
        left_layout.setSpacing(10)

        # Row 0: Slicing and Colormap (Left aligned)
        row0_left_widget = QtWidgets.QWidget()
        row0_left_layout = QtWidgets.QHBoxLayout(row0_left_widget)
        row0_left_layout.setContentsMargins(0, 0, 0, 0)
        row0_left_layout.setSpacing(10)

        self.slice_axis_label = QtWidgets.QLabel("Slicing direction:")
        self.slice_axis_label.setFixedWidth(label_width)
        self.slice_axis_combo = QtWidgets.QComboBox()
        self.slice_axis_combo.addItems(["Along Y (Z vs X)", "Along X (Z vs Y)", "Along Z (Y vs X)"])
        self.slice_axis_combo.currentIndexChanged.connect(self.on_slice_axis_changed)

        self.cmap_label = QtWidgets.QLabel("Colormap:")
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "jet", "gray"])
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        self.cmap_combo.setFixedWidth(input_width + 40)

        row0_left_layout.addWidget(self.slice_axis_label)
        row0_left_layout.addWidget(self.slice_axis_combo)
        row0_left_layout.addSpacing(20)
        row0_left_layout.addWidget(self.cmap_label)
        row0_left_layout.addWidget(self.cmap_combo)
        row0_left_layout.addStretch()

        left_layout.addWidget(row0_left_widget, 0, 0, 1, 4)

        # Row 1: Q Index
        self.q_label = QtWidgets.QLabel("Q Index:")
        self.q_label.setFixedWidth(label_width)
        self.q_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.q_slider.setRange(0, self.nq - 1)
        self.q_slider.setValue(self.current_q)

        self.q_input = QtWidgets.QSpinBox()
        self.q_input.setRange(0, self.nq - 1)
        self.q_input.setValue(self.current_q)
        self.q_input.setFixedWidth(input_width)
        self.q_input.setKeyboardTracking(False)

        self.q_slider.valueChanged.connect(self.on_q_changed)
        self.q_input.valueChanged.connect(self.on_q_changed)

        left_layout.addWidget(self.q_label, 1, 0)
        left_layout.addWidget(self.q_slider, 1, 1, 1, 2)
        left_layout.addWidget(self.q_input, 1, 3)

        # Row 2: Min Contrast
        self.min_label = QtWidgets.QLabel("Min Contrast (%):")
        self.min_label.setFixedWidth(label_width)
        self.min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_slider.setRange(0, 100)
        self.min_slider.setValue(self.clim_min_percentile)

        self.min_input = QtWidgets.QSpinBox()
        self.min_input.setRange(0, 100)
        self.min_input.setValue(self.clim_min_percentile)
        self.min_input.setFixedWidth(input_width)
        self.min_input.setKeyboardTracking(False)

        self.min_slider.valueChanged.connect(self.on_contrast_changed)
        self.min_input.valueChanged.connect(self.on_contrast_changed)

        left_layout.addWidget(self.min_label, 2, 0)
        left_layout.addWidget(self.min_slider, 2, 1, 1, 2)
        left_layout.addWidget(self.min_input, 2, 3)

        # Row 3: Min Clip
        self.min_clip_label = QtWidgets.QLabel("Min Threshold (%):")
        self.min_clip_label.setFixedWidth(label_width)
        self.min_clip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_clip_slider.setRange(0, 100)
        self.min_clip_slider.setValue(self.clip_min_percentile)

        self.min_clip_input = QtWidgets.QSpinBox()
        self.min_clip_input.setRange(0, 100)
        self.min_clip_input.setValue(self.clip_min_percentile)
        self.min_clip_input.setFixedWidth(input_width)
        self.min_clip_input.setKeyboardTracking(False)

        self.min_clip_slider.valueChanged.connect(self.on_clip_changed)
        self.min_clip_input.valueChanged.connect(self.on_clip_changed)

        left_layout.addWidget(self.min_clip_label, 3, 0)
        left_layout.addWidget(self.min_clip_slider, 3, 1, 1, 2)
        left_layout.addWidget(self.min_clip_input, 3, 3)

        left_layout.setColumnStretch(1, 1)
        left_layout.setColumnStretch(2, 1)
        layout.addWidget(left_widget, 2, 0)

        # --- Right Panel ---
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QGridLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 5, 0)
        right_layout.setSpacing(10)

        # Row 0: Y Index
        self.row_label = QtWidgets.QLabel("Y Index:")
        self.row_label.setFixedWidth(label_width)
        self.row_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.row_slider.setRange(0, self.ny - 1)
        self.row_slider.setValue(self.current_row)

        self.row_input = QtWidgets.QSpinBox()
        self.row_input.setRange(0, self.ny - 1)
        self.row_input.setValue(self.current_row)
        self.row_input.setFixedWidth(input_width)
        self.row_input.setKeyboardTracking(False)

        self.row_slider.valueChanged.connect(self.on_row_changed)
        self.row_input.valueChanged.connect(self.on_row_changed)

        right_layout.addWidget(self.row_label, 0, 0)
        right_layout.addWidget(self.row_slider, 0, 1, 1, 2)
        right_layout.addWidget(self.row_input, 0, 3)

        # Row 1: X Index
        self.col_label = QtWidgets.QLabel("X Index:")
        self.col_label.setFixedWidth(label_width)
        self.col_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.col_slider.setRange(0, self.nx - 1)
        self.col_slider.setValue(self.current_col)

        self.col_input = QtWidgets.QSpinBox()
        self.col_input.setRange(0, self.nx - 1)
        self.col_input.setValue(self.current_col)
        self.col_input.setFixedWidth(input_width)
        self.col_input.setKeyboardTracking(False)

        self.col_slider.valueChanged.connect(self.on_col_changed)
        self.col_input.valueChanged.connect(self.on_col_changed)

        right_layout.addWidget(self.col_label, 1, 0)
        right_layout.addWidget(self.col_slider, 1, 1, 1, 2)
        right_layout.addWidget(self.col_input, 1, 3)

        # Row 2: Max Contrast
        self.max_label = QtWidgets.QLabel("Max Contrast (%):")
        self.max_label.setFixedWidth(label_width)
        self.max_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_slider.setRange(0, 100)
        self.max_slider.setValue(self.clim_max_percentile)

        self.max_input = QtWidgets.QSpinBox()
        self.max_input.setRange(0, 100)
        self.max_input.setValue(self.clim_max_percentile)
        self.max_input.setFixedWidth(input_width)
        self.max_input.setKeyboardTracking(False)

        self.max_slider.valueChanged.connect(self.on_contrast_changed)
        self.max_input.valueChanged.connect(self.on_contrast_changed)

        right_layout.addWidget(self.max_label, 2, 0)
        right_layout.addWidget(self.max_slider, 2, 1, 1, 2)
        right_layout.addWidget(self.max_input, 2, 3)

        # Row 3: Max Clip
        self.max_clip_label = QtWidgets.QLabel("Max Threshold (%):")
        self.max_clip_label.setFixedWidth(label_width)
        self.max_clip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_clip_slider.setRange(0, 100)
        self.max_clip_slider.setValue(self.clip_max_percentile)

        self.max_clip_input = QtWidgets.QSpinBox()
        self.max_clip_input.setRange(0, 100)
        self.max_clip_input.setValue(self.clip_max_percentile)
        self.max_clip_input.setFixedWidth(input_width)
        self.max_clip_input.setKeyboardTracking(False)

        self.max_clip_slider.valueChanged.connect(self.on_clip_changed)
        self.max_clip_input.valueChanged.connect(self.on_clip_changed)

        right_layout.addWidget(self.max_clip_label, 3, 0)
        right_layout.addWidget(self.max_clip_slider, 3, 1, 1, 2)
        right_layout.addWidget(self.max_clip_input, 3, 3)

        right_layout.setColumnStretch(1, 1)
        right_layout.setColumnStretch(2, 1)
        layout.addWidget(right_widget, 2, 1)

        # Status Bar
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hover over the slice plot to see values")

        # Set stretch factors
        layout.setRowStretch(0, 10)
        layout.setRowStretch(1, 10)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

    def on_q_changed(self, value):
        sender = self.sender()
        if sender == self.q_input:
            self.q_slider.blockSignals(True)
            self.q_slider.setValue(value)
            self.q_slider.blockSignals(False)
        else:
            self.q_input.blockSignals(True)
            self.q_input.setValue(value)
            self.q_input.blockSignals(False)

        if value != self.current_q:
            self.current_q = value
            self.update_volume()
            self.update_slice()
            self.update_middle_left_plot()
            self.update_middle_right_plot()

    def on_row_changed(self, value):
        sender = self.sender()
        if sender == self.row_input:
            self.row_slider.blockSignals(True)
            self.row_slider.setValue(value)
            self.row_slider.blockSignals(False)
        else:
            self.row_input.blockSignals(True)
            self.row_input.setValue(value)
            self.row_input.blockSignals(False)

        if value != self.current_row:
            self.current_row = value
            self.update_slice()
            self.update_middle_left_plot()
            self.update_middle_right_plot()

    def on_col_changed(self, value):
        sender = self.sender()
        if sender == self.col_input:
            self.col_slider.blockSignals(True)
            self.col_slider.setValue(value)
            self.col_slider.blockSignals(False)
        else:
            self.col_input.blockSignals(True)
            self.col_input.setValue(value)
            self.col_input.blockSignals(False)

        if value != self.current_col:
            self.current_col = value
            self.update_slice()
            self.update_middle_left_plot()
            self.update_middle_right_plot()

    def on_contrast_changed(self, value):
        sender = self.sender()
        v_min = self.min_slider.value() if sender not in [self.min_slider, self.min_input] else value
        v_max = self.max_slider.value() if sender not in [self.max_slider, self.max_input] else value

        # Force Min < Max
        if sender in [self.min_slider, self.min_input]:
            if v_min >= self.max_slider.value():
                v_min = max(0, self.max_slider.value() - 1)
        else:
            if v_max <= self.min_slider.value():
                v_max = min(100, self.min_slider.value() + 1)

        # Sync widgets
        self.min_slider.blockSignals(True)
        self.min_slider.setValue(v_min)
        self.min_slider.blockSignals(False)
        self.min_input.blockSignals(True)
        self.min_input.setValue(v_min)
        self.min_input.blockSignals(False)

        self.max_slider.blockSignals(True)
        self.max_slider.setValue(v_max)
        self.max_slider.blockSignals(False)
        self.max_input.blockSignals(True)
        self.max_input.setValue(v_max)
        self.max_input.blockSignals(False)

        if v_min != self.clim_min_percentile or v_max != self.clim_max_percentile:
            self.clim_min_percentile = v_min
            self.clim_max_percentile = v_max
            self.update_volume()
            self.update_slice()
            self.update_middle_left_plot()

    def on_clip_changed(self, value):
        sender = self.sender()
        v_min = self.min_clip_slider.value() if sender not in [self.min_clip_slider, self.min_clip_input] else value
        v_max = self.max_clip_slider.value() if sender not in [self.max_clip_slider, self.max_clip_input] else value

        # Force Min < Max
        if sender in [self.min_clip_slider, self.min_clip_input]:
            if v_min >= self.max_clip_slider.value():
                v_min = max(0, self.max_clip_slider.value() - 1)
        else:
            if v_max <= self.min_clip_slider.value():
                v_max = min(100, self.min_clip_slider.value() + 1)

        # Sync widgets
        self.min_clip_slider.blockSignals(True)
        self.min_clip_slider.setValue(v_min)
        self.min_clip_slider.blockSignals(False)
        self.min_clip_input.blockSignals(True)
        self.min_clip_input.setValue(v_min)
        self.min_clip_input.blockSignals(False)

        self.max_clip_slider.blockSignals(True)
        self.max_clip_slider.setValue(v_max)
        self.max_clip_slider.blockSignals(False)
        self.max_clip_input.blockSignals(True)
        self.max_clip_input.setValue(v_max)
        self.max_clip_input.blockSignals(False)

        if v_min != self.clip_min_percentile or v_max != self.clip_max_percentile:
            self.clip_min_percentile = v_min
            self.clip_max_percentile = v_max
            self.update_volume()
            self.update_slice()
            self.update_middle_left_plot()

    def on_cmap_changed(self, value):
        self.current_cmap = value
        self.update_volume()
        self.update_slice()
        self.update_middle_left_plot()

    def on_slice_axis_changed(self, index):
        if index == 0:
            self.slice_axis = "nrow"
        elif index == 1:
            self.slice_axis = "ncol"
        else:
            self.slice_axis = "nz"
        self.update_middle_left_plot()

    def update_volume(self):
        vol = self.get_volume(self.current_q)
        # Reverse Row Index (axis 1 in vol (nz, ny, nx))
        vol = vol[:, ::-1, :]

        # Apply clipping
        cmin = np.percentile(vol, self.clip_min_percentile)
        cmax = np.percentile(vol, self.clip_max_percentile)
        vol[(vol < cmin) | (vol > cmax)] = 0

        if self.vol_actor is not None:
            self.plotter.remove_actor(self.vol_actor)

        # Percentile-based clim
        vmin = float(np.percentile(vol, self.clim_min_percentile))
        vmax = float(np.percentile(vol, self.clim_max_percentile))
        if vmax <= vmin: vmax = vmin + 1e-12

        grid = pv.ImageData(dimensions=np.array(vol.shape) + 1)
        # height (Z), row (Y), col (X)
        grid.spacing = (height_pixel_size, row_pixel_size, col_pixel_size)
        grid.origin = (0.0, 0.0, 0.0)
        grid.cell_data["values"] = vol.ravel(order="F")
        self.vol_actor = self.plotter.add_volume(
            grid, resolution=[1, 1, 1], scalars="values",
            cmap=self.current_cmap, opacity="linear", clim=(vmin, vmax),
            blending="composite", mapper="smart", shade=True, show_scalar_bar=False,
        )
        self.plotter.render()

    def update_slice(self):
        data_slice = self.dset[:, :, self.current_row, self.current_col]
        
        # Apply clipping
        cmin = np.percentile(data_slice, self.clip_min_percentile)
        cmax = np.percentile(data_slice, self.clip_max_percentile)
        data_slice = np.clip(data_slice, cmin, cmax)

        vmin = np.percentile(data_slice, self.clim_min_percentile)
        vmax = np.percentile(data_slice, self.clim_max_percentile)
        if vmin == vmax: vmax = vmin + 1e-12

        self.ax.clear()
        # x is Q, y is Z
        extent = [0, (self.nq - 1) * q_step, (self.nz - 1) * height_pixel_size, 0]
        self.ax.imshow(data_slice.T, aspect='auto', origin='upper', extent=extent, cmap=self.current_cmap, vmin=vmin, vmax=vmax)
        self.ax.set_title(f"Slice at Y Index={self.current_row}, X Index={self.current_col}")

        self.ax.set_xlabel("Q")
        self.ax.set_ylabel("Z [mm]")
        self.ax.axvline(x=self.current_q * q_step, color='r', linestyle='--', alpha=0.5)
        self.ax.axhline(y=self.current_clicked_z * height_pixel_size, color='g', linestyle='--', alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_middle_left_plot(self):
        self.ax_ml.clear()
        if self.slice_axis == "nrow":
            # (nq, nz, ny, nx) -> (nz, nx) -> (nx, nz)
            data_2d = self.dset[self.current_q, :, self.current_row, :].T
            title = f"X vs Z (Y Index={self.current_row}, Q={self.current_q * q_step:.4f})"
            extent = [0, (self.nz - 1) * height_pixel_size, (self.nx - 1) * col_pixel_size, 0]
            self.ax_ml.set_ylabel("X [mm]")
            self.ax_ml.set_xlabel("Z [mm]")
        elif self.slice_axis == "ncol":
            # (nq, nz, ny, nx) -> (nz, ny) -> (ny, nz)
            data_2d = self.dset[self.current_q, :, :, self.current_col].T
            title = f"Y vs Z (X Index={self.current_col}, Q={self.current_q * q_step:.4f})"
            extent = [0, (self.nz - 1) * height_pixel_size, (self.ny - 1) * row_pixel_size, 0]
            self.ax_ml.set_ylabel("Y [mm]")
            self.ax_ml.set_xlabel("Z [mm]")
        else:  # "nz"
            # (nq, nz, ny, nx) -> (ny, nx)
            data_2d = self.dset[self.current_q, self.current_clicked_z, :, :]
            title = f"Y vs X (Z={self.current_clicked_z * height_pixel_size:.2f} mm, Q={self.current_q * q_step:.4f})"
            extent = [0, (self.nx - 1) * col_pixel_size, (self.ny - 1) * row_pixel_size, 0]
            self.ax_ml.set_ylabel("Y [mm]")
            self.ax_ml.set_xlabel("X [mm]")

        # Apply clipping
        cmin = np.percentile(data_2d, self.clip_min_percentile)
        cmax = np.percentile(data_2d, self.clip_max_percentile)
        data_2d = np.clip(data_2d, cmin, cmax)

        vmin = np.percentile(data_2d, self.clim_min_percentile)
        vmax = np.percentile(data_2d, self.clim_max_percentile)
        if vmin == vmax: vmax = vmin + 1e-12

        self.ax_ml.imshow(data_2d, aspect='auto', origin='upper', extent=extent, cmap=self.current_cmap, vmin=vmin, vmax=vmax)
        self.ax_ml.set_title(title)
        self.fig_ml.tight_layout()
        self.canvas_ml.draw()

    def update_middle_right_plot(self):
        self.ax_mr.clear()
        profile = self.dset[:, self.current_clicked_z, self.current_row, self.current_col]
        q_axis = np.arange(self.nq) * q_step
        self.ax_mr.plot(q_axis, profile, color='b')
        self.ax_mr.axvline(x=self.current_q * q_step, color='r', linestyle='--', alpha=0.5)
        self.ax_mr.set_title(f"Intensity Profile at Z={self.current_clicked_z * height_pixel_size:.2f} mm")
        self.ax_mr.set_xlabel("Q")
        self.ax_mr.set_ylabel("Intensity")
        self.fig_mr.tight_layout()
        self.canvas_mr.draw()

    def get_volume(self, q_idx, half_width=3):
        q0 = max(0, q_idx - half_width)
        q1 = min(self.nq, q_idx + half_width + 1)
        vol = np.asarray(self.dset[q0:q1, :, :, :], dtype=np.float32).sum(axis=0)
        return np.clip(vol, 0, None)

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                ix = int(round(x / q_step))
                iy = int(round(y / height_pixel_size))
                if 0 <= ix < self.nq and 0 <= iy < self.nz:
                    val = self.dset[ix, iy, self.current_row, self.current_col]
                    self.status_bar.showMessage(f"Q: {x:.4f}, Z: {y:.2f} mm, Intensity: {val:.4f}")
                else: self.status_bar.showMessage("Outside bounds")

    def on_mouse_click(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if x is not None:
                ix = int(round(x / q_step))
                if 0 <= ix < self.nq: self.q_input.setValue(ix)
            if y is not None:
                iy = int(round(y / height_pixel_size))
                if 0 <= iy < self.nz:
                    self.current_clicked_z = iy
                    self.update_middle_right_plot()
                    self.update_slice()
                    self.update_middle_left_plot()

    def closeEvent(self, event):
        self.h5f.close()
        self.plotter.close()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(file_path)
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec())
