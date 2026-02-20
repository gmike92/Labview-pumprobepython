"""
K-Space Hyperspectral Window — scans NIREOS Gemini stage while acquiring
via LabVIEW Experiment_manager.vi, keeping EVERY pixel in the ROI.

Data flow:
  1. Scan Gemini stage across N positions
  2. At each position: acquire frame, extract ROI → 2D slice (h, w)
  3. Build datacube: shape (N, h, w)
  4. Per-pixel DFT → spectrum_cube (n_freq, h, w)
  5. Display:
     - Wavelength slider → 2D spatial map at selected λ
     - Click pixel → 1D spectrum at that spatial position

Usage:
    from sub_kspace_lw import KSpaceWindow
    window = KSpaceWindow(labview_manager, twins_stage, live_window=live_win)
    window.show()
"""

import os
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    raise ImportError("pyqtgraph required: pip install pyqtgraph pyqt6")

from labview_manager import LabVIEWManager, CMD_IDLE, CMD_MEASURE


# ============================================================================
# Default Parameters
# ============================================================================

DEFAULT_START_MM = 15.0
DEFAULT_STOP_MM  = 23.0
DEFAULT_N_STEPS  = 100
DEFAULT_APODIZATION = 0.2
DEFAULT_WL_START = 8.0       # µm
DEFAULT_WL_STOP  = 14.0      # µm

# Calibration file path
DEFAULT_CALIBRATION_FILE = r".\Twins\ASRC calibration\parameters_cal.txt"


# ============================================================================
# Per-Pixel Spectrum Processor
# ============================================================================

class HyperspectralProcessor:
    """
    Process a 3D datacube (n_positions, h, w) into a spectrum cube.
    Each pixel gets its own DFT independently.
    """

    def __init__(self, calibration_file=None):
        self.calibration_file = calibration_file or DEFAULT_CALIBRATION_FILE
        self.wavelength_cal = None
        self.reciprocal_cal = None
        self._load_calibration()

    def _load_calibration(self):
        try:
            import pandas as pd
            cal_path = Path(self.calibration_file)
            if not cal_path.exists():
                alt_paths = [
                    Path(r"C:\Users\mguizzardi\Desktop\Camera python\TWINS FILE\Twins\ASRC calibration\parameters_cal.txt"),
                    Path(r".\Twins\ASRC calibration\parameters_cal.txt"),
                ]
                for alt in alt_paths:
                    if alt.exists():
                        cal_path = alt
                        break
            if cal_path.exists():
                ref = pd.read_csv(cal_path, sep="\t", header=None)
                self.wavelength_cal = ref.iloc[0].to_numpy(dtype='float64')
                self.reciprocal_cal = ref.iloc[1].to_numpy(dtype='float64')
                print(f"[OK] KSpace: Loaded calibration: {cal_path.name}")
        except Exception as e:
            print(f"[WARN] KSpace calibration: {e}")

    def _get_frequency_limits(self, wl_start, wl_stop):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            return float(fn(1.0 / wl_stop)), float(fn(1.0 / wl_start))
        return 1.0 / wl_stop, 1.0 / wl_start

    def _freq_to_wavelength(self, frequencies):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(self.reciprocal_cal, 1.0 / self.wavelength_cal,
                          kind="linear", fill_value="extrapolate")
            return 1.0 / fn(frequencies)
        return 1.0 / frequencies

    def compute_hyperspectral(self, positions, datacube,
                               wl_start=8.0, wl_stop=14.0,
                               apod_width=0.2, n_freq=200):
        """
        Compute per-pixel DFT on a (n_pos, h, w) datacube.

        Returns:
            wavelengths : 1D array (n_freq,)
            spectrum_cube : 3D array (n_freq, h, w)
        """
        n_pos, h, w = datacube.shape
        if n_pos < 3:
            return None, None

        # Baseline removal: subtract per-pixel moving average
        window = max(1, n_pos // 5)
        kernel = np.ones(window) / window
        baseline = np.zeros_like(datacube)
        for r in range(h):
            for c in range(w):
                baseline[:, r, c] = np.convolve(
                    datacube[:, r, c], kernel, mode='same'
                )
        signal = datacube - baseline

        # Apodization (Gaussian window along position axis, same for all pixels)
        center_idx = n_pos // 2
        sigma = abs(positions[-1] - positions[0]) * apod_width
        if sigma > 0:
            apod_window = np.exp(-(positions - positions[center_idx])**2
                                  / (2.0 * sigma**2))
        else:
            apod_window = np.ones(n_pos)
        # Apply: broadcast (n_pos,) over (n_pos, h, w)
        signal = signal * apod_window[:, np.newaxis, np.newaxis]

        # Frequency grid
        start_freq, end_freq = self._get_frequency_limits(wl_start, wl_stop)
        frequencies = np.linspace(start_freq, end_freq, n_freq)
        wavelengths = self._freq_to_wavelength(frequencies)

        # DFT: for each frequency, sum over positions
        # phase: (n_pos, n_freq)
        pos = positions.reshape(-1, 1)
        dpos = np.diff(positions)
        dpos = np.append(dpos, dpos[-1] if len(dpos) > 0 else 0)

        phase = np.exp(-2j * np.pi * pos * frequencies)  # (n_pos, n_freq)
        weighted = signal * dpos[:, np.newaxis, np.newaxis]  # (n_pos, h, w)

        # Reshape for matrix multiply: (n_pos, h*w) @ phase would be wrong dim
        # We want: for each freq f, spectrum[f, r, c] = sum_i weighted[i, r, c] * phase[i, f]
        # Reshape weighted to (n_pos, h*w), then spectrum = phase.T @ weighted → (n_freq, h*w)
        flat = weighted.reshape(n_pos, -1)     # (n_pos, h*w)
        spec_flat = phase.conj().T @ flat      # (n_freq, h*w)
        spectrum_cube = np.abs(spec_flat).reshape(n_freq, h, w)

        return wavelengths, spectrum_cube


# ============================================================================
# K-Space Hyperspectral Window
# ============================================================================

class KSpaceWindow(QtWidgets.QWidget):
    """
    K-Space Hyperspectral imaging:
    scan Gemini stage, store every ROI pixel, per-pixel DFT → spectral cube.

    Layout:
      Left panel  – scan controls, processing, save/load
      Right panel – 2D spatial map at selected λ (top) + pixel spectrum (bottom)
    """

    def __init__(self, manager: LabVIEWManager, twins_stage,
                 live_window=None, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.stage = twins_stage
        self.live_window = live_window
        self.processor = HyperspectralProcessor()

        # Scan state
        self.scanning = False
        self.scan_positions = None
        self.datacube = None        # (n_pos, h, w)
        self.spectrum_cube = None   # (n_freq, h, w)
        self.wavelengths = None     # (n_freq,)
        self.scan_index = 0
        self.roi_shape = None       # (h, w) of extracted ROI

        # Selected pixel for spectrum display
        self.sel_row = 0
        self.sel_col = 0

        # Position update timer
        self.pos_timer = QtCore.QTimer()
        self.pos_timer.timeout.connect(self._update_position_display)
        self.pos_timer.start(500)

        self.setWindowTitle("K-Space Hyperspectral (LabVIEW)")
        self.resize(1200, 800)
        self._setup_ui()

    # =========================================================================
    #  UI
    # =========================================================================

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # ====== Left Panel: Controls ======
        ctrl_panel = QtWidgets.QWidget()
        ctrl_panel.setMaximumWidth(340)
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_panel)

        # Title
        title = QtWidgets.QLabel("K-Space Hyperspectral")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #00BCD4; padding: 4px;"
        )
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ctrl_layout.addWidget(title)

        # ====== Stage Position ======
        stage_group = QtWidgets.QGroupBox("Gemini Stage")
        stage_gl = QtWidgets.QGridLayout(stage_group)

        stage_gl.addWidget(QtWidgets.QLabel("Position:"), 0, 0)
        self.lbl_pos = QtWidgets.QLabel("-- mm")
        self.lbl_pos.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #2196F3;"
        )
        stage_gl.addWidget(self.lbl_pos, 0, 1)

        stage_gl.addWidget(QtWidgets.QLabel("Move To (mm):"), 1, 0)
        self.spin_move = QtWidgets.QDoubleSpinBox()
        self.spin_move.setRange(0.0, 50.0)
        self.spin_move.setValue(19.0)
        self.spin_move.setDecimals(2)
        stage_gl.addWidget(self.spin_move, 1, 1)

        self.btn_move = QtWidgets.QPushButton("Move Stage")
        self.btn_move.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 5px;"
        )
        self.btn_move.clicked.connect(self._move_stage)
        stage_gl.addWidget(self.btn_move, 2, 0, 1, 2)

        ctrl_layout.addWidget(stage_group)

        # ====== Scan Range ======
        range_group = QtWidgets.QGroupBox("Scan Range")
        range_gl = QtWidgets.QGridLayout(range_group)

        range_gl.addWidget(QtWidgets.QLabel("Start (mm):"), 0, 0)
        self.spin_start = QtWidgets.QDoubleSpinBox()
        self.spin_start.setRange(0.0, 50.0)
        self.spin_start.setValue(DEFAULT_START_MM)
        self.spin_start.setDecimals(2)
        self.spin_start.valueChanged.connect(self._update_info)
        range_gl.addWidget(self.spin_start, 0, 1)

        range_gl.addWidget(QtWidgets.QLabel("Stop (mm):"), 1, 0)
        self.spin_stop = QtWidgets.QDoubleSpinBox()
        self.spin_stop.setRange(0.0, 50.0)
        self.spin_stop.setValue(DEFAULT_STOP_MM)
        self.spin_stop.setDecimals(2)
        self.spin_stop.valueChanged.connect(self._update_info)
        range_gl.addWidget(self.spin_stop, 1, 1)

        range_gl.addWidget(QtWidgets.QLabel("Steps:"), 2, 0)
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(2, 5000)
        self.spin_steps.setValue(DEFAULT_N_STEPS)
        self.spin_steps.valueChanged.connect(self._update_info)
        range_gl.addWidget(self.spin_steps, 2, 1)

        range_gl.addWidget(QtWidgets.QLabel("Step Size:"), 3, 0)
        self.lbl_step = QtWidgets.QLabel("-- um")
        self.lbl_step.setStyleSheet("font-weight: bold;")
        range_gl.addWidget(self.lbl_step, 3, 1)

        ctrl_layout.addWidget(range_group)

        # ====== Acquisition ======
        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_gl = QtWidgets.QGridLayout(acq_group)

        acq_gl.addWidget(QtWidgets.QLabel("Frames/Point:"), 0, 0)
        self.spin_frames = QtWidgets.QSpinBox()
        self.spin_frames.setRange(2, 10000)
        self.spin_frames.setValue(100)
        self.spin_frames.setValue(100)
        acq_gl.addWidget(self.spin_frames, 0, 1)

        # Save Mode
        acq_gl.addWidget(QtWidgets.QLabel("Save Mode:"), 1, 0)
        self.cmb_save_mode = QtWidgets.QComboBox()
        self.cmb_save_mode.addItems(["Single Pixel", "ROI Average", "ROI (2D)", "Full Frame (2D)"])
        self.cmb_save_mode.setCurrentIndex(2) # Default ROI (2D) for K-Space
        acq_gl.addWidget(self.cmb_save_mode, 1, 1)

        # Plot Mode
        acq_gl.addWidget(QtWidgets.QLabel("Plot Mode:"), 2, 0)
        self.cmb_plot_mode = QtWidgets.QComboBox()
        self.cmb_plot_mode.addItems(["DeltaT (dT/T)", "Transmission (T)", "DeltaT (dT)"])
        self.cmb_plot_mode.setCurrentIndex(0) # Default dT/T
        acq_gl.addWidget(self.cmb_plot_mode, 2, 1)

        self.btn_bg = QtWidgets.QPushButton("Acquire Background")
        self.btn_bg.setStyleSheet("background-color: #607D8B; color: white;")
        self.btn_bg.clicked.connect(self._acquire_background)
        acq_gl.addWidget(self.btn_bg, 3, 0, 1, 2)
        
        self.lbl_roi_info = QtWidgets.QLabel("ROI: (use Live View)")
        self.lbl_roi_info.setStyleSheet("color: #888; font-style: italic;")
        acq_gl.addWidget(self.lbl_roi_info, 4, 0, 1, 2)

        # Data Saving
        save_group = QtWidgets.QGroupBox("Data to Save")
        save_layout = QtWidgets.QVBoxLayout(save_group)
        self.chk_save_t = QtWidgets.QCheckBox("Transmission (T)")
        self.chk_save_dt = QtWidgets.QCheckBox("DeltaT (dT)")
        self.chk_save_dtt = QtWidgets.QCheckBox("DeltaT/T")
        self.chk_save_raw = QtWidgets.QCheckBox("Raw (Odd/Even)")
        # Defaults
        self.chk_save_dtt.setChecked(True)
        
        save_layout.addWidget(self.chk_save_t)
        save_layout.addWidget(self.chk_save_dt)
        save_layout.addWidget(self.chk_save_dtt)
        save_layout.addWidget(self.chk_save_raw)
        
        acq_gl.addWidget(save_group, 6, 0, 1, 2)

        # Sample Name
        acq_gl.addWidget(QtWidgets.QLabel("Sample Name:"), 5, 0)
        self.txt_sample_name = QtWidgets.QLineEdit()
        self.txt_sample_name.setPlaceholderText("Enter sample name...")
        acq_gl.addWidget(self.txt_sample_name, 5, 1)

        ctrl_layout.addWidget(acq_group)

        # Start / Stop
        btn_group = QtWidgets.QGroupBox("Control")
        btn_layout = QtWidgets.QVBoxLayout(btn_group)

        self.btn_start = QtWidgets.QPushButton("START SCAN")
        self.btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px; border-radius: 6px;"
        )
        self.btn_start.clicked.connect(self._start_scan)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; "
            "padding: 10px; border-radius: 6px;"
        )
        self.btn_stop.clicked.connect(self._stop_scan)
        btn_layout.addWidget(self.btn_stop)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        btn_layout.addWidget(self.progress)

        ctrl_layout.addWidget(btn_group)

        # Processing
        proc_group = QtWidgets.QGroupBox("Spectrum Processing")
        proc_gl = QtWidgets.QGridLayout(proc_group)

        proc_gl.addWidget(QtWidgets.QLabel("Apodization:"), 0, 0)
        self.spin_apod = QtWidgets.QDoubleSpinBox()
        self.spin_apod.setRange(0.01, 1.0)
        self.spin_apod.setValue(DEFAULT_APODIZATION)
        self.spin_apod.setDecimals(2)
        proc_gl.addWidget(self.spin_apod, 0, 1)

        proc_gl.addWidget(QtWidgets.QLabel("WL Start (um):"), 1, 0)
        self.spin_wl_start = QtWidgets.QDoubleSpinBox()
        self.spin_wl_start.setRange(1.0, 30.0)
        self.spin_wl_start.setValue(DEFAULT_WL_START)
        self.spin_wl_start.setDecimals(1)
        proc_gl.addWidget(self.spin_wl_start, 1, 1)

        proc_gl.addWidget(QtWidgets.QLabel("WL Stop (um):"), 2, 0)
        self.spin_wl_stop = QtWidgets.QDoubleSpinBox()
        self.spin_wl_stop.setRange(1.0, 30.0)
        self.spin_wl_stop.setValue(DEFAULT_WL_STOP)
        self.spin_wl_stop.setDecimals(1)
        proc_gl.addWidget(self.spin_wl_stop, 2, 1)

        proc_gl.addWidget(QtWidgets.QLabel("Freq Points:"), 3, 0)
        self.spin_nfreq = QtWidgets.QSpinBox()
        self.spin_nfreq.setRange(50, 2000)
        self.spin_nfreq.setValue(200)
        proc_gl.addWidget(self.spin_nfreq, 3, 1)

        self.btn_compute = QtWidgets.QPushButton("Compute Hyperspectral")
        self.btn_compute.setStyleSheet(
            "background-color: #9C27B0; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 5px;"
        )
        self.btn_compute.clicked.connect(self._compute)
        proc_gl.addWidget(self.btn_compute, 4, 0, 1, 2)

        ctrl_layout.addWidget(proc_group)

        # Save / Load
        io_group = QtWidgets.QGroupBox("Data")
        io_layout = QtWidgets.QHBoxLayout(io_group)

        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.clicked.connect(self._save_data)
        io_layout.addWidget(self.btn_save)

        self.btn_load = QtWidgets.QPushButton("Load")
        self.btn_load.clicked.connect(self._load_data)
        io_layout.addWidget(self.btn_load)

        ctrl_layout.addWidget(io_group)

        # Status
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        ctrl_layout.addWidget(self.lbl_status)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl_panel)

        # ====== Right Panel: Display ======
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # Wavelength slider
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel("Wavelength:"))

        self.wl_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.wl_slider.setRange(0, 199)
        self.wl_slider.setValue(100)
        self.wl_slider.valueChanged.connect(self._on_slider_change)
        slider_row.addWidget(self.wl_slider, stretch=1)

        self.lbl_wl = QtWidgets.QLabel("-- um")
        self.lbl_wl.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #00BCD4; min-width: 80px;"
        )
        slider_row.addWidget(self.lbl_wl)

        right_layout.addLayout(slider_row)

        # Vertical splitter: spatial map (top) + pixel spectrum (bottom)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # -- Spatial map at selected λ --
        map_widget = pg.GraphicsLayoutWidget()
        self.map_plot = map_widget.addPlot(title="Spatial Map at Selected Wavelength")
        self.map_img = pg.ImageItem()
        self.map_plot.addItem(self.map_img)

        self.map_hist = pg.HistogramLUTItem()
        self.map_hist.setImageItem(self.map_img)
        map_widget.addItem(self.map_hist)

        # Crosshair for selected pixel
        self.map_hline = pg.InfiniteLine(angle=0, pen=pg.mkPen('y', width=1,
                                         style=QtCore.Qt.PenStyle.DashLine))
        self.map_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=1,
                                         style=QtCore.Qt.PenStyle.DashLine))
        self.map_hline.setVisible(False)
        self.map_vline.setVisible(False)
        self.map_plot.addItem(self.map_hline)
        self.map_plot.addItem(self.map_vline)

        # Click to select pixel
        self.map_img.scene().sigMouseClicked.connect(self._on_map_click)

        # Colormap: viridis-like
        try:
            lut = pg.colormap.get('viridis').getLookupTable()
            self.map_img.setLookupTable(lut)
        except Exception:
            pass

        splitter.addWidget(map_widget)

        # -- Pixel spectrum --
        self.spec_plot = pg.PlotWidget(
            title="Spectrum at Selected Pixel",
            labels={'left': 'Intensity', 'bottom': 'Wavelength (um)'}
        )
        self.spec_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spec_curve = self.spec_plot.plot(
            pen=pg.mkPen('#00BCD4', width=2)
        )
        self.lbl_pixel = QtWidgets.QLabel("Pixel: click map to select")
        self.lbl_pixel.setStyleSheet("color: #888; font-style: italic; padding: 2px;")

        spec_container = QtWidgets.QWidget()
        spec_vl = QtWidgets.QVBoxLayout(spec_container)
        spec_vl.setContentsMargins(0, 0, 0, 0)
        spec_vl.addWidget(self.lbl_pixel)
        spec_vl.addWidget(self.spec_plot)

        splitter.addWidget(spec_container)
        splitter.setSizes([500, 300])

        right_layout.addWidget(splitter, stretch=1)

        layout.addWidget(right_panel, stretch=1)

        # Init
        self._update_info()

    # =========================================================================
    #  Helpers
    # =========================================================================

    def _update_position_display(self):
        if self.stage and self.stage.is_connected:
            try:
                pos = self.stage.get_position()
                self.lbl_pos.setText(f"{pos:.3f} mm")
            except Exception:
                self.lbl_pos.setText("err")
        else:
            self.lbl_pos.setText("-- mm")

    def _update_info(self):
        start = self.spin_start.value()
        stop = self.spin_stop.value()
        n = self.spin_steps.value()
        if n > 1:
            step_um = abs(stop - start) / (n - 1) * 1000
            self.lbl_step.setText(f"{step_um:.1f} um")
        else:
            self.lbl_step.setText("-- um")

        # ROI info
        if self.live_window is not None:
            bounds = self.live_window.get_roi_bounds()
            if bounds:
                r0, r1, c0, c1 = bounds
                h, w = r1 - r0, c1 - c0
                self.lbl_roi_info.setText(f"ROI: {h}x{w} px ({h*w} pixels)")
                self.lbl_roi_info.setStyleSheet("color: #4CAF50; font-weight: bold;")

    def _move_stage(self):
        if not self.stage or not self.stage.is_connected:
            self.lbl_status.setText("Stage not connected!")
            return
        target = self.spin_move.value()
        self.lbl_status.setText(f"Moving to {target:.2f} mm...")
        self.btn_move.setEnabled(False)
        import threading
        def do_move():
            self.stage.move_to(target)
            self.stage.wait_for_stop()
        threading.Thread(target=do_move, daemon=True).start()
        QtCore.QTimer.singleShot(500, lambda: self.btn_move.setEnabled(True))
        QtCore.QTimer.singleShot(500, lambda: self.lbl_status.setText("Ready"))

    def _extract_to_save(self, img):
        """Extract data entity to SAVE in datacube based on mode. Always returns 2D array."""
        mode = self.cmb_save_mode.currentIndex()
        # 0=Single, 1=ROI Avg, 2=ROI, 3=Full
        
        if mode == 3: # Full Frame
            if img.ndim == 2: return img.copy()
            # Handle 1D flat?
            return img.copy() # Should be 2D

        if self.live_window:
            # Single Pixel
            if mode == 0:
                r, c = self.live_window.sel_row, self.live_window.sel_col
                if r is not None and c is not None:
                    h, w = img.shape
                    if 0 <= r < h and 0 <= c < w:
                        val = img[r, c]
                        return np.array([[val]])
                return np.array([[np.nanmean(img)]])
            
            # ROI modes
            if mode in [1, 2]:
                bounds = self.live_window.get_roi_bounds()
                if bounds:
                    r0, r1, c0, c1 = bounds
                    h, w = img.shape
                    r0, r1 = max(0, min(r0, h)), max(1, min(r1, h))
                    c0, c1 = max(0, min(c0, w)), max(1, min(c1, w))
                    slice_img = img[r0:r1, c0:c1]
                    
                    if mode == 1: # ROI Avg
                        val = np.mean(slice_img)
                        return np.array([[val]])
                    else: # ROI 2D
                        return slice_img.copy()
        
        # Fallback
        return img.copy()

    # =========================================================================
    #  Scan Logic (non-blocking, timer-based)
    # =========================================================================

    def _start_scan(self):
        if not self.stage or not self.stage.is_connected:
            self.lbl_status.setText("Gemini stage not connected!")
            return
        if not self.manager.vi:
            self.lbl_status.setText("LabVIEW not running!")
            return

        # Refresh ROI info
        self._update_info()

        start = self.spin_start.value()
        stop = self.spin_stop.value()
        n_steps = self.spin_steps.value()
        self.scan_positions = np.linspace(start, stop, n_steps)
        self.scan_index = 0
        self.datacube = None
        self.spectrum_cube = None
        self.roi_shape = None
        
        # Selective Cubes
        self.cube_t = None
        self.cube_dt = None
        self.cube_dtt = None
        self.cube_raw_odd = None
        self.cube_raw_even = None
        
        # Generate Standardized Paths
        timestamp = datetime.now()
        date_dir = timestamp.strftime(r"D:\pumpprobedata\%Y\%m\%d")
        os.makedirs(date_dir, exist_ok=True)
        
        sample = self.txt_sample_name.text().strip()
        if not sample: sample = "sample"
        # Sanitize
        sample = "".join(x for x in sample if x.isalnum() or x in " -_")
        
        # We don't save CSV for K-Space (too big?), just final NPZ
        # But we need a path for final save
        self.scan_npz_path = os.path.join(date_dir, f"{sample}_kspace_{timestamp.strftime('%H%M%S')}.npz")
        
        self.scanning = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.lbl_status.setText("Scanning...")

        print(f"[KSPACE] Starting scan: {start:.2f} -> {stop:.2f} mm, {n_steps} steps")
        print(f"[KSPACE] Starting scan: {start:.2f} -> {stop:.2f} mm, {n_steps} steps")
        QtCore.QTimer.singleShot(50, self._move_to_next)

    def _acquire_background(self):
        """Trigger a single acquisition for background."""
        if self.scanning:
            return
        
        self.lbl_status.setText("Status: Acquiring Global Background...")
        self._awaiting_background = True
        
        # Reuse trigger logic (manually)
        self.manager.vi.SetControlValue("N", self.spin_frames.value())
        self.manager.vi.SetControlValue("Acq Trigger", True)
        self.manager.vi.SetControlValue("Enum", CMD_MEASURE)
        
        self._acq_timer = QtCore.QTimer()
        self._acq_timer.timeout.connect(self._poll_acquire)
        self._acq_waited = 0.0
        self._acq_timer.start(50)

    def _stop_scan(self):
        self.scanning = False
        self.lbl_status.setText("Stopped")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        print("[KSPACE] Scan stopped by user")

    def _move_to_next(self):
        if not self.scanning:
            return
        if self.scan_index >= len(self.scan_positions):
            self._scan_complete()
            return

        target = self.scan_positions[self.scan_index]
        self.lbl_status.setText(
            f"Point {self.scan_index+1}/{len(self.scan_positions)}: "
            f"Moving to {target:.3f} mm"
        )

        try:
            self.stage.move_to(target)
        except Exception as e:
            self.lbl_status.setText(f"Move error: {e}")
            self.scan_index += 1
            QtCore.QTimer.singleShot(50, self._move_to_next)
            return

        self._stable_count = 0
        self._last_pos = None
        self._stage_timer = QtCore.QTimer()
        self._stage_timer.timeout.connect(self._poll_stage)
        self._stage_waited = 0.0
        self._stage_timer.start(50)

    def _poll_stage(self):
        if not self.scanning:
            self._stage_timer.stop()
            return

        self._stage_waited += 0.05
        if self._stage_waited > 30.0:
            self._stage_timer.stop()
            self.lbl_status.setText("Stage timeout!")
            self.scan_index += 1
            QtCore.QTimer.singleShot(50, self._move_to_next)
            return

        try:
            pos = self.stage.get_position()
        except Exception:
            return

        if self._last_pos is not None and abs(pos - self._last_pos) < 0.001:
            self._stable_count += 1
        else:
            self._stable_count = 0
        self._last_pos = pos

        if self._stable_count >= 3:
            self._stage_timer.stop()
            QtCore.QTimer.singleShot(50, self._trigger_acquire)

    def _trigger_acquire(self):
        if not self.scanning:
            return

        vi = self.manager.vi
        n = self.spin_frames.value()

        try:
            vi.SetControlValue("N", n)
            vi.SetControlValue("Acq Trigger", True)
            vi.SetControlValue("Enum", CMD_MEASURE)

            self._acq_timer = QtCore.QTimer()
            self._acq_timer.timeout.connect(self._poll_acquire)
            self._acq_waited = 0.0
            self._acq_timer.start(50)

        except Exception as e:
            self.lbl_status.setText(f"Acquire error: {e}")
            self.scan_index += 1
            QtCore.QTimer.singleShot(100, self._move_to_next)

    def _poll_acquire(self):
        if not self.scanning:
            self._acq_timer.stop()
            return

        self._acq_waited += 0.05
        if self._acq_waited > 60.0:
            self._acq_timer.stop()
            self.lbl_status.setText("Acquire timeout!")
            self.scan_index += 1
            QtCore.QTimer.singleShot(100, self._move_to_next)
            return

        try:
            if self.manager.vi.GetControlValue("Enum") != CMD_IDLE:
                return
        except Exception:
            return

        # Done — read result
        self._acq_timer.stop()

        try:
            odd_data = self.manager.vi.GetControlValue("Odd")
            even_data = self.manager.vi.GetControlValue("Even")
            if odd_data is not None and even_data is not None:
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                
                # Handle Background
                # Handle Background
                if hasattr(self, '_awaiting_background') and self._awaiting_background:
                    # Store BOTH Odd and Even for Scattering Correction
                    self.manager.background = (odd.copy(), even.copy())
                    self._awaiting_background = False
                    self.lbl_status.setText("Global Background Acquired (Scattering Mode)")
                    QtWidgets.QMessageBox.information(self, "Background", "Global Background acquired! (Odd/Even stored separately)")
                    return

                # Apply Background
                # Apply Background
                bg = self.manager.background
                if bg is not None:
                    # New Mode: Tuple
                    if isinstance(bg, (tuple, list)) and len(bg) == 2:
                        bg_odd, bg_even = bg
                        if bg_odd.shape == odd.shape and bg_even.shape == even.shape:
                            odd -= bg_odd
                            even -= bg_even
                    # Legacy
                    elif hasattr(bg, 'shape') and bg.shape == odd.shape:
                        odd -= bg
                        even -= bg
                
                # Compute All
                img_t = (odd + even) / 2.0
                img_dt = even - odd
                img_dtt = np.divide(even - odd, odd, out=np.zeros_like(odd), where=np.abs(odd) > 1.0)

                # Compute based on Plot Mode (for Display / Main Datacube)
                pmode = self.cmb_plot_mode.currentIndex()
                if pmode == 1:   img = img_t
                elif pmode == 2: img = img_dt
                else:            img = img_dtt
                
                if img.size == 0:
                    self.scan_index += 1
                    QtCore.QTimer.singleShot(50, self._move_to_next)
                    return

                # --- Helper to extract ROI slice ---
                def get_roi_slice(image):
                    s = self._extract_to_save(image)
                    if s.ndim == 0: s = np.array([[s]])
                    if s.ndim == 1: s = s.reshape(1, -1)
                    return s

                roi_slice = get_roi_slice(img)
                h, w = roi_slice.shape
                n_pos = len(self.scan_positions)

                # --- Allocate Cubes (First Frame) ---
                if self.datacube is None:
                    self.datacube = np.zeros((n_pos, h, w), dtype=np.float64)
                    self.roi_shape = (h, w)
                    self.lbl_roi_info.setText(f"ROI: {h}x{w} px ({h*w} pixels)")
                    print(f"[KSPACE] Main Datacube allocated: ({n_pos}, {h}, {w})")
                
                if self.chk_save_t.isChecked() and self.cube_t is None:
                    self.cube_t = np.zeros((n_pos, h, w), dtype=np.float64)
                if self.chk_save_dt.isChecked() and self.cube_dt is None:
                    self.cube_dt = np.zeros((n_pos, h, w), dtype=np.float64)
                if self.chk_save_dtt.isChecked() and self.cube_dtt is None:
                    self.cube_dtt = np.zeros((n_pos, h, w), dtype=np.float64)
                if self.chk_save_raw.isChecked() and self.cube_raw_odd is None:
                    self.cube_raw_odd = np.zeros((n_pos, h, w), dtype=np.float64)
                    self.cube_raw_even = np.zeros((n_pos, h, w), dtype=np.float64)

                # --- Store Data ---
                # Determine slice indices (handle resize)
                dh, dw = self.roi_shape
                r_end, c_end = min(h, dh), min(w, dw)
                
                self.datacube[self.scan_index, :r_end, :c_end] = roi_slice[:r_end, :c_end]

                if self.cube_t is not None:
                    sl = get_roi_slice(img_t)
                    self.cube_t[self.scan_index, :r_end, :c_end] = sl[:r_end, :c_end]
                if self.cube_dt is not None:
                    sl = get_roi_slice(img_dt)
                    self.cube_dt[self.scan_index, :r_end, :c_end] = sl[:r_end, :c_end]
                if self.cube_dtt is not None:
                    sl = get_roi_slice(img_dtt)
                    self.cube_dtt[self.scan_index, :r_end, :c_end] = sl[:r_end, :c_end]
                if self.cube_raw_odd is not None:
                    sl_odd = get_roi_slice(odd)
                    sl_even = get_roi_slice(even)
                    self.cube_raw_odd[self.scan_index, :r_end, :c_end] = sl_odd[:r_end, :c_end]
                    self.cube_raw_even[self.scan_index, :r_end, :c_end] = sl_even[:r_end, :c_end]

                print(f"[KSPACE] Point {self.scan_index+1}: "
                      f"{self.scan_positions[self.scan_index]:.3f} mm, "
                      f"ROI mean={np.mean(roi_slice):.4e}")

        except Exception as e:
            print(f"[KSPACE] Read error: {e}")

        self.scan_index += 1
        pct = int(100 * self.scan_index / len(self.scan_positions))
        self.progress.setValue(pct)
        QtCore.QTimer.singleShot(50, self._move_to_next)

    def _scan_complete(self):
        self.scanning = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(100)

        if self.datacube is not None:
            n_pos, h, w = self.datacube.shape
            self.lbl_status.setText(
                f"Scan complete! Datacube: ({n_pos}, {h}, {w}). "
                f"Click 'Compute' for FFT."
            )
            print(f"[KSPACE] Scan complete: datacube {self.datacube.shape}")

            # Auto-save raw datacube
            # Auto-save raw datacube
            if not hasattr(self, 'scan_npz_path'):
                # Fallback
                self.scan_npz_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    f"kspace_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
                )
            
            os.makedirs(os.path.dirname(self.scan_npz_path), exist_ok=True)
            np.savez(self.scan_npz_path,
                     positions=self.scan_positions,
                     datacube=self.datacube,
                     # Selective
                     data_t=self.cube_t if self.cube_t is not None else np.array([]),
                     data_dt=self.cube_dt if self.cube_dt is not None else np.array([]),
                     data_dtt=self.cube_dtt if self.cube_dtt is not None else np.array([]),
                     raw_odd=self.cube_raw_odd if self.cube_raw_odd is not None else np.array([]),
                     raw_even=self.cube_raw_even if self.cube_raw_even is not None else np.array([]))
            print(f"[KSPACE] Auto-saved raw: {self.scan_npz_path}")

            # Auto-compute
            self._compute()
        else:
            self.lbl_status.setText("Scan complete (no data)")

    # =========================================================================
    #  Hyperspectral Processing
    # =========================================================================

    def _compute(self):
        if self.datacube is None or self.scan_positions is None:
            self.lbl_status.setText("No datacube to process!")
            return

        self.lbl_status.setText("Computing per-pixel FFT...")
        QtCore.QTimer.singleShot(50, self._do_compute)

    def _do_compute(self):
        wavelengths, spectrum_cube = self.processor.compute_hyperspectral(
            self.scan_positions, self.datacube,
            wl_start=self.spin_wl_start.value(),
            wl_stop=self.spin_wl_stop.value(),
            apod_width=self.spin_apod.value(),
            n_freq=self.spin_nfreq.value()
        )

        if wavelengths is not None and spectrum_cube is not None:
            self.wavelengths = wavelengths
            self.spectrum_cube = spectrum_cube
            n_freq = len(wavelengths)

            # Update slider range
            self.wl_slider.setRange(0, n_freq - 1)
            self.wl_slider.setValue(n_freq // 2)

            self._update_map()
            self._update_spectrum()

            self.lbl_status.setText(
                f"Hyperspectral computed! "
                f"{spectrum_cube.shape[0]} freqs x "
                f"{spectrum_cube.shape[1]}x{spectrum_cube.shape[2]} pixels"
            )
            print(f"[KSPACE] Spectrum cube: {spectrum_cube.shape}")
        else:
            self.lbl_status.setText("Computation failed!")

    # =========================================================================
    #  Display Updates
    # =========================================================================

    def _on_slider_change(self, idx):
        self._update_map()
        self._update_spectrum()

    def _update_map(self):
        """Update 2D spatial map at the selected wavelength."""
        if self.spectrum_cube is None or self.wavelengths is None:
            return

        idx = self.wl_slider.value()
        if idx >= len(self.wavelengths):
            idx = len(self.wavelengths) - 1

        wl = self.wavelengths[idx]
        self.lbl_wl.setText(f"{wl:.2f} um")

        spatial_map = self.spectrum_cube[idx]
        self.map_img.setImage(spatial_map.T, autoLevels=True)
        self.map_plot.setTitle(f"Spatial Map at {wl:.2f} um")

    def _update_spectrum(self):
        """Update 1D spectrum plot at the selected pixel."""
        if self.spectrum_cube is None or self.wavelengths is None:
            return

        r, c = self.sel_row, self.sel_col
        h, w = self.spectrum_cube.shape[1], self.spectrum_cube.shape[2]

        if 0 <= r < h and 0 <= c < w:
            spectrum = self.spectrum_cube[:, r, c]
            self.spec_curve.setData(self.wavelengths, spectrum)
            self.spec_plot.setTitle(f"Spectrum at Pixel ({r}, {c})")

    def _on_map_click(self, event):
        """Click on spatial map → select pixel, show spectrum."""
        if self.spectrum_cube is None:
            return

        pos = event.scenePos()
        mouse_point = self.map_plot.vb.mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())

        h, w = self.spectrum_cube.shape[1], self.spectrum_cube.shape[2]
        if 0 <= row < h and 0 <= col < w:
            self.sel_row = row
            self.sel_col = col

            self.map_hline.setValue(row)
            self.map_vline.setValue(col)
            self.map_hline.setVisible(True)
            self.map_vline.setVisible(True)

            self.lbl_pixel.setText(f"Pixel: ({row}, {col})")
            self.lbl_pixel.setStyleSheet("color: #FFD600; font-weight: bold;")

            self._update_spectrum()

    # =========================================================================
    #  Save / Load
    # =========================================================================

    def _save_data(self):
        if self.datacube is None:
            self.lbl_status.setText("No data to save!")
            return

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Hyperspectral Data", "", "NumPy Files (*.npz);;All (*)"
        )
        if filepath:
            save_dict = {
                'positions': self.scan_positions,
                'datacube': self.datacube,
            }
            if self.wavelengths is not None:
                save_dict['wavelengths'] = self.wavelengths
            if self.spectrum_cube is not None:
                save_dict['spectrum_cube'] = self.spectrum_cube
            np.savez(filepath, **save_dict)
            self.lbl_status.setText(f"Saved: {Path(filepath).name}")

    def _load_data(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Hyperspectral Data", "", "NumPy Files (*.npz);;All (*)"
        )
        if filepath:
            try:
                d = np.load(filepath)
                self.scan_positions = d['positions']
                self.datacube = d['datacube']
                self.roi_shape = self.datacube.shape[1:]

                if 'wavelengths' in d and 'spectrum_cube' in d:
                    self.wavelengths = d['wavelengths']
                    self.spectrum_cube = d['spectrum_cube']
                    n_freq = len(self.wavelengths)
                    self.wl_slider.setRange(0, n_freq - 1)
                    self.wl_slider.setValue(n_freq // 2)
                    self._update_map()
                    self._update_spectrum()

                n_pos, h, w = self.datacube.shape
                self.lbl_status.setText(
                    f"Loaded: ({n_pos}, {h}, {w}) datacube"
                )
                self.lbl_roi_info.setText(f"ROI: {h}x{w} px ({h*w} pixels)")

            except Exception as e:
                self.lbl_status.setText(f"Load error: {e}")

    # =========================================================================
    #  Cleanup
    # =========================================================================

    def closeEvent(self, event):
        self.pos_timer.stop()
        if self.scanning:
            self._stop_scan()
        event.accept()
