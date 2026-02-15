"""
Twins Pump-Probe Window — nested hyperspectral pump-probe scan.

Outer loop: Thorlabs delay stage  (time delay in fs)
Inner loop: NIREOS Gemini stage   (interferogram scan in mm)

At each time delay, runs a full Gemini interferogram → DFT → spectrum.
Builds a 2D hyperspectral map: Wavelength × Time Delay.

Layout ported from opus camera sub_twins_pumpprobe.py.

Usage:
    from sub_twins_pumpprobe_lw import TwinsPumpProbeWindow
    window = TwinsPumpProbeWindow(manager, twins_stage, delay_stage)
    window.show()
"""

import os
import time
import csv
import numpy as np
from datetime import datetime

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    raise ImportError("pyqtgraph required: pip install pyqtgraph pyqt6")

from labview_manager import LabVIEWManager, CMD_IDLE, CMD_MEASURE


# ============================================================================
# Constants
# ============================================================================

SPEED_OF_LIGHT_MM_FS = 0.000299792458
GLOBAL_ZERO_POS_MM = 140.0

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_data")

DEFAULT_GEMINI_START = 15.0   # mm
DEFAULT_GEMINI_STOP = 23.0    # mm
DEFAULT_GEMINI_STEPS = 100


# ============================================================================
# Lightweight Spectrum Processor  (FFT-based, for speed during scan)
# ============================================================================

class SpectrumProcessor:
    """Process interferogram → spectrum via FFT with calibration."""

    def __init__(self):
        self.wavelength_cal = None
        self.reciprocal_cal = None
        self._load_calibration()

    def _load_calibration(self):
        try:
            import pandas as pd
            paths = [
                r".\Twins\ASRC calibration\parameters_cal.txt",
                r"C:\Users\mguizzardi\Desktop\Camera python\TWINS FILE\Twins\ASRC calibration\parameters_cal.txt",
            ]
            for path in paths:
                if os.path.exists(path):
                    ref = pd.read_csv(path, sep="\t", header=None)
                    self.wavelength_cal = ref.iloc[0].to_numpy(dtype='float64')
                    self.reciprocal_cal = ref.iloc[1].to_numpy(dtype='float64')
                    print(f"[OK] Loaded calibration from {path}")
                    break
        except Exception as e:
            print(f"[WARN] Calibration load failed: {e}")

    def compute_spectrum(self, positions, interferogram,
                         wl_start=8.0, wl_stop=14.0, n_points=None):
        """Compute spectrum from interferogram. Returns (wavelengths, power)."""
        if len(interferogram) < 10:
            return np.array([]), np.array([])

        # Remove baseline
        baseline = np.convolve(interferogram, np.ones(20)/20, mode='same')
        data = interferogram - baseline

        # Gaussian apodization
        center = len(data) // 2
        window = np.exp(-((np.arange(len(data)) - center) / (len(data) / 4)) ** 2)
        data = data * window

        # FFT
        if n_points and n_points > len(data):
            n_freq = n_points
        else:
            n_freq = len(data) * 4
            
        fft_result = np.fft.rfft(data, n=n_freq)
        power = np.abs(fft_result) ** 2

        dx = np.abs(positions[1] - positions[0]) if len(positions) > 1 else 0.08
        freq = np.fft.rfftfreq(n_freq, d=dx)

        # Convert to wavelength
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(self.reciprocal_cal, self.wavelength_cal,
                          kind='linear', fill_value='extrapolate', bounds_error=False)
            wavelengths = fn(freq)
        else:
            wavelengths = 1.0 / (freq + 1e-10)

        mask = (wavelengths >= wl_start) & (wavelengths <= wl_stop)
        return wavelengths[mask], power[mask]


# ============================================================================
# Twins Pump-Probe Window
# ============================================================================

class TwinsPumpProbeWindow(QtWidgets.QWidget):
    """
    Hyperspectral Pump-Probe Window.

    Layout:
      Left  – 3 stacked plots: last frame, hyperspectral map, current spectrum
      Right – tabs (Time Scan 3-interval + Gemini Scan) + acquisition controls
    """

    def __init__(self, manager: LabVIEWManager, twins_stage, delay_stage,
                 live_window=None, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.stage_gemini = twins_stage
        self.stage_delay = delay_stage
        self.live_window = live_window  # for ROI / pixel signal extraction
        self.processor = SpectrumProcessor()

        # Scan state
        self._scanning = False
        self._time_index = 0
        self._gemini_index = 0

        # Data storage
        self.time_points = np.array([])
        self.gemini_positions = np.array([])
        self.hyperspectral_map = None
        self.wavelengths = None
        self.reference_wavelengths = None
        self.current_interferogram = None

        self.setWindowTitle("Twins Pump-Probe (Hyperspectral)")
        self.resize(1400, 800)
        self._setup_ui()

        # Live preview timer
        self.preview_timer = QtCore.QTimer()
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.start(500)

    # =========================================================================
    #  UI
    # =========================================================================

    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # ====== Left: Plots ======
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Camera preview
        camera_group = QtWidgets.QGroupBox("Last Camera Frame")
        camera_layout = QtWidgets.QVBoxLayout(camera_group)

        self.img_widget = pg.GraphicsLayoutWidget()
        self.img_plot = self.img_widget.addPlot(title="")
        self.img_item = pg.ImageItem()
        self.img_plot.addItem(self.img_item)
        camera_layout.addWidget(self.img_widget)
        left_layout.addWidget(camera_group, stretch=1)

        # Hyperspectral map
        map_group = QtWidgets.QGroupBox("Hyperspectral Map (ΔT/T)")
        map_layout = QtWidgets.QVBoxLayout(map_group)

        self.map_widget = pg.GraphicsLayoutWidget()
        self.map_plot = self.map_widget.addPlot(title="")
        self.map_plot.setLabel('left', 'Wavelength Index')
        self.map_plot.setLabel('bottom', 'Time Step')
        self.map_item = pg.ImageItem()
        self.map_plot.addItem(self.map_item)
        map_layout.addWidget(self.map_widget)
        left_layout.addWidget(map_group, stretch=1)

        # Current spectrum plot
        spectrum_group = QtWidgets.QGroupBox("Current Spectrum")
        spectrum_layout = QtWidgets.QVBoxLayout(spectrum_group)
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Intensity')
        self.spectrum_plot.setLabel('bottom', 'Wavelength (µm)')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_curve = self.spectrum_plot.plot([], [], pen='c')
        self.reference_curve = self.spectrum_plot.plot([], [], pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        spectrum_layout.addWidget(self.spectrum_plot)
        left_layout.addWidget(spectrum_group, stretch=1)

        main_layout.addWidget(left_panel, stretch=2)

        # ====== Right: Controls ======
        right_panel = QtWidgets.QWidget()
        right_panel.setMaximumWidth(380)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Tabs
        tabs = QtWidgets.QTabWidget()

        # --- Tab 1: Time Scan (3 intervals) ---
        time_tab = QtWidgets.QWidget()
        time_layout = QtWidgets.QVBoxLayout(time_tab)

        # Zero position
        zero_group = QtWidgets.QGroupBox("Zero Position (t=0)")
        zero_layout = QtWidgets.QHBoxLayout(zero_group)
        zero_layout.addWidget(QtWidgets.QLabel("Position (mm):"))
        self.spin_zero = QtWidgets.QDoubleSpinBox()
        self.spin_zero.setRange(0, 300)
        self.spin_zero.setDecimals(3)
        self.spin_zero.setValue(GLOBAL_ZERO_POS_MM)
        zero_layout.addWidget(self.spin_zero)
        time_layout.addWidget(zero_group)

        # 3 intervals
        interval_configs = [
            ("Interval 1 (Fine near zero)", -1000, 0, 50),
            ("Interval 2 (Early dynamics)", 0, 10000, 500),
            ("Interval 3 (Late dynamics)", 10000, 100000, 5000),
        ]
        self.interval_spins = []  # [(start, end, step), ...]

        for label, s_def, e_def, st_def in interval_configs:
            grp = QtWidgets.QGroupBox(label)
            gl = QtWidgets.QGridLayout(grp)

            gl.addWidget(QtWidgets.QLabel("Start (fs):"), 0, 0)
            sp_s = QtWidgets.QDoubleSpinBox()
            sp_s.setRange(-1e6, 1e6)
            sp_s.setValue(s_def)
            gl.addWidget(sp_s, 0, 1)

            gl.addWidget(QtWidgets.QLabel("End (fs):"), 0, 2)
            sp_e = QtWidgets.QDoubleSpinBox()
            sp_e.setRange(-1e6, 1e6)
            sp_e.setValue(e_def)
            gl.addWidget(sp_e, 0, 3)

            gl.addWidget(QtWidgets.QLabel("Step (fs):"), 1, 0)
            sp_st = QtWidgets.QDoubleSpinBox()
            sp_st.setRange(1, 1e6)
            sp_st.setValue(st_def)
            gl.addWidget(sp_st, 1, 1)

            self.interval_spins.append((sp_s, sp_e, sp_st))
            time_layout.addWidget(grp)

        self.lbl_time_points = QtWidgets.QLabel("Time points: --")
        self.lbl_time_points.setStyleSheet("font-weight: bold;")
        time_layout.addWidget(self.lbl_time_points)
        tabs.addTab(time_tab, "Time Scan")

        # --- Tab 2: Gemini Scan ---
        gemini_tab = QtWidgets.QWidget()
        gemini_layout = QtWidgets.QGridLayout(gemini_tab)

        gemini_layout.addWidget(QtWidgets.QLabel("Start (mm):"), 0, 0)
        self.spin_gemini_start = QtWidgets.QDoubleSpinBox()
        self.spin_gemini_start.setRange(0, 50)
        self.spin_gemini_start.setDecimals(2)
        self.spin_gemini_start.setSingleStep(0.5)
        self.spin_gemini_start.setValue(DEFAULT_GEMINI_START)
        gemini_layout.addWidget(self.spin_gemini_start, 0, 1)

        gemini_layout.addWidget(QtWidgets.QLabel("Stop (mm):"), 1, 0)
        self.spin_gemini_stop = QtWidgets.QDoubleSpinBox()
        self.spin_gemini_stop.setRange(0, 50)
        self.spin_gemini_stop.setDecimals(2)
        self.spin_gemini_stop.setSingleStep(0.5)
        self.spin_gemini_stop.setValue(DEFAULT_GEMINI_STOP)
        gemini_layout.addWidget(self.spin_gemini_stop, 1, 1)

        gemini_layout.addWidget(QtWidgets.QLabel("Number of Steps:"), 2, 0)
        self.spin_gemini_steps = QtWidgets.QSpinBox()
        self.spin_gemini_steps.setRange(2, 10000)
        self.spin_gemini_steps.setValue(DEFAULT_GEMINI_STEPS)
        gemini_layout.addWidget(self.spin_gemini_steps, 2, 1)

        gemini_layout.addWidget(QtWidgets.QLabel("Step Size:"), 3, 0)
        self.lbl_step_size = QtWidgets.QLabel("-- µm")
        self.lbl_step_size.setStyleSheet("font-weight: bold;")
        gemini_layout.addWidget(self.lbl_step_size, 3, 1)

        tabs.addTab(gemini_tab, "Spectrum Scan")
        right_layout.addWidget(tabs)

        # Acquisition settings
        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_layout = QtWidgets.QGridLayout(acq_group)
        acq_layout.addWidget(QtWidgets.QLabel("Frames/point:"), 0, 0)
        self.spin_frames = QtWidgets.QSpinBox()
        self.spin_frames.setRange(2, 10000)
        self.spin_frames.setValue(100)
        acq_layout.addWidget(self.spin_frames, 0, 1)

        # Sample Name
        acq_layout.addWidget(QtWidgets.QLabel("Sample Name:"), 1, 0)
        self.txt_sample_name = QtWidgets.QLineEdit()
        self.txt_sample_name.setPlaceholderText("Enter sample name...")
        acq_layout.addWidget(self.txt_sample_name, 1, 1)

        self.lbl_total = QtWidgets.QLabel("Total: --")
        self.lbl_total.setStyleSheet("font-weight: bold;")
        acq_layout.addWidget(self.lbl_total, 1, 0, 1, 2)
        self.lbl_total.setStyleSheet("font-weight: bold;")
        acq_layout.addWidget(self.lbl_total, 1, 0, 1, 2)
        
        right_layout.addWidget(acq_group)

        # Save/Plot Settings
        save_mode_group = QtWidgets.QGroupBox("Save & Plot Settings")
        save_mode_layout = QtWidgets.QVBoxLayout(save_mode_group)
        
        # Save Mode
        save_layout_h = QtWidgets.QHBoxLayout()
        save_layout_h.addWidget(QtWidgets.QLabel("Save Mode:"))
        self.cmb_save_mode = QtWidgets.QComboBox()
        self.cmb_save_mode.addItems(["Single Pixel", "ROI Average", "ROI (2D)", "Full Frame (2D)"])
        self.cmb_save_mode.setCurrentIndex(1) # Default ROI Avg
        save_layout_h.addWidget(self.cmb_save_mode)
        save_mode_layout.addLayout(save_layout_h)
        
        # Plot Mode
        plot_layout_h = QtWidgets.QHBoxLayout()
        plot_layout_h.addWidget(QtWidgets.QLabel("Plot Mode:"))
        self.cmb_plot_mode = QtWidgets.QComboBox()
        self.cmb_plot_mode.addItems(["DeltaT (dT/T)", "Transmission (T)", "DeltaT (dT)"])
        self.cmb_plot_mode.setCurrentIndex(0) # Default dT/T
        plot_layout_h.addWidget(self.cmb_plot_mode)
        save_mode_layout.addLayout(plot_layout_h)

        self.btn_bg = QtWidgets.QPushButton("Acquire Background")
        self.btn_bg.setStyleSheet("background-color: #607D8B; color: white;")
        self.btn_bg.clicked.connect(self._acquire_background)
        save_mode_layout.addWidget(self.btn_bg)
        
        # Processing Settings
        proc_group = QtWidgets.QGroupBox("Processing Settings")
        proc_layout = QtWidgets.QGridLayout(proc_group)
        
        proc_layout.addWidget(QtWidgets.QLabel("Freq Points:"), 0, 0)
        self.spin_n_points = QtWidgets.QSpinBox()
        self.spin_n_points.setRange(0, 10000) # 0 = Auto (4x)
        self.spin_n_points.setValue(0)
        self.spin_n_points.setSpecialValueText("Auto (4x)")
        proc_layout.addWidget(self.spin_n_points, 0, 1)
        
        right_layout.addWidget(proc_group)
        
        right_layout.addWidget(save_mode_group)

        # Status + Progress
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        right_layout.addWidget(self.lbl_status)

        self.progress_bar = QtWidgets.QProgressBar()
        right_layout.addWidget(self.progress_bar)

        # Buttons
        self.btn_reference = QtWidgets.QPushButton("📸 Acquire Reference")
        self.btn_reference.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 10px; border-radius: 5px;"
        )
        self.btn_reference.clicked.connect(self._acquire_reference)
        right_layout.addWidget(self.btn_reference)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("▶ START SCAN")
        self.btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 12px; border-radius: 6px;"
        )
        self.btn_start.clicked.connect(self._start_scan)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("⏹ STOP")
        self.btn_stop.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; "
            "padding: 12px; border-radius: 6px;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_scan)
        btn_row.addWidget(self.btn_stop)
        right_layout.addLayout(btn_row)

        right_layout.addStretch()
        main_layout.addWidget(right_panel, stretch=1)

        # Connect signals for live count updates
        for sp_s, sp_e, sp_st in self.interval_spins:
            sp_s.valueChanged.connect(self._update_counts)
            sp_e.valueChanged.connect(self._update_counts)
            sp_st.valueChanged.connect(self._update_counts)
        self.spin_gemini_start.valueChanged.connect(self._update_counts)
        self.spin_gemini_stop.valueChanged.connect(self._update_counts)
        self.spin_gemini_steps.valueChanged.connect(self._update_counts)

        self._update_counts()

    # =========================================================================
    #  Helpers
    # =========================================================================

    def _generate_time_points(self):
        """Generate sorted, unique time points from 3 intervals."""
        points = []
        for sp_s, sp_e, sp_st in self.interval_spins:
            s, e, step = sp_s.value(), sp_e.value(), sp_st.value()
            if step > 0 and s < e:
                points.extend(np.arange(s, e, step).tolist())
        if len(self.interval_spins) == 3:
            # Include endpoint of last interval
            _, sp_e, _ = self.interval_spins[2]
            if sp_e.value() not in points:
                points.append(sp_e.value())
        return np.array(sorted(set(points)))

    def _generate_gemini_positions(self):
        start = self.spin_gemini_start.value()
        stop = self.spin_gemini_stop.value()
        n_steps = self.spin_gemini_steps.value()
        return np.linspace(start, stop, n_steps)

    def _update_counts(self):
        time_pts = self._generate_time_points()
        gemini_pts = self._generate_gemini_positions()
        self.lbl_time_points.setText(f"Time points: {len(time_pts)}")

        start = self.spin_gemini_start.value()
        stop = self.spin_gemini_stop.value()
        n_steps = self.spin_gemini_steps.value()
        if n_steps > 1:
            step_um = (stop - start) / (n_steps - 1) * 1000
            self.lbl_step_size.setText(f"{step_um:.1f} µm")
        else:
            self.lbl_step_size.setText("-- µm")

        total = len(time_pts) * len(gemini_pts)
        self.lbl_total.setText(f"Total: {total:,} acquisitions")

    # =========================================================================
    # Signal Extraction (ROI or Pixel from Live View)
    # =========================================================================
    
    # =========================================================================
    # Signal Extraction
    # =========================================================================
    
    def _extract(self, img):
        """Extract scalar signal for PLOTTING based on save mode."""
        mode = self.cmb_save_mode.currentIndex() 
        # 0=Single, 1=ROI Avg, 2=ROI(2D), 3=Full(2D)
        
        if self.live_window:
            # Single Pixel
            if mode == 0:
                r, c = self.live_window.sel_row, self.live_window.sel_col
                if r is not None and c is not None:
                    h, w = img.shape
                    if 0 <= r < h and 0 <= c < w:
                        return float(img[r, c])
            
            # ROI modes (Avg or 2D) -> Plot Mean of ROI
            if mode in [1, 2]:
                bounds = self.live_window.get_roi_bounds()
                if bounds:
                    r0, r1, c0, c1 = bounds
                    h, w = img.shape
                    r0, r1 = max(0, min(r0, h)), max(1, min(r1, h))
                    c0, c1 = max(0, min(c0, w)), max(1, min(c1, w))
                    return float(np.mean(img[r0:r1, c0:c1]))

        # Fallback / Full Frame -> Mean of everything
        return float(np.mean(img))
    
    def _extract_to_save(self, img):
        """Extract data entity to SAVE in datacube based on mode."""
        mode = self.cmb_save_mode.currentIndex()
        
        if mode == 3: # Full Frame
            return img.copy()
            
        if self.live_window:
            # ROI modes
            if mode in [1, 2]:
                bounds = self.live_window.get_roi_bounds()
                if bounds:
                    r0, r1, c0, c1 = bounds
                    h, w = img.shape
                    r0, r1 = max(0, min(r0, h)), max(1, min(r1, h))
                    c0, c1 = max(0, min(c0, w)), max(1, min(c1, w))
                    slice_img = img[r0:r1, c0:c1]
                    
                    if mode == 1: # ROI Avg -> Save scalar (or 1x1)
                        return float(np.mean(slice_img))
                    else: # ROI 2D -> Save slice
                        return slice_img.copy()
            
            # Single Pixel
            if mode == 0:
                r, c = self.live_window.sel_row, self.live_window.sel_col
                if r is not None and c is not None:
                    h, w = img.shape
                    if 0 <= r < h and 0 <= c < w:
                        return float(img[r, c])
        
        # Fallback
        return float(np.mean(img))
    
    def _fs_to_mm(self, time_fs):
        return time_fs * SPEED_OF_LIGHT_MM_FS / 2.0

    def _update_preview(self):
        """Read one frame from LabVIEW for live preview (when not scanning)."""
        if self._scanning:
            return
        if not self.manager.vi:
            return
        try:
            odd_data = self.manager.vi.GetControlValue("Odd")
            even_data = self.manager.vi.GetControlValue("Even")
            if odd_data is not None and even_data is not None:
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                img = (even - odd) / np.where(np.abs(odd) > 1e-10, odd, 1e-10)
                if img.ndim == 1:
                    side = int(np.sqrt(img.size))
                    if side * side == img.size:
                        img = img.reshape(side, side)
                if img.ndim == 2:
                    self.img_item.setImage(img.T, autoLevels=True)
        except Exception:
            pass

    # =========================================================================
    #  Reference Acquisition
    # =========================================================================

    def _acquire_reference(self):
        """Run a single Gemini interferogram → spectrum → store as reference."""
        if not self.stage_gemini or not self.stage_gemini.is_connected:
            self.lbl_status.setText("Gemini stage not connected!")
            return
        if not self.manager.vi:
            self.lbl_status.setText("LabVIEW Manager not running!")
            return

        self.lbl_status.setText("Acquiring reference...")
        self.lbl_status.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.btn_reference.setEnabled(False)
        self.btn_start.setEnabled(False)

        self.gemini_positions = self._generate_gemini_positions()
        self.current_interferogram = np.zeros(len(self.gemini_positions))
        self._gemini_index = 0
        self._ref_mode = True
        self._scanning = True
        self.preview_timer.stop()

        QtCore.QTimer.singleShot(50, self._gemini_move_next)

    # =========================================================================
    #  Full Scan
    # =========================================================================

    def _start_scan(self):
        if self._scanning:
            return
        if not self.stage_gemini or not self.stage_gemini.is_connected:
            self.lbl_status.setText("Gemini stage not connected!")
            return
        if not self.stage_delay or not self.stage_delay.is_connected:
            self.lbl_status.setText("Delay stage not connected!")
            return
        if not self.manager.vi:
            self.lbl_status.setText("LabVIEW Manager not running!")
            return
        if self.reference_spectrum is None:
            self.lbl_status.setText("Acquire reference first!")
            return

        self.time_points = self._generate_time_points()
        self.gemini_positions = self._generate_gemini_positions()

        n_time = len(self.time_points)
        self.hyperspectral_map = None  # Will be initialized after first spectrum

        # Generate Standardized Paths
        timestamp = datetime.now()
        date_dir = timestamp.strftime(r"D:\pumpprobedata\%Y\%m\%d")
        os.makedirs(date_dir, exist_ok=True)
        
        sample = self.txt_sample_name.text().strip()
        if not sample: sample = "sample"
        # Sanitize
        sample = "".join(x for x in sample if x.isalnum() or x in " -_")
        
        self._save_prefix = os.path.join(date_dir, f"{sample}_twins_pp_{timestamp.strftime('%H%M%S')}")

        # UI
        self._scanning = True
        self._ref_mode = False
        self._time_index = 0
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_reference.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Scanning...")
        self.lbl_status.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.preview_timer.stop()

        print(f"[TWINS-PP] Starting: {n_time} time × {len(self.gemini_positions)} gemini")
        QtCore.QTimer.singleShot(50, self._time_move_next)

    def _stop_scan(self):
        self._scanning = False
        self.lbl_status.setText("Stopped")
        self.lbl_status.setStyleSheet("color: #f44336; font-weight: bold;")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_reference.setEnabled(True)
        self.preview_timer.start(500)
        print("[TWINS-PP] Scan stopped by user")

    # =========================================================================
    #  Outer Loop: Time Delay
    # =========================================================================

    def _time_move_next(self):
        """Move delay stage to next time point."""
        if not self._scanning:
            return
        if self._time_index >= len(self.time_points):
            self._scan_complete()
            return

        delay_fs = self.time_points[self._time_index]
        pos_mm = self.spin_zero.value() + self._fs_to_mm(delay_fs)

        self.lbl_status.setText(
            f"Time {self._time_index+1}/{len(self.time_points)}: "
            f"{delay_fs:.0f} fs → {pos_mm:.3f} mm"
        )

        try:
            self.stage_delay.move_to(pos_mm, wait=False)
        except Exception as e:
            print(f"[TWINS-PP] Delay move error: {e}")
            self._time_index += 1
            QtCore.QTimer.singleShot(50, self._time_move_next)
            return

        # Poll delay stage until stable
        self._stable_count = 0
        self._last_pos = None
        self._delay_timer = QtCore.QTimer()
        self._delay_timer.timeout.connect(self._poll_delay_stage)
        self._delay_waited = 0.0
        self._delay_timer.start(100)

    def _poll_delay_stage(self):
        """Poll delay stage until stable, then start inner Gemini loop."""
        if not self._scanning:
            self._delay_timer.stop()
            return

        self._delay_waited += 0.1
        if self._delay_waited > 60.0:
            self._delay_timer.stop()
            print("[TWINS-PP] Delay stage timeout")
            self._time_index += 1
            QtCore.QTimer.singleShot(50, self._time_move_next)
            return

        try:
            pos = self.stage_delay.get_position()
        except Exception:
            return

        if self._last_pos is not None and abs(pos - self._last_pos) < 0.001:
            self._stable_count += 1
        else:
            self._stable_count = 0
        self._last_pos = pos

        if self._stable_count >= 3:
            self._delay_timer.stop()
            # Start inner Gemini loop
            self.current_interferogram = np.zeros(len(self.gemini_positions))
            self.current_roi_datacube = []  # ROI slices for this time step
            self._gemini_index = 0
            QtCore.QTimer.singleShot(50, self._gemini_move_next)

    # =========================================================================
    #  Inner Loop: Gemini Interferogram
    # =========================================================================
    
    def _acquire_background(self):
        """Trigger a single acquisition for background."""
        if self._scanning:
            return
        
        self.lbl_status.setText("Status: Acquiring Global Background...")
        self._awaiting_background = True
        
        # Reuse trigger logic (manually)
        self.manager.vi.SetControlValue("N", self.spin_frames.value())
        self.manager.vi.SetControlValue("Acq Trigger", True)
        self.manager.vi.SetControlValue("Enum", CMD_MEASURE)
        
        self._acq_timer = QtCore.QTimer()
        self._acq_timer.timeout.connect(self._gemini_poll_acquire) # Reuse existing poller
        self._acq_waited = 0.0
        self._acq_timer.start(50)

    def _gemini_move_next(self):
        """Move Gemini stage to next position in interferogram scan."""
        if not self._scanning:
            return
        if self._gemini_index >= len(self.gemini_positions):
            self._gemini_scan_done()
            return

        target = self.gemini_positions[self._gemini_index]

        try:
            self.stage_gemini.move_to(target)
        except Exception as e:
            print(f"[TWINS-PP] Gemini move error: {e}")
            self._gemini_index += 1
            QtCore.QTimer.singleShot(10, self._gemini_move_next)
            return

        # Since stage_gemini.move_to is blocking (synchronous),
        # we know motion is complete here.
        # No need to poll for stability unless vibration is severe.
        # Trigger acquire immediately (or with small settling time).
        QtCore.QTimer.singleShot(10, self._gemini_trigger_acquire)

    # _poll_gemini_stage removed (redundant)

    def _gemini_trigger_acquire(self):
        """Trigger LabVIEW Measure for one interferogram point."""
        if not self._scanning:
            return

        vi = self.manager.vi
        n = self.spin_frames.value()

        try:
            vi.SetControlValue("N", n)
            vi.SetControlValue("Acq Trigger", True)
            vi.SetControlValue("Enum", CMD_MEASURE)

            self._acq_timer = QtCore.QTimer()
            self._acq_timer.timeout.connect(self._gemini_poll_acquire)
            self._acq_waited = 0.0
            
            # Wait for acquisition (N ms) + 20ms buffer, then start polling every 20ms
            wait_ms = n + 20
            QtCore.QTimer.singleShot(wait_ms, lambda: self._acq_timer.start(20))
        except Exception as e:
            print(f"[TWINS-PP] Acquire error: {e}")
            self._gemini_index += 1
            QtCore.QTimer.singleShot(10, self._gemini_move_next)

    def _gemini_poll_acquire(self):
        """Poll LabVIEW until Idle, then read result."""
        if not self._scanning:
            self._acq_timer.stop()
            return

        self._acq_waited += 0.05
        if self._acq_waited > 60.0:
            self._acq_timer.stop()
            self._gemini_index += 1
            self._gemini_index += 1
            QtCore.QTimer.singleShot(10, self._gemini_move_next)
            return

        try:
            if self.manager.vi.GetControlValue("Enum") != CMD_IDLE:
                return
        except Exception:
            return

        self._acq_timer.stop()

        try:
            odd_data = self.manager.vi.GetControlValue("Odd")
            even_data = self.manager.vi.GetControlValue("Even")
            if odd_data is not None and even_data is not None:
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                
                # Handle Background
                if hasattr(self, '_awaiting_background') and self._awaiting_background:
                    self.manager.background = (odd + even) / 2.0
                    self._awaiting_background = False
                    self.lbl_status.setText("Global Background Acquired")
                    QtWidgets.QMessageBox.information(self, "Background", "Global Background acquired!")
                    return

                # Apply Background
                if self.manager.background is not None:
                    if self.manager.background.shape == odd.shape:
                        odd -= self.manager.background
                        even -= self.manager.background
                
                if self._ref_mode:
                    img = (odd + even) / 2.0 # Ref is Transmission
                else:
                    pmode = self.cmb_plot_mode.currentIndex()
                    if pmode == 1:   # Transmission
                        img = (odd + even) / 2.0
                    elif pmode == 2: # DeltaT (dT)
                        img = even - odd
                    else:            # DeltaT/T
                        img = np.divide(even - odd, odd, out=np.zeros_like(odd), where=np.abs(odd) > 1.0)
                signal = self._extract(img)
                self.current_interferogram[self._gemini_index] = signal

                # Store full ROI slice
                to_save = self._extract_to_save(img)
                self.current_roi_datacube.append(to_save)

                # Update camera preview every 10th point
                if self._gemini_index % 10 == 0:
                    if img.ndim == 1:
                        side = int(np.sqrt(img.size))
                        if side * side == img.size:
                            img = img.reshape(side, side)
                    if img.ndim == 2:
                        self.img_item.setImage(img.T, autoLevels=True)
        except Exception as e:
            print(f"[TWINS-PP] Read error: {e}")

        self._gemini_index += 1
        QtCore.QTimer.singleShot(50, self._gemini_move_next)

    # =========================================================================
    #  Post-Interferogram Processing
    # =========================================================================

    def _gemini_scan_done(self):
        """One full interferogram is complete — compute spectrum."""
        wl, spectrum = self.processor.compute_spectrum(
            self.gemini_positions, self.current_interferogram,
            n_points=self.spin_n_points.value() if self.spin_n_points.value() > 0 else None
        )

        if self._ref_mode:
            # Reference acquisition
            self.reference_wavelengths = wl
            self.reference_spectrum = spectrum
            self.reference_curve.setData(wl, spectrum)
            self.spectrum_curve.setData(wl, spectrum)

            self._scanning = False
            self._ref_mode = False
            self.btn_reference.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.preview_timer.start(500)
            self.lbl_status.setText("Reference acquired ✓")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            print("[TWINS-PP] Reference spectrum acquired")
            return

        # Full scan mode — compute ΔT/T
        # Full scan mode — compute Map
        # Note: spectrum is already processed based on Plot Mode (DT, T, or DT/T)
        # So we just treat it as the signal.
        
        # If we want to use Reference for Normalization (e.g. in T mode, T/T0), we could.
        # But for Pump-Probe, usually we care about the per-shot difference.
        delta_t = spectrum

        # Update spectrum plot
        self.spectrum_curve.setData(wl, spectrum)

        # Build / update hyperspectral map
        if self.hyperspectral_map is None:
            self.hyperspectral_map = np.zeros((len(self.time_points), len(delta_t)))
            self.wavelengths = wl

        if self._time_index < self.hyperspectral_map.shape[0]:
            n = min(len(delta_t), self.hyperspectral_map.shape[1])
            self.hyperspectral_map[self._time_index, :n] = delta_t[:n]

        self.map_item.setImage(self.hyperspectral_map.T, autoLevels=True)

        # Progress
        delay_fs = self.time_points[self._time_index]
        pct = int(100 * (self._time_index + 1) / len(self.time_points))
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(
            f"Time {self._time_index+1}/{len(self.time_points)}: {delay_fs:.0f} fs — done"
        )

        # Save this step
        self._save_time_step(self._time_index, delay_fs, wl, spectrum, delta_t)

        # Move to next time point
        self._time_index += 1
        QtCore.QTimer.singleShot(100, self._time_move_next)

    # =========================================================================
    #  Scan Complete
    # =========================================================================

    def _scan_complete(self):
        self._scanning = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_reference.setEnabled(True)
        self.preview_timer.start(500)
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Scan complete ✓")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

        # Save final combined data
        self._save_final()
        print("[TWINS-PP] Scan complete!")

    # =========================================================================
    #  Save
    # =========================================================================

    def _save_time_step(self, idx, delay_fs, wl, spectrum, delta_t):
        if not hasattr(self, '_save_prefix'):
            return
        filename = f"{self._save_prefix}_step{idx:03d}_{delay_fs:.0f}fs.npz"
        np.savez(filename,
                 delay_fs=delay_fs,
                 wavelengths=wl,
                 spectrum=spectrum,
                 delta_t=delta_t,
                 interferogram=self.current_interferogram,
                 gemini_positions=self.gemini_positions,
                 roi_datacube=np.array(self.current_roi_datacube) if self.current_roi_datacube else np.array([]))
        print(f"[SAVE] {os.path.basename(filename)}")

    def _save_final(self):
        if not hasattr(self, '_save_prefix') or self.hyperspectral_map is None:
            return
        try:
            np.savez(f"{self._save_prefix}_FINAL.npz",
                     time_points_fs=self.time_points[:self._time_index],
                     hyperspectral_map=self.hyperspectral_map,
                     wavelengths=self.wavelengths,
                     reference_wavelengths=self.reference_wavelengths,
                     reference_spectrum=self.reference_spectrum,
                     zero_mm=self.spin_zero.value())
            print(f"[SAVE] Final: {self._save_prefix}_FINAL.npz")
        except Exception as e:
            print(f"[ERROR] Final save failed: {e}")

    # =========================================================================
    #  Cleanup
    # =========================================================================

    def closeEvent(self, event):
        if self._scanning:
            self._stop_scan()
        self.preview_timer.stop()
        event.accept()
