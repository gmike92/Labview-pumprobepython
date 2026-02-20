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

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_data")

DEFAULT_GEMINI_START = 15.0   # mm
DEFAULT_GEMINI_STOP = 23.0    # mm
DEFAULT_GEMINI_STEPS = 100


# ============================================================================
# Lightweight Spectrum Processor  (FFT-based, for speed during scan)
# ============================================================================

class SpectrumProcessor:
    """Process interferogram → spectrum via FFT with calibration and phasing."""

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

    def get_dx(self, positions):
        if len(positions) > 1:
            return np.abs(positions[1] - positions[0])
        return 0.08 # Default estimation

    def compute_phase_correction(self, positions, reference, n_points=None, pad_factor=4):
        """Calculates phase correction from reference interferogram."""
        from scipy.fft import fft, fftshift, fftfreq
        
        n_raw = len(reference)
        # Windowing (Hanning)
        window = np.hanning(n_raw)
        windowed_ref = reference * window
        
        # Zero Padding
        if n_points:
            pad_length = n_points
        else:
            pad_length = n_raw * pad_factor
            
        # FFT and Shift
        ref_fft = fftshift(fft(windowed_ref, n=pad_length))
        phase_correction = np.angle(ref_fft)
        
        return phase_correction, pad_length

    def compute_phased_spectrum(self, positions, interferogram, phase_correction, pad_length=None, wl_start=8.0, wl_stop=14.0):
        """Computes phased spectrum (Absorption/Distorsion). Returns (wl, real, imag)."""
        from scipy.fft import fft, fftshift, fftfreq
        from scipy.interpolate import interp1d

        n_raw = interferogram.shape[-1]
        
        # Windowing
        window = np.hanning(n_raw)
        windowed_data = interferogram * window
        
        if pad_length is None:
            pad_length = n_raw * 4
            
        # FFT (Axis -1)
        dt_fft_raw = fftshift(fft(windowed_data, n=pad_length, axis=-1), axes=-1)
        
        # Apply Phase
        # Ensure phase broadcasting if data is 2D
        if phase_correction is not None:
            dt_phased = dt_fft_raw * np.exp(-1j * phase_correction)
        else:
            dt_phased = dt_fft_raw
            
        # Extract Real/Imag
        real_part = np.real(dt_phased)  # Absorption
        imag_part = np.imag(dt_phased)  # Dispersion
        
        # Frequency Axis
        dx = self.get_dx(positions)
        freq = fftshift(fftfreq(pad_length, d=dx))
        
        # Calibration (Freq -> WL)
        # If calibration exists:
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            # We must map freq array to wavelengths.
            # Calibration maps Reciprocal (Freq) -> Wavelength
            # We interp1d(Cal_Freq, Cal_WL)
            # But we need to be careful about range.
            
            # Note: fftfreq returns negative and positive frequencies.
            # Calibration typically is only for positive.
            # We normally take only positive half for display?
            # Or we map the whole thing?
            
            # The user's code returns the full FFT shift.
            # This puts Zero freq in center.
            # Positive freq on right (indices N/2 to N).
            # Negative freq on left.
            
            # If we mask, we usually only care about positive frequencies in range [wl_start, wl_stop].
            # Wavelength = 1/Freq.
            # Freq < 0 -> Negative Index? Physical meaning? usually we ignore.
            
            # Let's filter for positive frequencies first.
            pos_mask = freq > 0
            freq_pos = freq[pos_mask]
            real_pos = real_part[..., pos_mask]
            imag_pos = imag_part[..., pos_mask]
            
            fn = interp1d(self.reciprocal_cal, self.wavelength_cal,
                          kind='linear', fill_value='extrapolate', bounds_error=False)
            wavelengths = fn(freq_pos)
            
            # Interpolate to Linear Wavelength Grid (for ImageItem display)
            # Create a linear grid between start and stop
            n_out = len(wavelengths)
            wl_linear = np.linspace(wl_start, wl_stop, n_out)
            
            # Interpolate Data onto Linear Grid
            # Current wavelengths are monotonic? Typically yes (decreasing or increasing).
            # Sorting might be needed if not.
            # Assuming monotonic.
            
            # Interpolate Real
            # Use interp1d. x must be monotonic.
            
            # Sort just in case
            idx = np.argsort(wavelengths)
            wl_sorted = wavelengths[idx]
            real_sorted = real_pos[..., idx]
            imag_sorted = imag_pos[..., idx]
            
            fn_real = interp1d(wl_sorted, real_sorted, kind='linear', bounds_error=False, fill_value=0.0)
            fn_imag = interp1d(wl_sorted, imag_sorted, kind='linear', bounds_error=False, fill_value=0.0)
            
            real_interp = fn_real(wl_linear)
            imag_interp = fn_imag(wl_linear)
            
            return wl_linear, real_interp, imag_interp
            
        else:
            # No Cal -> 1/f
            pos_mask = freq > 1e-10
            freq_pos = freq[pos_mask]
            wavelengths = 1.0 / freq_pos
            
            # Interpolate to Linear Grid
            n_out = len(wavelengths)
            wl_linear = np.linspace(wl_start, wl_stop, n_out)
            
            idx = np.argsort(wavelengths)
            wl_sorted = wavelengths[idx]
            real_sorted = real_pos[..., idx]
            imag_sorted = imag_pos[..., idx]
            
            fn_real = interp1d(wl_sorted, real_sorted, kind='linear', bounds_error=False, fill_value=0.0)
            fn_imag = interp1d(wl_sorted, imag_sorted, kind='linear', bounds_error=False, fill_value=0.0)
            
            real_interp = fn_real(wl_linear)
            imag_interp = fn_imag(wl_linear)
            
            return wl_linear, real_interp, imag_interp

    def compute_spectrum(self, positions, interferogram, n_points=None, wl_start=8.0, wl_stop=14.0):
        """Legacy / Power Spectrum computation (Magnitude Squared)."""
        # We can implement this using the phased logic with phase=0 and taking abs**2
        # Or keep simplest logic.
        
        # Use simple logic for robustness if scanning for reference
        # But we need consistent axes.
        
        w, r, i = self.compute_phased_spectrum(positions, interferogram, None, n_points, wl_start, wl_stop)
        power = r**2 + i**2
        return w, power


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
        self.selected_wl_index = None # For Time Dynamics plot
        self.phase_correction = None
        self.pad_length = None

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

        # =====================================================================
        # LEFT SIDEBAR: Controls
        # =====================================================================
        sidebar_panel = QtWidgets.QWidget()
        sidebar_panel.setMaximumWidth(380)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)

        # Title / Status
        sidebar_layout.addWidget(QtWidgets.QLabel("<b>Control Panel</b>"))

        # Tabs
        tabs = QtWidgets.QTabWidget()

        # --- Tab 1: Time Scan (3 intervals) ---
        time_tab = QtWidgets.QWidget()
        time_layout = QtWidgets.QVBoxLayout(time_tab)

        # Zero position
        zero_group = QtWidgets.QGroupBox("Zero Position (t=0)")
        zero_layout = QtWidgets.QHBoxLayout(zero_group)
        zero_layout.addWidget(QtWidgets.QLabel("Pos (mm):"))
        self.spin_zero = QtWidgets.QDoubleSpinBox()
        self.spin_zero.setRange(0, 300)
        self.spin_zero.setDecimals(3)
        if self.stage_delay:
            self.spin_zero.setValue(self.stage_delay.zero_position)
        else:
            self.spin_zero.setValue(140.0)
            
        def on_zero_changed(val):
            if self.stage_delay:
                self.stage_delay.zero_position = val
        self.spin_zero.valueChanged.connect(on_zero_changed)
        
        zero_layout.addWidget(self.spin_zero)
        self.chk_probe = QtWidgets.QCheckBox("Probe")
        self.chk_probe.setToolTip("If checked, stage moves Probe (Delay = Zero - Pos). Else Pump (Delay = Pos - Zero).")
        zero_layout.addWidget(self.chk_probe)
        time_layout.addWidget(zero_group)

        # 3 intervals
        interval_configs = [
            ("Interval 1 (Fine)", -1000, 0, 50),
            ("Interval 2 (Early)", 0, 10000, 500),
            ("Interval 3 (Late)", 10000, 100000, 5000),
        ]
        self.interval_spins = []  # [(start, end, step), ...]

        for label, s_def, e_def, st_def in interval_configs:
            grp = QtWidgets.QGroupBox(label)
            if "Interval 1" not in label:
                grp.setCheckable(True)
                if "Interval 2" in label: grp.setChecked(True)
                else: grp.setChecked(False)
                grp.toggled.connect(self._update_counts)
                
            gl = QtWidgets.QGridLayout(grp)
            gl.setContentsMargins(2, 2, 2, 2)
            
            gl.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
            sp_s = QtWidgets.QDoubleSpinBox()
            sp_s.setRange(-1e6, 1e6); sp_s.setValue(s_def)
            gl.addWidget(sp_s, 0, 1)

            gl.addWidget(QtWidgets.QLabel("End:"), 0, 2)
            sp_e = QtWidgets.QDoubleSpinBox()
            sp_e.setRange(-1e6, 1e6); sp_e.setValue(e_def)
            gl.addWidget(sp_e, 0, 3)

            gl.addWidget(QtWidgets.QLabel("Step:"), 1, 0)
            sp_st = QtWidgets.QDoubleSpinBox()
            sp_st.setRange(0.1, 1e6); sp_st.setValue(st_def)
            gl.addWidget(sp_st, 1, 1)
            
            self.interval_spins.append((sp_s, sp_e, sp_st, grp))
            time_layout.addWidget(grp)

        self.lbl_time_points = QtWidgets.QLabel("Points: --")
        self.lbl_time_points.setStyleSheet("font-weight: bold;")
        time_layout.addWidget(self.lbl_time_points)
        time_layout.addStretch()
        tabs.addTab(time_tab, "Time")

        # --- Tab 2: Gemini Scan ---
        gemini_tab = QtWidgets.QWidget()
        gemini_layout = QtWidgets.QGridLayout(gemini_tab)

        gemini_layout.addWidget(QtWidgets.QLabel("Start (mm):"), 0, 0)
        self.spin_gemini_start = QtWidgets.QDoubleSpinBox()
        self.spin_gemini_start.setRange(0, 50); self.spin_gemini_start.setValue(DEFAULT_GEMINI_START)
        gemini_layout.addWidget(self.spin_gemini_start, 0, 1)

        gemini_layout.addWidget(QtWidgets.QLabel("Stop (mm):"), 1, 0)
        self.spin_gemini_stop = QtWidgets.QDoubleSpinBox()
        self.spin_gemini_stop.setRange(0, 50); self.spin_gemini_stop.setValue(DEFAULT_GEMINI_STOP)
        gemini_layout.addWidget(self.spin_gemini_stop, 1, 1)

        gemini_layout.addWidget(QtWidgets.QLabel("Steps:"), 2, 0)
        self.spin_gemini_steps = QtWidgets.QSpinBox()
        self.spin_gemini_steps.setRange(2, 10000); self.spin_gemini_steps.setValue(DEFAULT_GEMINI_STEPS)
        gemini_layout.addWidget(self.spin_gemini_steps, 2, 1)

        gemini_layout.addWidget(QtWidgets.QLabel("Size:"), 3, 0)
        self.lbl_step_size = QtWidgets.QLabel("-- µm")
        gemini_layout.addWidget(self.lbl_step_size, 3, 1)
        gemini_layout.setRowStretch(4, 1)
        tabs.addTab(gemini_tab, "Spectrum")
        
        # --- Tab 3: Settings ---
        settings_tab = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_tab)
        
        # Acquisition
        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_l = QtWidgets.QGridLayout(acq_group)
        acq_l.addWidget(QtWidgets.QLabel("Frames/pt:"), 0, 0)
        self.spin_frames = QtWidgets.QSpinBox()
        self.spin_frames.setRange(2, 10000); self.spin_frames.setValue(100)
        acq_l.addWidget(self.spin_frames, 0, 1)
        acq_l.addWidget(QtWidgets.QLabel("Sample:"), 1, 0)
        self.txt_sample_name = QtWidgets.QLineEdit()
        self.txt_sample_name.setPlaceholderText("Name...")
        acq_l.addWidget(self.txt_sample_name, 1, 1)
        self.lbl_total = QtWidgets.QLabel("Total Time: --")
        acq_l.addWidget(self.lbl_total, 2, 0, 1, 2)
        settings_layout.addWidget(acq_group)
        
        # Save/Plot
        sp_group = QtWidgets.QGroupBox("Save & Plot")
        sp_l = QtWidgets.QVBoxLayout(sp_group)
        
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Save:"))
        self.cmb_save_mode = QtWidgets.QComboBox()
        self.cmb_save_mode.addItems(["Single Pixel", "ROI Average", "ROI (2D)", "Full Frame (2D)"])
        self.cmb_save_mode.setCurrentIndex(1)
        h.addWidget(self.cmb_save_mode)
        sp_l.addLayout(h)
        
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("Plot:"))
        self.cmb_plot_mode = QtWidgets.QComboBox()
        self.cmb_plot_mode.addItems(["DeltaT (dT/T)", "Transmission (T)", "DeltaT (dT)"])
        self.cmb_plot_mode.setCurrentIndex(0)
        h2.addWidget(self.cmb_plot_mode)
        sp_l.addLayout(h2)
        
        self.btn_bg = QtWidgets.QPushButton("Acquire Background")
        self.btn_bg.setStyleSheet("background-color: #607D8B; color: white;")
        self.btn_bg.clicked.connect(self._acquire_background)
        sp_l.addWidget(self.btn_bg)
        settings_layout.addWidget(sp_group)
        
        # Data to Save
        save_group = QtWidgets.QGroupBox("Data Selection")
        sl = QtWidgets.QVBoxLayout(save_group)
        self.chk_save_t = QtWidgets.QCheckBox("Trans (T)")
        self.chk_save_dt = QtWidgets.QCheckBox("DeltaT (dT)")
        self.chk_save_dtt = QtWidgets.QCheckBox("DeltaT/T")
        self.chk_save_raw = QtWidgets.QCheckBox("Raw (Odd/Even)")
        self.chk_save_dtt.setChecked(True)
        sl.addWidget(self.chk_save_t); sl.addWidget(self.chk_save_dt)
        sl.addWidget(self.chk_save_dtt); sl.addWidget(self.chk_save_raw)
        settings_layout.addWidget(save_group)
        
        settings_layout.addStretch()
        tabs.addTab(settings_tab, "Settings")
        
        sidebar_layout.addWidget(tabs)

        # Processing (Freq Points) - Put in sidebar bottom or separate?
        # Put in sidebar
        proc_group = QtWidgets.QGroupBox("FFT Settings")
        pl = QtWidgets.QHBoxLayout(proc_group)
        pl.addWidget(QtWidgets.QLabel("Pts:"))
        self.spin_n_points = QtWidgets.QSpinBox()
        self.spin_n_points.setRange(0, 10000); self.spin_n_points.setValue(0)
        self.spin_n_points.setSpecialValueText("Auto")
        pl.addWidget(self.spin_n_points)
        
        pl.addWidget(QtWidgets.QLabel("Start:"))
        self.spin_wl_start = QtWidgets.QDoubleSpinBox()
        self.spin_wl_start.setRange(0, 20); self.spin_wl_start.setValue(8.0)
        pl.addWidget(self.spin_wl_start)
        
        pl.addWidget(QtWidgets.QLabel("Stop:"))
        self.spin_wl_stop = QtWidgets.QDoubleSpinBox()
        self.spin_wl_stop.setRange(0, 20); self.spin_wl_stop.setValue(14.0)
        pl.addWidget(self.spin_wl_stop)
        
        sidebar_layout.addWidget(proc_group)

        # Status & Progress
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        sidebar_layout.addWidget(self.lbl_status)
        self.progress_bar = QtWidgets.QProgressBar()
        sidebar_layout.addWidget(self.progress_bar)

        # Action Buttons
        self.btn_reference = QtWidgets.QPushButton("📸 Reference")
        self.btn_reference.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.btn_reference.clicked.connect(self._acquire_reference)
        sidebar_layout.addWidget(self.btn_reference)

        h_btn = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("▶ START")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.btn_start.clicked.connect(self._start_scan)
        h_btn.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("⏹ STOP")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_scan)
        h_btn.addWidget(self.btn_stop)
        sidebar_layout.addLayout(h_btn)

        main_layout.addWidget(sidebar_panel, stretch=0)

        # =====================================================================
        # RIGHT SIDE: PLOTS (Redesigned)
        # =====================================================================
        
        # Structure:
        # QVBoxLayout
        #   - Top: Reference (Collapsible/Small)
        #   - Center: Splitter (Transient Spectrum | Time Dynamics)
        #   - Bottom: Map (Delay vs WL)
        
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Top: Reference Spectrum
        self.ref_plot = pg.PlotWidget(title="Reference Spectrum")
        self.ref_plot.setMaximumHeight(150)
        self.ref_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ref_plot.setLabel('bottom', 'Wavelength (µm)')
        self.reference_curve = self.ref_plot.plot([], [], pen='r', name="Reference")
        right_layout.addWidget(self.ref_plot)
        
        # 2. Center: Split Spectrum and Dynamics
        splitter_center = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        
        # Left: Transient Spectrum (DT vs WL)
        self.spectrum_plot = pg.PlotWidget(title="Transient Spectrum (DT vs λ)")
        self.spectrum_plot.setLabel('left', 'Delta T/T')
        self.spectrum_plot.setLabel('bottom', 'Wavelength (µm)')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_curve = self.spectrum_plot.plot([], [], pen='c')
        splitter_center.addWidget(self.spectrum_plot)
        
        # Right: Time Dynamics (DT vs Delay)
        self.dynamics_plot = pg.PlotWidget(title="Time Dynamics (DT vs Delay)")
        self.dynamics_plot.setLabel('left', 'Delta T/T')
        self.dynamics_plot.setLabel('bottom', 'Delay (fs)')
        self.dynamics_plot.showGrid(x=True, y=True, alpha=0.3)
        self.dynamics_curve = self.dynamics_plot.plot([], [], pen='y')
        # Add a line indicating selected wavelength on spectrum?
        splitter_center.addWidget(self.dynamics_plot)
        
        right_layout.addWidget(splitter_center, stretch=1)
        
        # 3. Bottom: Hyperspectral Map
        self.map_widget = pg.GraphicsLayoutWidget()
        self.map_plot = self.map_widget.addPlot(title="Hyperspectral Map (X: Delay, Y: Wavelength)")
        self.map_plot.setLabel('left', 'Wavelength (µm)')
        self.map_plot.setLabel('bottom', 'Delay (fs)')
        self.map_item = pg.ImageItem()
        self.map_plot.addItem(self.map_item)
        
        # Colorbar / Histogram (Blue-White-Red)
        self.hist_item = pg.HistogramLUTItem()
        self.hist_item.setImageItem(self.map_item)
        
        # Define Blue-White-Red Colormap manually for robustness
        pos = np.array([0.0, 0.5, 1.0])
        color = np.array([
            [0, 0, 255, 255],     # Blue
            [255, 255, 255, 255], # White
            [255, 0, 0, 255]      # Red
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.hist_item.gradient.setColorMap(cmap)
        
        self.map_widget.addItem(self.hist_item)
        
        # Crosshair for interaction
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.map_plot.addItem(self.v_line, ignoreBounds=True)
        self.map_plot.addItem(self.h_line, ignoreBounds=True)
        
        # Click interaction
        self.map_plot.scene().sigMouseClicked.connect(self._on_map_click)
        
        right_layout.addWidget(self.map_widget, stretch=1)
        
        main_layout.addWidget(right_panel, stretch=1)

        # Connect signals for live count updates
        for sp_s, sp_e, sp_st, _ in self.interval_spins:
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
        for sp_s, sp_e, sp_st, grp in self.interval_spins:
            if grp.isCheckable() and not grp.isChecked():
                continue
                
            s, e, step = sp_s.value(), sp_e.value(), sp_st.value()
            if step > 0 and s < e:
                # Use inclusive range: [start, end]
                chunk = np.arange(s, e + step*0.001, step)
                points.extend(chunk.tolist())
            elif step > 0 and s == e:
                points.append(s)
                
        if not points:
            return np.array([])
            
        return np.array(sorted(set([float(f"{p:.3f}") for p in points]))) # Round to 3 decimal to avoid float jitter duplicate

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
        dist = time_fs * SPEED_OF_LIGHT_MM_FS / 2.0
        if self.chk_probe.isChecked():
            return -dist  # Probe: Move Closer (-mm) -> +Delay
        return dist       # Pump: Move Away (+mm) -> +Delay

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
                    # self.img_item.setImage(img.T, autoLevels=True)
                    pass
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
            
            # Selective buffers
            self.current_data_t = []
            self.current_data_dt = []
            self.current_data_dtt = []
            self.current_raw_odd = []
            self.current_raw_even = []
            
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
                    # Store BOTH Odd and Even for Scattering Correction
                    self.manager.background = (odd.copy(), even.copy())
                    self._awaiting_background = False
                    self.lbl_status.setText("Global Background Acquired (Scattering Mode)")
                    QtWidgets.QMessageBox.information(self, "Background", "Global Background acquired! (Odd/Even stored separately)")
                    return

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

                if self._ref_mode:
                    # User requested Odd frames (Pump Off) for Reference
                    img = odd
                else:
                    pmode = self.cmb_plot_mode.currentIndex()
                    if pmode == 1:   img = img_t
                    elif pmode == 2: img = img_dt
                    else:            img = img_dtt
                    
                signal = self._extract(img)
                self.current_interferogram[self._gemini_index] = signal

                # Legacy ROI datacube
                self.current_roi_datacube.append(self._extract_to_save(img))
                
                # Selective Save
                if self.chk_save_t.isChecked():
                    self.current_data_t.append(self._extract_to_save(img_t))
                if self.chk_save_dt.isChecked():
                    self.current_data_dt.append(self._extract_to_save(img_dt))
                if self.chk_save_dtt.isChecked():
                    self.current_data_dtt.append(self._extract_to_save(img_dtt))
                if self.chk_save_raw.isChecked():
                    self.current_raw_odd.append(self._extract_to_save(odd))
                    self.current_raw_even.append(self._extract_to_save(even))

                # Update camera preview every 10th point
                if self._gemini_index % 10 == 0:
                    if img.ndim == 1:
                        side = int(np.sqrt(img.size))
                        if side * side == img.size:
                            img = img.reshape(side, side)
                    if img.ndim == 2:
                        # self.img_item.setImage(img.T, autoLevels=True)
                        pass
        except Exception as e:
            print(f"[TWINS-PP] Read error: {e}")

        self._gemini_index += 1
        QtCore.QTimer.singleShot(50, self._gemini_move_next)

    # =========================================================================
    #  Post-Interferogram Processing
    # =========================================================================

    def _gemini_scan_done(self):
        """One full interferogram is complete — compute spectrum."""
        n_points = self.spin_n_points.value() if self.spin_n_points.value() > 0 else None
        w_start = self.spin_wl_start.value()
        w_stop = self.spin_wl_stop.value()

        if self._ref_mode:
            # Reference acquisition -> Compute Phase Correction
            try:
                phase, pad = self.processor.compute_phase_correction(
                    self.gemini_positions, self.current_interferogram, n_points=n_points
                )
                self.phase_correction = phase
                self.pad_length = pad
            except Exception as e:
                print(f"[ERROR] Phase cal failed: {e}")
            
            # Show Power Spectrum for Reference
            wl, spectrum = self.processor.compute_spectrum(
                self.gemini_positions, self.current_interferogram, n_points=n_points,
                wl_start=w_start, wl_stop=w_stop
            )

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
            print("[TWINS-PP] Reference spectrum acquired (Phase stored)")
            return

        # Full scan mode — compute Phase-Corrected ΔT/T
        if self.phase_correction is not None and self.pad_length is not None:
             wl, real, imag = self.processor.compute_phased_spectrum(
                self.gemini_positions, self.current_interferogram, 
                self.phase_correction, pad_length=self.pad_length,
                wl_start=w_start, wl_stop=w_stop
            )
             spectrum = real # Absorption Signal
        else:
             # Fallback
             wl, spectrum = self.processor.compute_spectrum(
                self.gemini_positions, self.current_interferogram, n_points=n_points,
                wl_start=w_start, wl_stop=w_stop
            )

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

        # Update Map (X=Time, Y=Wavelength)
        # ImageItem expects (X, Y).
        self.map_item.setImage(self.hyperspectral_map, autoLevels=(self._time_index == 0))
        
        # Fix Axis Scaling (Wavelength)
        if self.hyperspectral_map is not None:
             n_t, n_wl = self.hyperspectral_map.shape
             if n_wl > 1:
                 y_scale = (w_stop - w_start) / n_wl
                 y_origin = w_start
                 
                 self.map_item.resetTransform()
                 self.map_item.scale(1, y_scale)
                 self.map_item.setPos(0, y_origin)

        # Update Dynamics Plot if selected
        if self.selected_wl_index is not None and self.selected_wl_index < self.hyperspectral_map.shape[1]:
            dynamics = self.hyperspectral_map[:self._time_index+1, self.selected_wl_index]
            times = self.time_points[:self._time_index+1]
            self.dynamics_curve.setData(times, dynamics)
            
            # Update Vertical Line on Map
            # self.v_line.setPos(self.time_points[self._time_index]) # Optional: show current time
            

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
                 roi_datacube=np.array(self.current_roi_datacube) if self.current_roi_datacube else np.array([]),
                 # Selective
                 data_t=np.array(self.current_data_t) if hasattr(self,'current_data_t') and self.current_data_t else np.array([]),
                 data_dt=np.array(self.current_data_dt) if hasattr(self,'current_data_dt') and self.current_data_dt else np.array([]),
                 data_dtt=np.array(self.current_data_dtt) if hasattr(self,'current_data_dtt') and self.current_data_dtt else np.array([]),
                 raw_odd=np.array(self.current_raw_odd) if hasattr(self,'current_raw_odd') and self.current_raw_odd else np.array([]),
                 raw_even=np.array(self.current_raw_even) if hasattr(self,'current_raw_even') and self.current_raw_even else np.array([]))
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

    def _on_map_click(self, event):
        """Handle click on Hyperspectral Map to select Wavelength."""
        pos = event.scenePos()
        mouse_point = self.map_plot.vb.mapSceneToView(pos)
        
        # Map axes: X=Time (index?), Y=Wavelength (index?)
        # Wait, ImageItem uses INDICES (0..N, 0..M) unless scaled.
        # We haven't set scale/origin. So indices.
        
        t_idx = int(mouse_point.x())
        wl_val = mouse_point.y()
        
        if self.hyperspectral_map is None:
            return
            
        h, w = self.hyperspectral_map.shape # (Time, WL)
        
        # Convert physical WL to index
        w_start = self.spin_wl_start.value()
        w_stop = self.spin_wl_stop.value()
        
        if w > 1:
            scale = (w_stop - w_start) / w
            wl_idx = int((wl_val - w_start) / scale)
        else:
            wl_idx = 0
            
        # Clamp
        wl_idx = max(0, min(wl_idx, w - 1))
        
        if 0 <= wl_idx < w:
            self.selected_wl_index = wl_idx
            
            # Update Dynamics Plot
            dynamics = self.hyperspectral_map[:self._time_index+1, wl_idx]
            times = self.time_points[:self._time_index+1]
            # Use real time units for plot X
            self.dynamics_curve.setData(times, dynamics)
            if self.wavelengths is not None:
                # wl_val from click might be slightly off center of pixel, use actual WL
                # But self.wavelengths is the linear grid now, or approximate?
                # self.wavelengths is set to 'wl' in _scan_done.
                # If using linear interpolation, self.wavelengths is the linear grid.
                if wl_idx < len(self.wavelengths):
                    actual_wl = self.wavelengths[wl_idx]
                    self.dynamics_plot.setTitle(f"Time Dynamics @ {actual_wl:.3f} µm")
            
            # Update Crosshair
            self.h_line.setPos(wl_val) # Crosshair at physical Y
            self.v_line.setPos(t_idx)  # Crosshair at physical X (Index) 
            

    # =========================================================================
    #  Cleanup
    # =========================================================================

    def closeEvent(self, event):
        if self._scanning:
            self._stop_scan()
        self.preview_timer.stop()
        event.accept()
