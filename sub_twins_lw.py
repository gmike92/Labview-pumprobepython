"""
Twins FTIR Window — scans NIREOS Gemini stage while acquiring
via LabVIEW Experiment_manager.vi.

For each scan point:
  1. Move Gemini stage → wait for stop
  2. Trigger LabVIEW: Enum=Measure → poll Idle → read DeltaT
  3. Extract ROI mean → build interferogram
  4. Process: baseline removal, apodization, DFT → spectrum

Layout ported from opus camera sub_twins.py.

Usage:
    from sub_twins_lw import TwinsWindow
    window = TwinsWindow(labview_manager, twins_stage)
    window.show()
"""

import os
import csv
import time
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

DEFAULT_START_MM = 23.8     # Default start position (mm)
DEFAULT_STOP_MM = 24.8      # Default stop position (mm)
DEFAULT_N_STEPS = 120       # Default number of steps
DEFAULT_APODIZATION = 0.2   # Apodization width
DEFAULT_WL_START = 8.0      # Spectrum display start (µm)
DEFAULT_WL_STOP = 14.0      # Spectrum display stop (µm)

# Calibration file path
DEFAULT_CALIBRATION_FILE = r".\Twins\ASRC calibration\parameters_cal.txt"


# ============================================================================
# Spectrum Processor  (copied from opus camera)
# ============================================================================

class SpectrumProcessor:
    """
    Process interferogram to spectrum using DFT.
    Uses calibration file to convert pseudo-frequency to real wavelength.
    """

    def __init__(self, calibration_file=None):
        self.interferogram = None
        self.positions = None
        self.spectrum = None
        self.wavelengths = None
        self.freq = None

        self.calibration_file = calibration_file or DEFAULT_CALIBRATION_FILE
        self.wavelength_cal = None
        self.reciprocal_cal = None
        self._load_calibration()

    def _load_calibration(self):
        """Load calibration file for wavelength conversion."""
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
                print(f"[OK] Loaded calibration: {cal_path.name}")
                print(f"     Wavelength range: {self.wavelength_cal.min():.2f} - {self.wavelength_cal.max():.2f} µm")
            else:
                print(f"[WARN] Calibration file not found: {self.calibration_file}")
                print("       Using simple 1/frequency conversion")

        except Exception as e:
            print(f"[WARN] Error loading calibration: {e}")

    def set_data(self, positions, interferogram):
        self.positions = positions
        self.interferogram = interferogram

    def moving_average(self, data, window):
        if window < 2:
            return np.zeros_like(data)
        import pandas as pd
        ser = pd.Series(data)
        return ser.rolling(window=window, min_periods=1, center=True).mean().to_numpy()

    def apodization(self, data, positions, width=0.2):
        """Apply Gaussian apodization window (NIREOS formula)."""
        center_idx = np.argmax(np.abs(data))
        
        try:
            print(f"[SpectrumProcessor] Computed ZERO (burst center): {positions[center_idx]:.4f} mm (index {center_idx})")
        except:
            pass
            
        # Shift positions so that center burst is mathematically exactly 0
        shifted_positions = positions - positions[center_idx]
            
        left_pos = shifted_positions[:center_idx + 1]
        right_pos = shifted_positions[center_idx + 1:]

        if len(left_pos) > 0 and left_pos[0] != 0:
            left_gauss = np.exp(-np.power(left_pos, 2) /
                                (2 * np.power(left_pos[0] * width * 2, 2)))
        else:
            left_gauss = np.ones_like(left_pos)

        if len(right_pos) > 0 and right_pos[-1] != 0:
            right_gauss = np.exp(-np.power(right_pos, 2) /
                                 (2 * np.power(right_pos[-1] * width * 2, 2)))
        else:
            right_gauss = np.ones_like(right_pos)

        window = np.concatenate([left_gauss, right_gauss])

        if len(window) != len(data):
            window = np.interp(
                np.linspace(0, 1, len(data)),
                np.linspace(0, 1, len(window)),
                window
            )

        return data * window

    def _get_frequency_limits(self, wl_start, wl_stop):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            start_freq = fn(1.0 / wl_stop)
            end_freq = fn(1.0 / wl_start)
            return float(start_freq), float(end_freq)
        else:
            return 1.0 / wl_stop, 1.0 / wl_start

    def _freq_to_wavelength(self, frequencies):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(self.reciprocal_cal, 1.0 / self.wavelength_cal,
                          kind="linear", fill_value="extrapolate")
            inv_wavelength = fn(frequencies)
            return 1.0 / inv_wavelength
        else:
            return 1.0 / frequencies

    def compute_spectrum(self, wl_start=8.0, wl_stop=14.0,
                         apod_width=0.2, n_points=10000, invert=False, symmetrize=False):
        """Compute spectrum from interferogram using DFT."""
        if self.interferogram is None or self.positions is None:
            return None, None

        window_size = max(1, len(self.interferogram) // 5)
        baseline = self.moving_average(self.interferogram, window_size)
        signal = self.interferogram - baseline

        if invert:
            signal = -signal

        c_positions = self.positions
        
        if symmetrize:
            c_idx = np.argmax(np.abs(signal))
            left_len = c_idx
            right_len = len(signal) - 1 - c_idx
            
            if right_len > left_len:
                tail = signal[c_idx + 1:]
                sym_signal = np.concatenate([tail[::-1], [signal[c_idx]], tail])
                pos_diffs = c_positions[c_idx + 1:] - c_positions[c_idx]
                mirrored_pos = c_positions[c_idx] - pos_diffs[::-1]
                sym_positions = np.concatenate([mirrored_pos, [c_positions[c_idx]], c_positions[c_idx + 1:]])
            else:
                tail = signal[:c_idx]
                sym_signal = np.concatenate([tail, [signal[c_idx]], tail[::-1]])
                pos_diffs = c_positions[c_idx] - c_positions[:c_idx]
                mirrored_pos = c_positions[c_idx] + pos_diffs[::-1]
                sym_positions = np.concatenate([c_positions[:c_idx], [c_positions[c_idx]], mirrored_pos])
            
            signal = sym_signal
            c_positions = sym_positions
            self.center_idx = len(signal) // 2

        self.symmetrized_signal = signal
        apodized = self.apodization(signal, c_positions, apod_width)

        start_freq, end_freq = self._get_frequency_limits(wl_start, wl_stop)
        frequencies = np.linspace(end_freq, start_freq, n_points)

        pos = c_positions.reshape(-1, 1)
        dpos = np.diff(c_positions)
        dpos = np.append(dpos, dpos[-1] if len(dpos) > 0 else 0)

        phase = -2j * np.pi * pos * frequencies
        spectrum = (dpos * apodized).dot(np.exp(phase))
        spectrum = np.abs(spectrum)

        wavelengths = self._freq_to_wavelength(frequencies)

        self.freq = frequencies
        self.wavelengths = wavelengths
        self.spectrum = spectrum
        self.apodized_signal = apodized
        self.apodized_positions = c_positions

        return wavelengths, spectrum


# ============================================================================
# Twins Window  (LabVIEW version)
# ============================================================================

class TwinsWindow(QtWidgets.QWidget):
    """
    Twins FTIR window: scan Gemini stage + LabVIEW camera acquisition.

    Layout:
      Left panel  – stage control, scan range, acquisition, processing, save/load
      Right panel – interferogram plot (top) + spectrum plot (bottom)
    """

    def __init__(self, manager: LabVIEWManager, twins_stage, live_window=None, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.stage = twins_stage
        self.live_window = live_window  # for ROI / pixel signal extraction
        self.processor = SpectrumProcessor()

        # Scan state
        self.scanning = False
        self.scan_positions = None
        self.interferogram = None
        self.wavelengths = None
        self.spectrum = None
        self.scan_index = 0
        self.scan_index = 0
        self.scan_csv_path = None

        # Position update timer
        self.pos_timer = QtCore.QTimer()
        self.pos_timer.timeout.connect(self._update_position_display)
        self.pos_timer.start(500)

        self.setWindowTitle("Twins FTIR (LabVIEW)")
        self.resize(1100, 750)
        self._setup_ui()

    # =========================================================================
    #  UI
    # =========================================================================

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # ====== Left Panel: Controls ======
        control_panel = QtWidgets.QWidget()
        control_panel.setMaximumWidth(380)
        control_layout = QtWidgets.QVBoxLayout(control_panel)

        # Title
        title = QtWidgets.QLabel("NIREOS Gemini FTIR")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #9C27B0; padding: 4px;"
        )
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(title)

        # ====== Stage Control ======
        stage_group = QtWidgets.QGroupBox("Stage Control")
        stage_layout = QtWidgets.QGridLayout(stage_group)

        stage_layout.addWidget(QtWidgets.QLabel("Current Position:"), 0, 0)
        self.lbl_current_pos = QtWidgets.QLabel("-- mm")
        self.lbl_current_pos.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #2196F3;"
        )
        stage_layout.addWidget(self.lbl_current_pos, 0, 1)

        stage_layout.addWidget(QtWidgets.QLabel("Move To (mm):"), 1, 0)
        self.spin_move_to = QtWidgets.QDoubleSpinBox()
        self.spin_move_to.setRange(0.0, 50.0)
        self.spin_move_to.setValue(19.0)
        self.spin_move_to.setDecimals(2)
        self.spin_move_to.setSingleStep(0.1)
        stage_layout.addWidget(self.spin_move_to, 1, 1)

        self.btn_move = QtWidgets.QPushButton("🎯 Move Stage")
        self.btn_move.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 5px;"
        )
        self.btn_move.clicked.connect(self._move_stage)
        stage_layout.addWidget(self.btn_move, 2, 0, 1, 2)

        control_layout.addWidget(stage_group)

        # ====== Scan Range ======
        range_group = QtWidgets.QGroupBox("Scan Range")
        range_layout = QtWidgets.QGridLayout(range_group)

        range_layout.addWidget(QtWidgets.QLabel("Start (mm):"), 0, 0)
        self.spin_start = QtWidgets.QDoubleSpinBox()
        self.spin_start.setRange(0.0, 50.0)
        self.spin_start.setValue(DEFAULT_START_MM)
        self.spin_start.setDecimals(2)
        self.spin_start.setSingleStep(0.5)
        self.spin_start.valueChanged.connect(self._update_step_display)
        range_layout.addWidget(self.spin_start, 0, 1)

        range_layout.addWidget(QtWidgets.QLabel("Stop (mm):"), 1, 0)
        self.spin_stop = QtWidgets.QDoubleSpinBox()
        self.spin_stop.setRange(0.0, 50.0)
        self.spin_stop.setValue(DEFAULT_STOP_MM)
        self.spin_stop.setDecimals(2)
        self.spin_stop.setSingleStep(0.5)
        self.spin_stop.valueChanged.connect(self._update_step_display)
        range_layout.addWidget(self.spin_stop, 1, 1)

        range_layout.addWidget(QtWidgets.QLabel("Number of Steps:"), 2, 0)
        self.spin_n_steps = QtWidgets.QSpinBox()
        self.spin_n_steps.setRange(2, 10000)
        self.spin_n_steps.setValue(DEFAULT_N_STEPS)
        self.spin_n_steps.valueChanged.connect(self._update_step_display)
        range_layout.addWidget(self.spin_n_steps, 2, 1)
        
        # Sample Name
        range_layout.addWidget(QtWidgets.QLabel("Sample Name:"), 3, 0)
        self.txt_sample_name = QtWidgets.QLineEdit()
        self.txt_sample_name.setPlaceholderText("Enter sample name...")
        range_layout.addWidget(self.txt_sample_name, 3, 1)

        range_layout.addWidget(QtWidgets.QLabel("Step Size:"), 4, 0)
        self.lbl_step_size = QtWidgets.QLabel("-- µm")
        self.lbl_step_size.setStyleSheet("font-weight: bold;")
        range_layout.addWidget(self.lbl_step_size, 3, 1)

        control_layout.addWidget(range_group)

        # ====== Acquisition Settings ======
        acq_settings = QtWidgets.QGroupBox("Acquisition Settings")
        acq_layout = QtWidgets.QGridLayout(acq_settings)

        acq_layout.addWidget(QtWidgets.QLabel("Frames/Point:"), 0, 0)
        self.spin_frames = QtWidgets.QSpinBox()
        self.spin_frames.setRange(2, 10000)
        self.spin_frames.setValue(100)
        acq_layout.addWidget(self.spin_frames, 0, 1)

        control_layout.addWidget(acq_settings)

        # ====== Processing Settings ======
        proc_settings = QtWidgets.QGroupBox("Processing Settings")
        proc_layout = QtWidgets.QGridLayout(proc_settings)
        
        proc_layout.addWidget(QtWidgets.QLabel("Wavelength Start:"), 0, 0)
        self.spin_wl_start = QtWidgets.QDoubleSpinBox()
        self.spin_wl_start.setRange(0.1, 50.0)
        self.spin_wl_start.setValue(8.0)
        proc_layout.addWidget(self.spin_wl_start, 0, 1)

        proc_layout.addWidget(QtWidgets.QLabel("Wavelength Stop:"), 1, 0)
        self.spin_wl_stop = QtWidgets.QDoubleSpinBox()
        self.spin_wl_stop.setRange(0.1, 50.0)
        self.spin_wl_stop.setValue(14.0)
        proc_layout.addWidget(self.spin_wl_stop, 1, 1)

        proc_layout.addWidget(QtWidgets.QLabel("Freq Points:"), 2, 0)
        self.spin_n_points = QtWidgets.QSpinBox()
        self.spin_n_points.setRange(50, 10000)
        self.spin_n_points.setValue(1000)
        proc_layout.addWidget(self.spin_n_points, 2, 1)

        proc_layout.addWidget(QtWidgets.QLabel("Apodization:"), 3, 0)
        self.spin_apod = QtWidgets.QDoubleSpinBox()
        self.spin_apod.setRange(0.01, 5.0)
        self.spin_apod.setValue(0.2)
        self.spin_apod.setSingleStep(0.1)
        proc_layout.addWidget(self.spin_apod, 3, 1)

        self.chk_invert_ifg = QtWidgets.QCheckBox("Invert IFG (+45/-45)")
        self.chk_invert_ifg.setChecked(False)
        proc_layout.addWidget(self.chk_invert_ifg, 4, 0, 1, 2)

        self.chk_asymmetric = QtWidgets.QCheckBox("Asymmetric Scan (Symmetrize)")
        self.chk_asymmetric.setChecked(False)
        proc_layout.addWidget(self.chk_asymmetric, 5, 0, 1, 2)

        control_layout.addWidget(proc_settings)
        
        # Save Configure
        save_mode_group = QtWidgets.QGroupBox("Data Configuration")
        save_mode_layout = QtWidgets.QVBoxLayout(save_mode_group)
        
        # Row 1: Combos
        combo_row = QtWidgets.QHBoxLayout()
        combo_row.addWidget(QtWidgets.QLabel("ROI Mode:"))
        self.cmb_save_mode = QtWidgets.QComboBox()
        self.cmb_save_mode.addItems(["Single Pixel", "ROI Average", "ROI (2D)", "Full Frame (2D)"])
        self.cmb_save_mode.setCurrentIndex(1) # Default ROI Avg
        combo_row.addWidget(self.cmb_save_mode)
        
        combo_row.addWidget(QtWidgets.QLabel("Plot:"))
        self.cmb_plot_mode = QtWidgets.QComboBox()
        self.cmb_plot_mode.addItems(["DeltaT (dT/T)", "Transmission (T)", "DeltaT (dT)"])
        self.cmb_plot_mode.setCurrentIndex(1) # Default to Transmission (T)
        combo_row.addWidget(self.cmb_plot_mode)
        
        save_mode_layout.addLayout(combo_row)
        
        # Load and bind QSettings
        self._settings = QtCore.QSettings('Polimi', 'HybridCamera')
        
        # Load values
        try:
            self.spin_start.setValue(float(self._settings.value('twins_start', self.spin_start.value())))
            self.spin_stop.setValue(float(self._settings.value('twins_stop', self.spin_stop.value())))
            self.spin_n_steps.setValue(int(self._settings.value('twins_steps', self.spin_n_steps.value())))
            self.spin_wl_start.setValue(float(self._settings.value('twins_wl_start', self.spin_wl_start.value())))
            self.spin_wl_stop.setValue(float(self._settings.value('twins_wl_stop', self.spin_wl_stop.value())))
            self.spin_n_points.setValue(int(self._settings.value('twins_n_points', self.spin_n_points.value())))
            self.spin_apod.setValue(float(self._settings.value('twins_apod', self.spin_apod.value())))
            self.chk_invert_ifg.setChecked(str(self._settings.value('twins_invert_ifg', self.chk_invert_ifg.isChecked())).lower() == 'true')
            self.chk_asymmetric.setChecked(str(self._settings.value('twins_asymmetric', self.chk_asymmetric.isChecked())).lower() == 'true')
            self.txt_sample_name.setText(str(self._settings.value('twins_sample_name', self.txt_sample_name.text())))
        except Exception:
            pass
            
        def save_settings(*args):
            self._settings.setValue('twins_start', self.spin_start.value())
            self._settings.setValue('twins_stop', self.spin_stop.value())
            self._settings.setValue('twins_steps', self.spin_n_steps.value())
            self._settings.setValue('twins_wl_start', self.spin_wl_start.value())
            self._settings.setValue('twins_wl_stop', self.spin_wl_stop.value())
            self._settings.setValue('twins_n_points', self.spin_n_points.value())
            self._settings.setValue('twins_apod', self.spin_apod.value())
            self._settings.setValue('twins_invert_ifg', self.chk_invert_ifg.isChecked())
            self._settings.setValue('twins_asymmetric', self.chk_asymmetric.isChecked())
            self._settings.setValue('twins_sample_name', self.txt_sample_name.text())
            
        self.spin_start.valueChanged.connect(save_settings)
        self.spin_stop.valueChanged.connect(save_settings)
        self.spin_n_steps.valueChanged.connect(save_settings)
        self.spin_wl_start.valueChanged.connect(save_settings)
        self.spin_wl_stop.valueChanged.connect(save_settings)
        self.spin_n_points.valueChanged.connect(save_settings)
        self.spin_apod.valueChanged.connect(save_settings)
        self.chk_invert_ifg.toggled.connect(save_settings)
        self.chk_asymmetric.toggled.connect(save_settings)
        self.txt_sample_name.textChanged.connect(save_settings)
        
        # Row 2: Checkboxes
        check_row = QtWidgets.QHBoxLayout()
        self.chk_save_t = QtWidgets.QCheckBox("Save T")
        self.chk_save_dt = QtWidgets.QCheckBox("Save dT")
        self.chk_save_dtt = QtWidgets.QCheckBox("Save dT/T")
        self.chk_save_raw = QtWidgets.QCheckBox("Save Raw")
        # Default
        self.chk_save_dtt.setChecked(True)
        
        check_row.addWidget(self.chk_save_t)
        check_row.addWidget(self.chk_save_dt)
        check_row.addWidget(self.chk_save_dtt)
        check_row.addWidget(self.chk_save_raw)
        
        save_mode_layout.addLayout(check_row)

        self.btn_bg = QtWidgets.QPushButton("Acquire Background")
        self.btn_bg.setStyleSheet("background-color: #607D8B; color: white;")
        self.btn_bg.clicked.connect(self._acquire_background)
        save_mode_layout.addWidget(self.btn_bg)
        
        control_layout.addWidget(save_mode_group)

        # ====== Acquisition Controls ======
        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_ctrl_layout = QtWidgets.QVBoxLayout(acq_group)

        self.btn_start = QtWidgets.QPushButton("▶ Start Scan")
        self.btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px; border-radius: 6px;"
        )
        self.btn_start.clicked.connect(self._start_scan)
        acq_ctrl_layout.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("■ Stop Scan")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; "
            "padding: 10px; border-radius: 6px;"
        )
        self.btn_stop.clicked.connect(self._stop_scan)
        acq_ctrl_layout.addWidget(self.btn_stop)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        acq_ctrl_layout.addWidget(self.progress_bar)

        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        acq_ctrl_layout.addWidget(self.lbl_status)

        control_layout.addWidget(acq_group)

        # ====== Processing ======
        proc_group = QtWidgets.QGroupBox("Spectrum Processing")
        proc_layout = QtWidgets.QGridLayout(proc_group)

        proc_layout.addWidget(QtWidgets.QLabel("Apodization:"), 0, 0)
        self.spin_apod = QtWidgets.QDoubleSpinBox()
        self.spin_apod.setRange(0.01, 1.0)
        self.spin_apod.setValue(DEFAULT_APODIZATION)
        self.spin_apod.setDecimals(2)
        self.spin_apod.setSingleStep(0.05)
        proc_layout.addWidget(self.spin_apod, 0, 1)

        proc_layout.addWidget(QtWidgets.QLabel("WL Start (µm):"), 1, 0)
        self.spin_wl_start = QtWidgets.QDoubleSpinBox()
        self.spin_wl_start.setRange(1.0, 30.0)
        self.spin_wl_start.setValue(DEFAULT_WL_START)
        self.spin_wl_start.setDecimals(1)
        proc_layout.addWidget(self.spin_wl_start, 1, 1)

        proc_layout.addWidget(QtWidgets.QLabel("WL Stop (µm):"), 2, 0)
        self.spin_wl_stop = QtWidgets.QDoubleSpinBox()
        self.spin_wl_stop.setRange(1.0, 30.0)
        self.spin_wl_stop.setValue(DEFAULT_WL_STOP)
        self.spin_wl_stop.setDecimals(1)
        proc_layout.addWidget(self.spin_wl_stop, 2, 1)

        self.btn_process = QtWidgets.QPushButton("🔬 Compute Spectrum")
        self.btn_process.clicked.connect(self._compute_spectrum)
        proc_layout.addWidget(self.btn_process, 3, 0, 1, 2)

        control_layout.addWidget(proc_group)

        # ====== Save / Load ======
        io_group = QtWidgets.QGroupBox("Data")
        io_layout = QtWidgets.QHBoxLayout(io_group)

        self.btn_save = QtWidgets.QPushButton("💾 Save")
        self.btn_save.clicked.connect(self._save_data)
        io_layout.addWidget(self.btn_save)

        self.btn_load = QtWidgets.QPushButton("📂 Load")
        self.btn_load.clicked.connect(self._load_data)
        io_layout.addWidget(self.btn_load)

        control_layout.addWidget(io_group)

        control_layout.addStretch()
        layout.addWidget(control_panel)

        # ====== Right Panel: Plots ======
        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Interferogram plot
        interf_widget = QtWidgets.QWidget()
        interf_layout = QtWidgets.QVBoxLayout(interf_widget)
        interf_layout.setContentsMargins(0, 0, 0, 0)

        interf_label = QtWidgets.QLabel("Interferogram")
        interf_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        interf_layout.addWidget(interf_label)

        self.plot_interf = pg.PlotWidget()
        self.plot_interf.setLabel('left', 'Intensity (mean ΔT/T)')
        self.plot_interf.setLabel('bottom', 'Position (mm)')
        self.plot_interf.showGrid(x=True, y=True, alpha=0.3)
        self.curve_interf = self.plot_interf.plot(pen='y', name="Raw IFG")
        self.curve_sym = self.plot_interf.plot(pen='c', name="Symmetric IFG")
        self.curve_apod = self.plot_interf.plot(pen='r', name="Apodized IFG")
        interf_layout.addWidget(self.plot_interf)

        plot_splitter.addWidget(interf_widget)

        # Spectrum plot
        spec_widget = QtWidgets.QWidget()
        spec_layout = QtWidgets.QVBoxLayout(spec_widget)
        spec_layout.setContentsMargins(0, 0, 0, 0)

        spec_label = QtWidgets.QLabel("Spectrum")
        spec_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        spec_layout.addWidget(spec_label)

        self.plot_spec = pg.PlotWidget()
        self.plot_spec.setLabel('left', 'Intensity')
        self.plot_spec.setLabel('bottom', 'Wavelength (µm)')
        self.plot_spec.showGrid(x=True, y=True, alpha=0.3)
        self.curve_spec = self.plot_spec.plot(pen='c')
        spec_layout.addWidget(self.plot_spec)

        plot_splitter.addWidget(spec_widget)

        layout.addWidget(plot_splitter, stretch=1)

        # Initialize step display
        self._update_step_display()

    # =========================================================================
    #  Helpers
    # =========================================================================

    def _update_position_display(self):
        if self.stage and self.stage.is_connected:
            try:
                pos = self.stage.get_position()
                self.lbl_current_pos.setText(f"{pos:.3f} mm")
            except Exception:
                self.lbl_current_pos.setText("err")
        else:
            self.lbl_current_pos.setText("-- mm")

    def _update_step_display(self):
        start = self.spin_start.value()
        stop = self.spin_stop.value()
        n_steps = self.spin_n_steps.value()
        if n_steps > 1:
            step_mm = abs(stop - start) / (n_steps - 1)
            step_um = step_mm * 1000
            self.lbl_step_size.setText(f"{step_um:.1f} µm")
        else:
            self.lbl_step_size.setText("-- µm")

    def _move_stage(self):
        if not self.stage or not self.stage.is_connected:
            self.lbl_status.setText("Stage not connected!")
            return
        target = self.spin_move_to.value()
        self.lbl_status.setText(f"Moving to {target:.2f} mm...")
        self.btn_move.setEnabled(False)

        import threading
        def do_move():
            self.stage.move_to(target)
            self.stage.wait_for_stop()

        t = threading.Thread(target=do_move, daemon=True)
        t.start()
        QtCore.QTimer.singleShot(500, lambda: self.btn_move.setEnabled(True))
        QtCore.QTimer.singleShot(500, lambda: self.lbl_status.setText("Ready"))

    # =========================================================================
    #  Scan Logic  (non-blocking, timer-based like sub_pumpprobe_lw)
    # =========================================================================

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
    
    def _start_scan(self):
        if not self.stage or not self.stage.is_connected:
            self.lbl_status.setText("Stage not connected!")
            return
        if not self.manager.vi:
            self.lbl_status.setText("LabVIEW Manager not running!")
            return

        start = self.spin_start.value()
        stop = self.spin_stop.value()
        n_steps = self.spin_n_steps.value()
        self.scan_positions = np.linspace(start, stop, n_steps)
        self.interferogram = np.zeros(n_steps)
        self.roi_datacube = []  # list of 2D ROI slices per point
        
        # Selective Lists
        self.data_t = []
        self.data_dt = []
        self.data_dtt = []
        self.raw_odd = []
        self.raw_even = []
        
        self.scan_index = 0

        # Generate Standardized Paths
        timestamp = datetime.now()
        date_dir = timestamp.strftime(r"D:\pumpprobedata\%Y\%m\%d")
        os.makedirs(date_dir, exist_ok=True)
        
        sample = self.txt_sample_name.text().strip()
        if not sample: sample = "sample"
        # Sanitize
        sample = "".join(x for x in sample if x.isalnum() or x in " -_")
        
        base_name = f"{sample}_twins_{timestamp.strftime('%H%M%S')}"
        self.scan_csv_path = os.path.join(date_dir, base_name + ".csv")
        self.scan_npy_path = os.path.join(date_dir, base_name + ".npy") # Store for end of scan

        with open(self.scan_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["position_mm", "actual_mm", "mean_signal"])

        # UI
        self.scanning = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Scanning...")
        self.curve_interf.setData([], [])
        self.curve_spec.setData([], [])

        print(f"[TWINS] Starting scan: {start:.2f} → {stop:.2f} mm, {n_steps} steps")
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
        print("[TWINS] Scan stopped by user")

    def _move_to_next(self):
        """Move stage to next scan position."""
        if not self.scanning:
            return
        if self.scan_index >= len(self.scan_positions):
            self._scan_complete()
            return

        target_mm = self.scan_positions[self.scan_index]
        self.lbl_status.setText(
            f"Point {self.scan_index+1}/{len(self.scan_positions)}: "
            f"Moving to {target_mm:.3f} mm"
        )

        try:
            self.stage.move_to(target_mm)
        except Exception as e:
            self.lbl_status.setText(f"Stage error: {e}")
            self.scan_index += 1
            QtCore.QTimer.singleShot(10, self._move_to_next)
            return

        # Stage move is blocking. Trigger acquire immediately.
        # No need to poll for stability unless vibration is severe.
        QtCore.QTimer.singleShot(10, self._trigger_acquire)

    # _poll_stage removed (redundant)

    def _trigger_acquire(self):
        """Trigger LabVIEW Measure at current stage position."""
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
            
            # Wait N ms + 20ms buffer, then poll every 20ms
            wait_ms = n + 20
            QtCore.QTimer.singleShot(wait_ms, lambda: self._acq_timer.start(20))

        except Exception as e:
            self.lbl_status.setText(f"Acquire error: {e}")
            self.scan_index += 1
            QtCore.QTimer.singleShot(10, self._move_to_next)

    def _poll_acquire(self):
        """Poll LabVIEW until Enum → Idle, then read DeltaT."""
        if not self.scanning:
            self._acq_timer.stop()
            return

        self._acq_waited += 0.05
        if self._acq_waited > 60.0:
            self._acq_timer.stop()
            self.lbl_status.setText("Acquire timeout!")
            self.scan_index += 1
            QtCore.QTimer.singleShot(10, self._move_to_next)
            return

        try:
            if self.manager.vi.GetControlValue("Enum") != CMD_IDLE:
                return
        except Exception:
            return

        # Done — read result
        self._acq_timer.stop()
        pos_mm = self.scan_positions[self.scan_index]

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
                img_dtt = np.divide(even - odd, odd, out=np.zeros_like(odd), where=np.abs(odd) > 1.0) * 100.0
                
                # Compute based on Plot Mode
                pmode = self.cmb_plot_mode.currentIndex()
                if pmode == 1:   img = img_t
                elif pmode == 2: img = img_dt
                else:            img = img_dtt
                
                # Extract Signal for Plot
                val = self._extract(img)
                self.interferogram[self.scan_index] = val
                
                # Update Interferogram Plot
                if self.scan_index > 0:
                     self.curve_interf.setData(
                         self.scan_positions[:self.scan_index+1],
                         self.interferogram[:self.scan_index+1]
                     )

                # Store Selective Data
                # Use _extract_to_save to get ROI/Pixel/Full
                if self.chk_save_t.isChecked():
                    self.data_t.append(self._extract_to_save(img_t))
                if self.chk_save_dt.isChecked():
                    self.data_dt.append(self._extract_to_save(img_dt))
                if self.chk_save_dtt.isChecked():
                    self.data_dtt.append(self._extract_to_save(img_dtt))
                if self.chk_save_raw.isChecked():
                    self.raw_odd.append(self._extract_to_save(odd))
                    self.raw_even.append(self._extract_to_save(even))

                # Legacy ROI datacube (stores whatever is plotted/selected? Or just img?)
                # Original code didn't append to roi_datacube in _poll_acquire?!
                # Wait, looking at sub_twins_lw.py lines 643... it inits roi_datacube=[]
                # But where is it appended?
                # I don't see it appended in the viewed code!
                # Maybe I should append 'img' (selected mode) to roi_datacube for backward compat?
                self.roi_datacube.append(self._extract_to_save(img))
                
                print(f"[TWINS] Point {self.scan_index+1}: {pos_mm:.3f} mm, val={val:.4e}")
                
                # CSV append
                try:
                    actual_mm = self.stage.get_position()
                except Exception:
                    actual_mm = pos_mm
                with open(self.scan_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{pos_mm:.4f}", f"{actual_mm:.4f}", f"{val:.8e}"])

                print(f"[TWINS] Point {self.scan_index+1}: {pos_mm:.3f} mm = {val:.4e}")
            else:
                print(f"[TWINS] No data at point {self.scan_index+1}")

        except Exception as e:
            print(f"[TWINS] Read error: {e}")

        # Next
        self.scan_index += 1
        pct = int(100 * self.scan_index / len(self.scan_positions))
        self.progress_bar.setValue(pct)

        QtCore.QTimer.singleShot(50, self._move_to_next)

    def _scan_complete(self):
        """Scan finished — compute spectrum and save."""
        self.scanning = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Scan complete!")

        # Auto-compute spectrum
        self._compute_spectrum()

        # Save final npz
        if self.scan_csv_path:
            # Change extension to .npz
            npz_path = self.scan_csv_path.replace('.csv', '.npz')
            
            save_dict = {
                'positions': self.scan_positions,
                'interferogram': self.interferogram,
                'wavelengths': self.wavelengths,
                'spectrum': self.spectrum,
                'roi_datacube': np.array(self.roi_datacube) if self.roi_datacube else np.array([]),
                # Selective
                'data_t': np.array(self.data_t) if hasattr(self,'data_t') and self.data_t else np.array([]),
                'data_dt': np.array(self.data_dt) if hasattr(self,'data_dt') and self.data_dt else np.array([]),
                'data_dtt': np.array(self.data_dtt) if hasattr(self,'data_dtt') and self.data_dtt else np.array([]),
                'raw_odd': np.array(self.raw_odd) if hasattr(self,'raw_odd') and self.raw_odd else np.array([]),
                'raw_even': np.array(self.raw_even) if hasattr(self,'raw_even') and self.raw_even else np.array([])
            }

            np.savez(npz_path, **save_dict)
            print(f"[TWINS] Saved final npz: {npz_path}")

    # =========================================================================
    #  Spectrum Processing
    # =========================================================================

    def _compute_spectrum(self):
        if self.interferogram is None or self.scan_positions is None:
            self.lbl_status.setText("No interferogram data!")
            return

        self.lbl_status.setText("Computing spectrum...")
        try:
            self.processor.set_data(self.scan_positions, self.interferogram)
        except AttributeError:
            self.processor.positions = self.scan_positions
            self.processor.interferogram = self.interferogram
        
        sym_flag = hasattr(self, 'chk_asymmetric') and self.chk_asymmetric.isChecked()

        wavelengths, spectrum = self.processor.compute_spectrum(
            wl_start=self.spin_wl_start.value(),
            wl_stop=self.spin_wl_stop.value(),
            apod_width=self.spin_apod.value(),
            n_points=self.spin_n_points.value(),
            invert=self.chk_invert_ifg.isChecked(),
            symmetrize=sym_flag
        )

        if wavelengths is not None and spectrum is not None:
            self.wavelengths = wavelengths
            self.spectrum = spectrum
            self.curve_spec.setData(wavelengths, spectrum)
            
            apod_pos = getattr(self.processor, 'apodized_positions', self.scan_positions)
            
            if sym_flag and hasattr(self.processor, 'symmetrized_signal'):
                self.curve_sym.setData(apod_pos, self.processor.symmetrized_signal)
                self.curve_sym.setVisible(True)
            else:
                self.curve_sym.setData([], [])
                self.curve_sym.setVisible(False)
            
            if hasattr(self.processor, 'apodized_signal') and self.processor.apodized_signal is not None:
                self.curve_apod.setData(apod_pos, self.processor.apodized_signal)
                
            self.lbl_status.setText("Spectrum computed!")
        else:
            self.lbl_status.setText("Spectrum computation failed")

    # =========================================================================
    #  Save / Load
    # =========================================================================

    def _save_data(self):
        if self.interferogram is None:
            self.lbl_status.setText("No data to save!")
            return

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", "", "NumPy Files (*.npy);;All Files (*)"
        )
        if filepath:
            data = {
                'positions': self.scan_positions,
                'interferogram': self.interferogram,
                'wavelengths': self.wavelengths,
                'spectrum': self.spectrum,
                'start_mm': self.spin_start.value(),
                'stop_mm': self.spin_stop.value(),
                'n_steps': self.spin_n_steps.value(),
                'apodization': self.spin_apod.value(),
            }
            np.save(filepath, data, allow_pickle=True)
            self.lbl_status.setText(f"Saved: {Path(filepath).name}")

    def _load_data(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Data", "", "NumPy Files (*.npy);;All Files (*)"
        )
        if filepath:
            try:
                data = np.load(filepath, allow_pickle=True).item()
                self.scan_positions = data.get('positions')
                self.interferogram = data.get('interferogram')
                self.wavelengths = data.get('wavelengths')
                self.spectrum = data.get('spectrum')

                if self.scan_positions is not None and self.interferogram is not None:
                    self.curve_interf.setData(self.scan_positions, self.interferogram)
                if self.wavelengths is not None and self.spectrum is not None:
                    self.curve_spec.setData(self.wavelengths, self.spectrum)

                if 'start_mm' in data:
                    self.spin_start.setValue(data['start_mm'])
                if 'stop_mm' in data:
                    self.spin_stop.setValue(data['stop_mm'])
                if 'n_steps' in data:
                    self.spin_n_steps.setValue(data['n_steps'])
                if 'apodization' in data:
                    self.spin_apod.setValue(data['apodization'])

                self.lbl_status.setText(f"Loaded: {Path(filepath).name}")

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
