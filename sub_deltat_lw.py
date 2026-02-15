"""
DeltaT Window — Live camera view with Thorlabs delay stage control.

Combines continuous LabVIEW Getframe streaming (like Live View) with
manual delay stage position controls (absolute move + relative steps).

Features:
  - Continuous ΔT/T or T image display via LabVIEW
  - Delay stage: position display (mm + fs), absolute move, relative step buttons
  - Click pixel → row/column profiles
  - Autoscale + Save buttons

Layout ported from opus camera sub_deltat.py.

Usage:
    from sub_deltat_lw import DeltaTWindow
    window = DeltaTWindow(labview_manager, delay_stage)
    window.show()
"""

import numpy as np
import os
from datetime import datetime

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from labview_manager import LabVIEWManager, CMD_IDLE, CMD_GETFRAME, CMD_MEASURE

# Physics
SPEED_OF_LIGHT_MM_FS = 0.000299792458
GLOBAL_ZERO_POS_MM = 140.0

# Performance
PROFILE_THROTTLE = 3
MAX_HISTORY = 500


class DeltaTWindow(QtWidgets.QWidget):
    """
    Live ΔT/T view with Thorlabs delay stage control.

    Layout:
      Left  – acquisition controls + display settings + pixel info
      Middle – delay stage controls (position, move, step buttons)
      Right – image display + time trace
    """

    def __init__(self, manager: LabVIEWManager, delay_stage, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.delay_stage = delay_stage

        # Acquisition state
        self.acquiring = False
        self.batch_count = 0
        self.acquiring = False
        self.batch_count = 0
        self.current_img = None
        
        # Background subtraction
        self._awaiting_background = False
        self.background_img = None

        # Pixel tracking
        self.sel_row = None
        self.sel_col = None
        self._first_frame = True

        # Pre-allocated temporal history
        self._hist_len = 0
        self._pixel_hist = np.zeros(MAX_HISTORY, dtype=np.float64)

        self.setWindowTitle("Delta T (LabVIEW + Stage)")
        self.resize(1200, 800)
        self._setup_ui()

        # Stage position update timer
        self.stage_timer = QtCore.QTimer()
        self.stage_timer.timeout.connect(self._update_stage_position)
        self.stage_timer.start(500)

    # =========================================================================
    #  UI
    # =========================================================================

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # ====== Left Panel: Acquisition Controls ======
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(280)
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        # Title
        title = QtWidgets.QLabel("DeltaT / T Measurement")
        title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #FF9800; padding: 5px;"
        )
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        # Mode selector
        mode_group = QtWidgets.QGroupBox("Display Mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["DeltaT (dT/T)", "Transmission (T)", "DeltaT (dT)"])
        mode_layout.addWidget(self.mode_combo)
        left_layout.addWidget(mode_group)

        # Settings
        settings_group = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QGridLayout(settings_group)

        settings_layout.addWidget(QtWidgets.QLabel("Frames:"), 0, 0)
        self.spin_frames = QtWidgets.QSpinBox()
        self.spin_frames.setRange(2, 10000)
        self.spin_frames.setValue(100)
        self.spin_frames.setSingleStep(2)
        settings_layout.addWidget(self.spin_frames, 0, 1)

        self.chk_invert = QtWidgets.QCheckBox("Invert Phase")
        settings_layout.addWidget(self.chk_invert, 1, 0, 1, 2)

        left_layout.addWidget(settings_group)

        # Acquisition buttons
        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)

        self.btn_start = QtWidgets.QPushButton(" Start Live")
        self.btn_start.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; "
            "font-size: 14px; padding: 10px; border-radius: 5px;"
        )
        self.btn_start.clicked.connect(self.start_live)
        acq_layout.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton(" Stop")
        self.btn_stop.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; "
            "padding: 10px; border-radius: 5px;"
        )
        self.btn_stop.clicked.connect(self.stop_live)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setEnabled(False)
        acq_layout.addWidget(self.btn_stop)
        
        self.btn_bg = QtWidgets.QPushButton("🚫 Background")
        self.btn_bg.setStyleSheet("background-color: #607D8B; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_bg.clicked.connect(self._acquire_background)
        self.btn_bg.clicked.connect(self._acquire_background)
        acq_layout.addWidget(self.btn_bg)

        # Sample Name
        acq_layout.addWidget(QtWidgets.QLabel("Sample Name:"))
        self.txt_sample_name = QtWidgets.QLineEdit()
        self.txt_sample_name.setPlaceholderText("Enter sample name...")
        acq_layout.addWidget(self.txt_sample_name)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_autoscale = QtWidgets.QPushButton("Autoscale")
        self.btn_autoscale.setStyleSheet(
            "background-color: #7C4DFF; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_autoscale.clicked.connect(self._autoscale)
        btn_row.addWidget(self.btn_autoscale)

        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_save.clicked.connect(self._save_image)
        btn_row.addWidget(self.btn_save)
        acq_layout.addLayout(btn_row)

        left_layout.addWidget(acq_group)

        # Pixel info
        pixel_group = QtWidgets.QGroupBox("Pixel Info")
        pixel_layout = QtWidgets.QVBoxLayout(pixel_group)

        self.lbl_pixel = QtWidgets.QLabel("Click image to select pixel")
        self.lbl_pixel.setStyleSheet("font-weight: bold;")
        pixel_layout.addWidget(self.lbl_pixel)

        self.lbl_pixel_value = QtWidgets.QLabel("Value: --")
        pixel_layout.addWidget(self.lbl_pixel_value)

        self.lbl_frame_count = QtWidgets.QLabel("Frames: 0")
        pixel_layout.addWidget(self.lbl_frame_count)

        left_layout.addWidget(pixel_group)

        # Status
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        left_layout.addWidget(self.lbl_status)

        left_layout.addStretch()
        layout.addWidget(left_panel)

        # ====== Middle Panel: Delay Stage Controls ======
        stage_panel = QtWidgets.QWidget()
        stage_panel.setMaximumWidth(220)
        stage_layout = QtWidgets.QVBoxLayout(stage_panel)

        stage_title = QtWidgets.QLabel("Translation Stage")
        stage_title.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #2196F3;"
        )
        stage_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        stage_layout.addWidget(stage_title)

        # Position display
        pos_group = QtWidgets.QGroupBox("Position")
        pos_layout = QtWidgets.QGridLayout(pos_group)

        pos_layout.addWidget(QtWidgets.QLabel("Actual (mm):"), 0, 0)
        self.lbl_stage_mm = QtWidgets.QLabel("---.---")
        self.lbl_stage_mm.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #2196F3;"
        )
        pos_layout.addWidget(self.lbl_stage_mm, 0, 1)

        pos_layout.addWidget(QtWidgets.QLabel("Actual (fs):"), 1, 0)
        self.lbl_stage_fs = QtWidgets.QLabel("------")
        self.lbl_stage_fs.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #9C27B0;"
        )
        pos_layout.addWidget(self.lbl_stage_fs, 1, 1)

        stage_layout.addWidget(pos_group)

        # Absolute move
        abs_group = QtWidgets.QGroupBox("Absolute Move")
        abs_layout = QtWidgets.QGridLayout(abs_group)

        abs_layout.addWidget(QtWidgets.QLabel("Position (mm):"), 0, 0)
        self.spin_abs_mm = QtWidgets.QDoubleSpinBox()
        self.spin_abs_mm.setRange(0.0, 300.0)
        self.spin_abs_mm.setValue(GLOBAL_ZERO_POS_MM)
        self.spin_abs_mm.setDecimals(3)
        self.spin_abs_mm.setSingleStep(0.01)
        abs_layout.addWidget(self.spin_abs_mm, 0, 1)

        self.btn_go = QtWidgets.QPushButton("Go")
        self.btn_go.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_go.clicked.connect(self._move_absolute)
        abs_layout.addWidget(self.btn_go, 1, 0, 1, 2)

        stage_layout.addWidget(abs_group)

        # Relative step buttons
        step_group = QtWidgets.QGroupBox("Relative Steps")
        step_layout_grid = QtWidgets.QGridLayout(step_group)

        # Negative steps (top row)
        steps_neg = [("-10 ps", -10000), ("-1 ps", -1000), ("-100 fs", -100)]
        for col, (label, fs) in enumerate(steps_neg):
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet("padding: 6px;")
            btn.clicked.connect(lambda checked, d=fs: self._move_relative_fs(d))
            step_layout_grid.addWidget(btn, 0, col)

        # Positive steps (bottom row)
        steps_pos = [("+100 fs", 100), ("+1 ps", 1000), ("+10 ps", 10000)]
        for col, (label, fs) in enumerate(steps_pos):
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet("padding: 6px;")
            btn.clicked.connect(lambda checked, d=fs: self._move_relative_fs(d))
            step_layout_grid.addWidget(btn, 1, col)

        stage_layout.addWidget(step_group)

        # Stage status
        self.lbl_stage_status = QtWidgets.QLabel("Not connected")
        self.lbl_stage_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_stage_status.setStyleSheet("color: #888;")
        stage_layout.addWidget(self.lbl_stage_status)

        stage_layout.addStretch()
        layout.addWidget(stage_panel)

        # ====== Right Panel: Display ======
        display_panel = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_panel)

        # Image + Histogram
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.img_view = self.graphics_widget.addPlot(title="Delta T / T")
        self.img_item = pg.ImageItem()
        self.img_view.addItem(self.img_item)

        # Histogram
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.graphics_widget.addItem(self.hist)

        # Click handler
        self.img_view.scene().sigMouseClicked.connect(self._on_image_click)

        # Crosshairs
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen='y')
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.img_view.addItem(self.crosshair_v)
        self.img_view.addItem(self.crosshair_h)

        # Colormap — diverging for DeltaT
        try:
            cmap = pg.colormap.get('RdBu')
            self.img_item.setLookupTable(cmap.getLookupTable())
        except Exception:
            pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            color = np.array([
                [0, 0, 180, 255], [100, 150, 255, 255],
                [255, 255, 255, 255],
                [255, 150, 100, 255], [180, 0, 0, 255],
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(pos, color)
            self.img_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

        display_layout.addWidget(self.graphics_widget, stretch=3)

        # Time trace plot
        trace_group = QtWidgets.QGroupBox("Pixel Time Trace")
        trace_layout_box = QtWidgets.QVBoxLayout(trace_group)
        self.trace_plot = pg.PlotWidget()
        self.trace_plot.setLabel('left', 'Intensity')
        self.trace_plot.setLabel('bottom', 'Frame #')
        self.trace_plot.showGrid(x=True, y=True, alpha=0.3)
        self.trace_curve = self.trace_plot.plot(pen='y')
        trace_layout_box.addWidget(self.trace_plot)
        display_layout.addWidget(trace_group, stretch=1)

        # Stats bar
        stats_layout = QtWidgets.QHBoxLayout()
        self.lbl_min = QtWidgets.QLabel("Min: --")
        self.lbl_max = QtWidgets.QLabel("Max: --")
        self.lbl_mean = QtWidgets.QLabel("Mean: --")
        stats_layout.addWidget(self.lbl_min)
        stats_layout.addWidget(self.lbl_max)
        stats_layout.addWidget(self.lbl_mean)
        stats_layout.addStretch()
        display_layout.addLayout(stats_layout)

        layout.addWidget(display_panel, stretch=1)

        # Poll timer (started on Start Live)
        self.poll_timer = QtCore.QTimer()
        self.poll_timer.timeout.connect(self._poll_data)

    # =========================================================================
    #  Stage Control
    # =========================================================================

    def _update_stage_position(self):
        if self.delay_stage and self.delay_stage.is_connected:
            try:
                pos_mm = self.delay_stage.get_position()
                self.lbl_stage_mm.setText(f"{pos_mm:.3f}")
                delay_fs = (pos_mm - GLOBAL_ZERO_POS_MM) / SPEED_OF_LIGHT_MM_FS * 2.0
                self.lbl_stage_fs.setText(f"{delay_fs:.0f}")
                self.lbl_stage_status.setText("Connected")
                self.lbl_stage_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            except Exception:
                self.lbl_stage_mm.setText("err")
                self.lbl_stage_fs.setText("err")
        else:
            self.lbl_stage_status.setText("Not connected")
            self.lbl_stage_status.setStyleSheet("color: #888;")

    def _move_absolute(self):
        if not self.delay_stage or not self.delay_stage.is_connected:
            self.lbl_status.setText("Stage not connected!")
            return
        target = self.spin_abs_mm.value()
        self.lbl_status.setText(f"Moving to {target:.3f} mm...")
        self.btn_go.setEnabled(False)

        import threading
        def do_move():
            self.delay_stage.move_to(target)
        t = threading.Thread(target=do_move, daemon=True)
        t.start()
        QtCore.QTimer.singleShot(1000, lambda: self.btn_go.setEnabled(True))
        QtCore.QTimer.singleShot(1000, lambda: self.lbl_status.setText("Ready"))

    def _move_relative_fs(self, delta_fs):
        if not self.delay_stage or not self.delay_stage.is_connected:
            self.lbl_status.setText("Stage not connected!")
            return
        delta_mm = delta_fs * SPEED_OF_LIGHT_MM_FS / 2.0
        self.lbl_status.setText(f"Step: {delta_fs:+.0f} fs ({delta_mm:+.4f} mm)...")

        import threading
        def do_move():
            self.delay_stage.move_relative(delta_mm)
        t = threading.Thread(target=do_move, daemon=True)
        t.start()
        QtCore.QTimer.singleShot(500, lambda: self.lbl_status.setText("Ready"))

    # =========================================================================
    #  Live Acquisition (Getframe streaming, same as sub_live_lw.py)
    # =========================================================================

    def start_live(self):
        if not self.manager.vi:
            self.lbl_status.setText("Manager not running!")
            return

        vi = self.manager.vi
        n = self.spin_frames.value()
        vi.SetControlValue("N", n)
        vi.SetControlValue("Acq Trigger", True)
        vi.SetControlValue("Stoplive", False)
        vi.SetControlValue("Enum", CMD_GETFRAME)

        self.acquiring = True
        self.batch_count = 0
        self._first_frame = True
        self._hist_len = 0

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Live")
        self.lbl_status.setStyleSheet("color: #FF9800; font-weight: bold;")

        self.poll_timer.start(50)

    def stop_live(self):
        self.acquiring = False
        self.poll_timer.stop()

        if self.manager.vi:
            try:
                self.manager.vi.SetControlValue("Stoplive", True)
            except Exception:
                pass

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopped")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

    def _poll_data(self):
        if not self.acquiring:
            self.poll_timer.stop()
            return
            
        vi = self.manager.vi
        mode = self.mode_combo.currentIndex()
        
        # Read Odd/Even frames (Python calculation is more robust than getting "T"/"DeltaT")
        odd_val = vi.GetControlValue("Odd")
        even_val = vi.GetControlValue("Even")
        
        if odd_val is None or even_val is None:
            self.lbl_status.setText("Status: Waiting for Odd/Even data...")
            return

        try:
            odd = np.array(odd_val, dtype=float)
            even = np.array(even_val, dtype=float)
            
            # Handle Background Acquisition (Before subtraction)
            if self._awaiting_background:
                if mode == 0: # DeltaT: use Odd as background (approx)
                    bg_frame = odd.copy()
                else: # T: use Average of Odd/Even
                    bg_frame = (odd + even) / 2.0
                
                self.manager.background = bg_frame
                self._awaiting_background = False
                self.lbl_status.setText("Status: Background acquired")
                QtWidgets.QMessageBox.information(self, "Background", "Background acquired!")

            # Apply Background Subtraction
            bg = self.manager.background
            debug_bg_mean = 0.0
            
            if bg is not None and bg.shape == odd.shape:
                odd -= bg
                even -= bg
                debug_bg_mean = np.mean(bg)
                
            # Compute Image based on mode
            if mode == 0:  # DeltaT (dT/T)
                # (Even - Odd) / Odd. Zero out where Odd is low (background).
                img = np.divide(even - odd, odd, out=np.zeros_like(odd), where=np.abs(odd) > 1.0)
            elif mode == 2: # DeltaT (dT)
                # Even - Odd
                img = even - odd
            else:  # T
                # (Odd + Even) / 2
                img = (odd + even) / 2.0
            
            # Update Status with Debug Info
            debug_t_mean = np.mean(img)
            self.lbl_status.setText(f"Status: Plot Mean={debug_t_mean:.2f} | Bkg Mean={debug_bg_mean:.2f}")

            # Skip empty data
            if img.size == 0:
                return

            if img.ndim == 1:
                side = int(np.sqrt(img.size))
                if side * side == img.size:
                    img = img.reshape(side, side)
                else:
                    return
            elif img.ndim != 2:
                return

            self.current_img = img
            self.batch_count += 1

            # Image update
            if self._first_frame:
                self.img_item.setImage(img.T, autoLevels=True)
                self._first_frame = False
            else:
                self.img_item.setImage(img.T, autoLevels=False)

            self.lbl_frame_count.setText(f"Frames: {self.batch_count}")

            # Stats
            if self.batch_count % PROFILE_THROTTLE == 0:
                self.lbl_min.setText(f"Min: {np.nanmin(img):.4e}")
                self.lbl_max.setText(f"Max: {np.nanmax(img):.4e}")
                self.lbl_mean.setText(f"Mean: {np.nanmean(img):.4e}")
                self._update_time_trace()

        except Exception as e:
            self.lbl_status.setText(f"Read error: {e}")

    def _acquire_background(self):
        """Prompt user to block beam, then acquire background frame. Or clear."""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Acquire Background")
        msg.setText("Please block the probe beam.\n\nClick OK to acquire new background.\nClick Reset to clear existing background.")
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Ok | 
            QtWidgets.QMessageBox.StandardButton.Cancel | 
            QtWidgets.QMessageBox.StandardButton.Reset
        )
        msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
        
        reply = msg.exec()
        
        if reply == QtWidgets.QMessageBox.StandardButton.Ok:
            self._awaiting_background = True
            self.lbl_status.setText("Status: Acquiring background...")
        elif reply == QtWidgets.QMessageBox.StandardButton.Reset:
            self.background_img = None
            self._awaiting_background = False
            self.lbl_status.setText("Status: Background cleared")

    # =========================================================================
    #  Pixel Tracking
    # =========================================================================

    def _on_image_click(self, event):
        vb = self.img_view.vb
        pos = event.scenePos()
        if not self.img_view.sceneBoundingRect().contains(pos):
            return
        mp = vb.mapSceneToView(pos)
        col, row = int(mp.x()), int(mp.y())

        if self.current_img is not None:
            h, w = self.current_img.shape
            if 0 <= row < h and 0 <= col < w:
                self.sel_row = row
                self.sel_col = col

                self.crosshair_v.setPos(col + 0.5)
                self.crosshair_h.setPos(row + 0.5)
                self.crosshair_v.setVisible(True)
                self.crosshair_h.setVisible(True)

                val = self.current_img[row, col]
                self.lbl_pixel.setText(f"Pixel: ({row}, {col})")
                self.lbl_pixel_value.setText(f"Value: {val:.6e}")

                self._hist_len = 0

    def _update_time_trace(self):
        if self.sel_row is None or self.current_img is None:
            return
        r, c = self.sel_row, self.sel_col
        h, w = self.current_img.shape
        if r >= h or c >= w:
            return

        val = self.current_img[r, c]
        if self._hist_len < MAX_HISTORY:
            self._pixel_hist[self._hist_len] = val
            self._hist_len += 1
        else:
            self._pixel_hist[:-1] = self._pixel_hist[1:]
            self._pixel_hist[-1] = val

        self.trace_curve.setData(self._pixel_hist[:self._hist_len])
        self.lbl_pixel_value.setText(f"Value: {val:.6e}")

    # =========================================================================
    #  Autoscale / Save
    # =========================================================================

    def _autoscale(self):
        if self.current_img is not None:
            self.img_item.setImage(self.current_img.T, autoLevels=True)
        self.trace_plot.enableAutoRange()

    def _save_image(self):
        if self.current_img is None:
            return

        # Generate Standardized Paths
        timestamp = datetime.now()
        date_dir = timestamp.strftime(r"D:\pumpprobedata\%Y\%m\%d")
        os.makedirs(date_dir, exist_ok=True)
        
        sample = self.txt_sample_name.text().strip()
        if not sample: sample = "sample"
        # Sanitize
        sample = "".join(x for x in sample if x.isalnum() or x in " -_")

        mode_tag = "dT" if self.mode_combo.currentIndex() == 0 else "T"
        # timestamp = time.strftime("%Y%m%d_%H%M%S") # Use global ts

        # Include stage position in filename
        pos_tag = ""
        if self.delay_stage and self.delay_stage.is_connected:
            try:
                pos = self.delay_stage.get_position()
                pos_tag = f"_{pos:.3f}mm"
            except Exception:
                pass

        filename = f"{sample}_deltat_{mode_tag}{pos_tag}_{timestamp.strftime('%H%M%S')}.npy"
        filepath = os.path.join(date_dir, filename)

        np.save(filepath, self.current_img)
        self.lbl_status.setText(f"Saved: {filename}")
        print(f"[Live] Saved image to {filepath}")

    # =========================================================================
    #  Cleanup
    # =========================================================================

    def closeEvent(self, event):
        self.stage_timer.stop()
        if self.acquiring:
            self.stop_live()
        event.accept()
