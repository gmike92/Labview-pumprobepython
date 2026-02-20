"""
Live View Window — continuous camera feed via LabVIEW Experiment_manager.vi.

Features:
  - Continuous Getframe streaming with Stoplive control
  - Click on pixel → row profile, column profile, temporal evolution
  - ROI selection → time evolution of ROI mean
  - Crosshair overlay on selected pixel

Performance notes:
  - Image autoLevels disabled after first frame (histogram stays manual)
  - Side-plots throttled to every 3rd frame
  - ROI uses direct numpy slicing (not getArrayRegion)
  - Temporal traces use pre-allocated numpy arrays

Usage:
    from sub_live_lw import LiveViewWindow
    window = LiveViewWindow(labview_manager)
    window.show()
"""

import os
import time
import numpy as np
import threading
from collections import deque

try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    import pyqtgraph as pg
    # pyqtgraph.Qt is often used for compatibility, but if PyQt6 is explicitly imported,
    # pyqtgraph might automatically use it. Keeping the original pyqtgraph.Qt import
    # for robustness if pyqtgraph doesn't auto-detect.
    from pyqtgraph.Qt import QtCore, QtWidgets as pg_QtWidgets # Renamed to avoid conflict
except ImportError:
    # Fallback if PyQt6 is not directly available or pyqtgraph is missing
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets, QtGui # QtGui might be needed by pyqtgraph.Qt
    except ImportError:
        raise ImportError("pyqtgraph required: pip install pyqtgraph pyqt6")

from labview_manager import LabVIEWManager, CMD_IDLE, CMD_GETFRAME


# Max history length for temporal traces
MAX_HISTORY = 500

# Update side-plots every N frames (1 = every frame, 3 = every 3rd)
PROFILE_THROTTLE = 3


class LiveViewWindow(QtWidgets.QWidget):
    """
    Live camera view with pixel analysis and ROI tracking.
    """
    
    def __init__(self, manager, delay_stage=None):
        super().__init__()
        self.manager = manager
        self.delay_stage = delay_stage
        self.acquiring = False
        self.batch_count = 0
        
        # Current image (kept for profile extraction)
        self.current_img = None
        
        # Selected pixel
        self.sel_row = None
        self.sel_col = None
        
        # Pre-allocated temporal history (numpy arrays, faster than deque→list)
        self._hist_len = 0
        self._pixel_hist = np.zeros(MAX_HISTORY, dtype=np.float64)
        self._roi_hist = np.zeros(MAX_HISTORY, dtype=np.float64)
        self._time_hist = np.zeros(MAX_HISTORY, dtype=np.float64)
        
        # Pre-allocated profile arrays (avoid re-creating each frame)
        self._row_x = None
        self._col_x = None
        
        # First-frame flag (for autoLevels)
        self._first_frame = True
        
        self.setWindowTitle("Live View (LabVIEW)")
        self.resize(1200, 900)
        self._setup_ui()
        self._init_stage_polling()
    
    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # =====================================================================
        # Controls
        # =====================================================================
        ctrl_group = QtWidgets.QGroupBox("Acquisition Controls")
        ctrl_layout = QtWidgets.QGridLayout(ctrl_group)
        main_layout.addWidget(ctrl_group)
        
        # Row 0: Frames + Mode + Invert
        ctrl_layout.addWidget(QtWidgets.QLabel("Frames (N):"), 0, 0)
        self.frames_spin = QtWidgets.QSpinBox()
        self.frames_spin.setRange(2, 10000)
        self.frames_spin.setValue(100)
        self.frames_spin.setSingleStep(2)
        ctrl_layout.addWidget(self.frames_spin, 0, 1)
        
        ctrl_layout.addWidget(self.frames_spin, 0, 1)
        
        # Sample Name
        ctrl_layout.addWidget(QtWidgets.QLabel("Sample:"), 0, 2)
        self.txt_sample_name = QtWidgets.QLineEdit()
        self.txt_sample_name.setPlaceholderText("Sample name...")
        ctrl_layout.addWidget(self.txt_sample_name, 0, 3)

        ctrl_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 4)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["ΔT/T (dT)", "Transmission (T)", "DeltaT (dT)", "Even (Pump On)"])
        self.mode_combo.currentIndexChanged.connect(self._change_mode)
        ctrl_layout.addWidget(self.mode_combo, 0, 5)
        
        self.invert_chk = QtWidgets.QCheckBox("Invert Phase")
        self.invert_chk = QtWidgets.QCheckBox("Invert Phase")
        ctrl_layout.addWidget(self.invert_chk, 0, 6)
        
        # ROI / Pixel toggle
        self.roi_toggle = QtWidgets.QPushButton("Mode: ROI")
        self.roi_toggle.setCheckable(True)
        self.roi_toggle.setChecked(True)   # ROI is default
        self.roi_toggle.setStyleSheet(
            "QPushButton { background-color: #009688; color: white; "
            "font-weight: bold; padding: 6px 12px; border-radius: 4px; }"
            "QPushButton:checked { background-color: #009688; }"
            "QPushButton:!checked { background-color: #E91E63; }"
        )
        self.roi_toggle.clicked.connect(self._toggle_signal_mode)
        ctrl_layout.addWidget(self.roi_toggle, 0, 7)
        
        # Row 1: Start / Stop + info
        self.start_btn = QtWidgets.QPushButton(" ▶  START LIVE  ")
        self.start_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; "
            "font-size: 14px; padding: 8px 20px;"
        )
        self.start_btn.clicked.connect(self.start_live)
        ctrl_layout.addWidget(self.start_btn, 1, 0, 1, 2)
        
        self.stop_btn = QtWidgets.QPushButton("■ STOP")
        self.stop_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; "
            "font-size: 14px; padding: 8px 20px;"
        )
        self.stop_btn.clicked.connect(self.stop_live)
        self.stop_btn.setEnabled(False)
        ctrl_layout.addWidget(self.stop_btn, 1, 2, 1, 2)
        
        self.autoscale_btn = QtWidgets.QPushButton("⟳ Autoscale")
        self.autoscale_btn.setStyleSheet(
            "background-color: #7C4DFF; color: white; font-weight: bold; "
            "font-size: 12px; padding: 6px 12px;"
        )
        self.autoscale_btn.clicked.connect(self._autoscale)
        ctrl_layout.addWidget(self.autoscale_btn, 1, 4)
        
        self.save_btn = QtWidgets.QPushButton("💾 Save Image")
        self.save_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "font-size: 12px; padding: 6px 12px;"
        )
        self.save_btn.clicked.connect(self._save_image)
        ctrl_layout.addWidget(self.save_btn, 1, 5)
        
        self.bg_btn = QtWidgets.QPushButton("🚫 Background")
        self.bg_btn.setToolTip("Acquire background (block probe)")
        self.bg_btn.clicked.connect(self._acquire_background)
        ctrl_layout.addWidget(self.bg_btn, 1, 6)
        
        self.batch_label = QtWidgets.QLabel("Frames: 0")
        ctrl_layout.addWidget(self.batch_label, 1, 7)
        
        # =====================================================================
        # Status + Pixel Info
        # =====================================================================
        info_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(info_layout)
        
        self.status_label = QtWidgets.QLabel("Status: Ready")
        info_layout.addWidget(self.status_label)
        
        self.pixel_label = QtWidgets.QLabel("Pixel: click on image")
        self.pixel_label.setStyleSheet("color: #888; font-style: italic;")
        info_layout.addWidget(self.pixel_label)
        info_layout.addStretch()
        
        # =====================================================================
        # Main Area: Image + Profile Plots
        # =====================================================================
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)
        
        # --- Left: Camera image with crosshair ---
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.img_layout_widget = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.img_layout_widget)
        
        self.img_view = self.img_layout_widget.addPlot(title="Camera Feed")
        self.img_view.setAspectLocked(False)
        self.img_item = pg.ImageItem()
        self.img_view.addItem(self.img_item)
        
        # Crosshair lines
        self.h_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('y', width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.v_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.h_line.setVisible(False)
        self.v_line.setVisible(False)
        self.img_view.addItem(self.h_line)
        self.img_view.addItem(self.v_line)
        
        # ROI (draggable rectangle)
        self.roi = pg.RectROI([20, 20], [30, 30], pen=pg.mkPen('c', width=2))
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.img_view.addItem(self.roi)
        
        # Histogram — disable auto-range for speed
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.img_layout_widget.addItem(self.hist)
        
        # Connect click
        self.img_item.scene().sigMouseClicked.connect(self._on_click)
        
        splitter.addWidget(left_widget)
        
        # --- Right: Profile plots (stacked) ---
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Row profile
        self.row_plot = pg.PlotWidget(title="Row Profile")
        self.row_plot.setLabel('bottom', 'Column')
        self.row_plot.setLabel('left', 'Value')
        self.row_plot.showGrid(x=True, y=True, alpha=0.3)
        self.row_plot.disableAutoRange()
        self.row_curve = self.row_plot.plot(pen=pg.mkPen('#FF5722', width=1.5))
        right_layout.addWidget(self.row_plot)
        
        # Column profile
        self.col_plot = pg.PlotWidget(title="Column Profile")
        self.col_plot.setLabel('bottom', 'Row')
        self.col_plot.setLabel('left', 'Value')
        self.col_plot.showGrid(x=True, y=True, alpha=0.3)
        self.col_plot.disableAutoRange()
        self.col_curve = self.col_plot.plot(pen=pg.mkPen('#4CAF50', width=1.5))
        right_layout.addWidget(self.col_plot)
        
        # Temporal evolution (pixel + ROI)
        self.time_plot = pg.PlotWidget(title="Temporal Evolution")
        self.time_plot.setLabel('bottom', 'Frame #')
        self.time_plot.setLabel('left', 'Value')
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.time_plot.addLegend()
        self.pixel_time_curve = self.time_plot.plot(
            pen=pg.mkPen('#2196F3', width=1.5), name='Pixel'
        )
        self.roi_time_curve = self.time_plot.plot(
            pen=pg.mkPen('#FF9800', width=1.5), name='ROI mean'
        )
        right_layout.addWidget(self.time_plot)
        
        splitter.addWidget(right_widget)
        
        # =====================================================================
        # Stage Control Panel (Right Side, collapsible or fixed?)
        # Let's add it to the right of the plot area or below controls
        # The user requested "same as live+stage" (sub_deltat_lw.py style)
        # =====================================================================
        
        stage_group = QtWidgets.QGroupBox("Delay Stage Control")
        stage_layout = QtWidgets.QVBoxLayout(stage_group)
        
        # 1. Position Display
        self.lbl_stage_pos = QtWidgets.QLabel("Position: -- mm\nDelay: -- fs")
        self.lbl_stage_pos.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_stage_pos.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
        stage_layout.addWidget(self.lbl_stage_pos)

        # 2. Pump/Probe Toggle (Moved up)
        pp_layout = QtWidgets.QHBoxLayout()
        self.rb_pump = QtWidgets.QRadioButton("Pump (on Stage)")
        self.rb_probe = QtWidgets.QRadioButton("Probe (on Stage)")
        self.rb_pump.setChecked(True)
        self.rb_pump.setToolTip("Stage moves Pump path")
        self.rb_probe.setToolTip("Stage moves Probe path")
        self.rb_pump.toggled.connect(self._update_stage_ui)
        
        pp_layout.addWidget(self.rb_pump)
        pp_layout.addWidget(self.rb_probe)
        stage_layout.addLayout(pp_layout)

        # (Removed old Pump/Probe toggle location)
        
        # 4. Absolute Move
        abs_layout = QtWidgets.QHBoxLayout()
        self.spin_stage_target = QtWidgets.QDoubleSpinBox()
        self.spin_stage_target.setRange(-50.0, 350.0) # mm
        self.spin_stage_target.setDecimals(4)
        self.spin_stage_target.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.btn_stage_go = QtWidgets.QPushButton("Go")
        self.btn_stage_go.clicked.connect(self._move_stage_abs)
        abs_layout.addWidget(QtWidgets.QLabel("Target:"))
        abs_layout.addWidget(self.spin_stage_target)
        abs_layout.addWidget(self.btn_stage_go)
        stage_layout.addLayout(abs_layout)

        # 5. Relative Steps (Specific fs values)
        # c = 0.00029979 mm/fs. Delay = 2*dx/c => dx = Delay*c/2
        # 1 fs -> 0.00015 mm
        # 10 fs -> 0.0015 mm
        # 100 fs -> 0.015 mm
        # 1 ps (1000 fs) -> 0.15 mm
        # 10 ps (10000 fs) -> 1.5 mm
        
        step_layout = QtWidgets.QGridLayout()
        # Labels
        step_layout.addWidget(QtWidgets.QLabel("Step (fs):"), 0, 0, 1, 4)
        
        steps_fs = [10, 100, 1000, 10000]
        c = 0.000299792458
        
        # Negative Steps
        # Negative Steps
        for i, fs in enumerate(steps_fs):
            lbl = f"-{fs}fs" if fs < 1000 else f"-{fs/1000:.0f}ps"
            btn = QtWidgets.QPushButton(lbl)
            btn.clicked.connect(lambda _, val=-fs: self._move_stage_by_fs(val))
            step_layout.addWidget(btn, 1, i)

        # Positive Steps
        for i, fs in enumerate(steps_fs):
            lbl = f"+{fs}fs" if fs < 1000 else f"+{fs/1000:.0f}ps"
            btn = QtWidgets.QPushButton(lbl)
            btn.clicked.connect(lambda _, val=fs: self._move_stage_by_fs(val))
            step_layout.addWidget(btn, 2, i)
            
        stage_layout.addLayout(step_layout)
        
        # 6. Zero
        zero_layout = QtWidgets.QHBoxLayout()
        self.lbl_zero = QtWidgets.QLabel("Zero: 0.000 mm")
        self.spin_zero = QtWidgets.QDoubleSpinBox()
        self.spin_zero.setDecimals(4)
        self.spin_zero.setRange(-50, 350)
        if self.delay_stage:
            self.spin_zero.setValue(self.delay_stage.zero_position)
        else:
            self.spin_zero.setValue(140.0) # Default fallback
            
        def on_zero_changed(val):
            if self.delay_stage:
                self.delay_stage.zero_position = val
            self._update_stage_ui()
            
        self.spin_zero.valueChanged.connect(on_zero_changed)
        zero_layout.addWidget(QtWidgets.QLabel("Zero:"))
        zero_layout.addWidget(self.spin_zero)
        stage_layout.addLayout(zero_layout)

        right_layout.addWidget(stage_group)
        # Move plots down or adjust layout? 
        # Actually, adding stage_group to right_layout puts it BELOW the plots.
        # Ideally it should be separate or collapsible.
        # But this works for "adding control".


        
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 500])
        
        # Set initial colormap
        self._change_mode(0)
        
        # Poll timer
        self.poll_timer = QtCore.QTimer()
        self.poll_timer.timeout.connect(self._poll_data)
    
    # =========================================================================
    # Click Handling
    # =========================================================================
    
    def _on_click(self, event):
        """Handle click on image → select pixel, update crosshair."""
        if self.current_img is None:
            return
        
        pos = event.scenePos()
        mouse_point = self.img_view.vb.mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        
        h, w = self.current_img.shape
        if 0 <= row < h and 0 <= col < w:
            self.sel_row = row
            self.sel_col = col
            
            # Update crosshair
            self.h_line.setValue(row)
            self.v_line.setValue(col)
            self.h_line.setVisible(True)
            self.v_line.setVisible(True)
            
            val = self.current_img[row, col]
            self.pixel_label.setText(f"Pixel: ({row}, {col}) = {val:.6e}")
            self.pixel_label.setStyleSheet("color: #FFD600; font-weight: bold;")
            
            # Reset temporal history for new pixel
            self._hist_len = 0
            
            # Update profiles immediately
            self._update_profiles()
            self._enable_auto_range_once()
    
    def _enable_auto_range_once(self):
        """Re-enable autoRange on profile plots for one update, then disable."""
        self.row_plot.enableAutoRange()
        self.col_plot.enableAutoRange()
    
    def _autoscale(self):
        """Recalculate image levels and re-enable autoRange on all plots."""
        # Re-apply autoLevels on current image
        if self.current_img is not None:
            self.img_item.setImage(self.current_img.T, autoLevels=True)
        
        # Re-enable autoRange on all side-plots
        self.row_plot.enableAutoRange()
        self.col_plot.enableAutoRange()
        self.time_plot.enableAutoRange()
    
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
            self.status_label.setText("Status: Acquiring global background...")
        elif reply == QtWidgets.QMessageBox.StandardButton.Reset:
            self.manager.background = None
            self._awaiting_background = False
            self.status_label.setText("Status: Global Background cleared")
    
    def _save_image(self):
        """Save the current frame to disk as .npy (numpy array)."""
        if self.current_img is None:
            self.status_label.setText("Status: No image to save")
            return
        
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_images")
        os.makedirs(save_dir, exist_ok=True)
        
        idx = self.mode_combo.currentIndex()
        tags = ["dT_T", "T", "dT", "Even"]
        mode_tag = tags[idx] if idx < len(tags) else "img"
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"liveview_{mode_tag}_{timestamp}.npy"
        filepath = os.path.join(save_dir, filename)
        
        np.save(filepath, self.current_img)
        self.status_label.setText(f"Status: Saved → {filename}")
        print(f"[SAVE] {filepath}")
    
    # =========================================================================
    # Live View Control
    # =========================================================================
    
    def start_live(self):
        """Start continuous Getframe loop."""
        if not self.manager.vi:
            self.status_label.setText("Status: Manager not started!")
            return
        if not self.manager.camera_initialized:
            self.status_label.setText("Status: Camera not initialized!")
            return
        
        self.acquiring = True
        self.batch_count = 0
        self._hist_len = 0
        self._first_frame = True
        self._awaiting_background = False
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        vi = self.manager.vi
        n = self.frames_spin.value()
        
        try:
            vi.SetControlValue("N", n)
            vi.SetControlValue("Acq Trigger", True)
            vi.SetControlValue("Stoplive", False)
            vi.SetControlValue("Enum", CMD_GETFRAME) 
            
            self.status_label.setText("Status: Live view active")
            self.poll_timer.start(100) # Slower poll is fine for Getframe
            
            # Stats
            if self.batch_count % PROFILE_THROTTLE == 0:
                self._update_profiles()
                
        except Exception as e:
            self.status_label.setText(f"Status: Error — {e}")
            self.acquiring = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def _poll_data(self):
        """Read latest frame and update plots. Side-plots throttled."""
        if not self.acquiring:
            self.poll_timer.stop()
            return
        
        vi = self.manager.vi
        mode = self.mode_combo.currentIndex()
        
        try:
            # For CMD_GETFRAME, LabVIEW loops continuously. 
            # We just peek at the controls.
            # No need to check CMD_IDLE (it will stay in Getframe/2).
            
            # --- READ DATA ---
            # --- READ DATA (from previous measurement) ---
            bg = self.manager.background
            
            # All modes use Odd/Even
            odd_val = vi.GetControlValue("Odd")
            even_val = vi.GetControlValue("Even")
            
            if odd_val is None or even_val is None:
                self.status_label.setText("Status: Waiting for Odd/Even data...")
                return

            odd = np.array(odd_val, dtype=float)
            even = np.array(even_val, dtype=float)
            
            # Handle Background Acquisition (Global)
            if self._awaiting_background:
                # Store average of Odd/Even as background (Dark Frame)
                # Or use different logic per mode? 
                # Ideally background is blocked beam -> Dark Frame.
                # So Avg(Odd, Even) is good estimate of dark noise.
                # Store BOTH Odd and Even for Scattering Correction
                self.manager.background = (odd.copy(), even.copy())
                bg_data = self.manager.background
                self._awaiting_background = False
                self.status_label.setText("Status: Global Background acquired (Scattering Mode)")
                QtWidgets.QMessageBox.information(self, "Background", "Global Background acquired! (Odd/Even stored separately)")
            
            debug_bg_mean = 0.0
            # Subtract Background
            if bg is not None:
                # New Mode: Tuple (odd_bg, even_bg)
                if isinstance(bg, (tuple, list)) and len(bg) == 2:
                    bg_odd, bg_even = bg
                    if bg_odd.shape == odd.shape and bg_even.shape == even.shape:
                        odd -= bg_odd
                        even -= bg_even
                        debug_bg_mean = np.mean(bg_even)
                # Legacy / Single Frame
                elif hasattr(bg, 'shape') and bg.shape == odd.shape:
                    odd -= bg
                    even -= bg
                    debug_bg_mean = np.mean(bg)
            
            # --- COMPUTE IMAGE ---
            if mode == 0: # dT/T
                # (Even - Odd) / Odd. Zero out where Odd is low.
                denom = np.where(np.abs(odd) > 1e-10, odd, 1e-10)
                img = (even - odd) / denom
                
            elif mode == 2: # dT (Reference - Signal approx, or just diff)
                img = even - odd 
                
            elif mode == 3: # Even (Pump On)
                img = even
                
            else: # T (mode 1)
                img = (odd + even) / 2.0
            
            # Update Status with Debug Info
            debug_t_mean = np.mean(img)
            self.status_label.setText(f"Status: Mode={mode} | Mean={debug_t_mean:.2e} | Bkg={debug_bg_mean:.2e}")
                

            
            # Skip empty data
            if img.size == 0:
                return
            
            # Reshape if 1D
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
            
            # --- IMAGE UPDATE (every frame) ---
            # First frame: autoLevels so histogram sets good range
            # After that: autoLevels=False (fast, reuses existing levels)
            if self._first_frame:
                self.img_item.setImage(img.T, autoLevels=True)
                self._first_frame = False
                # Cache profile x-axes
                h, w = img.shape
                self._row_x = np.arange(w, dtype=np.float64)
                self._col_x = np.arange(h, dtype=np.float64)
                # Enable auto-range for first profile update
                self._enable_auto_range_once()
            else:
                self.img_item.setImage(img.T, autoLevels=False)
            
            self.batch_label.setText(f"Frames: {self.batch_count}")
            
            # --- SIDE-PLOTS (throttled: every Nth frame) ---
            if self.batch_count % PROFILE_THROTTLE == 0:
                self._update_profiles()
                self._update_temporal()
                
        except Exception as e:
            self.status_label.setText(f"Status: Read error — {e}")
    
    def _update_profiles(self):
        """Update row/column profiles for the selected pixel."""
        if self.current_img is None or self.sel_row is None:
            return
        
        img = self.current_img
        r, c = self.sel_row, self.sel_col
        h, w = img.shape
        
        if 0 <= r < h and 0 <= c < w:
            # Row profile — reuse cached x-axis
            if self._row_x is not None and len(self._row_x) == w:
                self.row_curve.setData(self._row_x, img[r, :])
            
            # Column profile — reuse cached x-axis
            if self._col_x is not None and len(self._col_x) == h:
                self.col_curve.setData(self._col_x, img[:, c])
            
            val = img[r, c]
            self.pixel_label.setText(f"Pixel: ({r}, {c}) = {val:.6e}")
    
    def _update_temporal(self):
        """Append pixel and ROI values using pre-allocated numpy arrays."""
        if self.current_img is None:
            return
        
        img = self.current_img
        h, w = img.shape
        idx = self._hist_len
        
        # Roll buffer if full
        if idx >= MAX_HISTORY:
            self._time_hist[:-1] = self._time_hist[1:]
            self._pixel_hist[:-1] = self._pixel_hist[1:]
            self._roi_hist[:-1] = self._roi_hist[1:]
            idx = MAX_HISTORY - 1
            self._hist_len = MAX_HISTORY
        else:
            self._hist_len = idx + 1
        
        self._time_hist[idx] = self.batch_count
        n = self._hist_len
        t_slice = self._time_hist[:n]
        
        # Pixel temporal trace
        if self.sel_row is not None and self.sel_col is not None:
            r, c = self.sel_row, self.sel_col
            if 0 <= r < h and 0 <= c < w:
                self._pixel_hist[idx] = img[r, c]
                self.pixel_time_curve.setData(t_slice, self._pixel_hist[:n])
        
        # ROI temporal trace — fast numpy slicing instead of getArrayRegion
        try:
            roi_state = self.roi.state
            x0 = max(0, int(roi_state['pos'].x()))
            y0 = max(0, int(roi_state['pos'].y()))
            rw = max(1, int(roi_state['size'].x()))
            rh = max(1, int(roi_state['size'].y()))
            x1 = min(w, x0 + rw)
            y1 = min(h, y0 + rh)
            
            if x1 > x0 and y1 > y0:
                # ROI coords are in image-T space (col, row), so swap
                roi_slice = img[y0:y1, x0:x1]
                self._roi_hist[idx] = np.mean(roi_slice)
                self.roi_time_curve.setData(t_slice, self._roi_hist[:n])
        except Exception:
            pass
    
    def stop_live(self):
        """Stop live view — Stoplive=True → LabVIEW exits Getframe → Idle."""
        self.acquiring = False
        self.poll_timer.stop()
        
        vi = self.manager.vi
        if vi:
            try:
                vi.SetControlValue("Stoplive", True)
                self.status_label.setText("Status: Stopping...")
                QtCore.QTimer.singleShot(200, self._check_stopped)
            except Exception as e:
                self.status_label.setText(f"Status: Stop error — {e}")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def _check_stopped(self):
        vi = self.manager.vi
        if vi:
            try:
                if vi.GetControlValue("Enum") == CMD_IDLE:
                    self.status_label.setText("Status: Stopped")
                else:
                    QtCore.QTimer.singleShot(200, self._check_stopped)
            except Exception:
                self.status_label.setText("Status: Stopped (timeout)")
    
    # =========================================================================
    # Colormap
    # =========================================================================
    
    def _change_mode(self, index):
        if index == 0 or index == 2:  # DT/T or DT: Blue → White → Red
            pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            color = np.array([
                [0,   0,   180, 255],
                [100, 150, 255, 255],
                [255, 255, 255, 255],
                [255, 150, 100, 255],
                [180, 0,   0,   255],
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(pos, color)
            self.img_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        else:  # T: Magma
            try:
                self.img_item.setLookupTable(pg.colormap.get('magma').getLookupTable())
            except Exception:
                pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                color = np.array([
                    [0, 0, 128, 255], [0, 255, 255, 255], [0, 255, 0, 255],
                    [255, 255, 0, 255], [255, 0, 0, 255]
                ], dtype=np.ubyte)
                cmap = pg.ColorMap(pos, color)
                self.img_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
    

    # =========================================================================
    # Stage Control Logic
    # =========================================================================

    def _init_stage_polling(self):
        """Start polling stage position if stage exists."""
        if self.delay_stage:
             self._poll_stage_pos()
             self.stage_timer = QtCore.QTimer()
             self.stage_timer.timeout.connect(self._poll_stage_pos)
             self.stage_timer.start(500) # 2Hz poll

    def _poll_stage_pos(self):
        """Poll stage position periodically."""
        if not self.delay_stage:
            self.lbl_stage_pos.setText("No Stage Driver")
            return
            
        if not self.delay_stage.is_connected:
            self.lbl_stage_pos.setText("Stage Not Connected")
            return

        try:
            pos_mm = self.delay_stage.get_position()
            
            # Use Shared Zero
            zero_mm = self.delay_stage.zero_position if self.delay_stage else 0.0
            
            # Sync Spinbox if changed externally (optional, but good)
            if abs(self.spin_zero.value() - zero_mm) > 0.0001:
                self.spin_zero.blockSignals(True)
                self.spin_zero.setValue(zero_mm)
                self.spin_zero.blockSignals(False)
            c = 0.000299792458
            
            if self.rb_pump.isChecked():
                # Pump on stage: Delay = (Pos - Zero) * 2 / c
                delay_fs = (pos_mm - zero_mm) * 2 / c
            else:
                # Probe on stage: Delay = (Zero - Pos) * 2 / c
                delay_fs = (zero_mm - pos_mm) * 2 / c
            
            # Display both units
            self.lbl_stage_pos.setText(f"Pos: {pos_mm:.4f} mm\nDelay: {delay_fs:.1f} fs")
                
        except Exception as e:
            self.lbl_stage_pos.setText(f"Stage Error: {e}")
            
    def _update_stage_ui(self):
        """Force update of stage UI (e.g. when zero changed)."""
        self._poll_stage_pos()

    def _move_stage_abs(self):
        if not self.delay_stage or not self.delay_stage.is_connected:
             return
             
        target = self.spin_stage_target.value()
        self.status_label.setText(f"Moving to {target:.4f} mm...")
        
        def do_move():
            try:
                print(f"[LIVE] Moving stage to {target:.4f} mm")
                self.delay_stage.move_to(target, wait=True)
                print(f"[LIVE] Move complete")
            except Exception as e:
                print(f"[LIVE] Move failed: {e}")
            
        t = threading.Thread(target=do_move, daemon=True)
        t.start()

    def _move_stage_by_fs(self, fs_val):
        """Move stage by femtoseconds, accounting for Pump/Probe config. (User requested Invert)"""
        c = 0.000299792458
        
        if self.rb_pump.isChecked():
            mm_val = (fs_val * c / 2.0)   # Pump: +mm
        else:
            mm_val = -(fs_val * c / 2.0)  # Probe: -mm
            
        self._move_stage_rel(mm_val)
        
    def _move_stage_rel(self, delta):
        if not self.delay_stage or not self.delay_stage.is_connected:
             return
             
        self.status_label.setText(f"Moving relative {delta:+.4f} mm...")
        
        def do_move():
            self.delay_stage.move_relative(delta)
            
        t = threading.Thread(target=do_move, daemon=True)
        t.start()

    # =========================================================================
    #  Saving
    # =========================================================================

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

        mode_idx = self.mode_combo.currentIndex()
        if mode_idx == 0: mode_tag = "dT_T"
        elif mode_idx == 1: mode_tag = "T"
        else: mode_tag = "dT"

        filename = f"{sample}_live_{mode_tag}_{timestamp.strftime('%H%M%S')}.npy"
        filepath = os.path.join(date_dir, filename)

        np.save(filepath, self.current_img)
        self.status_label.setText(f"Status: Saved {filename}")
        print(f"[Live] Saved image to {filepath}")

    # =========================================================================
    # Signal Mode Toggle + Public API
    # =========================================================================
    
    def _toggle_signal_mode(self):
        """Toggle between ROI and Single Pixel mode."""
        if self.roi_toggle.isChecked():
            self.roi_toggle.setText("Mode: ROI")
            self.roi.setVisible(True)
        else:
            self.roi_toggle.setText("Mode: Pixel")
            self.roi.setVisible(False)
    
    @property
    def use_roi(self) -> bool:
        """True if ROI mode is active, False for single-pixel mode."""
        return self.roi_toggle.isChecked()
    
    def extract_signal(self, img: np.ndarray) -> float:
        """
        Extract signal from an image based on current toggle state.
        
        ROI mode  → mean of the ROI rectangle
        Pixel mode → value at the selected pixel
        
        Called by scan windows (pump-probe, twins, twins-pp) to compute
        the signal at each scan point using the same region the user
        selected in Live View.
        
        Falls back to full-image mean if no selection exists.
        """
        if img is None or img.size == 0:
            return 0.0
        
        # Ensure 2D
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            if side * side == img.size:
                img = img.reshape(side, side)
            else:
                return float(np.nanmean(img))
        
        h, w = img.shape
        
        if self.use_roi:
            # ROI mode
            try:
                roi_state = self.roi.state
                x0 = max(0, int(roi_state['pos'].x()))
                y0 = max(0, int(roi_state['pos'].y()))
                rw = max(1, int(roi_state['size'].x()))
                rh = max(1, int(roi_state['size'].y()))
                x1 = min(w, x0 + rw)
                y1 = min(h, y0 + rh)
                if x1 > x0 and y1 > y0:
                    return float(np.nanmean(img[y0:y1, x0:x1]))
            except Exception:
                pass
            return float(np.nanmean(img))
        else:
            # Single pixel mode
            if self.sel_row is not None and self.sel_col is not None:
                r, c = self.sel_row, self.sel_col
                if 0 <= r < h and 0 <= c < w:
                    return float(img[r, c])
            return float(np.nanmean(img))
    
    def get_roi_bounds(self):
        """Return current ROI bounds as (row_start, row_end, col_start, col_end) or None."""
        try:
            roi_state = self.roi.state
            x0 = max(0, int(roi_state['pos'].x()))
            y0 = max(0, int(roi_state['pos'].y()))
            rw = max(1, int(roi_state['size'].x()))
            rh = max(1, int(roi_state['size'].y()))
            return (y0, y0 + rh, x0, x0 + rw)
        except Exception:
            return None
    
    def closeEvent(self, event):
        if self.acquiring:
            self.stop_live()
        if hasattr(self, 'stage_timer'):
             self.stage_timer.stop()
        event.accept()
