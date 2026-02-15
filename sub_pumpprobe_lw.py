"""
Pump-Probe Delay Scan — scans Thorlabs delay stage while acquiring
via LabVIEW Experiment_manager.vi.

For each scan point:
  1. Move delay stage → poll position stability (3 reads within 1µm)
  2. Settle 50ms for optical vibrations
  3. Trigger LabVIEW: Enum=Measure (single-shot) → poll Idle → read DeltaT
  4. CSV append (incremental safety save)
  5. Update live ΔT/T vs Delay plot → next point

Usage:
    from sub_pumpprobe_lw import PumpProbeScanWindow
    window = PumpProbeScanWindow(labview_manager, delay_stage)
    window.show()
"""

import csv
import numpy as np
from datetime import datetime

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    raise ImportError("pyqtgraph required: pip install pyqtgraph pyqt6")

from labview_manager import LabVIEWManager, CMD_IDLE, CMD_MEASURE

# Physics
SPEED_OF_LIGHT_MM_FS = 0.000299792458
GLOBAL_ZERO_POS_MM = 140.0


class PumpProbeScanWindow(QtWidgets.QWidget):
    """
    Pump-probe delay scan using Thorlabs stage + LabVIEW camera.
    """
    
    def __init__(self, manager: LabVIEWManager, delay_stage, live_window=None, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.delay_stage = delay_stage
        self.live_window = live_window  # for ROI / pixel signal extraction
        
        # Scan state
        self.scanning = False
        self.scan_points_fs = []
        self.scan_delays = []
        self.scan_signals = []
        self.scan_index = 0
        self.scan_csv_path = None
        self.roi_datacube = []    # list of 2D ROI slices per point
        
        # New State
        self.interval_unit = 'fs'  # 'fs' or 'mm'
        self.interval_checks = []  # Checkboxes for intervals
        
        self.setWindowTitle("Pump-Probe Delay Scan (LabVIEW)")
        self.resize(900, 800)
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # ====== Left Panel: Scan Controls ======
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        # Title
        title = QtWidgets.QLabel("Pump-Probe Delay Scan")
        title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #4CAF50; padding: 5px;"
        )
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        # Zero position + Probe Toggle
        zero_group = QtWidgets.QGroupBox("Stage Settings")
        zero_layout = QtWidgets.QGridLayout(zero_group)
        
        zero_layout.addWidget(QtWidgets.QLabel("Zero Pos (mm):"), 0, 0)
        self.zero_spin = QtWidgets.QDoubleSpinBox()
        self.zero_spin.setRange(0, 300)
        self.zero_spin.setDecimals(3)
        self.zero_spin.setValue(GLOBAL_ZERO_POS_MM)
        zero_layout.addWidget(self.zero_spin, 0, 1)
        
        self.chk_probe = QtWidgets.QCheckBox("Probe on Stage")
        self.chk_probe.setToolTip("If checked, stage moves Probe (Delay = Zero - Pos). Else Pump (Delay = Pos - Zero).")
        self.chk_probe.toggled.connect(self._update_stage_display)
        zero_layout.addWidget(self.chk_probe, 1, 0, 1, 2)
        
        left_layout.addWidget(zero_group)

        # 3 Intervals
        # Unit switch
        unit_layout = QtWidgets.QHBoxLayout()
        unit_layout.addWidget(QtWidgets.QLabel("Interval Units:"))
        self.rb_fs = QtWidgets.QRadioButton("fs")
        self.rb_mm = QtWidgets.QRadioButton("mm")
        self.rb_fs.setChecked(True)
        self.rb_fs.toggled.connect(self._update_units)
        unit_layout.addWidget(self.rb_fs)
        unit_layout.addWidget(self.rb_mm)
        unit_layout.addStretch()
        left_layout.addLayout(unit_layout)

        interval_configs = [
            ("Interval 1 (Fine)", -1000, 0, 50, True),
            ("Interval 2 (Mid)", 0, 10000, 100, False),
            ("Interval 3 (Coarse)", 10000, 100000, 1000, False),
        ]
        self.interval_spins = []
        self.interval_checks = [] # Reset

        for i, (label, s_def, e_def, st_def, always_on) in enumerate(interval_configs):
            grp = QtWidgets.QGroupBox(label)
            grp.setCheckable(not always_on)
            if not always_on:
                grp.setChecked(False) 
            
            # Use the groupbox checkbox itself if possible, or add one.
            # QGroupBox.setCheckable adds a checkbox in title. Pefect.
            
            gl = QtWidgets.QGridLayout(grp)

            self.lbl_start = QtWidgets.QLabel("Start (fs):")
            gl.addWidget(self.lbl_start, 0, 0)
            sp_s = QtWidgets.QDoubleSpinBox()
            sp_s.setRange(-1e6, 1e6)
            sp_s.setValue(s_def)
            gl.addWidget(sp_s, 0, 1)

            self.lbl_end = QtWidgets.QLabel("End (fs):")
            gl.addWidget(self.lbl_end, 0, 2)
            sp_e = QtWidgets.QDoubleSpinBox()
            sp_e.setRange(-1e6, 1e6)
            sp_e.setValue(e_def)
            gl.addWidget(sp_e, 0, 3)

            self.lbl_step = QtWidgets.QLabel("Step (fs):")
            gl.addWidget(self.lbl_step, 1, 0)
            sp_st = QtWidgets.QDoubleSpinBox()
            sp_st.setRange(0.0001, 1e6) # Allow small steps for mm
            sp_st.setValue(st_def)
            gl.addWidget(sp_st, 1, 1)

            self.interval_spins.append((sp_s, sp_e, sp_st, self.lbl_start, self.lbl_end, self.lbl_step))
            self.interval_checks.append(grp) # The groupbox itself is the check
            left_layout.addWidget(grp)

        # Acquisition settings
        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_layout = QtWidgets.QGridLayout(acq_group)

        acq_layout.addWidget(QtWidgets.QLabel("Frames/point:"), 0, 0)
        self.frames_spin = QtWidgets.QSpinBox()
        self.frames_spin.setRange(2, 10000)
        self.frames_spin.setValue(100)
        acq_layout.addWidget(self.frames_spin, 0, 1)

        self.points_label = QtWidgets.QLabel("Points: --")
        self.points_label.setStyleSheet("font-weight: bold;")
        acq_layout.addWidget(self.points_label, 1, 0, 1, 2)
        
        # Sample Name
        acq_layout.addWidget(QtWidgets.QLabel("Sample Name:"), 2, 0)
        self.txt_sample_name = QtWidgets.QLineEdit()
        self.txt_sample_name.setPlaceholderText("Enter sample name...")
        acq_layout.addWidget(self.txt_sample_name, 2, 1)
        
        # Save Mode
        acq_layout.addWidget(QtWidgets.QLabel("Save Mode:"), 3, 0)
        self.cmb_save_mode = QtWidgets.QComboBox()
        self.cmb_save_mode.addItems(["Single Pixel", "ROI Average", "ROI (2D)", "Full Frame (2D)"])
        self.cmb_save_mode.setCurrentIndex(1) # Default ROI Avg
        acq_layout.addWidget(self.cmb_save_mode, 3, 1)
        
        # Plot Mode
        acq_layout.addWidget(QtWidgets.QLabel("Plot Mode:"), 4, 0)
        self.cmb_plot_mode = QtWidgets.QComboBox()
        self.cmb_plot_mode.addItems(["DeltaT (dT/T)", "Transmission (T)", "DeltaT (dT)"])
        self.cmb_plot_mode.setCurrentIndex(0) # Default dT/T
        acq_layout.addWidget(self.cmb_plot_mode, 4, 1)

        self.btn_bg = QtWidgets.QPushButton("Acquire Background")
        self.btn_bg.setStyleSheet("background-color: #607D8B; color: white;")
        self.btn_bg.clicked.connect(self._acquire_background)
        acq_layout.addWidget(self.btn_bg, 5, 0, 1, 2)
        
        left_layout.addWidget(acq_group)

        # Start / Stop buttons
        btn_row = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("START SCAN")
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px; border-radius: 6px;"
        )
        self.start_btn.clicked.connect(self.start_scan)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("STOP")
        self.stop_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px; border-radius: 6px;"
        )
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)
        left_layout.addLayout(btn_row)

        # Progress + Status
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        left_layout.addWidget(self.progress)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # ====== Middle Panel: Delay Stage ======
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
        self.stage_label = QtWidgets.QLabel("---.---")
        self.stage_label.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #2196F3;"
        )
        pos_layout.addWidget(self.stage_label, 0, 1)

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

        self.btn_go_abs = QtWidgets.QPushButton("Go")
        self.btn_go_abs.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_go_abs.clicked.connect(self._move_absolute)
        abs_layout.addWidget(self.btn_go_abs, 1, 0, 1, 2)

        stage_layout.addWidget(abs_group)
        
        # Stage connection status
        self.lbl_stage_status = QtWidgets.QLabel("Not connected")
        self.lbl_stage_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_stage_status.setStyleSheet("color: #888;")
        stage_layout.addWidget(self.lbl_stage_status)

        stage_layout.addStretch()
        main_layout.addWidget(stage_panel)

        # ====== Right Panel: Display ======
        display_panel = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_panel)

        # Vertical splitter: Camera frame on top, scan plot on bottom
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Camera frame
        cam_widget = pg.GraphicsLayoutWidget()
        self.img_view = cam_widget.addPlot(title="Last Acquired Frame (DeltaT/T)")
        self.img_item = pg.ImageItem()
        self.img_view.addItem(self.img_item)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        cam_widget.addItem(self.hist)

        # DT colormap
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array([
            [0, 0, 180, 255], [100, 150, 255, 255],
            [255, 255, 255, 255],
            [255, 150, 100, 255], [180, 0, 0, 255]
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.img_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

        splitter.addWidget(cam_widget)

        # Scan plot
        self.scan_plot = pg.PlotWidget(
            title="DeltaT/T vs Delay",
            labels={'left': 'DeltaT/T (mean)', 'bottom': 'Delay (fs)'}
        )
        self.scan_plot.showGrid(x=True, y=True, alpha=0.3)
        self.scan_curve = self.scan_plot.plot(
            pen=pg.mkPen('#2196F3', width=2), symbol='o', symbolSize=4,
            symbolBrush='#2196F3'
        )
        splitter.addWidget(self.scan_plot)

        splitter.setSizes([400, 300])
        display_layout.addWidget(splitter)

        main_layout.addWidget(display_panel, stretch=1)

        # Stage position update timer
        self._stage_timer = QtCore.QTimer()
        self._stage_timer.timeout.connect(self._update_stage_display)
        self._stage_timer.start(500)
    
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
    
    # =========================================================================
    # Unit Conversions
    # =========================================================================
    
    # =========================================================================
    # Unit Conversions
    # =========================================================================
    
    def _fs_to_mm_dist(self, val, to_mm=True):
        """Pure unit conversion (fs <-> mm) without sign flip."""
        if to_mm:
            return val * SPEED_OF_LIGHT_MM_FS / 2.0
        else:
            return val * 2.0 / SPEED_OF_LIGHT_MM_FS

    def _fs_to_mm(self, time_fs):
        """Convert delay (fs) to stage shift (mm). Reverses if Probe on Stage."""
        dist = self._fs_to_mm_dist(time_fs, to_mm=True)
        if self.chk_probe.isChecked():
            return -dist
        return dist
    
    def _mm_to_fs(self, distance_mm):
        """Convert stage shift (mm) to delay (fs). Reverses if Probe on Stage."""
        base_fs = self._fs_to_mm_dist(distance_mm, to_mm=False)
        if self.chk_probe.isChecked():
            return -base_fs
        return base_fs
    
    def _update_units(self):
        """Convert spinbox values when unit changes."""
        to_mm = self.rb_mm.isChecked()
        if (to_mm and self.interval_unit == 'mm') or (not to_mm and self.interval_unit == 'fs'):
            return
        
        self.interval_unit = 'mm' if to_mm else 'fs'
        unit_lbl = "mm" if to_mm else "fs"

        for sp_s, sp_e, sp_st, lbl_s, lbl_e, lbl_step in self.interval_spins:
            s_old, e_old, st_old = sp_s.value(), sp_e.value(), sp_st.value()
            
            # Convert values (Pure magnitude/unit)
            # Use _fs_to_mm_dist logic (no sign flip for Unit Toggle)
            if to_mm:
                s_new = self._fs_to_mm_dist(s_old, True)
                e_new = self._fs_to_mm_dist(e_old, True)
                st_new = self._fs_to_mm_dist(st_old, True)
            else:
                s_new = self._fs_to_mm_dist(s_old, False)
                e_new = self._fs_to_mm_dist(e_old, False)
                st_new = self._fs_to_mm_dist(st_old, False)
            
            # Update Spinboxes
            sp_s.setValue(s_new)
            sp_e.setValue(e_new)
            sp_st.setValue(st_new)
            
            # Update Labels
            lbl_s.setText(f"Start ({unit_lbl}):")
            lbl_e.setText(f"End ({unit_lbl}):")
            lbl_step.setText(f"Step ({unit_lbl}):")
    
    # =========================================================================
    # Stage Control (middle panel)
    # =========================================================================
    
    def _update_stage_display(self):
        """Periodic stage position update."""
        if self.delay_stage and self.delay_stage.is_connected:
            try:
                pos_mm = self.delay_stage.get_position()
                self.stage_label.setText(f"{pos_mm:.3f}")
                
                # Calculate Delay: (Pos - Zero) or (Zero - Pos)
                zero = self.zero_spin.value()
                diff = pos_mm - zero
                if self.chk_probe.isChecked():
                    diff = -diff # Probe on Stage: Pos < Zero means Positive Delay
                
                delay_fs = diff * 2.0 / SPEED_OF_LIGHT_MM_FS
                self.lbl_stage_fs.setText(f"{delay_fs:.0f}")
                
                self.lbl_stage_status.setText("Connected")
                self.lbl_stage_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            except Exception:
                self.stage_label.setText("err")
                self.lbl_stage_fs.setText("err")
        else:
            self.lbl_stage_status.setText("Not connected")
            self.lbl_stage_status.setStyleSheet("color: #888;")
    
    def _move_absolute(self):
        if not self.delay_stage or not self.delay_stage.is_connected:
            self.status_label.setText("Status: Stage not connected!")
            return
        target = self.spin_abs_mm.value()
        self.status_label.setText(f"Status: Moving to {target:.3f} mm...")
        self.btn_go_abs.setEnabled(False)
        import threading
        def do_move():
            self.delay_stage.move_to(target)
        threading.Thread(target=do_move, daemon=True).start()
        QtCore.QTimer.singleShot(1000, lambda: self.btn_go_abs.setEnabled(True))
        QtCore.QTimer.singleShot(1000, lambda: self.status_label.setText("Status: Ready"))
    

    
    # =========================================================================
    # Scan Point Generation
    # =========================================================================
    
    def _generate_scan_points(self):
        """Generate scan positions from enabled intervals."""
        points = []
        is_mm = (self.interval_unit == 'mm')
        
        for i, (sp_s, sp_e, sp_st, _, _, _) in enumerate(self.interval_spins):
            # Check enabled
            if i < len(self.interval_checks) and not self.interval_checks[i].isChecked():
                continue
                
            s, e, st = sp_s.value(), sp_e.value(), sp_st.value()
            
            # Convert to fs if in mm mode
            if is_mm:
                s = self._fs_to_mm_dist(s, False) # mm->fs
                e = self._fs_to_mm_dist(e, False)
                st = self._fs_to_mm_dist(st, False)
            
            # Generate points
            # If s > e (reverse scan), handle it
            if st <= 0: continue
            
            if s <= e:
                pts = np.arange(s, e, st)
            else:
                pts = np.arange(s, e, -st) # Descending scan?
                # Usually step is positive magnitude in UI.
                
            points.extend(pts.tolist())
            
        # Add final endpoint of LAST enabled interval
        if points and self.interval_spins:
            # Find last enabled interval info
            last_enabled = -1
            for i in range(len(self.interval_checks)-1, -1, -1):
                if self.interval_checks[i].isChecked():
                    last_enabled = i
                    break
            
            if last_enabled >= 0:
                e_last = self.interval_spins[last_enabled][1].value()
                if is_mm:
                    e_last = self._fs_to_mm_dist(e_last, False)
                points.append(e_last)

        return np.unique(np.array(points)) # Unique sorts them. Careful if user wants specific order?
        # Pump-Probe usually monotonic. np.unique sorts ascending.
        # If user defined Intervals 1,2,3 in overlapping/strange ways, unique helps.
        # But if they wanted a specific sequence, unique destroys it.
        # Standard usage: Interval 1 (-100 to 0), Interval 2 (0 to 100).
        # np.unique is safe and good to remove duplicates at boundaries.
        if len(points) == 0: return np.array([])
        return np.sort(np.unique(np.array(points)))
    
    # =========================================================================
    # Scan Control
    # =========================================================================
    
    def start_scan(self):
        """Start pump-probe delay scan."""
        if self.delay_stage is None or not self.delay_stage.is_connected:
            self.status_label.setText("Status: Connect delay stage first!")
            return
        if not self.manager.camera_initialized:
            self.status_label.setText("Status: Initialize camera first!")
            return
        
        # Generate scan points
        self.scan_points_fs = self._generate_scan_points()
        if len(self.scan_points_fs) == 0:
            self.status_label.setText("Status: No scan points! Check intervals.")
            return
        
        # Reset state
        self.scan_delays = []
        self.scan_signals = []
        self.scan_index = 0
        self.scanning = True
        self.scan_curve.setData([], [])
        
        # Generate Standardized Paths
        timestamp = datetime.now()
        date_dir = timestamp.strftime(r"D:\pumpprobedata\%Y\%m\%d")
        os.makedirs(date_dir, exist_ok=True)
        
        sample = self.txt_sample_name.text().strip()
        if not sample: sample = "sample"
        # Sanitize
        sample = "".join(x for x in sample if x.isalnum() or x in " -_")
        
        base_name = f"{sample}_pumpprobe_{timestamp.strftime('%H%M%S')}"
        self.scan_csv_path = os.path.join(date_dir, base_name + ".csv")
        self.scan_npz_path = os.path.join(date_dir, base_name + ".npz") # Stored for end of scan

        with open(self.scan_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["delay_fs", "pos_mm", "signal_dT_T"])
        
        # UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        
        n_pts = len(self.scan_points_fs)
        self.points_label.setText(f"Points: 0/{n_pts}")
        self.status_label.setText(
            f"Status: Scanning {n_pts} points, "
            f"{self.scan_points_fs[0]:.0f} to {self.scan_points_fs[-1]:.0f} fs"
        )
        print(f"[SCAN] Starting {n_pts}-point scan")
        
        # Begin
        QtCore.QTimer.singleShot(100, self._move_stage)
    
    def _acquire_background(self):
        """Trigger a single acquisition for background."""
        if self.scanning:
            return
        
        self.status_label.setText("Status: Acquiring Global Background...")
        self._awaiting_background = True
        
        # Reuse trigger logic (manually)
        self.manager.vi.SetControlValue("N", self.frames_spin.value())
        self.manager.vi.SetControlValue("Acq Trigger", True)
        self.manager.vi.SetControlValue("Enum", CMD_MEASURE)
        
        self._acq_timer = QtCore.QTimer()
        self._acq_timer.timeout.connect(self._poll_acquire)
        self._acq_waited = 0.0
        self._acq_timer.start(50)
    
    def _move_stage(self):
        """Move stage to next scan point."""
        if not self.scanning or self.scan_index >= len(self.scan_points_fs):
            self._scan_complete()
            return
        
        delay_fs = self.scan_points_fs[self.scan_index]
        zero_mm = self.zero_spin.value()
        target_mm = zero_mm + self._fs_to_mm(delay_fs)
        
        self.status_label.setText(
            f"Status: Point {self.scan_index+1}/{len(self.scan_points_fs)}: "
            f"{delay_fs:.0f} fs → {target_mm:.4f} mm"
        )
        self.stage_label.setText(f"Stage: → {target_mm:.3f} mm")
        
        # Non-blocking move
        self.delay_stage.move_to(target_mm, wait=False)
        
        # Poll for position stability
        self._poll_count = 0
        self._last_pos = None
        self._stable = 0
        self._poll_timer = QtCore.QTimer()
        self._poll_timer.timeout.connect(self._poll_stage)
        self._poll_timer.start(100)
    
    def _poll_stage(self):
        """Poll stage position until stable → trigger acquire."""
        if not self.scanning:
            self._poll_timer.stop()
            return
        
        self._poll_count += 1
        if self._poll_count > 600:  # 60s timeout
            self._poll_timer.stop()
            self.status_label.setText("Status: WARNING — Stage timeout, skipping")
            self.scan_index += 1
            QtCore.QTimer.singleShot(100, self._move_stage)
            return
        
        pos = self.delay_stage.get_position()
        self.stage_label.setText(f"Stage: {pos:.4f} mm")
        
        # 3 consecutive reads within 1µm = stable
        if self._last_pos is not None:
            if abs(pos - self._last_pos) < 0.001:
                self._stable += 1
                if self._stable >= 3:
                    self._poll_timer.stop()
                    # Settle for vibrations
                    QtCore.QTimer.singleShot(50, self._trigger_acquire)
                    return
            else:
                self._stable = 0
        
        self._last_pos = pos
    
    def _trigger_acquire(self):
        """Trigger one LabVIEW acquisition after stage settled."""
        if not self.scanning:
            return
        
        vi = self.manager.vi
        n = self.frames_spin.value()
        
        try:
            vi.SetControlValue("N", n)
            vi.SetControlValue("Acq Trigger", True)
            # Use Measure (single-shot) — acquires once then returns to Idle
            vi.SetControlValue("Enum", CMD_MEASURE)
            
            # Poll for Idle (single-shot acquire)
            self._acq_timer = QtCore.QTimer()
            self._acq_timer.timeout.connect(self._poll_acquire)
            self._acq_waited = 0.0
            self._acq_timer.start(50)
            
        except Exception as e:
            self.status_label.setText(f"Status: Acquire error — {e}")
            self.scan_index += 1
            QtCore.QTimer.singleShot(100, self._move_stage)
    
    def _poll_acquire(self):
        """Poll LabVIEW until Enum == Idle → read result."""
        if not self.scanning:
            self._acq_timer.stop()
            return
        
        self._acq_waited += 0.05
        if self._acq_waited > 60.0:
            self._acq_timer.stop()
            self.status_label.setText("Status: WARNING — Acquire timeout")
            self.scan_index += 1
            QtCore.QTimer.singleShot(100, self._move_stage)
            return
        
        try:
            if self.manager.vi.GetControlValue("Enum") != CMD_IDLE:
                return  # Still acquiring
        except Exception:
            return
        
        # Done — read result
        self._acq_timer.stop()
        delay_fs = self.scan_points_fs[self.scan_index]
        
        try:
            odd_data = self.manager.vi.GetControlValue("Odd")
            even_data = self.manager.vi.GetControlValue("Even")
            if odd_data is not None and even_data is not None:
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                
                # Handle Background
                if hasattr(self, '_awaiting_background') and self._awaiting_background:
                    # Store average of Odd/Even as background
                    self.manager.background = (odd + even) / 2.0
                    self._awaiting_background = False
                    self.status_label.setText("Status: Global Background Acquired")
                    QtWidgets.QMessageBox.information(self, "Background", "Global Background acquired!")
                    return

                # Apply Background
                if self.manager.background is not None:
                    if self.manager.background.shape == odd.shape:
                        odd -= self.manager.background
                        even -= self.manager.background
                
                # Compute based on Plot Mode
                
                # Compute based on Plot Mode
                pmode = self.cmb_plot_mode.currentIndex()
                if pmode == 1:   # Transmission (T) -> (Odd + Even) / 2
                    img = (odd + even) / 2.0
                elif pmode == 2: # DeltaT (dT) -> Even - Odd
                    img = even - odd
                else:            # DeltaT/T -> (Even - Odd) / Odd
                    img = np.divide(even - odd, odd, out=np.zeros_like(odd), where=np.abs(odd) > 1.0)
                
                # Skip empty data
                if img.size == 0:
                    print(f"[SCAN] Empty data at point {self.scan_index+1}")
                    self.scan_index += 1
                    QtCore.QTimer.singleShot(50, self._move_stage)
                    return
                if img.ndim == 1:
                    side = int(np.sqrt(img.size))
                    if side * side == img.size:
                        img = img.reshape(side, side)
                
                if img.ndim == 2:
                    self.img_item.setImage(img.T)
                
                signal = self._extract(img)
                to_save = self._extract_to_save(img)
                self.roi_datacube.append(to_save)
                
                self.scan_delays.append(delay_fs)
                self.scan_signals.append(signal)
                self.scan_curve.setData(self.scan_delays, self.scan_signals)
                
                # CSV save
                actual_mm = self.delay_stage.get_position()
                with open(self.scan_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{delay_fs:.2f}", f"{actual_mm:.4f}", f"{signal:.8e}"])
                
                print(f"[SCAN] Point {self.scan_index+1}: {delay_fs:.0f} fs = {signal:.4e}")
            else:
                print(f"[SCAN] No data at point {self.scan_index+1}")
                
        except Exception as e:
            print(f"[SCAN] Read error: {e}")
        
        # Update progress
        self.scan_index += 1
        pct = int(100 * self.scan_index / len(self.scan_points_fs))
        self.progress.setValue(pct)
        self.points_label.setText(
            f"Points: {self.scan_index}/{len(self.scan_points_fs)}"
        )
        
        QtCore.QTimer.singleShot(50, self._move_stage)
    
    def _scan_complete(self):
        """Save final data and reset UI."""
        self.scanning = False
        
        if self.scan_delays:
            if not hasattr(self, 'scan_npz_path'):
                # Fallback if somehow not set (should be set in start_scan)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.scan_npz_path = f"pumpprobe_{ts}_final.npz"

            np.savez(
                self.scan_npz_path,
                delays_fs=np.array(self.scan_delays),
                signals=np.array(self.scan_signals),
                zero_mm=self.zero_spin.value(),
                frames_per_point=self.frames_spin.value(),
                roi_datacube=np.array(self.roi_datacube) if self.roi_datacube else np.array([])
            )
            self.status_label.setText(
                f"Status: Scan complete! Saved to {os.path.basename(self.scan_npz_path)}"
            )
            print(f"[SCAN] Final data: {self.scan_npz_path}")
        else:
            self.status_label.setText("Status: Scan complete (no data)")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def stop_scan(self):
        """Emergency stop."""
        print("[SCAN] STOP requested")
        self.scanning = False
        
        if hasattr(self, '_poll_timer'):
            self._poll_timer.stop()
        if hasattr(self, '_acq_timer'):
            self._acq_timer.stop()
        
        self._scan_complete()
    
    def closeEvent(self, event):
        self._stage_timer.stop()
        if self.scanning:
            self.stop_scan()
        event.accept()
