import numpy as np
import h5py
import time
import sys
import os
import csv
from datetime import datetime
from threading import Thread

# Try importing pyqtgraph
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("Error: pyqtgraph is not installed. Please install it with 'pip install pyqtgraph pyqt6'")
    sys.exit(1)

# Try importing pywin32 for LabVIEW ActiveX automation
try:
    import win32com.client
    import pythoncom
    HAS_WIN32COM = True
except ImportError:
    print("[WARN] pywin32 not installed. LabVIEW automation disabled. Install with: pip install pywin32")
    HAS_WIN32COM = False

# Try importing stage drivers
try:
    from stage_delay import DelayStageDriver
    HAS_DELAY_STAGE = True
except ImportError:
    HAS_DELAY_STAGE = False
    print("[WARN] stage_delay.py not found. Delay stage disabled.")

try:
    from stage_driver import StageDriver
    HAS_TWINS_STAGE = True
except ImportError:
    HAS_TWINS_STAGE = False
    print("[WARN] stage_driver.py not found. Twins stage disabled.")

# =============================================================================
# Configuration
# =============================================================================
NUM_FRAMES = 100
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
GLOBAL_ZERO_POS_MM = 140.0  # Delay stage zero position (t=0) in mm
SPEED_OF_LIGHT_MM_FS = 0.000299792458  # mm per femtosecond

# Path to the persistent Manager VI
DEFAULT_VI_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Experiment_manager.vi"
)

# Command Enum values (must match LabVIEW enum)
CMD_IDLE = 0
CMD_INIT = 1
CMD_GETFRAME = 2
CMD_CLOSE = 3


# =============================================================================
# LabVIEW Manager Controller (Puppeteer Pattern)
# =============================================================================
class LabVIEWManager:
    """
    Controls the persistent Experiment_manager.vi via ActiveX.
    
    The Manager VI runs continuously with a While Loop + Case Structure.
    Python sends commands by setting the "Enum" control,
    then polls until the VI sets it back to "Idle" (0).
    
    Enum values:
        0 = Idle       (do nothing, wait)
        1 = Init       (open camera → returns to Idle)
        2 = Getframe   (continuous loop: acquire → T, DeltaT)
        3 = Close      (close camera → returns to Idle)
    
    Controls:
        N          (I32)  — number of frames per acquisition
        Acq Trigger (Bool) — acquisition trigger
        Stoplive   (Bool) — set True to exit Getframe loop → Idle
        End        (Bool) — stop the Manager VI after Close
    
    Indicators:
        T       — transmission data
        DeltaT  — delta T/T data
    """
    
    def __init__(self, vi_path: str = DEFAULT_VI_PATH):
        self.vi_path = vi_path
        self.lv = None
        self.vi = None
        self.is_running = False
    
    def start(self):
        """Launch LabVIEW, load Manager VI, and run it (non-blocking)."""
        if not HAS_WIN32COM:
            print("[ERROR] pywin32 not available")
            return False
        
        try:
            # Connect to or launch LabVIEW
            try:
                self.lv = win32com.client.GetActiveObject("LabVIEW.Application")
                print("[OK] Connected to existing LabVIEW instance")
            except Exception:
                self.lv = win32com.client.Dispatch("LabVIEW.Application")
                print("[OK] Launched new LabVIEW instance")
            
            # Load the Manager VI
            if not os.path.exists(self.vi_path):
                print(f"[ERROR] VI not found: {self.vi_path}")
                return False
            
            self.vi = self.lv.GetVIReference(self.vi_path)
            
            # Tell COM these are methods, not properties (late-binding fix)
            self.vi._FlagAsMethod("Run")
            self.vi._FlagAsMethod("Abort")
            self.vi._FlagAsMethod("SetControlValue")
            self.vi._FlagAsMethod("GetControlValue")
            
            self.vi.FPWinOpen = True
            print(f"[OK] Loaded: {os.path.basename(self.vi_path)}")
            
            # Run non-blocking — VI stays alive with its While Loop
            try:
                self.vi.Run(True)  # True = async (returns immediately)
                print("[OK] Manager VI started (async)")
            except Exception as run_err:
                print(f"[WARN] Run(False) failed: {run_err}")
                print("[INFO] Please click the Run button in LabVIEW manually.")
                print("[INFO] Then click 'Initialize Camera' in Python.")
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start Manager: {e}")
            return False
    
    def send_command(self, cmd, timeout=30.0):
        """
        Send a command and wait for the VI to return to Idle.
        
        Args:
            cmd: CMD_IDLE, CMD_INITIALIZE, CMD_ACQUIRE, or CMD_EXIT
            timeout: Max seconds to wait for completion
            
        Returns:
            True if command completed (returned to Idle), False on timeout
        """
        if not self.vi:
            print("[ERROR] Manager VI not loaded")
            return False
        
        try:
            self.vi.SetControlValue("Enum", cmd)
            
            # Poll until Enum returns to Idle
            waited = 0.0
            poll_interval = 0.05  # 50ms
            while waited < timeout:
                time.sleep(poll_interval)
                waited += poll_interval
                current = self.vi.GetControlValue("Enum")
                if current == CMD_IDLE:
                    return True
            
            print(f"[WARN] Command {cmd} timed out after {timeout}s")
            return False
            
        except Exception as e:
            print(f"[ERROR] send_command failed: {e}")
            return False
    
    def initialize_camera(self):
        """Send Init command (enum = 1). Camera opens and stays open."""
        print("[CMD] Init camera...")
        success = self.send_command(CMD_INIT, timeout=30.0)
        if success:
            print("[OK] Camera initialized")
        else:
            print("[ERROR] Camera initialization failed or timed out")
        return success
    
    def acquire_map(self, n_frames=100, acq_trigger=True):
        """
        Set parameters, send Getframe command, read result.
        
        Args:
            n_frames: Number of frames
            acq_trigger: Acquisition trigger boolean
            
        Returns:
            dict with 'T' and 'DeltaT' numpy arrays, or None
        """
        if not self.vi:
            return None
        
        try:
            # Set parameters before acquiring
            self.vi.SetControlValue("N", n_frames)
            self.vi.SetControlValue("Acq Trigger", acq_trigger)
            
            print(f"[CMD] Getframe (N={n_frames}, trigger={acq_trigger})...")
            
            # Send Getframe command and wait
            success = self.send_command(CMD_GETFRAME, timeout=60.0)
            if not success:
                print("[ERROR] Getframe timed out")
                return None
            
            # Read both results from VI
            result = {}
            try:
                t_data = self.vi.GetControlValue("T")
            except Exception:
                t_data = None
            
            odd_data = self.vi.GetControlValue("Odd")
            even_data = self.vi.GetControlValue("Even")
            
            if odd_data is None: print("[SERVER] 'Odd' is None")
            if even_data is None: print("[SERVER] 'Even' is None")
            
            if t_data is not None:
                result['T'] = np.array(t_data)
            if odd_data is not None and even_data is not None:
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                print(f"[SERVER] Odd: {odd.shape}, Even: {even.shape}")
                result['Odd'] = odd
                result['Even'] = even
                result['DeltaT'] = (even - odd) / np.where(np.abs(odd) > 1e-10, odd, 1e-10)
            
            return result if result else None
            
        except Exception as e:
            print(f"[ERROR] acquire_map failed: {e}")
            return None
    
    def shutdown(self):
        """Close camera (enum=3), then set End=True to stop the Manager VI."""
        if not self.is_running or not self.vi:
            return
        
        try:
            # Step 1: Close camera
            print("[CMD] Closing camera...")
            self.vi.SetControlValue("Enum", CMD_CLOSE)
            
            # Poll for Idle
            waited = 0.0
            while waited < 15.0:
                time.sleep(0.1)
                waited += 0.1
                try:
                    if self.vi.GetControlValue("Enum") == CMD_IDLE:
                        break
                except Exception:
                    break
            
            # Step 2: End the Manager VI
            print("[CMD] Ending Manager VI...")
            self.vi.SetControlValue("end", True)
            time.sleep(1.0)
            
            self.is_running = False
            print("[OK] Manager VI stopped")
        except Exception as e:
            print(f"[WARN] Shutdown error: {e}")
        
        self.vi = None
        self.lv = None
    
    def close(self):
        """Clean up — send Exit if still running."""
        if self.is_running:
            self.shutdown()


# =============================================================================
# Main Window
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    # Signals for thread-safe UI updates
    new_data_signal = QtCore.pyqtSignal(object)
    status_signal = QtCore.pyqtSignal(str)
    acq_finished_signal = QtCore.pyqtSignal()
    camera_connected_signal = QtCore.pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Camera Server")
        self.resize(900, 800)
        
        # Controller
        self.manager = LabVIEWManager()
        self.acquiring = False
        self.camera_initialized = False
        self.scanning = False
        
        # Stage drivers (singletons)
        self.delay_stage = DelayStageDriver() if HAS_DELAY_STAGE else None
        self.twins_stage = StageDriver() if HAS_TWINS_STAGE else None
        
        # Scan state
        self.scan_points_fs = []
        self.scan_delays = []
        self.scan_signals = []
        self.scan_index = 0
        self.scan_csv_path = None
        
        # Connect signals
        self.new_data_signal.connect(self.update_plot)
        self.status_signal.connect(self.update_status)
        self.acq_finished_signal.connect(self._on_acq_finished)
        self.camera_connected_signal.connect(self._on_camera_connected)
        
        # Central Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # =====================================================================
        # Connection & Camera
        # =====================================================================
        hw_group = QtWidgets.QGroupBox("Camera (via Experiment_Manager.vi)")
        hw_layout = QtWidgets.QGridLayout(hw_group)
        layout.addWidget(hw_group)
        
        # Row 0: VI Path
        hw_layout.addWidget(QtWidgets.QLabel("Manager VI:"), 0, 0)
        self.vi_path_edit = QtWidgets.QLineEdit(DEFAULT_VI_PATH)
        hw_layout.addWidget(self.vi_path_edit, 0, 1, 1, 3)
        
        self.browse_vi_btn = QtWidgets.QPushButton("Browse...")
        self.browse_vi_btn.clicked.connect(self.browse_vi)
        hw_layout.addWidget(self.browse_vi_btn, 0, 4)
        
        # Row 1: Start Manager / Initialize Camera / Status
        self.start_mgr_btn = QtWidgets.QPushButton("▶ Start Manager")
        self.start_mgr_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.start_mgr_btn.clicked.connect(self.start_manager)
        hw_layout.addWidget(self.start_mgr_btn, 1, 0)
        
        self.init_cam_btn = QtWidgets.QPushButton("🔌 Initialize Camera")
        self.init_cam_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.init_cam_btn.clicked.connect(self.initialize_camera)
        self.init_cam_btn.setEnabled(False)
        hw_layout.addWidget(self.init_cam_btn, 1, 1)
        
        self.shutdown_btn = QtWidgets.QPushButton("⏹ Shutdown")
        self.shutdown_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.shutdown_btn.clicked.connect(self.shutdown_manager)
        self.shutdown_btn.setEnabled(False)
        hw_layout.addWidget(self.shutdown_btn, 1, 2)
        
        self.hw_status_label = QtWidgets.QLabel("Status: Not Started")
        self.hw_status_label.setStyleSheet("color: red; font-weight: bold;")
        hw_layout.addWidget(self.hw_status_label, 1, 3, 1, 2)
        
        # =====================================================================
        # Acquisition Controls
        # =====================================================================
        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_layout = QtWidgets.QGridLayout(acq_group)
        layout.addWidget(acq_group)
        
        # Row 0: N, Mode, Invert Phase
        acq_layout.addWidget(QtWidgets.QLabel("Frames (N):"), 0, 0)
        self.frames_spinbox = QtWidgets.QSpinBox()
        self.frames_spinbox.setRange(2, 10000)
        self.frames_spinbox.setValue(NUM_FRAMES)
        self.frames_spinbox.setSingleStep(2)
        acq_layout.addWidget(self.frames_spinbox, 0, 1)
        
        acq_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 2)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["ΔT/T (dT)", "Transmission (T)"])
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        acq_layout.addWidget(self.mode_combo, 0, 3)
        
        self.invert_chk = QtWidgets.QCheckBox("Invert Phase")
        self.invert_chk.setToolTip("Swap On/Off phase assignment (XOR in LabVIEW)")
        acq_layout.addWidget(self.invert_chk, 0, 4)
        
        # Row 1: Acquire + Stop
        self.acquire_btn = QtWidgets.QPushButton(" ▶  ACQUIRE ")
        self.acquire_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; "
            "font-size: 14px; padding: 8px 20px;"
        )
        self.acquire_btn.clicked.connect(self.start_acquisition)
        self.acquire_btn.setEnabled(False)
        acq_layout.addWidget(self.acquire_btn, 1, 0, 1, 2)
        
        self.stop_acq_btn = QtWidgets.QPushButton("■ STOP")
        self.stop_acq_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; "
            "font-size: 14px; padding: 8px 20px;"
        )
        self.stop_acq_btn.clicked.connect(self.stop_acquisition)
        self.stop_acq_btn.setEnabled(False)
        acq_layout.addWidget(self.stop_acq_btn, 1, 2, 1, 2)
        
        self.batch_label = QtWidgets.QLabel("Batches: 0")
        acq_layout.addWidget(self.batch_label, 1, 4)
        
        # =====================================================================
        # Pump-Probe Scan Configuration
        # =====================================================================
        scan_group = QtWidgets.QGroupBox("Pump-Probe Delay Scan")
        scan_layout = QtWidgets.QGridLayout(scan_group)
        layout.addWidget(scan_group)
        
        # Row 0: Stage connection + Zero position
        self.connect_stage_btn = QtWidgets.QPushButton("🔌 Connect Delay Stage")
        self.connect_stage_btn.setStyleSheet(
            "background-color: #9C27B0; color: white; font-weight: bold;"
        )
        self.connect_stage_btn.clicked.connect(self.connect_delay_stage)
        self.connect_stage_btn.setEnabled(HAS_DELAY_STAGE)
        scan_layout.addWidget(self.connect_stage_btn, 0, 0, 1, 2)
        
        scan_layout.addWidget(QtWidgets.QLabel("Zero (mm):"), 0, 2)
        self.zero_spin = QtWidgets.QDoubleSpinBox()
        self.zero_spin.setRange(0, 300)
        self.zero_spin.setDecimals(3)
        self.zero_spin.setValue(GLOBAL_ZERO_POS_MM)
        self.zero_spin.setSuffix(" mm")
        scan_layout.addWidget(self.zero_spin, 0, 3)
        
        self.stage_status_label = QtWidgets.QLabel("Stage: Not connected")
        self.stage_status_label.setStyleSheet("color: gray;")
        scan_layout.addWidget(self.stage_status_label, 0, 4)
        
        # Row 1-3: 3 Intervals (Start, End, Step in fs)
        interval_labels = ["Interval 1 (fine)", "Interval 2 (mid)", "Interval 3 (coarse)"]
        defaults = [
            (-1000, 0, 50),      # Pre-zero, fine
            (0, 10000, 100),     # Early dynamics
            (10000, 100000, 1000) # Late dynamics
        ]
        self.interval_spins = []
        for row_i, (label, (s, e, st)) in enumerate(zip(interval_labels, defaults)):
            r = row_i + 1
            scan_layout.addWidget(QtWidgets.QLabel(label), r, 0)
            
            start_spin = QtWidgets.QDoubleSpinBox()
            start_spin.setRange(-1e6, 1e6)
            start_spin.setValue(s)
            start_spin.setSuffix(" fs")
            scan_layout.addWidget(start_spin, r, 1)
            
            end_spin = QtWidgets.QDoubleSpinBox()
            end_spin.setRange(-1e6, 1e6)
            end_spin.setValue(e)
            end_spin.setSuffix(" fs")
            scan_layout.addWidget(end_spin, r, 2)
            
            step_spin = QtWidgets.QDoubleSpinBox()
            step_spin.setRange(1, 1e6)
            step_spin.setValue(st)
            step_spin.setSuffix(" fs")
            scan_layout.addWidget(step_spin, r, 3)
            
            self.interval_spins.append((start_spin, end_spin, step_spin))
        
        # Header labels
        for col, txt in enumerate(["Range", "Start", "End", "Step"]):
            lbl = QtWidgets.QLabel(txt)
            lbl.setStyleSheet("font-weight: bold; font-size: 10px; color: #666;")
        
        # Row 4: Scan controls
        self.scan_frames_spin = QtWidgets.QSpinBox()
        self.scan_frames_spin.setRange(2, 10000)
        self.scan_frames_spin.setValue(100)
        scan_layout.addWidget(QtWidgets.QLabel("Frames/point:"), 4, 0)
        scan_layout.addWidget(self.scan_frames_spin, 4, 1)
        
        self.scan_points_label = QtWidgets.QLabel("Points: --")
        scan_layout.addWidget(self.scan_points_label, 4, 2)
        
        self.start_scan_btn = QtWidgets.QPushButton("▶ START SCAN")
        self.start_scan_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "font-size: 14px; padding: 8px;"
        )
        self.start_scan_btn.clicked.connect(self.start_scan)
        self.start_scan_btn.setEnabled(False)
        scan_layout.addWidget(self.start_scan_btn, 4, 3)
        
        self.stop_scan_btn = QtWidgets.QPushButton("■ STOP SCAN")
        self.stop_scan_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold;"
        )
        self.stop_scan_btn.clicked.connect(self.stop_scan)
        self.stop_scan_btn.setEnabled(False)
        scan_layout.addWidget(self.stop_scan_btn, 4, 4)
        
        # Row 5: Progress bar
        self.scan_progress = QtWidgets.QProgressBar()
        self.scan_progress.setRange(0, 100)
        self.scan_progress.setValue(0)
        scan_layout.addWidget(self.scan_progress, 5, 0, 1, 5)
        
        # =====================================================================
        # Status Bar
        # =====================================================================
        self.status_label = QtWidgets.QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
        # =====================================================================
        # Plotting Area (tabs: Live View + Delay Scan)
        # =====================================================================
        plot_tabs = QtWidgets.QTabWidget()
        layout.addWidget(plot_tabs)
        
        # Tab 1: Camera Feed
        cam_tab = QtWidgets.QWidget()
        cam_layout = QtWidgets.QVBoxLayout(cam_tab)
        self.plot_widget = pg.GraphicsLayoutWidget()
        cam_layout.addWidget(self.plot_widget)
        
        self.img_view = self.plot_widget.addPlot(title="Camera Feed")
        self.img_item = pg.ImageItem()
        self.img_view.addItem(self.img_item)
        
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.plot_widget.addItem(self.hist)
        plot_tabs.addTab(cam_tab, "📹 Camera Feed")
        
        # Tab 2: Delay Scan Plot
        scan_tab = QtWidgets.QWidget()
        scan_tab_layout = QtWidgets.QVBoxLayout(scan_tab)
        self.scan_plot_widget = pg.PlotWidget(
            title="ΔT/T vs Delay",
            labels={'left': 'ΔT/T (mean ROI)', 'bottom': 'Delay (fs)'}
        )
        self.scan_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.scan_curve = self.scan_plot_widget.plot(
            pen=pg.mkPen('#2196F3', width=2), symbol='o', symbolSize=4,
            symbolBrush='#2196F3'
        )
        scan_tab_layout.addWidget(self.scan_plot_widget)
        plot_tabs.addTab(scan_tab, "📊 Delay Scan")
        
        # Set initial colormap
        self.change_mode(0)
    
    # =========================================================================
    # Manager Control Slots
    # =========================================================================
    def browse_vi(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Manager VI", "", "LabVIEW VI (*.vi);;All Files (*)"
        )
        if path:
            self.vi_path_edit.setText(path)
    
    def start_manager(self):
        """Start the persistent Manager VI (non-blocking)."""
        if not HAS_WIN32COM:
            self.update_status("ERROR: pywin32 not installed")
            return
        
        self.hw_status_label.setText("Status: Starting LabVIEW...")
        self.hw_status_label.setStyleSheet("color: orange; font-weight: bold;")
        QtWidgets.QApplication.processEvents()
        
        self.manager.vi_path = self.vi_path_edit.text()
        
        if self.manager.start():
            self.hw_status_label.setText("Status: Manager Running (Camera not init)")
            self.hw_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
            self.start_mgr_btn.setEnabled(False)
            self.init_cam_btn.setEnabled(True)
            self.shutdown_btn.setEnabled(True)
            self.vi_path_edit.setEnabled(False)
            self.browse_vi_btn.setEnabled(False)
        else:
            self.hw_status_label.setText("Status: FAILED to start")
            self.hw_status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def initialize_camera(self):
        """Send Init command to Manager VI (main thread, with processEvents)."""
        self.hw_status_label.setText("Status: Initializing camera...")
        self.hw_status_label.setStyleSheet("color: orange; font-weight: bold;")
        QtWidgets.QApplication.processEvents()
        
        vi = self.manager.vi
        
        # Diagnostic: test each control name individually
        test_controls = [
            ("Enum", CMD_INIT),
            ("N", 100),
            ("Acq Trigger", True),
            ("Stoplive", False),
        ]
        for name, val in test_controls:
            try:
                vi.SetControlValue(name, val)
                print(f"  [OK] SetControlValue(\"{name}\", {val})")
            except Exception as e:
                print(f"  [FAIL] SetControlValue(\"{name}\", {val}) → {e}")
        
        # Test reading indicators
        test_indicators = ["Enum", "T", "DeltaT"]
        for name in test_indicators:
            try:
                val = vi.GetControlValue(name)
                print(f"  [OK] GetControlValue(\"{name}\") = {val}")
            except Exception as e:
                print(f"  [FAIL] GetControlValue(\"{name}\") → {e}")
        
        # If Command was set successfully, poll for Idle
        try:
            waited = 0.0
            while waited < 30.0:
                time.sleep(0.1)
                waited += 0.1
                QtWidgets.QApplication.processEvents()
                try:
                    if vi.GetControlValue("Enum") == CMD_IDLE:
                        self._on_camera_connected(True)
                        return
                except Exception:
                    pass
            
            self._on_camera_connected(False)
        except Exception as e:
            print(f"[ERROR] Init: {e}")
            self._on_camera_connected(False)
    
    def _on_camera_connected(self, success):
        """Thread-safe UI update after camera init."""
        if success:
            self.camera_initialized = True
            self.hw_status_label.setText("Status: Camera Ready ✓")
            self.hw_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.init_cam_btn.setEnabled(False)
            self.acquire_btn.setEnabled(True)
            # Enable scan if stage is also connected
            if self.delay_stage and self.delay_stage.is_connected:
                self.start_scan_btn.setEnabled(True)
        else:
            self.hw_status_label.setText("Status: Init FAILED")
            self.hw_status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def shutdown_manager(self):
        """Send Exit command and clean up."""
        self.acquiring = False
        self.manager.shutdown()
        self.camera_initialized = False
        self.hw_status_label.setText("Status: Not Started")
        self.hw_status_label.setStyleSheet("color: red; font-weight: bold;")
        self.start_mgr_btn.setEnabled(True)
        self.init_cam_btn.setEnabled(False)
        self.shutdown_btn.setEnabled(False)
        self.acquire_btn.setEnabled(False)
        self.stop_acq_btn.setEnabled(False)
        self.vi_path_edit.setEnabled(True)
        self.browse_vi_btn.setEnabled(True)
    
    # =========================================================================
    # Acquisition Slots
    # =========================================================================
    def start_acquisition(self):
        """Start continuous live view — send Getframe once, LabVIEW loops internally."""
        self.acquiring = True
        self.batch_count = 0
        self.acquire_btn.setEnabled(False)
        self.stop_acq_btn.setEnabled(True)
        
        vi = self.manager.vi
        n = self.frames_spinbox.value()
        
        try:
            # Set parameters
            vi.SetControlValue("N", n)
            vi.SetControlValue("Acq Trigger", True)
            vi.SetControlValue("Stoplive", False)
            
            # Send Getframe ONCE — LabVIEW loops internally
            vi.SetControlValue("Enum", CMD_GETFRAME)
            
            self.update_status(f"Live view started (N={n})...")
            
            # Start polling timer to read latest data from VI
            self._poll_timer = QtCore.QTimer()
            self._poll_timer.timeout.connect(self._poll_live_data)
            self._poll_timer.start(50)  # Read every 50ms
            
        except Exception as e:
            self.update_status(f"Acquire error: {e}")
            self._on_acq_finished()
    
    def _poll_live_data(self):
        """Read the latest T/DeltaT from the continuously-running VI."""
        if not self.acquiring:
            self._poll_timer.stop()
            return
        
        mode = "dt" if self.mode_combo.currentIndex() == 0 else "t"
        indicator = "DeltaT" if mode == "dt" else "T"
        
        try:
            if mode == "t":
                # T mode: Read Odd/Even and compute (robust)
                odd_val = self.manager.vi.GetControlValue("Odd")
                even_val = self.manager.vi.GetControlValue("Even")
                
                if odd_val is not None and even_val is not None:
                    odd = np.array(odd_val, dtype=float)
                    even = np.array(even_val, dtype=float)
                    img = (odd + even) / 2.0
                    
                    self.batch_count += 1
                    self.update_plot(img)
                    self.update_status(f"Live #{self.batch_count} ({img.shape}, T from Odd/Even)")
            else:
                # DeltaT mode: Read Odd/Even and compute
                odd_data = self.manager.vi.GetControlValue("Odd")
                even_data = self.manager.vi.GetControlValue("Even")
                
                if odd_data is not None and even_data is not None:
                    odd = np.array(odd_data, dtype=float)
                    even = np.array(even_data, dtype=float)
                    img = (even - odd) / np.where(np.abs(odd) > 1e-10, odd, 1e-10)
                    
                    self.batch_count += 1
                    self.update_plot(img)
                    self.update_status(f"Live #{self.batch_count} ({img.shape}, DeltaT)")
                    
        except Exception as e:
            self.update_status(f"Read error: {e}")
    
    def _on_acq_finished(self):
        """UI reset after acquisition ends."""
        self.acquire_btn.setEnabled(True)
        self.stop_acq_btn.setEnabled(False)
    
    def stop_acquisition(self):
        """Stop live view — set Stoplive=True, LabVIEW exits Getframe loop → Idle."""
        self.acquiring = False
        
        if hasattr(self, '_poll_timer'):
            self._poll_timer.stop()
        
        try:
            self.manager.vi.SetControlValue("Stoplive", True)
            self.update_status("Stopping live view...")
            
            # Wait for LabVIEW to return to Idle
            waited = 0.0
            while waited < 10.0:
                time.sleep(0.1)
                waited += 0.1
                QtWidgets.QApplication.processEvents()
                try:
                    if self.manager.vi.GetControlValue("Enum") == CMD_IDLE:
                        break
                except Exception:
                    break
            
            self.update_status("Live view stopped.")
        except Exception as e:
            self.update_status(f"Stop error: {e}")
        
        self._on_acq_finished()
    
    def save_data(self, img, mode):
        """Save result to HDF5."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"scan_data_{timestamp}.h5"
        try:
            with h5py.File(filename, 'w') as f:
                f.create_dataset("image", data=img)
                f.attrs['mode'] = mode
        except Exception as e:
            print(f"Save error: {e}")
    
    # =========================================================================
    # Stage Control
    # =========================================================================
    def connect_delay_stage(self):
        """Connect to Thorlabs delay stage."""
        if self.delay_stage is None:
            self.update_status("Delay stage driver not available")
            return
        
        self.update_status("Connecting to delay stage...")
        QtWidgets.QApplication.processEvents()
        
        if self.delay_stage.connect():
            self.stage_status_label.setText("Stage: Connected")
            self.stage_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.connect_stage_btn.setEnabled(False)
            self.start_scan_btn.setEnabled(self.camera_initialized)
            self.update_status("Delay stage connected. Homing...")
            QtWidgets.QApplication.processEvents()
            
            # Home the stage (blocking but with processEvents)
            if self.delay_stage.home(timeout_s=60.0):
                pos = self.delay_stage.get_position()
                self.stage_status_label.setText(f"Stage: {pos:.3f} mm")
                self.update_status(f"Stage homed at {pos:.3f} mm")
            else:
                self.update_status("WARNING: Homing failed")
        else:
            self.stage_status_label.setText("Stage: FAILED")
            self.stage_status_label.setStyleSheet("color: red; font-weight: bold;")
            self.update_status("Failed to connect to delay stage")
    
    def _fs_to_mm(self, time_fs):
        """Convert femtosecond delay to mm position (double-pass)."""
        return time_fs * SPEED_OF_LIGHT_MM_FS / 2.0
    
    def _mm_to_fs(self, distance_mm):
        """Convert mm distance to femtosecond delay (double-pass)."""
        return distance_mm * 2.0 / SPEED_OF_LIGHT_MM_FS
    
    # =========================================================================
    # Pump-Probe Scan
    # =========================================================================
    def _generate_scan_points(self):
        """Generate scan positions from 3 intervals."""
        points = []
        for start_spin, end_spin, step_spin in self.interval_spins:
            s = start_spin.value()
            e = end_spin.value()
            st = step_spin.value()
            if s < e and st > 0:
                pts = np.arange(s, e, st)
                points.extend(pts.tolist())
        # Add final endpoint
        if self.interval_spins:
            points.append(self.interval_spins[-1][1].value())
        return np.array(points)
    
    def start_scan(self):
        """Start pump-probe delay scan."""
        if self.delay_stage is None or not self.delay_stage.is_connected:
            self.update_status("Connect delay stage first!")
            return
        if not self.camera_initialized:
            self.update_status("Initialize camera first!")
            return
        
        # Generate scan points
        self.scan_points_fs = self._generate_scan_points()
        if len(self.scan_points_fs) == 0:
            self.update_status("No scan points! Check interval config.")
            return
        
        # Reset scan state
        self.scan_delays = []
        self.scan_signals = []
        self.scan_index = 0
        self.scanning = True
        self.scan_curve.setData([], [])
        
        # Setup CSV for incremental saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scan_csv_path = f"pumpprobe_{timestamp}.csv"
        with open(self.scan_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["delay_fs", "pos_mm", "signal_dT_T"])
        
        # UI state
        self.start_scan_btn.setEnabled(False)
        self.stop_scan_btn.setEnabled(True)
        self.acquire_btn.setEnabled(False)
        self.scan_progress.setValue(0)
        self.scan_points_label.setText(f"Points: 0/{len(self.scan_points_fs)}")
        
        n_pts = len(self.scan_points_fs)
        self.update_status(
            f"Scan started: {n_pts} points, "
            f"{self.scan_points_fs[0]:.0f} to {self.scan_points_fs[-1]:.0f} fs"
        )
        print(f"[SCAN] Starting {n_pts}-point scan")
        
        # Start first point
        QtCore.QTimer.singleShot(100, self._scan_move_stage)
    
    def _scan_move_stage(self):
        """Move stage to next scan point."""
        if not self.scanning or self.scan_index >= len(self.scan_points_fs):
            self._scan_complete()
            return
        
        delay_fs = self.scan_points_fs[self.scan_index]
        zero_mm = self.zero_spin.value()
        target_mm = zero_mm + self._fs_to_mm(delay_fs)
        
        self.update_status(
            f"Point {self.scan_index+1}/{len(self.scan_points_fs)}: "
            f"{delay_fs:.0f} fs → {target_mm:.4f} mm"
        )
        
        # Send move command (non-blocking)
        self.delay_stage.move_to(target_mm, wait=False)
        
        # Start polling for position stability
        self._stage_poll_count = 0
        self._stage_last_pos = None
        self._stage_stable = 0
        self._stage_timer = QtCore.QTimer()
        self._stage_timer.timeout.connect(self._scan_poll_stage)
        self._stage_timer.start(100)  # Check every 100ms
    
    def _scan_poll_stage(self):
        """Poll stage position until stable, then trigger acquisition."""
        if not self.scanning:
            self._stage_timer.stop()
            return
        
        self._stage_poll_count += 1
        
        # Timeout after 60s
        if self._stage_poll_count > 600:
            self._stage_timer.stop()
            self.update_status("WARNING: Stage timeout, skipping point")
            self.scan_index += 1
            QtCore.QTimer.singleShot(100, self._scan_move_stage)
            return
        
        current_pos = self.delay_stage.get_position()
        
        # Position stability: 3 consecutive reads within 1µm
        if self._stage_last_pos is not None:
            if abs(current_pos - self._stage_last_pos) < 0.001:
                self._stage_stable += 1
                if self._stage_stable >= 3:
                    self._stage_timer.stop()
                    # Settle time for optical vibrations
                    QtCore.QTimer.singleShot(50, self._scan_trigger_acquire)
                    return
            else:
                self._stage_stable = 0
        
        self._stage_last_pos = current_pos
    
    def _scan_trigger_acquire(self):
        """Trigger one acquisition via LabVIEW after stage is settled."""
        if not self.scanning:
            return
        
        vi = self.manager.vi
        n = self.scan_frames_spin.value()
        
        try:
            vi.SetControlValue("N", n)
            vi.SetControlValue("Acq Trigger", True)
            vi.SetControlValue("Stoplive", False)
            vi.SetControlValue("Enum", CMD_GETFRAME)
            
            # Poll for Idle (single-shot, not continuous)
            self._acq_poll_timer = QtCore.QTimer()
            self._acq_poll_timer.timeout.connect(self._scan_poll_acquire)
            self._acq_poll_waited = 0.0
            self._acq_poll_timer.start(50)
            
        except Exception as e:
            self.update_status(f"Acquire error at point {self.scan_index}: {e}")
            self.scan_index += 1
            QtCore.QTimer.singleShot(100, self._scan_move_stage)
    
    def _scan_poll_acquire(self):
        """Poll LabVIEW until Enum returns to Idle, then read result."""
        if not self.scanning:
            self._acq_poll_timer.stop()
            return
        
        self._acq_poll_waited += 0.05
        
        if self._acq_poll_waited > 60.0:
            self._acq_poll_timer.stop()
            self.update_status("WARNING: Acquire timeout, skipping point")
            self.scan_index += 1
            QtCore.QTimer.singleShot(100, self._scan_move_stage)
            return
        
        try:
            if self.manager.vi.GetControlValue("Enum") != CMD_IDLE:
                return  # Still acquiring
        except Exception:
            return
        
        # Done! Read result
        self._acq_poll_timer.stop()
        
        delay_fs = self.scan_points_fs[self.scan_index]
        
        try:
            odd_data = self.manager.vi.GetControlValue("Odd")
            even_data = self.manager.vi.GetControlValue("Even")
            if odd_data is not None and even_data is not None:
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                img = (even - odd) / np.where(np.abs(odd) > 1e-10, odd, 1e-10)
                
                # Update camera feed
                self.update_plot(img)
                
                # Calculate mean ROI signal (full frame for now)
                signal = float(np.nanmean(img))
                
                # Store
                self.scan_delays.append(delay_fs)
                self.scan_signals.append(signal)
                
                # Update live plot
                self.scan_curve.setData(self.scan_delays, self.scan_signals)
                
                # Incremental CSV save
                actual_mm = self.delay_stage.get_position()
                with open(self.scan_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{delay_fs:.2f}", f"{actual_mm:.4f}", f"{signal:.8e}"])
                
                print(
                    f"[SCAN] Point {self.scan_index+1}: "
                    f"{delay_fs:.0f} fs = {signal:.4e}"
                )
            else:
                print(f"[SCAN] No data at point {self.scan_index+1}")
                
        except Exception as e:
            print(f"[SCAN] Read error: {e}")
        
        # Update progress
        self.scan_index += 1
        pct = int(100 * self.scan_index / len(self.scan_points_fs))
        self.scan_progress.setValue(pct)
        self.scan_points_label.setText(
            f"Points: {self.scan_index}/{len(self.scan_points_fs)}"
        )
        
        # Next point
        QtCore.QTimer.singleShot(50, self._scan_move_stage)
    
    def _scan_complete(self):
        """Scan finished — save final data and reset UI."""
        self.scanning = False
        
        # Save final NPZ
        if self.scan_delays:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            npz_path = f"pumpprobe_{timestamp}_final.npz"
            np.savez(
                npz_path,
                delays_fs=np.array(self.scan_delays),
                signals=np.array(self.scan_signals),
                zero_mm=self.zero_spin.value(),
                frames_per_point=self.scan_frames_spin.value()
            )
            self.update_status(
                f"Scan complete! {len(self.scan_delays)} points saved to {npz_path}"
            )
            print(f"[SCAN] Final data saved: {npz_path}")
        else:
            self.update_status("Scan complete (no data collected)")
        
        self.start_scan_btn.setEnabled(True)
        self.stop_scan_btn.setEnabled(False)
        self.acquire_btn.setEnabled(True)
    
    def stop_scan(self):
        """Emergency stop — abort scan."""
        print("[SCAN] STOP requested")
        self.scanning = False
        
        # Stop any active timers
        if hasattr(self, '_stage_timer'):
            self._stage_timer.stop()
        if hasattr(self, '_acq_poll_timer'):
            self._acq_poll_timer.stop()
        
        self._scan_complete()
    
    # =========================================================================
    # Display Slots
    # =========================================================================
    def change_mode(self, index):
        if index == 0:  # DT mode — Blue → White (zero) → Red
            pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            color = np.array([
                [0,   0,   180, 255],   # dark blue
                [100, 150, 255, 255],   # light blue
                [255, 255, 255, 255],   # white (zero)
                [255, 150, 100, 255],   # light red
                [180, 0,   0,   255],   # dark red
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(pos, color)
            self.img_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        else:  # T mode — Magma (or jet fallback)
            try:
                self.img_item.setLookupTable(pg.colormap.get('magma').getLookupTable())
            except Exception:
                try:
                    self.img_item.setLookupTable(pg.colormap.get('CET-L8').getLookupTable())
                except Exception:
                    # Manual jet-like fallback
                    pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                    color = np.array([
                        [0, 0, 128, 255], [0, 255, 255, 255], [0, 255, 0, 255],
                        [255, 255, 0, 255], [255, 0, 0, 255]
                    ], dtype=np.ubyte)
                    cmap = pg.ColorMap(pos, color)
                    self.img_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        
    def update_plot(self, img_data):
        self.img_item.setImage(img_data.T)
        
    def update_status(self, msg):
        self.status_label.setText(f"Status: {msg}")
        
    def closeEvent(self, event):
        self.acquiring = False
        self.scanning = False
        
        # Disconnect stages
        if self.delay_stage and self.delay_stage.is_connected:
            try:
                self.delay_stage.disconnect()
            except Exception:
                pass
        if self.twins_stage and hasattr(self.twins_stage, 'is_connected') and self.twins_stage.is_connected:
            try:
                self.twins_stage.disconnect()
            except Exception:
                pass
        
        self.manager.close()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
