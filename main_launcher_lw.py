"""
Main Launcher for LabVIEW Hybrid Camera System.

Entry point: launches LabVIEW Manager and opens sub-windows
for Live View, Pump-Probe scanning, etc.

Usage:
    python main_launcher_lw.py
"""

import sys
import os

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
except ImportError:
    print("Error: pyqtgraph required. pip install pyqtgraph pyqt6")
    sys.exit(1)

# Local modules
from labview_manager import LabVIEWManager

# Optional stage drivers
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
# LED Status Widget
# =============================================================================

class LEDIndicator(QtWidgets.QWidget):
    """Colored LED circle for hardware status."""
    
    def __init__(self, label="", parent=None):
        super().__init__(parent)
        self._color = "#888888"  # gray = not connected
        self._label = label
        self.setFixedSize(120, 40)
    
    def set_color(self, color):
        self._color = color
        self.update()
    
    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor, QFont
        from PyQt6.QtCore import Qt
        
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw circle
        p.setBrush(QColor(self._color))
        p.setPen(QColor("#333333"))
        p.drawEllipse(5, 10, 20, 20)
        
        # Draw label
        p.setPen(QColor("#CCCCCC"))
        p.setFont(QFont("Arial", 9))
        p.drawText(30, 8, 85, 24, Qt.AlignmentFlag.AlignVCenter, self._label)
        
        p.end()


# =============================================================================
# Main Launcher Window
# =============================================================================

class MainLauncher(QtWidgets.QWidget):
    """
    Main entry point: connect hardware, then launch sub-windows.
    """
    
    def __init__(self):
        super().__init__()
        
        # Shared drivers (singletons)
        self.manager = LabVIEWManager()
        self.delay_stage = DelayStageDriver() if HAS_DELAY_STAGE else None
        self.twins_stage = StageDriver() if HAS_TWINS_STAGE else None
        
        # Sub-windows
        self.live_window = None
        self.pp_window = None
        self.twins_window = None
        self.twins_pp_window = None
        self.deltat_window = None
        self.kspace_window = None
        
        self.setWindowTitle("LabVIEW Hybrid Camera — Main Launcher")
        self.setMinimumSize(460, 500)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Dark theme
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #e0e0e0; font-family: Arial; }
            QGroupBox { 
                border: 1px solid #555; border-radius: 6px; margin-top: 12px;
                padding-top: 16px; font-weight: bold; font-size: 12px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; padding: 2px 8px;
                background-color: #1e1e1e; 
            }
            QPushButton {
                background-color: #333; color: #e0e0e0; border: 1px solid #555;
                border-radius: 4px; padding: 8px 16px; font-size: 12px;
            }
            QPushButton:hover { background-color: #444; }
            QPushButton:disabled { color: #666; background-color: #2a2a2a; }
        """)
        
        # =================================================================
        # Title
        # =================================================================
        title = QtWidgets.QLabel("LabVIEW Hybrid Camera System")
        title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #64B5F6; "
            "padding: 10px; text-align: center;"
        )
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # =================================================================
        # Hardware Status LEDs
        # =================================================================
        led_group = QtWidgets.QGroupBox("Hardware Status")
        led_layout = QtWidgets.QHBoxLayout(led_group)
        layout.addWidget(led_group)
        
        self.led_lv = LEDIndicator("LabVIEW")
        self.led_cam = LEDIndicator("Camera")
        self.led_delay = LEDIndicator("Delay Stage")
        self.led_twins = LEDIndicator("Twins Stage")
        
        led_layout.addWidget(self.led_lv)
        led_layout.addWidget(self.led_cam)
        led_layout.addWidget(self.led_delay)
        led_layout.addWidget(self.led_twins)
        
        # =================================================================
        # Manager Controls
        # =================================================================
        mgr_group = QtWidgets.QGroupBox("LabVIEW Manager")
        mgr_layout = QtWidgets.QGridLayout(mgr_group)
        layout.addWidget(mgr_group)
        
        self.start_mgr_btn = QtWidgets.QPushButton("🚀 Start Manager")
        self.start_mgr_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; font-size: 13px;"
        )
        self.start_mgr_btn.clicked.connect(self._start_manager)
        mgr_layout.addWidget(self.start_mgr_btn, 0, 0)
        
        self.init_cam_btn = QtWidgets.QPushButton("📷 Init Camera")
        self.init_cam_btn.clicked.connect(self._init_camera)
        self.init_cam_btn.setEnabled(False)
        mgr_layout.addWidget(self.init_cam_btn, 0, 1)
        
        self.shutdown_btn = QtWidgets.QPushButton("⏹ Shutdown")
        self.shutdown_btn.setStyleSheet(
            "background-color: #555; color: #ccc;"
        )
        self.shutdown_btn.clicked.connect(self._shutdown)
        self.shutdown_btn.setEnabled(False)
        mgr_layout.addWidget(self.shutdown_btn, 0, 2)
        
        # =================================================================
        # Stage Controls
        # =================================================================
        stage_group = QtWidgets.QGroupBox("Stage Controls")
        stage_layout = QtWidgets.QGridLayout(stage_group)
        layout.addWidget(stage_group)
        
        self.connect_delay_btn = QtWidgets.QPushButton("🔗 Connect Delay Stage")
        self.connect_delay_btn.clicked.connect(self._connect_delay)
        self.connect_delay_btn.setEnabled(HAS_DELAY_STAGE)
        stage_layout.addWidget(self.connect_delay_btn, 0, 0)
        
        self.home_delay_btn = QtWidgets.QPushButton("🏠 Home Delay")
        self.home_delay_btn.clicked.connect(self._home_delay)
        self.home_delay_btn.setEnabled(False)
        stage_layout.addWidget(self.home_delay_btn, 0, 1)
        
        self.connect_twins_btn = QtWidgets.QPushButton("🔗 Connect Twins Stage")
        self.connect_twins_btn.clicked.connect(self._connect_twins)
        self.connect_twins_btn.setEnabled(HAS_TWINS_STAGE)
        stage_layout.addWidget(self.connect_twins_btn, 1, 0)
        
        # =================================================================
        # Sub-Window Launchers
        # =================================================================
        launch_group = QtWidgets.QGroupBox("Sub-Windows")
        launch_layout = QtWidgets.QHBoxLayout(launch_group)
        layout.addWidget(launch_group)
        
        self.live_btn = QtWidgets.QPushButton("📹 Live View")
        self.live_btn.setStyleSheet(
            "background-color: #009688; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px;"
        )
        self.live_btn.clicked.connect(self._open_live)
        launch_layout.addWidget(self.live_btn)
        
        self.deltat_btn = QtWidgets.QPushButton("\u0394T Live + Stage")
        self.deltat_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px;"
        )
        self.deltat_btn.clicked.connect(self._open_deltat)
        launch_layout.addWidget(self.deltat_btn)
        
        self.pp_btn = QtWidgets.QPushButton("🔬 Pump-Probe Scan")
        self.pp_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px;"
        )
        self.pp_btn.clicked.connect(self._open_pumpprobe)
        launch_layout.addWidget(self.pp_btn)
        
        self.twins_btn = QtWidgets.QPushButton("🌊 Twins FTIR")
        self.twins_btn.setStyleSheet(
            "background-color: #9C27B0; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px;"
        )
        self.twins_btn.clicked.connect(self._open_twins)
        launch_layout.addWidget(self.twins_btn)
        
        self.twins_pp_btn = QtWidgets.QPushButton("🌈 Twins Pump-Probe")
        self.twins_pp_btn.setStyleSheet(
            "background-color: #E91E63; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px;"
        )
        self.twins_pp_btn.clicked.connect(self._open_twins_pumpprobe)
        launch_layout.addWidget(self.twins_pp_btn)
        
        self.kspace_btn = QtWidgets.QPushButton("K-Space Hyperspectral")
        self.kspace_btn.setStyleSheet(
            "background-color: #00BCD4; color: white; font-weight: bold; "
            "font-size: 14px; padding: 12px;"
        )
        self.kspace_btn.clicked.connect(self._open_kspace)
        launch_layout.addWidget(self.kspace_btn)
        
        # =================================================================
        # Log
        # =================================================================
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet(
            "font-family: Consolas; font-size: 10px; "
            "background-color: #111; color: #aaa;"
        )
        layout.addWidget(self.log_text)
        
        self._log("Ready. Start Manager to begin.")
    
    # =========================================================================
    # Logging
    # =========================================================================
    
    def _log(self, msg):
        self.log_text.append(msg)
        print(msg)
    
    # =========================================================================
    # Manager
    # =========================================================================
    
    def _start_manager(self):
        self._log("Starting LabVIEW Manager...")
        self.start_mgr_btn.setEnabled(False)
        
        # Run on a timer so the UI doesn't freeze
        QtCore.QTimer.singleShot(100, self._do_start_manager)
    
    def _do_start_manager(self):
        success = self.manager.start()
        if success:
            self.led_lv.set_color("#4CAF50")  # green
            self.init_cam_btn.setEnabled(True)
            self.shutdown_btn.setEnabled(True)
            self._log("[OK] Manager started")
        else:
            self.led_lv.set_color("#f44336")  # red
            self.start_mgr_btn.setEnabled(True)
            self._log("[FAIL] Manager failed to start")
    
    def _init_camera(self):
        self._log("Initializing camera...")
        self.init_cam_btn.setEnabled(False)
        QtCore.QTimer.singleShot(100, self._do_init_camera)
    
    def _do_init_camera(self):
        success = self.manager.initialize_camera()
        if success:
            self.led_cam.set_color("#4CAF50")
            self.live_btn.setEnabled(True)
            # PP scan needs delay stage too
            if self.delay_stage and self.delay_stage.is_connected and self.delay_stage.is_homed:
                self.pp_btn.setEnabled(True)
            self._log("[OK] Camera initialized")
        else:
            self.led_cam.set_color("#f44336")
            self.init_cam_btn.setEnabled(True)
            self._log("[FAIL] Camera init failed")
    
    def _shutdown(self):
        self._log("Shutting down Manager...")
        self.manager.shutdown()
        self.led_lv.set_color("#888888")
        self.led_cam.set_color("#888888")
        self.live_btn.setEnabled(False)
        self.pp_btn.setEnabled(False)
        self.init_cam_btn.setEnabled(False)
        self.shutdown_btn.setEnabled(False)
        self.start_mgr_btn.setEnabled(True)
        self._log("[OK] Shutdown complete")
    
    # =========================================================================
    # Stages
    # =========================================================================
    
    def _connect_delay(self):
        if not self.delay_stage:
            return
        self._log("Connecting delay stage...")
        self.connect_delay_btn.setEnabled(False)
        QtCore.QTimer.singleShot(100, self._do_connect_delay)
    
    def _do_connect_delay(self):
        success = False
        try:
            success = self.delay_stage.connect()
        except Exception as e:
            self._log(f"[WARN] Real connect failed: {e}")
        
        if success:
            self.led_delay.set_color("#FFC107")
            self.home_delay_btn.setEnabled(True)
            self._log("[OK] Delay stage connected")
        else:
            # Simulated connection for UI testing
            self.delay_stage.is_connected = True
            self.delay_stage.is_homed = True
            self.led_delay.set_color("#2196F3")  # blue = simulated
            self.home_delay_btn.setEnabled(True)
            self._log("[SIM] Delay stage — simulated connection (no hardware)")
    
    def _home_delay(self):
        self._log("Homing delay stage (may take ~60s)...")
        self.home_delay_btn.setEnabled(False)
        QtCore.QTimer.singleShot(100, self._do_home_delay)
    
    def _do_home_delay(self):
        if self.delay_stage.home():
            self.led_delay.set_color("#4CAF50")  # green = homed
            if self.manager.camera_initialized:
                self.pp_btn.setEnabled(True)
            self._log("[OK] Delay stage homed")
        else:
            self.led_delay.set_color("#FF9800")  # orange = error
            self.home_delay_btn.setEnabled(True)
            self._log("[FAIL] Homing failed")
    
    def _connect_twins(self):
        if not self.twins_stage:
            return
        self._log("Connecting Twins stage...")
        self.connect_twins_btn.setEnabled(False)
        QtCore.QTimer.singleShot(100, self._do_connect_twins)
    
    def _do_connect_twins(self):
        success = False
        try:
            success = self.twins_stage.connect()
        except Exception as e:
            self._log(f"[WARN] Real connect failed: {e}")
        
        if success:
            self.led_twins.set_color("#4CAF50")
            self._log("[OK] Twins stage connected")
        else:
            # Simulated connection for UI testing
            self.twins_stage.is_connected = True
            self.led_twins.set_color("#2196F3")  # blue = simulated
            self._log("[SIM] Twins stage — simulated connection (no hardware)")
    
    # =========================================================================
    # Sub-Windows
    # =========================================================================
    
    def _open_live(self):
        from sub_live_lw import LiveViewWindow
        
        if self.live_window is None or not self.live_window.isVisible():
            self.live_window = LiveViewWindow(self.manager, self.delay_stage)
        self.live_window.show()
        self.live_window.raise_()
        self._log("Opened Live View window")
    
    def _open_pumpprobe(self):
        from sub_pumpprobe_lw import PumpProbeScanWindow
        
        if self.pp_window is None or not self.pp_window.isVisible():
            self.pp_window = PumpProbeScanWindow(
                self.manager, self.delay_stage, live_window=self.live_window
            )
        self.pp_window.show()
        self.pp_window.raise_()
        self._log("Opened Pump-Probe Scan window")
    
    def _open_deltat(self):
        from sub_deltat_lw import DeltaTWindow
        
        if self.deltat_window is None or not self.deltat_window.isVisible():
            self.deltat_window = DeltaTWindow(self.manager, self.delay_stage)
        self.deltat_window.show()
        self.deltat_window.raise_()
        self._log("Opened DeltaT (Live + Stage) window")
    
    def _open_twins(self):
        from sub_twins_lw import TwinsWindow
        
        if self.twins_window is None or not self.twins_window.isVisible():
            self.twins_window = TwinsWindow(
                self.manager, self.twins_stage, live_window=self.live_window
            )
        self.twins_window.show()
        self.twins_window.raise_()
        self._log("Opened Twins FTIR window")
    
    def _open_twins_pumpprobe(self):
        from sub_twins_pumpprobe_lw import TwinsPumpProbeWindow
        
        if self.twins_pp_window is None or not self.twins_pp_window.isVisible():
            self.twins_pp_window = TwinsPumpProbeWindow(
                self.manager, self.twins_stage, self.delay_stage,
                live_window=self.live_window
            )
        self.twins_pp_window.show()
        self.twins_pp_window.raise_()
        self._log("Opened Twins Pump-Probe window")
    
    def _open_kspace(self):
        from sub_kspace_lw import KSpaceWindow
        
        if self.kspace_window is None or not self.kspace_window.isVisible():
            self.kspace_window = KSpaceWindow(
                self.manager, self.twins_stage, live_window=self.live_window
            )
        self.kspace_window.show()
        self.kspace_window.raise_()
        self._log("Opened K-Space Hyperspectral window")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def closeEvent(self, event):
        """Clean shutdown: close sub-windows, disconnect hardware."""
        # Close sub-windows
        if self.live_window:
            self.live_window.close()
        if self.pp_window:
            self.pp_window.close()
        if self.twins_window:
            self.twins_window.close()
        if self.twins_pp_window:
            self.twins_pp_window.close()
        if self.deltat_window:
            self.deltat_window.close()
        if self.kspace_window:
            self.kspace_window.close()
        
        # Disconnect stages
        if self.delay_stage and self.delay_stage.is_connected:
            self.delay_stage.disconnect()
        if self.twins_stage and self.twins_stage.is_connected:
            self.twins_stage.disconnect()
        
        # Shutdown manager
        self.manager.close()
        
        event.accept()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    launcher = MainLauncher()
    launcher.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
