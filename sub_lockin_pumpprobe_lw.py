"""
sub_lockin_pumpprobe_lw.py
Time-resolved scan at a single wavelength using Lock-In.
Ported from TWINSMCT app to Camera hybrid app.
"""

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
import sys, os

from driver_lockin import LockInDriver
from sub_interval_lw import DelayIntervalWidget
from sub_lockin_lw import LockInControlWidget

class PumpProbeWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    data_update = QtCore.pyqtSignal(float, float, float, float) # mm_delay, mm_abs, val, phase
    finished_scan = QtCore.pyqtSignal()
    
    def __init__(self, points_mm, delay_stage, channel_code=0, wait_time=0.0):
        super().__init__()
        self.points = points_mm
        self.delay = delay_stage
        self.wait_time = wait_time
        
        # Use stage zero position (or actual home if required)
        self.home_pos = self.delay.zero_position if self.delay else 0.0
        print(f"[Worker] Using Global Zero: {self.home_pos:.4f} mm")
        
        self.channel_code = channel_code # 0=X, 1=Y, 2=R, 3=Theta
        self.lockin = LockInDriver()
        self.is_running = True
    
    def run(self):
        total = len(self.points)
        
        for i, pos in enumerate(self.points):
            if not self.is_running: break
            
            # Move (Home + Relative Offset)
            target = self.home_pos + pos
            if self.delay:
                self.delay.move_to(target)
            
            # Wait before reading Lock-In
            if self.wait_time > 0:
                import time
                time.sleep(self.wait_time)
            
            # Read
            # Use channel code
            val = self.lockin.read_value(channel_code=self.channel_code, samples=3)
            # Also read Phase (Channel 3 = Theta)
            phase = self.lockin.read_value(channel_code=3, samples=3)
            
            self.data_update.emit(pos, target, val, phase)
            self.progress.emit(i+1, total)
            
        self.finished_scan.emit()

    def stop(self):
        self.is_running = False

class LockInPumpProbeWindow(QtWidgets.QWidget):
    def __init__(self, delay_stage=None):
        super().__init__()
        self.delay_stage = delay_stage
        self.setWindowTitle("Lock-In Pump Probe Scan")
        self.setMinimumSize(900, 600)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        
        # Scroll area for controls
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(360)
        
        ctrl = QtWidgets.QWidget()
        ctrl_l = QtWidgets.QVBoxLayout(ctrl)
        
        # 1. Delay Config
        self.delay_widget = DelayIntervalWidget()
        ctrl_l.addWidget(self.delay_widget)
        
        # 2. Configuration Group (Stage + Lockin + Wait)
        gb_conf = QtWidgets.QGroupBox("Configuration")
        gc_l = QtWidgets.QVBoxLayout(gb_conf)
        
        # Pump/Probe selection
        self.chk_pump_stage = QtWidgets.QCheckBox("Pump on Stage (Inverted Axis)")
        self.chk_pump_stage.setToolTip("Checked: Moving Pump (Time increases -> Stage increases path)\nUnchecked: Moving Probe (Time increases -> Stage decreases path)")
        gc_l.addWidget(self.chk_pump_stage)
        
        # Global Home Display
        self.lbl_home_info = QtWidgets.QLabel("Global Zero: (Checking...)")
        self.lbl_home_info.setStyleSheet("color: #2196F3; font-weight: bold;")
        gc_l.addWidget(self.lbl_home_info)
        
        # Lock-in
        self.lockin_widget = LockInControlWidget()
        gc_l.addWidget(self.lockin_widget)
        
        # Wait
        self.chk_auto_wait = QtWidgets.QCheckBox("Auto Wait (2 * TimeConstant)")
        self.chk_auto_wait.setChecked(True)
        self.chk_auto_wait.stateChanged.connect(self.update_wait_config)
        gc_l.addWidget(self.chk_auto_wait)
        
        self.lbl_wait_val = QtWidgets.QLabel("Wait: 0.000 s")
        gc_l.addWidget(self.lbl_wait_val)
        
        ctrl_l.addWidget(gb_conf)
        
        # 3. Scan Actions
        g_run = QtWidgets.QGroupBox("Scan Actions")
        gr_l = QtWidgets.QVBoxLayout(g_run)
        
        self.btn_run = QtWidgets.QPushButton("▶ Run Scan")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run_scan)
        gr_l.addWidget(self.btn_run)
        
        self.btn_stop = QtWidgets.QPushButton("⏹ Stop")
        self.btn_stop.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_stop.setEnabled(False) 
        gr_l.addWidget(self.btn_stop)
        
        self.prog = QtWidgets.QProgressBar()
        gr_l.addWidget(self.prog)
        
        # Sample Name
        gr_l.addWidget(QtWidgets.QLabel("Sample Name:"))
        self.txt_sample_name = QtWidgets.QLineEdit("measure")
        gr_l.addWidget(self.txt_sample_name)
        
        self.btn_save = QtWidgets.QPushButton("💾 Save Data")
        self.btn_save.clicked.connect(self.save_data)
        self.btn_save.setEnabled(False)
        gr_l.addWidget(self.btn_save)
        
        ctrl_l.addWidget(g_run)
        
        # 4. Live Stage Control
        g_stage = QtWidgets.QGroupBox("Live Stage Control")
        gs_l = QtWidgets.QVBoxLayout(g_stage)
        
        # Absolute Move
        h_abs = QtWidgets.QHBoxLayout()
        h_abs.addWidget(QtWidgets.QLabel("Abs (mm):"))
        self.spin_stage_abs = QtWidgets.QDoubleSpinBox()
        self.spin_stage_abs.setRange(-200.0, 300.0)
        self.spin_stage_abs.setDecimals(3)
        self.spin_stage_abs.setSingleStep(0.01)
        h_abs.addWidget(self.spin_stage_abs)
        self.btn_stage_go = QtWidgets.QPushButton("Go")
        self.btn_stage_go.clicked.connect(self._move_stage_abs)
        h_abs.addWidget(self.btn_stage_go)
        gs_l.addLayout(h_abs)
        
        # Relative Step
        h_rel = QtWidgets.QHBoxLayout()
        h_rel.addWidget(QtWidgets.QLabel("Step (fs):"))
        self.spin_stage_rel = QtWidgets.QDoubleSpinBox()
        self.spin_stage_rel.setRange(0.1, 10000.0)
        self.spin_stage_rel.setValue(100.0)
        h_rel.addWidget(self.spin_stage_rel)
        
        self.btn_stage_neg = QtWidgets.QPushButton("- Step")
        self.btn_stage_neg.clicked.connect(lambda: self._move_stage_rel(-1))
        h_rel.addWidget(self.btn_stage_neg)
        
        self.btn_stage_pos = QtWidgets.QPushButton("+ Step")
        self.btn_stage_pos.clicked.connect(lambda: self._move_stage_rel(1))
        h_rel.addWidget(self.btn_stage_pos)
        gs_l.addLayout(h_rel)
        
        ctrl_l.addWidget(g_stage)
        
        ctrl_l.addStretch()
        scroll.setWidget(ctrl)
        layout.addWidget(scroll)
        
        # Plot
        self.plot = pg.PlotWidget(title="Pump-Probe Trace (Lock-In)")
        self.plot.setLabel('left', 'Signal', 'V')
        self.plot.setLabel('bottom', 'Delay', 'fs') 
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen('#FF5722', width=2), symbol='o', symbolSize=4, symbolBrush='#FF5722')
        layout.addWidget(self.plot)

        # Init Wait Label
        self.update_wait_config()
        
    def showEvent(self, event):
        super().showEvent(event)
        self.update_home_display()
        
    def update_home_display(self):
        try:
            if self.delay_stage:
                h = self.delay_stage.zero_position
                self.lbl_home_info.setText(f"Global Zero: {h:.4f} mm")
            else:
                self.lbl_home_info.setText("Global Zero: (No Stage)")
        except:
            self.lbl_home_info.setText("Global Zero: (Error)")
    
    def update_wait_config(self):
        if self.chk_auto_wait.isChecked():
            # Use the existing driver instance through the widget to prevent instantiation issues
            l = self.lockin_widget.lockin
            if l and l.is_connected:
                tc = l.get_time_constant()
                wait = 2.0 * tc
                self.lbl_wait_val.setText(f"Wait: {wait:.4f} s (TC={tc:.4f}s)")
            else:
                self.lbl_wait_val.setText("Wait: (Connect Lock-in first)")
        else:
            self.lbl_wait_val.setText("Wait: 0.000 s (Manual)")

    def get_wait_time(self):
        l = self.lockin_widget.lockin
        if hasattr(self, 'chk_auto_wait') and self.chk_auto_wait.isChecked() and l and l.is_connected:
            return 2.0 * l.get_time_constant()
        return 0.0


    # --- Live Stage Methods ---
    def _move_stage_abs(self):
        if not self.delay_stage or not self.delay_stage.is_connected:
            print("[LockIn] Delay stage not connected.")
            return
        target = self.spin_stage_abs.value()
        self.delay_stage.move_to(target)
        self.update_home_display()

    def _move_stage_rel(self, direction):
        if not self.delay_stage or not self.delay_stage.is_connected:
            print("[LockIn] Delay stage not connected.")
            return
            
        step_fs = self.spin_stage_rel.value()
        
        # Convert fs to mm. (1 fs = ~1.5e-4 mm distance change, but remember optical path is *2)
        # using the same math as interval_widget:
        SPEED_OF_LIGHT_MM_FS = 0.000299792458
        step_mm = (step_fs * SPEED_OF_LIGHT_MM_FS / 2.0)
        
        is_pump = self.chk_pump_stage.isChecked()
        if not is_pump:
            # Probe mode: +fs delay means moving stage backwards (-mm)
            step_mm = -step_mm
            
        step_mm *= direction
        
        current_pos = self.delay_stage.get_position()
        target = current_pos + step_mm
        self.delay_stage.move_to(target)
        self.update_home_display()
        
        # Update abs spinbox to match target
        self.spin_stage_abs.setValue(target)

    def run_scan(self):
        try:
            wait_time = self.get_wait_time()
            
            # Use points from widget
            is_pump = self.chk_pump_stage.isChecked()
            points = self.delay_widget.get_scan_points_mm(invert_stage=is_pump)
            
            if len(points) == 0:
                print("[LockIn PP] No points generated")
                return

            self.btn_run.setEnabled(False)
            self.curve.setData([], [])
            
            # Init Data Storage
            self.scan_data_mm = []
            self.scan_data_fs = []
            self.scan_data_signal = []
            self.scan_data_phase = []
            self.scan_data_stage_mm = []
            
            chan = self.lockin_widget.get_channel_code()
            self.plot.setLabel('left', f"Signal ({self.lockin_widget.get_channel_name()})")
            
            # Worker fetches home_pos internally
            self.worker = PumpProbeWorker(points, self.delay_stage, channel_code=chan, wait_time=wait_time)
            self.worker.data_update.connect(lambda p, abs_p, v, ph: self.update_plot(p, abs_p, v, ph, is_pump))
            
            self.prog.setMaximum(len(points))
            self.worker.progress.connect(self.prog.setValue)
            
            self.worker.finished_scan.connect(self.on_scan_finished)
            self.worker.start()
            
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_save.setEnabled(False)
            
        except Exception as e:
            print(f"[LockIn PP] Run Error: {e}")

    def stop_scan(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            print("[LockIn PP] Stop requested...")

    def on_scan_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        print("[LockIn PP] Scan finished.")

    def save_data(self):
        if not hasattr(self, 'scan_data_signal') or len(self.scan_data_signal) == 0:
            return

        from datetime import datetime
        import os
        
        # Base Directory
        base_dir = r"D:\pumpprobedata"
        now = datetime.now()
        
        # Structure: Year / Month / Day
        target_dir = os.path.join(base_dir, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
        
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except:
                target_dir = os.getcwd()
                
        # Filename
        prefix = now.strftime("%H%M%S_")
        name = self.txt_sample_name.text().strip()
        if not name: name = "measure"
        default_name = f"{name}_LockIn_{prefix}.npz"
        
        initial_path = os.path.join(target_dir, default_name)
        
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save LockIn Data", initial_path, "NPZ Files (*.npz)")
        
        if fname:
            try:
                np.savez(fname, 
                         delay_mm=np.array(self.scan_data_mm),
                         delay_fs=np.array(self.scan_data_fs),
                         signal=np.array(self.scan_data_signal),
                         phase=np.array(self.scan_data_phase),
                         stage_pos_mm=np.array(self.scan_data_stage_mm))
                print(f"[LockIn PP] Saved to: {fname}")
            except Exception as e:
                print(f"[LockIn PP] Save Error: {e}")

    def update_plot(self, pos_mm, abs_mm, val, phase, is_pump):
        # Convert mm to fs for plotting
        pos_fs = self.delay_widget.get_fs_from_mm(pos_mm, invert_stage=is_pump)
        
        # Store Data
        if hasattr(self, 'scan_data_mm'):
            self.scan_data_mm.append(pos_mm)
            self.scan_data_fs.append(pos_fs)
            self.scan_data_signal.append(val)
            self.scan_data_phase.append(phase)
            if not hasattr(self, 'scan_data_stage_mm'): self.scan_data_stage_mm = []
            self.scan_data_stage_mm.append(abs_mm)
        
        cur_x, cur_y = self.curve.getData()
        if cur_x is None: cur_x = []
        if cur_y is None: cur_y = []
        self.curve.setData(np.append(cur_x, pos_fs), np.append(cur_y, val))
