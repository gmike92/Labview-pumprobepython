from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from driver_lockin import LockInDriver

class LockInControlWidget(QtWidgets.QWidget):
    """
    Sub-UI for controlling the SR865A Lock-In amplifier.
    Ported from TWINSMCT app.
    """
    def __init__(self):
        super().__init__()
        self.lockin = LockInDriver()
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("LI Chan:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["X (0)", "Y (1)", "R (2)", "Theta (3)"])
        self.combo.setCurrentIndex(0) # Default X? Or R?
        h1.addWidget(self.combo)
        layout.addLayout(h1)
        
        h2 = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Not Conn")
        h2.addWidget(self.lbl_status)
        
        self.txt_resource = QtWidgets.QLineEdit("USB0::0xB506::0x2000::004418::INSTR")
        h2.addWidget(self.txt_resource)
        
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self.connect_lockin)
        h2.addWidget(self.btn_connect)
        
        self.btn_update = QtWidgets.QPushButton("Refresh")
        self.btn_update.clicked.connect(self.update_info)
        h2.addWidget(self.btn_update)
        layout.addLayout(h2)
        
        # Harmonic
        h_harm = QtWidgets.QHBoxLayout()
        h_harm.addWidget(QtWidgets.QLabel("Harm:"))
        self.spin_harm = QtWidgets.QSpinBox()
        self.spin_harm.setRange(1, 10)
        self.spin_harm.setValue(1)
        # Connect to driver
        self.spin_harm.valueChanged.connect(self.set_harmonic)
        h_harm.addWidget(self.spin_harm)
        layout.addLayout(h_harm)
        
        # Time Constant
        h_tc = QtWidgets.QHBoxLayout()
        h_tc.addWidget(QtWidgets.QLabel("TC:"))
        self.combo_tc = QtWidgets.QComboBox()
        # Common TC values (1, 3 sequence)
        self.tc_values = [
            1e-6, 3e-6, 10e-6, 30e-6, 100e-6, 300e-6, 
            1e-3, 3e-3, 10e-3, 30e-3, 100e-3, 300e-3,
            1.0, 3.0, 10.0, 30.0
        ]
        self.tc_labels = [
            "1 us", "3 us", "10 us", "30 us", "100 us", "300 us",
            "1 ms", "3 ms", "10 ms", "30 ms", "100 ms", "300 ms",
            "1 s", "3 s", "10 s", "30 s"
        ]
        self.combo_tc.addItems(self.tc_labels)
        self.combo_tc.currentIndexChanged.connect(self.set_time_constant)
        h_tc.addWidget(self.combo_tc)
        layout.addLayout(h_tc)
        
        # Live Readout
        h_live = QtWidgets.QHBoxLayout()
        self.chk_live = QtWidgets.QCheckBox("Live Monitor")
        self.chk_live.stateChanged.connect(self.toggle_live)
        h_live.addWidget(self.chk_live)
        self.lbl_live_val = QtWidgets.QLabel("R: - | P: -")
        h_live.addWidget(self.lbl_live_val)
        layout.addLayout(h_live)
        
        # Timer
        self.timer_live = QtCore.QTimer()
        self.timer_live.setInterval(500) # 2 Hz
        self.timer_live.timeout.connect(self.update_live_readout)
        
        self.lbl_sens = QtWidgets.QLabel("Sens: -")
        layout.addWidget(self.lbl_sens)
        
        # Live Plot
        self.plot_live = pg.PlotWidget(title="Live Monitor (R)")
        self.plot_live.setLabel('left', 'Magnitude (V)')
        self.plot_live.setLabel('bottom', 'Time steps')
        self.plot_live.showGrid(x=True, y=True, alpha=0.3)
        self.plot_live.setMinimumHeight(200)
        self.curve_live = self.plot_live.plot(pen='y')
        layout.addWidget(self.plot_live)
        
        # Data storage
        self.live_data = []
        
        self.update_info() # Sets initial disconnected state
             
    def connect_lockin(self):
        res = self.txt_resource.text().strip()
        if self.lockin.connect(res):
            self.lbl_status.setText("Connected")
            self.update_info()
        else:
            self.lbl_status.setText("Conn Failed")
            
    def update_info(self):
        if self.lockin.is_connected:
            try:
                # Read TC
                val = self.lockin.get_time_constant()
                # Find closest index
                idx = 0
                min_diff = float('inf')
                for i, v in enumerate(self.tc_values):
                    diff = abs(val - v)
                    if diff < min_diff:
                        min_diff = diff
                        idx = i
                
                # Block signals to avoid re-triggering set
                self.combo_tc.blockSignals(True)
                self.combo_tc.setCurrentIndex(idx)
                self.combo_tc.blockSignals(False)
                
                # Read Harmonic (Driver needs get_harmonic if we want to sync back, assume 1 for now or add get)
                # For now, just show connection status
                self.lbl_status.setText("Connected")
            except:
                pass
            
            self.lbl_sens.setText("Sens: (Auto)")
        else:
            self.lbl_status.setText("Not Conn")
            self.lbl_sens.setText("Sens: -")

    def set_harmonic(self, val):
        self.lockin.set_harmonic(val)
        
    def set_time_constant(self, idx):
        if idx >= 0 and idx < len(self.tc_values):
            val = self.tc_values[idx]
            self.lockin.set_time_constant(val)
             
    def get_channel_code(self):
        return self.combo.currentIndex()

    def get_channel_name(self):
        return self.combo.currentText()
        
    def toggle_live(self, state):
        if self.chk_live.isChecked():
            self.live_data = []
            self.curve_live.setData([])
            self.timer_live.start()
        else:
            self.timer_live.stop()
            self.lbl_live_val.setText("R: - | P: -")
            
    def update_live_readout(self):
        if not self.lockin.is_connected: return
        try:
            # Read R (2) and Theta (3)
            # Access driver directly
            r = self.lockin.read_value(2, samples=1)
            p = self.lockin.read_value(3, samples=1)
            self.lbl_live_val.setText(f"R: {r:.4e} V | P: {p:.2f} deg")
            
            # Update Plot
            self.live_data.append(r)
            if len(self.live_data) > 300: # Keep last 300 pts (~150 sec at 2Hz)
                self.live_data.pop(0)
            self.curve_live.setData(self.live_data)
            
        except:
            pass
