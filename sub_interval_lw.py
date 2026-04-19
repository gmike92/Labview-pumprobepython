from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np

class DelayIntervalWidget(QtWidgets.QWidget):
    """
    Ported from TWINSMCT app. Provides 3 zones for Delay Stage scanning.
    """
    def __init__(self):
        super().__init__()
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        # --- Header Row: Units ---
        h_units = QtWidgets.QHBoxLayout()
        h_units.addWidget(QtWidgets.QLabel("Units:"))
        self.combo_units = QtWidgets.QComboBox()
        self.combo_units.addItems(["fs", "mm"])
        # Set default to fs, which is more common
        self.combo_units.setCurrentIndex(0)
        self.combo_units.currentIndexChanged.connect(self.on_unit_changed)
        h_units.addWidget(self.combo_units)
        h_units.addStretch()
        layout.addLayout(h_units)
        
        gb = QtWidgets.QGroupBox("Scan Intervals")
        grid = QtWidgets.QGridLayout(gb)
        grid.setContentsMargins(5, 5, 5, 5)
        
        # --- Headers ---
        self.lbl_start = QtWidgets.QLabel("Start (fs)")
        self.lbl_stop = QtWidgets.QLabel("Stop (fs)")
        
        grid.addWidget(self.lbl_start, 0, 1)
        grid.addWidget(self.lbl_stop, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Step (fs)"), 0, 3) # Keep step in fs
        
        # --- Zone 1 ---
        grid.addWidget(QtWidgets.QLabel("Zone 1:"), 1, 0)
        self.z1_start = QtWidgets.QDoubleSpinBox(); self.z1_start.setRange(-200000, 200000); self.z1_start.setValue(-1000.0)
        grid.addWidget(self.z1_start, 1, 1)
        
        self.z1_stop = QtWidgets.QDoubleSpinBox(); self.z1_stop.setRange(-200000, 200000); self.z1_stop.setValue(0.0)
        self.z1_stop.valueChanged.connect(self.update_continuity)
        grid.addWidget(self.z1_stop, 1, 2)
        
        self.z1_step = QtWidgets.QDoubleSpinBox(); self.z1_step.setRange(0.1, 10000); self.z1_step.setValue(10.0)
        grid.addWidget(self.z1_step, 1, 3)
        
        # --- Zone 2 ---
        self.chk_z2 = QtWidgets.QCheckBox("Zone 2:")
        self.chk_z2.stateChanged.connect(self.toggle_zones)
        grid.addWidget(self.chk_z2, 2, 0)
        
        self.z2_start = QtWidgets.QDoubleSpinBox(); self.z2_start.setRange(-200000, 200000); self.z2_start.setReadOnly(True)
        self.z2_start.setStyleSheet("background-color: #f0f0f0; color: #555;")
        self.z2_start.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        grid.addWidget(self.z2_start, 2, 1)
        
        self.z2_stop = QtWidgets.QDoubleSpinBox(); self.z2_stop.setRange(-200000, 200000); self.z2_stop.setValue(2000.0)
        self.z2_stop.valueChanged.connect(self.update_continuity)
        grid.addWidget(self.z2_stop, 2, 2)
        
        self.z2_step = QtWidgets.QDoubleSpinBox(); self.z2_step.setRange(0.1, 10000); self.z2_step.setValue(20.0)
        grid.addWidget(self.z2_step, 2, 3)
        
        # --- Zone 3 ---
        self.chk_z3 = QtWidgets.QCheckBox("Zone 3:")
        self.chk_z3.stateChanged.connect(self.toggle_zones)
        grid.addWidget(self.chk_z3, 3, 0)
        
        self.z3_start = QtWidgets.QDoubleSpinBox(); self.z3_start.setRange(-200000, 200000); self.z3_start.setReadOnly(True)
        self.z3_start.setStyleSheet("background-color: #f0f0f0; color: #555;")
        self.z3_start.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        grid.addWidget(self.z3_start, 3, 1)
        
        self.z3_stop = QtWidgets.QDoubleSpinBox(); self.z3_stop.setRange(-200000, 200000); self.z3_stop.setValue(10000.0)
        grid.addWidget(self.z3_stop, 3, 2)
        
        self.z3_step = QtWidgets.QDoubleSpinBox(); self.z3_step.setRange(0.1, 10000); self.z3_step.setValue(100.0)
        grid.addWidget(self.z3_step, 3, 3)
        
        layout.addWidget(gb)
        
        self.current_unit = "fs"
        self.update_continuity()
        self.toggle_zones()

    def on_unit_changed(self, index):
        new_unit = self.combo_units.currentText()
        if new_unit == self.current_unit: return
        
        # Convert values
        factor = 1.0
        if self.current_unit == "mm" and new_unit == "fs":
            factor = 6666.6 # 1 mm = 6666.6 fs approx
        elif self.current_unit == "fs" and new_unit == "mm":
            factor = 1.0 / 6666.6
            
        def conv(spin):
            val = spin.value()
            spin.setDecimals(3 if new_unit == "mm" else 1)
            spin.setValue(val * factor)
            
        # Block signals to prevent improved continuity recursion mess
        self.blockSignals(True)
        conv(self.z1_start); conv(self.z1_stop)
        conv(self.z2_stop) # z2_start follows z1
        conv(self.z3_stop) # z3_start follows z2
        self.blockSignals(False)
        
        self.current_unit = new_unit
        
        # Update Labels
        self.lbl_start.setText(f"Start ({new_unit})")
        self.lbl_stop.setText(f"Stop ({new_unit})")
        
        self.update_continuity()

    def update_continuity(self):
        # Link Z2 Start to Z1 Stop
        val1 = self.z1_stop.value()
        self.z2_start.setValue(val1)
        
        # Link Z3 Start to Z2 Stop
        val2 = self.z2_stop.value()
        self.z3_start.setValue(val2)

    def toggle_zones(self):
        # Zone 2 logic
        z2_active = self.chk_z2.isChecked()
        self.z2_start.setEnabled(z2_active)
        self.z2_stop.setEnabled(z2_active)
        self.z2_step.setEnabled(z2_active)
        
        z3_active = self.chk_z3.isChecked()
        self.z3_start.setEnabled(z3_active)
        self.z3_stop.setEnabled(z3_active)
        self.z3_step.setEnabled(z3_active)
        
        self.update_continuity()

    def get_scan_points_mm(self, invert_stage=False):
        """ Returns the points in mm offset from home, suitable for the stage_delay.move_to """
        points = []
        is_fs_mode = (self.current_unit == "fs")
        SPEED_OF_LIGHT_MM_FS = 0.000299792458
        
        def to_mm_dist(val):
            if is_fs_mode:
                return val * SPEED_OF_LIGHT_MM_FS / 2.0
            return val
            
        def add_zone(start, end, step_fs):
            # Start/End are in current units (mm or fs)
            # Step is ALWAYS fs
            start_mm = to_mm_dist(start)
            end_mm = to_mm_dist(end)
            step_mm = (step_fs * SPEED_OF_LIGHT_MM_FS / 2.0)
            
            if invert_stage:
                # If stage is pump, +delay = +mm. Distance is directly added.
                pass
            else:
                # If stage is probe, +delay (moving delay lines back) = -mm path difference 
                start_mm = -start_mm
                end_mm = -end_mm
            
            if step_mm <= 0: return
            
            steps = int(abs(end_mm - start_mm) / step_mm)
            if steps > 0:
                p = np.linspace(start_mm, end_mm, steps+1) 
                points.extend(p)

        # Zone 1
        add_zone(self.z1_start.value(), self.z1_stop.value(), self.z1_step.value())
        
        # Zone 2
        if self.chk_z2.isChecked():
            add_zone(self.z2_start.value(), self.z2_stop.value(), self.z2_step.value())
            
        # Zone 3
        if self.chk_z3.isChecked():
            add_zone(self.z3_start.value(), self.z3_stop.value(), self.z3_step.value())

        return np.unique(points) 

    def get_fs_from_mm(self, pos_mm, invert_stage=False):
        SPEED_OF_LIGHT_MM_FS = 0.000299792458
        val = pos_mm * 2.0 / SPEED_OF_LIGHT_MM_FS
        if invert_stage:
            return val # Pump: +mm = +fs
        else:
            return -val # Probe: -mm = +fs
