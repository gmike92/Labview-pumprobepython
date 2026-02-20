import sys
import os
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore

class DataViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Camera Data Viewer")
        self.resize(1000, 700)
        
        # Central Widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("📂 Load .npz File")
        self.btn_load.setStyleSheet("font-size: 14px; padding: 8px; font-weight: bold;")
        self.btn_load.clicked.connect(self.load_file)
        toolbar.addWidget(self.btn_load)
        
        self.cmb_datasets = QtWidgets.QComboBox()
        self.cmb_datasets.currentIndexChanged.connect(self.update_view)
        self.cmb_datasets.setMinimumWidth(200)
        toolbar.addWidget(QtWidgets.QLabel("Dataset:"))
        toolbar.addWidget(self.cmb_datasets)
        
        self.lbl_info = QtWidgets.QLabel("No file loaded")
        self.lbl_info.setStyleSheet("color: #666; font-style: italic; margin-left: 10px;")
        toolbar.addWidget(self.lbl_info)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Content Area (Stack)
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack)
        
        # 1. Image View (for 3D/2D data)
        self.img_view = pg.ImageView()
        self.stack.addWidget(self.img_view)
        
        # 2. Plot Widget (for 1D data)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.stack.addWidget(self.plot_widget)
        
        # Initial State
        self.stack.setCurrentWidget(self.img_view)
        
        self.current_data = {}

    def load_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Data File", "", "NPZ Files (*.npz);;All Files (*)"
        )
        if not fname:
            return
        
        try:
            raw = np.load(fname)
            self.lbl_info.setText(f"Loaded: {os.path.basename(fname)}")
            
            # Identify useful arrays
            self.current_data = {}
            self.cmb_datasets.clear()
            
            # Common axes
            delays = None
            positions = None
            if 'delays_fs' in raw: delays = raw['delays_fs']
            if 'positions' in raw: positions = raw['positions']
            # Fallback
            x_axis = delays if delays is not None else positions
            
            # Store metadata
            self.current_x = x_axis
            
            # Scan keys
            for key in raw.files:
                arr = raw[key]
                if arr.ndim >= 1 and key not in ['delays_fs', 'positions', 'zero_mm', 'frames_per_point']:
                    self.current_data[key] = arr
                    self.cmb_datasets.addItem(f"{key} {arr.shape}")
            
            # Auto-select interesting one
            priority = ['roi_datacube', 'datacube', 'raw_odd', 'raw_even', 'signals']
            for p in priority:
                for i in range(self.cmb_datasets.count()):
                    if self.cmb_datasets.itemText(i).startswith(p):
                        self.cmb_datasets.setCurrentIndex(i)
                        return

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def update_view(self):
        txt = self.cmb_datasets.currentText()
        if not txt:
            return
            
        key = txt.split(' ')[0] # naive parse "key (shape)"
        if key not in self.current_data:
            return
            
        data = self.current_data[key]
        x_axis = self.current_x
        
        print(f"Showing {key}: {data.shape}")
        
        # 3D / 2D -> ImageView
        if data.ndim >= 2:
            # Check if 1D list of scalars wrapped in 2D? (N, 1)
            if data.ndim == 2 and (data.shape[1] == 1):
                 # Plot as 1D
                 self.show_1d(x_axis, data.flatten(), title=key)
            else:
                self.show_3d(data, title=key)
        else:
            # 1D -> Plot
            self.show_1d(x_axis, data, title=key)

    def show_3d(self, cube, title=""):
        self.stack.setCurrentWidget(self.img_view)
        if cube.ndim == 2:
            # (H, W) -> (1, H, W)
            cube = cube.reshape(1, *cube.shape)
            
        self.img_view.setImage(cube)
        self.img_view.view.setTitle(title)

    def show_1d(self, x, y, title=""):
        self.stack.setCurrentWidget(self.plot_widget)
        self.plot_widget.clear()
        self.plot_widget.setTitle(title)
        
        if x is not None and len(x) == len(y):
            self.plot_widget.plot(x, y, pen=pg.mkPen('b', width=2), symbol='o')
        else:
            self.plot_widget.plot(y, pen=pg.mkPen('r', width=2))
        
        self.plot_widget.setLabel('bottom', "Delay / Index")
        self.plot_widget.setLabel('left', "Signal")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = DataViewer()
    viewer.show()
    sys.exit(app.exec())
