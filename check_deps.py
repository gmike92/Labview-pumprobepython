import sys
try:
    import numpy
    print("numpy: OK", numpy.__version__)
except ImportError:
    print("numpy: MISSING")

try:
    import h5py
    print("h5py: OK", h5py.__version__)
except ImportError:
    print("h5py: MISSING")

try:
    import pyqtgraph
    print("pyqtgraph: OK", pyqtgraph.__version__)
except ImportError:
    print("pyqtgraph: MISSING")

try:
    from PyQt6 import QtWidgets
    print("PyQt6: OK")
except ImportError:
    try:
        from PyQt5 import QtWidgets
        print("PyQt5: OK")
    except ImportError:
        print("PyQt: MISSING (Need PyQt6 or PyQt5)")

try:
    from PySide6 import QtWidgets
    print("PySide6: OK")
except ImportError:
    print("PySide6: MISSING")
