import numpy as np
import time
from qcodes.instrument_drivers.stanford_research import SR865A
from qcodes.instrument import Instrument

class LockInDriver:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LockInDriver, cls).__new__(cls)
            cls._instance.lockin = None
            cls._instance.is_connected = False
            cls._instance.port = None
            import threading
            cls._instance._lock = threading.Lock()
        return cls._instance

    def connect(self, resource_str="USB0::0xB506::0x2000::004418::INSTR"):
        with self._lock:
            # If already connected to the same port, return
            if self.is_connected and self.port == resource_str:
                return True
            
            try:
                # Check if QCoDeS instrument with this name already exists
                try:
                    # If it exists, it might be in a bad state (locked/TMO). 
                    # Better to close it and start fresh.
                    old_inst = Instrument.find_instrument('lockin')
                    print(f"[LockIn] Found existing instrument 'lockin'. Closing it...")
                    if hasattr(old_inst, 'visa_handle'):
                        try:
                            old_inst.visa_handle.close()
                        except:
                            pass
                    old_inst.close()
                    
                    if 'lockin' in Instrument._all_instruments:
                        del Instrument._all_instruments['lockin']
                except KeyError:
                    pass
                
                # Create new instance
                print(f"[LockIn] Connecting via QCoDeS to {resource_str}...")
                self.lockin = SR865A('lockin', resource_str)
                
                self.port = resource_str
                self.is_connected = True
                
                # --- Robustness Config ---
                # Now that 'Locked' error is fixed, try standard \n termination to avoid UserWarning
                if hasattr(self.lockin, 'visa_handle'):
                    self.lockin.visa_handle.timeout = 20000 
                    self.lockin.visa_handle.read_termination = '\n'
                    self.lockin.visa_handle.write_termination = '\n'
                    self.lockin.visa_handle.clear()
                
                # Clear Error Status
                try:
                    self.lockin.write('*CLS')
                except:
                    pass
                
                print(f"[LockIn] Connected: {self.lockin.get_idn()}")
                return True
                
            except Exception as e:
                print(f"[LockIn] Connection Failed: {e}")
                self.is_connected = False
                return False

    def get_time_constant(self):
        with self._lock:
            if not self.is_connected: return 0.1
            try:
                return self.lockin.time_constant()
            except:
                return 0.1

    def set_time_constant(self, val):
        with self._lock:
            if not self.is_connected: return
            try:
                # QCoDeS SR865A accepts float (e.g. 10e-6 for 10us)
                self.lockin.time_constant(val)
                print(f"[LockIn] Set Time Constant: {val}s")
            except Exception as e:
                print(f"[LockIn] Error setting TC: {e}")

    def read_value(self, channel_code=0, samples=1):
        # channel_code mappings from user request:
        # 0=X, 1=Y, 2=R, 3=Theta
        with self._lock:
            if not self.is_connected:
                return 0.0
            
            try:
                # Use SNAP for simultaneous acquisition if possible, or simple get()
                # The QCoDeS driver has parameters: X, Y, R, P (Phase/Theta)
                val = 0.0
                vals = []
                
                for _ in range(samples):
                    if channel_code == 0:
                        v = self.lockin.X()
                    elif channel_code == 1:
                        v = self.lockin.Y()
                    elif channel_code == 2:
                        v = self.lockin.R()
                    elif channel_code == 3:
                        v = self.lockin.P()
                    else:
                        v = 0.0
                    vals.append(v)
                    
                return np.mean(vals)
            except Exception as e:
                print(f"[LockIn] Read Error: {e}")
                return 0.0

    def read_voltage(self):
        # Default to R (Magnitude)
        return self.read_value(channel_code=2)
        
    def read_phase(self):
        # Helper for Theta/Phase
        return self.read_value(channel_code=3)

    def set_harmonic(self, harm=1):
        with self._lock:
            if not self.is_connected: return
            try:
                self.lockin.harmonic(harm)
            except:
                pass
            
    def set_wait_time(self, seconds):
        time.sleep(seconds)
