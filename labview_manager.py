"""
LabVIEW Manager - Singleton controller for Experiment_manager.vi via ActiveX.

The Manager VI runs continuously with a While Loop + Case Structure.
Python sends commands by setting the "Enum" control,
then polls until the VI sets it back to "Idle" (0).

Enum values:
    0 = Idle       (do nothing, wait)
    1 = Init       (open camera → returns to Idle)
    2 = Getframe   (continuous loop: acquire → T, DeltaT — for Live View)
    3 = Close      (close camera → returns to Idle)
    4 = Measure    (acquire once → T, DeltaT → returns to Idle — for scanning)

Controls:
    N          (I32)  — number of frames per acquisition
    Acq Trigger (Bool) — acquisition trigger
    Stoplive   (Bool) — set True to exit Getframe loop → Idle
    End        (Bool) — stop the Manager VI after Close

Indicators:
    T       — transmission data
    DeltaT  — delta T/T data

Usage:
    from labview_manager import LabVIEWManager
    mgr = LabVIEWManager()
    mgr.start()
    mgr.initialize_camera()
    mgr.vi.SetControlValue("N", 100)
    mgr.shutdown()
"""

import os
import time
import numpy as np

# Try importing pywin32 for LabVIEW ActiveX automation
try:
    import win32com.client
    import pythoncom
    HAS_WIN32COM = True
except ImportError:
    print("[WARN] pywin32 not installed. LabVIEW automation disabled.")
    HAS_WIN32COM = False


# =============================================================================
# Configuration
# =============================================================================

# Path to the persistent Manager VI
DEFAULT_VI_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Experiment_manager.vi"
)

# Command Enum values (must match LabVIEW enum)
CMD_IDLE = 0
CMD_INIT = 1
CMD_GETFRAME = 2   # Continuous streaming (Live View) — use Stoplive to exit
CMD_CLOSE = 3
CMD_MEASURE = 4    # Single-shot acquire → returns to Idle (for scanning)


# =============================================================================
# LabVIEW Manager Controller (Singleton)
# =============================================================================

class LabVIEWManager:
    """
    Singleton controller for Experiment_manager.vi via ActiveX.
    
    Shared across all sub-windows — only one LabVIEW connection exists.
    """
    
    _instance = None
    
    def __new__(cls, vi_path: str = DEFAULT_VI_PATH):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, vi_path: str = DEFAULT_VI_PATH):
        if self._initialized:
            return
        self._initialized = True
        
        self.vi_path = vi_path
        self.lv = None
        self.vi = None
        self.is_running = False
        self.camera_initialized = False
        self.background = None # Global Background Frame
        print("[LabVIEWManager] Singleton initialized")
    
    def start(self):
        """Launch LabVIEW, load Manager VI, and run it (non-blocking)."""
        if not HAS_WIN32COM:
            print("[ERROR] pywin32 not available")
            return False
        
        if self.is_running:
            print("[INFO] Manager already running")
            return True
        
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
                print(f"[WARN] Run() failed: {run_err}")
                print("[INFO] Please click the Run button in LabVIEW manually.")
            
            # List all controls for debugging
            print("\n[DEBUG] --- VI Control Inspection ---")
            try:
                # 1. Sanity check: try reading straightforwardly
                try:
                    o = self.vi.GetControlValue("Odd")
                    print(f"[DEBUG] GetControlValue('Odd') -> Success (Type: {type(o)})")
                except Exception as e:
                    print(f"[DEBUG] GetControlValue('Odd') -> FAILED: {e}")
                
                try:
                    e_val = self.vi.GetControlValue("Even")
                    print(f"[DEBUG] GetControlValue('Even') -> Success (Type: {type(e_val)})")
                except Exception as e:
                    print(f"[DEBUG] GetControlValue('Even') -> FAILED: {e}")

                # 2. List all controls to find mismatch
                fp = self.vi.FrontPanel
                ctrls = fp.Controls
                count = ctrls.Count
                names = []
                print(f"[DEBUG] Found {count} controls on Front Panel:")
                for i in range(count):
                    try:
                        c = ctrls.Item(i)
                        name = c.Name
                        label = c.Label.Text
                        print(f"  - Index {i}: Name='{name}' | Label='{label}'")
                        names.append(name)
                    except:
                        pass
                print("[DEBUG] -------------------------------")
            except Exception as e:
                print(f"[DEBUG] Could not inspect controls: {e}")

            self.is_running = True
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start Manager: {e}")
            return False
    
    def send_command(self, cmd, timeout=30.0):
        """
        Send a command and wait for the VI to return to Idle.
        
        Args:
            cmd: CMD_IDLE, CMD_INIT, CMD_GETFRAME, or CMD_CLOSE
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
            self.camera_initialized = True
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
            self.vi.SetControlValue("N", n_frames)
            self.vi.SetControlValue("Acq Trigger", acq_trigger)
            
            print(f"[CMD] Measure (N={n_frames}, trigger={acq_trigger})...")
            # Use CMD_MEASURE (4) because CMD_GETFRAME (2) loops forever, causing send_command to timeout.
            # CMD_MEASURE acquires once and returns to Idle, allowing us to read data.
            success = self.send_command(CMD_MEASURE, timeout=60.0)
            if not success:
                print("[ERROR] Getframe timed out")
                return None
            
            result = {}
            try:
                t_data = self.vi.GetControlValue("T")
            except Exception:
                t_data = None
            
            odd_data = self.vi.GetControlValue("Odd")
            even_data = self.vi.GetControlValue("Even")
            
            # Debug prints
            if odd_data is None: print("[DEBUG] 'Odd' control returns None")
            if even_data is None: print("[DEBUG] 'Even' control returns None")
            
            if t_data is not None:
                result['T'] = np.array(t_data)
            if odd_data is not None and even_data is not None:
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                print(f"[DEBUG] Odd shape: {odd.shape}, Even shape: {even.shape}, Mean(Odd): {np.mean(odd):.2f}, Mean(Even): {np.mean(even):.2f}")
                result['Odd'] = odd
                result['Even'] = even
                result['DeltaT'] = (even - odd) / np.where(np.abs(odd) > 1e-10, odd, 1e-10)
            
            return result if result else None
            
        except Exception as e:
            print(f"[ERROR] acquire_map failed: {e}")
            return None
    
    def measure(self, n_frames=100, acq_trigger=True):
        """
        Single-shot measurement: acquire once → return to Idle.
        
        Use this for scanning workflows (pump-probe, twins).
        Unlike acquire_map/Getframe, this does NOT loop — LabVIEW
        acquires N frames, computes T/DeltaT, and sets Enum back to Idle.
        
        Args:
            n_frames: Number of frames to average
            acq_trigger: Acquisition trigger boolean
            
        Returns:
            dict with 'T' and 'DeltaT' numpy arrays, or None
        """
        if not self.vi:
            return None
        
        try:
            self.vi.SetControlValue("N", n_frames)
            self.vi.SetControlValue("Acq Trigger", acq_trigger)
            
            print(f"[CMD] Measure (N={n_frames}, trigger={acq_trigger})...")
            success = self.send_command(CMD_MEASURE, timeout=60.0)
            if not success:
                print("[ERROR] Measure timed out")
                return None
            
            result = {}
            try:
                t_data = self.vi.GetControlValue("T")
            except Exception:
                t_data = None
            
            odd_data = self.vi.GetControlValue("Odd")
            even_data = self.vi.GetControlValue("Even")
            
            if odd_data is None: print("[DEBUG-MEASURE] 'Odd' is None")
            if even_data is None: print("[DEBUG-MEASURE] 'Even' is None")
            
            if t_data is not None:
                result['T'] = np.array(t_data)
            if odd_data is not None and even_data is not None:
                odd = np.array(odd_data, dtype=float)
                even = np.array(even_data, dtype=float)
                print(f"[DEBUG-MEASURE] Odd: {odd.shape}, Even: {even.shape}")
                result['Odd'] = odd
                result['Even'] = even
                result['DeltaT'] = (even - odd) / np.where(np.abs(odd) > 1e-10, odd, 1e-10)
            
            return result if result else None
            
        except Exception as e:
            print(f"[ERROR] measure failed: {e}")
            return None
    
    def shutdown(self):
        """Shutdown sequence: Close camera (enum=3) → wait Idle → set End=True."""
        if not self.is_running or not self.vi:
            return
        
        try:
            # Step 1: Send Close command to shut down the camera
            if self.camera_initialized:
                print("[CMD] Sending Close (enum=3) to shut camera...")
                self.vi.SetControlValue("Enum", CMD_CLOSE)
                
                # Poll until VI returns to Idle
                waited = 0.0
                idle_reached = False
                while waited < 15.0:
                    time.sleep(0.1)
                    waited += 0.1
                    try:
                        if self.vi.GetControlValue("Enum") == CMD_IDLE:
                            idle_reached = True
                            break
                    except Exception:
                        break
                
                if idle_reached:
                    print(f"[OK] Close complete — back to Idle after {waited:.1f}s")
                else:
                    print(f"[WARN] Close did not return to Idle within {waited:.1f}s")
            
            # Step 2: Set End=True to stop the Manager VI's While Loop
            print("[CMD] Setting End=True → stopping Manager VI...")
            self.vi.SetControlValue("end", True)
            time.sleep(1.0)
            
            self.is_running = False
            self.camera_initialized = False
            print("[OK] Manager VI stopped")
        except Exception as e:
            print(f"[WARN] Shutdown error: {e}")
        
        self.vi = None
        self.lv = None
    
    def close(self):
        """Clean up — send Exit if still running."""
        if self.is_running:
            self.shutdown()
