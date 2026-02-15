"""
DelayStageDriver - Singleton driver for Thorlabs Kinesis translation stage.

This module controls the Thorlabs delay line for pump-probe experiments.
Uses pylablib for simplified Thorlabs Kinesis control.

Physics:
- Speed of light: c = 0.000299792 mm/fs
- Double-pass delay: Δt (fs) = 2 × Δx (mm) / c
- Time-to-distance: Δx (mm) = Δt (fs) × c / 2

Usage:
    from stage_delay import DelayStageDriver
    
    stage = DelayStageDriver()
    stage.connect()
    stage.home()
    stage.move_to(10.0)  # Move to 10mm
    print(f"Position: {stage.get_position()} mm")
    print(f"Delay: {stage.mm_to_fs(stage.get_position())} fs")
    stage.disconnect()
"""

import time
from typing import Optional

# Physics constants
SPEED_OF_LIGHT_MM_FS = 0.000299792458  # mm per femtosecond
SPEED_OF_LIGHT_UM_PS = 299.792458      # µm per picosecond


# ============================================================================
# Delay Stage Driver (Singleton)
# ============================================================================

class DelayStageDriver:
    """
    Singleton driver for Thorlabs Kinesis translation stage.
    
    Controls:
        - Connect/disconnect to stage
        - Home (required before movement)
        - Absolute and relative positioning
        - Position readout in mm and fs
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # State
        self.is_connected = False
        self.is_homed = False
        self._position_mm = 0.0
        
        # Thorlabs stage object (from pylablib)
        self._stage = None
        self._serial_number = None
        self._stage_type = None  # 'kinesis' or 'apt'
        
        # Configuration
        self.double_pass = True  # True = retroreflector (factor of 2)
        
        # Scaling factor: 20000 counts per mm for Kinesis stages
        self.COUNTS_PER_MM = 20000
        
        # Default serial number for probe delay line (from lab notes)
        self.DEFAULT_SERIAL = "45835224"
        
        self._initialized = True
        print("[DelayStageDriver] Singleton initialized")
    
    # ========================================================================
    # Connection
    # ========================================================================
    
    def connect(self, serial_number: str = None) -> bool:
        """
        Connect to Thorlabs Kinesis stage.
        
        Args:
            serial_number: Optional serial number. If None, uses first available.
            
        Returns:
            True if connected successfully
        """
        if self.is_connected:
            print("[INFO] Delay stage already connected")
            return True
        
        # Try APT first (more reliable for BBD30X Brushless Motor Controller)
        print("[INFO] Trying APT library first...")
        if self._connect_apt(serial_number):
            return True
        
        # Fallback to Kinesis
        print("[INFO] APT failed, trying Kinesis...")
        if self._connect_kinesis(serial_number):
            return True
        
        # Last resort: pythonnet
        print("[INFO] Kinesis failed, trying pythonnet...")
        return self._connect_pythonnet(serial_number)
    
    def _connect_kinesis(self, serial_number: str = None) -> bool:
        """Connect using pylablib Kinesis library."""
        try:
            from pylablib.devices import Thorlabs
            
            devices = Thorlabs.list_kinesis_devices()
            print(f"[INFO] Found Thorlabs Kinesis devices: {devices}")
            
            if not devices:
                print("[WARN] No Kinesis devices found")
                return False
            
            if serial_number:
                sn = serial_number
            elif devices:
                sn = devices[0][0]
            else:
                sn = self.DEFAULT_SERIAL
            
            self._serial_number = sn
            device_type = devices[0][1] if devices else "Unknown"
            print(f"[INFO] Connecting to Kinesis device: {sn} ({device_type})")
            
            try:
                self._stage = Thorlabs.KinesisMotor(sn)
            except Exception as e1:
                print(f"[WARN] Default mode failed: {e1}")
                try:
                    self._stage = Thorlabs.KinesisMotor(sn, is_rack_system=False)
                except Exception as e2:
                    print(f"[WARN] is_rack_system=False failed: {e2}")
                    self._stage = Thorlabs.KinesisMotor(sn, is_rack_system=True)
            
            self._stage_type = 'kinesis'
            self.is_connected = True
            print(f"[OK] Connected to Kinesis delay stage: {sn}")
            return True
            
        except ImportError as e:
            print(f"[WARN] pylablib not available ({e})")
            return False
        except Exception as e:
            print(f"[ERROR] Kinesis connection failed: {e}")
            return False
    
    def _connect_apt(self, serial_number: str = None) -> bool:
        """Connect using thorlabs_apt library."""
        try:
            import sys
            import os
            
            # Add APT DLL path (from TWINS FILE folder)
            apt_dll_paths = [
                r"C:\Users\mguizzardi\Desktop\Camera python\TWINS FILE\Twins\APT dll",
                os.path.join(os.path.dirname(__file__), "..", "TWINS FILE", "Twins", "APT dll"),
                os.path.join(os.path.dirname(__file__), "APT dll"),
            ]
            
            for apt_path in apt_dll_paths:
                if os.path.exists(apt_path) and apt_path not in sys.path:
                    sys.path.insert(0, apt_path)
                    print(f"[INFO] Added APT DLL path: {apt_path}")
                    break
            
            import thorlabs_apt as apt
            
            devices = apt.list_available_devices()
            print(f"[INFO] Found APT devices: {devices}")
            
            if not devices:
                print("[ERROR] No APT devices found")
                return False
            
            if serial_number:
                sn = int(serial_number)
            else:
                sn = devices[0][1]
            
            self._serial_number = str(sn)
            print(f"[INFO] Connecting to APT device: {sn}")
            
            self._stage = apt.Motor(sn)
            self._stage_type = 'apt'
            
            self.is_connected = True
            print(f"[OK] Connected to APT delay stage: {sn}")
            return True
            
        except ImportError as e:
            print(f"[WARN] thorlabs_apt not available ({e})")
            return self._connect_pythonnet(serial_number)
        except Exception as e:
            print(f"[ERROR] APT connection failed: {e}")
            return False
    
    def _connect_pythonnet(self, serial_number: str = None) -> bool:
        """Alternative connection using pythonnet (Thorlabs .NET DLLs)."""
        try:
            import clr
            import sys
            
            kinesis_path = r"C:\Program Files\Thorlabs\Kinesis"
            if kinesis_path not in sys.path:
                sys.path.append(kinesis_path)
            
            clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
            clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
            clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
            
            from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
            from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
            
            DeviceManagerCLI.BuildDeviceList()
            devices = DeviceManagerCLI.GetDeviceList()
            
            if devices.Count == 0:
                print("[ERROR] No Thorlabs devices found")
                return False
            
            if serial_number:
                sn = serial_number
            else:
                sn = str(devices[0])
            
            self._serial_number = sn
            print(f"[INFO] Connecting to device: {sn}")
            
            self._stage = KCubeDCServo.CreateKCubeDCServo(sn)
            self._stage.Connect(sn)
            
            time.sleep(0.5)
            self._stage.StartPolling(250)
            time.sleep(0.5)
            self._stage.EnableDevice()
            
            self.is_connected = True
            print(f"[OK] Connected via pythonnet: {sn}")
            return True
            
        except Exception as e:
            print(f"[ERROR] pythonnet connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from stage."""
        if not self.is_connected:
            return True
        
        try:
            if self._stage is not None:
                if hasattr(self._stage, 'close'):
                    self._stage.close()
                elif hasattr(self._stage, 'Disconnect'):
                    self._stage.StopPolling()
                    self._stage.Disconnect()
                
            self._stage = None
            self.is_connected = False
            self.is_homed = False
            print("[OK] Delay stage disconnected")
            return True
            
        except Exception as e:
            print(f"[ERROR] Disconnect failed: {e}")
            return False
    
    # ========================================================================
    # Homing (CRITICAL - Required before any movement!)
    # ========================================================================
    
    def home(self, timeout_s: float = 60.0) -> bool:
        """
        Home the stage. REQUIRED before any movement.
        
        Args:
            timeout_s: Timeout for homing operation
            
        Returns:
            True if homing completed successfully
        """
        if not self.is_connected:
            print("[ERROR] Stage not connected")
            return False
        
        if self.is_homed:
            print("[INFO] Stage already homed")
            return True
        
        try:
            print("[INFO] Starting homing sequence...")
            
            if hasattr(self._stage, 'home'):
                self._stage.home(sync=True, timeout=timeout_s)
            elif hasattr(self._stage, 'Home'):
                self._stage.Home(int(timeout_s * 1000))
            else:
                print("[WARN] No home method found, assuming already homed")
            
            self.is_homed = True
            self._position_mm = 0.0
            print("[OK] Homing complete")
            return True
            
        except Exception as e:
            print(f"[ERROR] Homing failed: {e}")
            return False
    
    # ========================================================================
    # Movement
    # ========================================================================
    
    def move_to(self, position_mm: float, wait: bool = True, 
                timeout_s: float = 30.0) -> bool:
        """
        Move to absolute position.
        
        Args:
            position_mm: Target position in mm
            wait: If True, wait for move to complete
            timeout_s: Timeout for movement
            
        Returns:
            True if move completed (or started if wait=False)
        """
        if not self.is_connected:
            print("[ERROR] Stage not connected")
            return False
        
        if not self.is_homed:
            print("[ERROR] Stage not homed - call home() first!")
            return False
        
        try:
            print(f"[INFO] Moving to {position_mm:.4f} mm...")
            
            if self._stage_type == 'kinesis':
                self._stage.move_to(position_mm * self.COUNTS_PER_MM)
            elif self._stage_type == 'apt':
                self._stage.move_to(float(position_mm), blocking=True)
            elif hasattr(self._stage, 'MoveTo'):
                self._stage.MoveTo(self._mm_to_device_units(position_mm),
                                  int(timeout_s * 1000))
            else:
                print("[ERROR] No move method found")
                return False
            
            self._position_mm = position_mm
            print(f"[OK] Moved to {position_mm:.4f} mm")
            return True
            
        except Exception as e:
            print(f"[ERROR] Move failed: {e}")
            return False
    
    def move_relative(self, distance_mm: float, wait: bool = True,
                      timeout_s: float = 30.0) -> bool:
        """
        Move relative to current position.
        
        Args:
            distance_mm: Distance to move (positive = forward)
            wait: If True, wait for move to complete
            timeout_s: Timeout for movement
            
        Returns:
            True if move completed
        """
        if not self.is_connected:
            print("[ERROR] Stage not connected")
            return False
        
        if not self.is_homed:
            print("[ERROR] Stage not homed - call home() first!")
            return False
        
        try:
            current = self.get_position()
            target = current + distance_mm
            print(f"[INFO] Moving relative: {distance_mm:+.4f} mm to {target:.4f} mm")
            
            if self._stage_type == 'kinesis':
                self._stage.move_by(distance_mm * self.COUNTS_PER_MM)
            elif self._stage_type == 'apt':
                self._stage.move_by(float(distance_mm), blocking=True)
            else:
                return self.move_to(target, wait, timeout_s)
            
            self._position_mm = target
            return True
            
        except Exception as e:
            print(f"[ERROR] Relative move failed: {e}")
            return False
    
    def move_relative_fs(self, delta_fs: float, wait: bool = True) -> bool:
        """
        Move relative by time delay (femtoseconds).
        
        Args:
            delta_fs: Time step in femtoseconds (positive = more delay)
            wait: If True, wait for move to complete
            
        Returns:
            True if move completed
        """
        distance_mm = self.fs_to_mm(delta_fs)
        print(f"[INFO] Moving {delta_fs:+.1f} fs = {distance_mm:+.6f} mm")
        return self.move_relative(distance_mm, wait)
    
    # ========================================================================
    # Position
    # ========================================================================
    
    def get_position(self) -> float:
        """
        Get current position in mm.
        
        Returns:
            Position in millimeters
        """
        if not self.is_connected:
            return self._position_mm
        
        try:
            if self._stage_type == 'kinesis':
                pos = self._stage.get_position() * (1.0 / self.COUNTS_PER_MM)
            elif self._stage_type == 'apt':
                pos = self._stage.position
            elif hasattr(self._stage, 'Position'):
                pos = self._device_units_to_mm(self._stage.Position)
            else:
                pos = self._position_mm
            
            self._position_mm = float(pos)
            return self._position_mm
            
        except Exception as e:
            print(f"[WARN] Could not read position: {e}")
            return self._position_mm
    
    def get_position_fs(self) -> float:
        """Get current position as time delay in femtoseconds."""
        return self.mm_to_fs(self.get_position())
    
    def is_moving(self) -> bool:
        """Check if stage is currently moving."""
        if not self.is_connected:
            return False
        
        try:
            if hasattr(self._stage, 'is_moving'):
                return self._stage.is_moving()
            elif hasattr(self._stage, 'IsMoving'):
                return bool(self._stage.IsMoving)
            else:
                return False
        except:
            return False
    
    # ========================================================================
    # Unit Conversions
    # ========================================================================
    
    def mm_to_fs(self, distance_mm: float) -> float:
        """Convert distance (mm) to time delay (femtoseconds). Double-pass."""
        factor = 2.0 if self.double_pass else 1.0
        return distance_mm * factor / SPEED_OF_LIGHT_MM_FS
    
    def fs_to_mm(self, time_fs: float) -> float:
        """Convert time delay (femtoseconds) to distance (mm). Double-pass."""
        factor = 2.0 if self.double_pass else 1.0
        return time_fs * SPEED_OF_LIGHT_MM_FS / factor
    
    def mm_to_ps(self, distance_mm: float) -> float:
        """Convert distance (mm) to picoseconds."""
        return self.mm_to_fs(distance_mm) / 1000.0
    
    def ps_to_mm(self, time_ps: float) -> float:
        """Convert picoseconds to distance (mm)."""
        return self.fs_to_mm(time_ps * 1000.0)
    
    # ========================================================================
    # Device Unit Conversions (for pythonnet)
    # ========================================================================
    
    def _mm_to_device_units(self, mm: float) -> int:
        """Convert mm to device units (encoder counts)."""
        COUNTS_PER_MM = 34304
        return int(mm * COUNTS_PER_MM)
    
    def _device_units_to_mm(self, counts: int) -> float:
        """Convert device units to mm."""
        COUNTS_PER_MM = 34304
        return counts / COUNTS_PER_MM


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Thorlabs Delay Stage Driver Test")
    print("=" * 60)
    
    stage = DelayStageDriver()
    
    print("\n--- Conversion Test ---")
    print(f"1 mm = {stage.mm_to_fs(1.0):.2f} fs = {stage.mm_to_ps(1.0):.2f} ps")
    print(f"100 fs = {stage.fs_to_mm(100):.6f} mm")
    print(f"1 ps = {stage.fs_to_mm(1000):.4f} mm")
    
    print("\n--- Connecting to Stage ---")
    if stage.connect():
        print(f"\nPosition: {stage.get_position():.4f} mm")
        
        print("\n--- Homing ---")
        if stage.home():
            print("\n--- Test Move ---")
            stage.move_to(1.0)
            print(f"Position: {stage.get_position():.4f} mm = {stage.get_position_fs():.0f} fs")
        
        stage.disconnect()
    else:
        print("[WARN] Could not connect - check hardware")
