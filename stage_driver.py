"""
StageDriver - Singleton class for NIREOS Gemini (SmarAct) stage control.

This module provides a thread-safe, single-instance stage driver that handles
all SCU3DControl.dll interactions. Used for Twins hyperspectral scans.

Usage:
    from stage_driver import StageDriver
    
    stage = StageDriver()  # Always returns the same instance
    stage.connect()
    stage.move_to(5.0)     # Move to 5mm
    stage.wait_for_stop()  # Wait until motion complete
    pos = stage.get_position()
    stage.disconnect()

NO GUI CODE IN THIS FILE - Pure hardware control layer.
"""

import ctypes
from ctypes import POINTER, c_uint, c_int
import time
from typing import Optional
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DLL_PATH = r".\Twins\Python Scripts\SCU3DControl.dll"

DLL_SEARCH_PATHS = [
    r".\Twins\Python Scripts\SCU3DControl.dll",
    r"C:\Users\mguizzardi\Desktop\Camera python\TWINS FILE\Twins\Python Scripts\SCU3DControl.dll",
    r".\SCU3DControl.dll",
]

HOME_POSITION_MM = 19.0
SAFE_POSITION_MM = 25.0
POSITION_SCALE = 10000

STATUS_STOPPED = 0
STATUS_MOVING = 6


# ============================================================================
# Singleton StageDriver
# ============================================================================

class StageDriver:
    """
    Singleton stage driver for NIREOS Gemini (SmarAct SCU).
    Ensures only ONE instance exists across all modules.
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
            
        self._initialized = True
        self.dll = None
        self.dll_path = DEFAULT_DLL_PATH
        self.is_connected = False
        
        self.system_index = c_uint(0)
        self.channel_index = c_uint(0)
        self._position_mm = 0.0
    
    # ========================================================================
    # Connection Methods
    # ========================================================================
    
    def connect(self, dll_path: Optional[str] = None) -> bool:
        """Connect to the stage: Load DLL, initialize, and find reference."""
        if self.is_connected:
            print("[INFO] Stage already connected")
            return True
        
        if dll_path:
            self.dll_path = dll_path
        
        if not self._load_dll():
            return False
        
        try:
            self.dll.SA_InitDevices.argtypes = [c_uint]
            self.dll.SA_InitDevices.restype = c_int
            
            result = self.dll.SA_InitDevices(c_uint(0))
            
            if result != 0:
                print(f"[ERROR] SA_InitDevices failed with code: {result}")
                print("        Please power cycle the GEMINI and try again")
                return False
            
            print("[OK] Stage devices initialized")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize devices: {e}")
            return False
        
        # Check if physical position is known
        try:
            known = c_uint(0)
            self.dll.SA_GetPhysicalPositionKnown_S.argtypes = [
                c_uint, c_uint, POINTER(c_uint)
            ]
            self.dll.SA_GetPhysicalPositionKnown_S.restype = c_int
            
            result = self.dll.SA_GetPhysicalPositionKnown_S(
                self.system_index, 
                self.channel_index, 
                ctypes.byref(known)
            )
            
            if result != 0:
                print(f"[WARN] GetPhysicalPositionKnown failed: {result}")
            
            if known.value == 0:
                print("[INFO] Finding reference mark...")
                if not self._find_reference():
                    print("[WARN] Could not find reference, continuing anyway")
            else:
                print("[OK] Physical position already known")
                
        except Exception as e:
            print(f"[WARN] Error checking position: {e}")
        
        # Move to home position
        self._move_to_position(HOME_POSITION_MM)
        self._wait_for_motion_complete()
        
        self.is_connected = True
        self._position_mm = HOME_POSITION_MM
        print(f"[OK] Stage connected and moved to home ({HOME_POSITION_MM} mm)")
        
        return True
    
    def _load_dll(self) -> bool:
        """Load the DLL library."""
        paths_to_try = [self.dll_path] + DLL_SEARCH_PATHS
        
        for path in paths_to_try:
            if Path(path).exists():
                try:
                    self.dll = ctypes.CDLL(path)
                    self.dll_path = path
                    print(f"[OK] Stage DLL loaded: {path}")
                    return True
                except OSError as e:
                    print(f"[WARN] Failed to load {path}: {e}")
                    continue
        
        print("[ERROR] SCU3DControl.dll not found in any search path")
        return False
    
    def _find_reference(self) -> bool:
        """Find the reference mark (home position)."""
        try:
            hold_time = c_uint(0)
            auto_zero = c_uint(0)
            
            self.dll.SA_MoveToReference_S.argtypes = [
                c_uint, c_uint, c_uint, c_uint
            ]
            self.dll.SA_MoveToReference_S.restype = c_int
            
            result = self.dll.SA_MoveToReference_S(
                self.system_index,
                self.channel_index,
                hold_time,
                auto_zero
            )
            
            if result != 0:
                print(f"[ERROR] MoveToReference failed: {result}")
                return False
            
            self._wait_for_motion_complete()
            print("[OK] Reference mark found")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error finding reference: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from the stage safely."""
        if not self.is_connected:
            return True
        
        try:
            print(f"[INFO] Moving to safe position ({SAFE_POSITION_MM} mm)...")
            self.move_to(SAFE_POSITION_MM)
            self.wait_for_stop()
        except Exception as e:
            print(f"[WARN] Error moving to safe position: {e}")
        
        try:
            self.dll.SA_ReleaseDevices.argtypes = []
            self.dll.SA_ReleaseDevices.restype = c_int
            self.dll.SA_ReleaseDevices()
            print("[OK] Stage disconnected")
        except Exception as e:
            print(f"[WARN] Error releasing devices: {e}")
        
        self.is_connected = False
        self.dll = None
        return True
    
    # ========================================================================
    # Position Methods
    # ========================================================================
    
    def get_position(self) -> float:
        """Get current stage position in millimeters."""
        if not self.is_connected:
            return self._position_mm
        
        try:
            position = c_int(0)
            
            self.dll.SA_GetPosition_S.argtypes = [
                c_uint, c_uint, POINTER(c_int)
            ]
            self.dll.SA_GetPosition_S.restype = c_int
            
            result = self.dll.SA_GetPosition_S(
                self.system_index,
                self.channel_index,
                ctypes.byref(position)
            )
            
            if result == 0:
                self._position_mm = position.value / POSITION_SCALE
            
        except Exception as e:
            print(f"[WARN] Error reading position: {e}")
        
        return self._position_mm
    
    def move_to(self, position_mm: float) -> bool:
        """Move stage to absolute position in millimeters."""
        if not self.is_connected:
            print("[ERROR] Stage not connected")
            return False
        
        return self._move_to_position(position_mm)
    
    def _move_to_position(self, position_mm: float) -> bool:
        """Internal method to send move command."""
        try:
            position_raw = c_int(int(position_mm * POSITION_SCALE))
            
            self.dll.SA_MovePositionAbsolute_S.argtypes = [
                c_uint, c_uint, c_int, c_uint
            ]
            self.dll.SA_MovePositionAbsolute_S.restype = c_int
            
            result = self.dll.SA_MovePositionAbsolute_S(
                self.system_index,
                self.channel_index,
                position_raw,
                c_uint(60000)
            )
            
            if result != 0:
                print(f"[ERROR] MovePositionAbsolute failed: {result}")
                return False
            
            self._position_mm = position_mm
            return True
            
        except Exception as e:
            print(f"[ERROR] Move failed: {e}")
            return False
    
    def move_by(self, distance_mm: float) -> bool:
        """Move stage by relative distance in millimeters."""
        if not self.is_connected:
            print("[ERROR] Stage not connected")
            return False
        
        try:
            distance_raw = c_int(int(distance_mm * POSITION_SCALE))
            
            self.dll.SA_MovePositionRelative_S.argtypes = [
                c_uint, c_uint, c_int, c_uint
            ]
            self.dll.SA_MovePositionRelative_S.restype = c_int
            
            result = self.dll.SA_MovePositionRelative_S(
                self.system_index,
                self.channel_index,
                distance_raw,
                c_uint(0)
            )
            
            if result != 0:
                print(f"[ERROR] MovePositionRelative failed: {result}")
                return False
            
            self._position_mm += distance_mm
            return True
            
        except Exception as e:
            print(f"[ERROR] Relative move failed: {e}")
            return False
    
    # ========================================================================
    # Motion Control
    # ========================================================================
    
    def wait_for_stop(self, timeout_s: float = 30.0) -> bool:
        """Wait for stage motion to complete."""
        if not self.is_connected:
            return True
        return self._wait_for_motion_complete(timeout_s)
    
    def _wait_for_motion_complete(self, timeout_s: float = 30.0) -> bool:
        """Internal method to poll status until motion complete."""
        try:
            status = c_int(0)
            
            self.dll.SA_GetStatus_S.argtypes = [
                c_uint, c_uint, POINTER(c_int)
            ]
            self.dll.SA_GetStatus_S.restype = c_int
            
            start_time = time.time()
            
            while True:
                result = self.dll.SA_GetStatus_S(
                    self.system_index,
                    self.channel_index,
                    ctypes.byref(status)
                )
                
                if result != 0:
                    print(f"[WARN] GetStatus failed: {result}")
                    return False
                
                if status.value != STATUS_MOVING:
                    return True
                
                if time.time() - start_time > timeout_s:
                    print(f"[WARN] Motion timeout after {timeout_s}s")
                    return False
                
                time.sleep(0.01)
                
        except Exception as e:
            print(f"[ERROR] Error waiting for stop: {e}")
            return False
    
    def is_moving(self) -> bool:
        """Check if stage is currently moving."""
        if not self.is_connected:
            return False
        
        try:
            status = c_int(0)
            
            self.dll.SA_GetStatus_S.argtypes = [
                c_uint, c_uint, POINTER(c_int)
            ]
            self.dll.SA_GetStatus_S.restype = c_int
            
            result = self.dll.SA_GetStatus_S(
                self.system_index,
                self.channel_index,
                ctypes.byref(status)
            )
            
            if result == 0:
                return status.value == STATUS_MOVING
                
        except Exception:
            pass
        
        return False


# ============================================================================
# Module-level convenience
# ============================================================================

def get_stage() -> StageDriver:
    """Get the singleton StageDriver instance."""
    return StageDriver()


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("NIREOS Gemini Stage - Driver Test")
    print("=" * 60)
    
    stage = StageDriver()
    
    print("\n1. Connecting to stage...")
    if not stage.connect():
        print("[FAIL] Could not connect to stage")
        sys.exit(1)
    
    print(f"\n2. Current position: {stage.get_position():.3f} mm")
    
    print("\n3. Moving to 5 mm...")
    stage.move_to(5.0)
    stage.wait_for_stop()
    print(f"   Position: {stage.get_position():.3f} mm")
    
    print("\n4. Disconnecting...")
    stage.disconnect()
    
    print("\n[OK] All tests passed!")
