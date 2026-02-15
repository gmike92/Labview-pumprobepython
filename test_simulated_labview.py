import socket
import numpy as np
import struct
import time

HOST = '127.0.0.1'
PORT = 5555
NUM_FRAMES = 100
HEIGHT = 128
WIDTH = 128

def generate_dummy_burst(counter, num_frames):
    # Generate N frames
    # Pumped frames (even indices): 0, 2, 4... -> High intensity + Signal
    # Unpumped frames (odd indices): 1, 3, 5... -> High intensity
    
    frames = np.zeros((num_frames, HEIGHT, WIDTH), dtype=np.uint16)
    
    # Base background (simulate raw counts ~1000)
    background = 1000 + np.random.normal(0, 10, (num_frames, HEIGHT, WIDTH))
    
    # Pump Signal (Gaussian spot in center)
    x = np.linspace(-5, 5, WIDTH)
    y = np.linspace(-5, 5, HEIGHT)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-(X**2 + Y**2)/2.0) * 100 * np.sin(counter * 0.1) # Oscillating signal
    
    # Add signal only to even frames (Pumped)
    # Note: Logic in server is: Pumped = 0::2 if not flipped.
    background[0::2] += gaussian.astype(np.uint16)
    
    return background.astype(np.uint16)

def main():
    print(f"Connecting to {HOST}:{PORT}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        print("Connected.")
        
        # Handshake: Read NUM_FRAMES from server (Big-Endian U32)
        header = sock.recv(4)
        if len(header) < 4:
            print("Error: Failed to receive frame count from server.")
            return
        num_frames = struct.unpack('>I', header)[0]
        print(f"Server requested N = {num_frames} frames per burst.")
        
        counter = 0
        while True:
            # 1. Generate Data using N from server
            data = generate_dummy_burst(counter, num_frames)
            
            # 2. Serialize (Big-Endian U16)
            payload = data.astype('>u2').tobytes()
            
            # 3. Create Header (4 bytes size)
            size = len(payload)
            header = struct.pack('>I', size)
            
            # 4. Send
            sock.sendall(header + payload)
            print(f"Sent batch {counter}: {num_frames} frames ({size} bytes)")
            
            counter += 1
            time.sleep(0.5) # Slower rate to see changes
            
    except ConnectionRefusedError:
        print("Connection refused. Is the server running?")
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
