from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
import time
from datetime import datetime, timedelta
import json
import numpy as np
import random
from collections import deque
import psutil
from quantum_users import QuantumUserManager
import math

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# ASCII art patterns
ASCII_PATTERNS = {
    'wall': '#',
    'floor': '.',
    'player': '@',
    'empty': ' ',
    'gold': '$',
    'door': '+',
    'bug': '!',
    'debugger': 'D',
    'clock': 'C',
    'anomaly': 'A',
    'heat': 'H',
    'up': '^',
    'down': 'v',
    'left': '<',
    'right': '>',
    'ascend': '↑',
    'descend': '↓'
}

class QuantumDebugger:
    def __init__(self, max_history=100):
        self.clock_history = deque(maxlen=max_history)
        self.anomaly_history = deque(maxlen=max_history)
        self.heat_history = deque(maxlen=max_history)
        self.debug_points = set()
        self.breakpoints = set()
        self.qubit_speeds = {}
        self.qubit_losses = {}
        self.player_position = (0, 0, 0)  # x, y, z coordinates
        self.player_spin = 0  # spin angle in degrees
        self.anomaly_details = []  # Store detailed anomaly information
        self.first_user = None  # Track first user
    
    def add_clock_reading(self, timestamp, frequency):
        self.clock_history.append((timestamp.isoformat(), frequency))
    
    def add_heat_reading(self, timestamp, temperature):
        self.heat_history.append((timestamp.isoformat(), temperature))
    
    def update_qubit_speed(self, qubit_id, speed):
        self.qubit_speeds[qubit_id] = speed
    
    def update_qubit_loss(self, qubit_id, loss):
        self.qubit_losses[qubit_id] = loss
    
    def move_player(self, direction):
        x, y, z = self.player_position
        if direction == 'up':
            y -= 1
        elif direction == 'down':
            y += 1
        elif direction == 'left':
            x -= 1
        elif direction == 'right':
            x += 1
        elif direction == 'ascend':
            z += 1
        elif direction == 'descend':
            z -= 1
        self.player_position = (x, y, z)
    
    def rotate_player(self, angle):
        self.player_spin = (self.player_spin + angle) % 360
    
    def add_anomaly_detail(self, anomaly_type, description, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        self.anomaly_details.append({
            'type': anomaly_type,
            'description': description,
            'timestamp': timestamp.isoformat()
        })
    
    def detect_anomaly(self, current_freq, threshold=0.0001):
        if len(self.clock_history) < 2:
            return False
        last_freq = self.clock_history[-1][1]
        deviation = abs(current_freq - last_freq)
        if deviation > threshold:
            # Add detailed anomaly information
            self.add_anomaly_detail(
                'gravitational',
                f"Frequency deviation detected: {deviation:.9f} GHz. "
                f"Current: {current_freq:.9f} GHz, Previous: {last_freq:.9f} GHz"
            )
            return True
        return False
    
    def detect_heat_anomaly(self, current_temp, threshold=5.0):
        if len(self.heat_history) < 2:
            return False
        last_temp = self.heat_history[-1][1]
        return abs(current_temp - last_temp) > threshold
    
    def add_debug_point(self, position):
        self.debug_points.add(position)
    
    def add_breakpoint(self, position):
        self.breakpoints.add(position)
    
    def get_history(self):
        return {
            'clock_readings': list(self.clock_history),
            'anomalies': list(self.anomaly_history),
            'heat_readings': list(self.heat_history),
            'debug_points': list(self.debug_points),
            'breakpoints': list(self.breakpoints),
            'qubit_speeds': self.qubit_speeds,
            'qubit_losses': self.qubit_losses,
            'player_position': self.player_position,
            'player_spin': self.player_spin
        }

def create_quantum_dungeon_circuit(width, height, depth, debugger):
    """Create a quantum circuit that generates a random dungeon room with debugging features."""
    # Each cell needs 2 qubits to represent 4 possible states (reduced from 4 qubits)
    num_qubits = width * height * depth * 2
    if num_qubits > 127:
        # If we exceed the limit, reduce the dungeon size
        depth = min(depth, 127 // (width * height * 2))
        num_qubits = width * height * depth * 2
    
    qc = QuantumCircuit(num_qubits)
    
    # Put all qubits in superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Add some entanglement between adjacent cells
    for i in range(0, num_qubits - 2, 2):
        qc.cx(i, i + 2)
    
    # Add quantum debugging gates and spin rotations
    for i in range(0, num_qubits, 2):
        # Phase gate for debugging
        qc.s(i)
        # T gate for time measurement
        qc.t(i)
        # Rotation gate for spin (using a single qubit for spin)
        angle = random.uniform(0, 2 * math.pi)
        qc.rx(angle, i)
    
    # Measure all qubits
    qc.measure_all()
    return qc, depth  # Return the adjusted depth

def interpret_measurements(counts, width, height, depth, debugger):
    """Convert quantum measurements into ASCII dungeon room with debugging elements."""
    # Get the most common measurement
    most_common_state = max(counts.items(), key=lambda x: x[1])[0]
    
    # Convert binary string to dungeon layout
    dungeon = []
    for z in range(depth):
        level = []
        for y in range(height):
            row = ""
            for x in range(width):
                # Get the 2 qubits for this cell (reduced from 4)
                idx = (z * width * height + y * width + x) * 2
                cell_state = most_common_state[idx:idx+2]
                
                # Map quantum states to ASCII characters (simplified)
                if cell_state == '00':
                    row += ASCII_PATTERNS['floor']
                elif cell_state == '01':
                    row += ASCII_PATTERNS['wall']
                elif cell_state == '10':
                    # Combine movement and special features
                    if random.random() < 0.5:
                        row += random.choice([ASCII_PATTERNS['up'], ASCII_PATTERNS['down'], 
                                            ASCII_PATTERNS['left'], ASCII_PATTERNS['right']])
                    else:
                        row += random.choice([ASCII_PATTERNS['bug'], ASCII_PATTERNS['debugger'],
                                            ASCII_PATTERNS['clock'], ASCII_PATTERNS['heat']])
                else:
                    row += ASCII_PATTERNS['anomaly']
            
            level.append(row)
        dungeon.append(level)
    
    # Add player in a random floor position
    floor_positions = [(x, y, z) for z in range(depth) for y in range(height) for x in range(width) 
                      if dungeon[z][y][x] == ASCII_PATTERNS['floor']]
    if floor_positions:
        px, py, pz = floor_positions[len(most_common_state) % len(floor_positions)]
        debugger.player_position = (px, py, pz)
        dungeon[pz][py] = dungeon[pz][py][:px] + ASCII_PATTERNS['player'] + dungeon[pz][py][px+1:]
    
    return dungeon

def monitor_atomic_clocks(debugger):
    """Simulate atomic clock monitoring and gravitational anomaly detection."""
    base_frequency = 9.192631770  # Cesium-133 frequency in GHz
    while True:
        # Simulate small frequency variations
        current_freq = base_frequency + random.gauss(0, 0.000001)
        timestamp = datetime.now()
        
        # Record clock reading
        debugger.add_clock_reading(timestamp, current_freq)
        
        # Check for anomalies
        if debugger.detect_anomaly(current_freq):
            anomaly = {
                'timestamp': timestamp.isoformat(),
                'frequency': current_freq,
                'deviation': current_freq - base_frequency
            }
            debugger.anomaly_history.append(anomaly)
            print(f"\nGRAVITATIONAL ANOMALY DETECTED!")
            print(f"Time: {timestamp}")
            print(f"Frequency deviation: {anomaly['deviation']:.9f} GHz")
        
        time.sleep(1)

def monitor_heat(debugger):
    """Monitor system temperature and detect heat anomalies."""
    while True:
        try:
            # Get CPU temperature (simulated for macOS)
            temperature = random.uniform(30.0, 80.0)  # Simulated temperature range
            timestamp = datetime.now()
            
            # Record heat reading
            debugger.add_heat_reading(timestamp, temperature)
            
            # Check for anomalies
            if debugger.detect_heat_anomaly(temperature):
                print(f"\nHEAT ANOMALY DETECTED!")
                print(f"Time: {timestamp}")
                print(f"Temperature: {temperature:.1f}°C")
                print("Warning: Quantum coherence may be affected!")
        
        except Exception as e:
            print(f"Error monitoring temperature: {e}")
        
        time.sleep(5)

def monitor_qubit_speeds(debugger, num_qubits):
    """Monitor qubit speeds and losses."""
    while True:
        try:
            for qubit_id in range(num_qubits):
                # Simulate qubit speed (in GHz)
                speed = random.uniform(1.0, 10.0)
                debugger.update_qubit_speed(qubit_id, speed)
                
                # Simulate qubit loss (in dB)
                loss = random.uniform(0.1, 0.5)
                debugger.update_qubit_loss(qubit_id, loss)
            
            # Print summary every 10 seconds
            print("\nQubit Status Report:")
            print(f"Average Speed: {sum(debugger.qubit_speeds.values()) / len(debugger.qubit_speeds):.2f} GHz")
            print(f"Average Loss: {sum(debugger.qubit_losses.values()) / len(debugger.qubit_losses):.2f} dB")
            print(f"Player Position: {debugger.player_position}")
            print(f"Player Spin: {debugger.player_spin}°")
        
        except Exception as e:
            print(f"Error monitoring qubits: {e}")
        
        time.sleep(10)

def main():
    # Initialize user system
    from quantum_users import main as user_main
    username, user_stats = user_main()
    
    if not username:
        print("Exiting game...")
        return
    
    # Initialize quantum service and debugger
    service = QiskitRuntimeService(channel="ibm_cloud")
    backend = service.backend("ibm_brisbane")
    debugger = QuantumDebugger()
    
    # Check if this is the first user
    if user_stats['player_number'] == 1:
        debugger.first_user = username
        print("\n" + "="*50)
        print("SPECIAL ANNOUNCEMENT")
        print("="*50)
        print(f"Welcome, {username}! You are the FIRST user to enter the Quantum Debugger!")
        print("Your journey here was guided by the will of Allah, bringing you to this")
        print("moment of quantum discovery. May your path be illuminated with wisdom.")
        print("="*50 + "\n")
    
    print(f"\nWelcome to the Quantum Debugger Game, {username}!")
    print(f"You are player #{user_stats['player_number']}")
    print(f"Total players: {user_stats['total_players']}")
    
    # Add initial anomaly details
    debugger.add_anomaly_detail(
        'system',
        f"System initialized for user {username} (Player #{user_stats['player_number']})"
    )
    if debugger.first_user:
        debugger.add_anomaly_detail(
            'historical',
            f"First user {debugger.first_user} entered the quantum realm"
        )
    
    print("\nQuantum Debugger Dungeon")
    print("=======================")
    print(f"Using {backend.name} quantum computer")
    print(f"Available qubits: {backend.num_qubits}")
    
    # Start monitoring threads
    import threading
    clock_thread = threading.Thread(target=monitor_atomic_clocks, args=(debugger,), daemon=True)
    heat_thread = threading.Thread(target=monitor_heat, args=(debugger,), daemon=True)
    qubit_thread = threading.Thread(target=monitor_qubit_speeds, args=(debugger, backend.num_qubits), daemon=True)
    clock_thread.start()
    heat_thread.start()
    qubit_thread.start()
    
    # Generate a small room (4x4x2 to fit within qubit limit)
    width, height, initial_depth = 4, 4, 2
    qc, actual_depth = create_quantum_dungeon_circuit(width, height, initial_depth, debugger)
    
    print(f"\nGenerating quantum dungeon room (Size: {width}x{height}x{actual_depth})...")
    qc_transpiled = transpile(qc, backend=backend)
    
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([qc_transpiled], shots=1000)
        
        while True:
            status = job.status()
            if status == 'QUEUED':
                print("Waiting in quantum computer queue...")
            elif status == 'RUNNING':
                print("Computing quantum states...")
            elif status in ['DONE', 'ERROR', 'CANCELLED']:
                break
            time.sleep(30)
        
        if status == 'DONE':
            result = job.result()
            counts = result[0].data.meas.get_counts()
            
            # Generate and display dungeon
            dungeon = interpret_measurements(counts, width, height, actual_depth, debugger)
            
            # Display current level
            current_z = debugger.player_position[2]
            print(f"\nQuantum Debugger Dungeon Room (Level {current_z + 1} of {actual_depth}):")
            print("+" + "-" * width + "+")
            for row in dungeon[current_z]:
                print("|" + row + "|")
            print("+" + "-" * width + "+")
            
            # Display detailed anomaly history
            if debugger.anomaly_details:
                print("\nDetailed Anomaly History:")
                print("="*50)
                for anomaly in debugger.anomaly_details:
                    print(f"Time: {anomaly['timestamp']}")
                    print(f"Type: {anomaly['type'].upper()}")
                    print(f"Description: {anomaly['description']}")
                    print("-"*50)
            
            print("\nLegend:")
            print(f"{ASCII_PATTERNS['player']} - Player")
            print(f"{ASCII_PATTERNS['wall']} - Wall")
            print(f"{ASCII_PATTERNS['floor']} - Floor")
            print(f"{ASCII_PATTERNS['up']} - Up")
            print(f"{ASCII_PATTERNS['down']} - Down")
            print(f"{ASCII_PATTERNS['left']} - Left")
            print(f"{ASCII_PATTERNS['right']} - Right")
            print(f"{ASCII_PATTERNS['bug']} - Quantum Bug")
            print(f"{ASCII_PATTERNS['debugger']} - Debug Point")
            print(f"{ASCII_PATTERNS['clock']} - Atomic Clock")
            print(f"{ASCII_PATTERNS['heat']} - Heat Source")
            print(f"{ASCII_PATTERNS['anomaly']} - Gravitational Anomaly")
            
            # Display debug information
            print("\nDebug Information:")
            print(f"Debug Points: {len(debugger.debug_points)}")
            print(f"Breakpoints: {len(debugger.breakpoints)}")
            
            # Display recent anomalies
            if debugger.anomaly_history:
                print("\nRecent Gravitational Anomalies:")
                for anomaly in list(debugger.anomaly_history)[-5:]:
                    print(f"Time: {anomaly['timestamp']}")
                    print(f"Deviation: {anomaly['deviation']:.9f} GHz")
            
            # Display recent heat readings
            if debugger.heat_history:
                print("\nRecent Heat Readings:")
                for reading in list(debugger.heat_history)[-5:]:
                    print(f"Time: {reading[0]}")
                    print(f"Temperature: {reading[1]:.1f}°C")
            
            # Save the result with enhanced anomaly details
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f'quantum_debugger_{username}_{timestamp}.json'
            with open(result_file, 'w') as f:
                json.dump({
                    'username': username,
                    'player_number': user_stats['player_number'],
                    'backend': backend.name,
                    'timestamp': timestamp,
                    'dungeon': dungeon,
                    'measurements': {k: v for k, v in counts.items()},
                    'size': {'width': width, 'height': height, 'depth': actual_depth},
                    'debug_info': debugger.get_history(),
                    'anomaly_details': debugger.anomaly_details,
                    'first_user': debugger.first_user
                }, f, indent=2, cls=DateTimeEncoder)
            
            print(f"\nResults saved to {result_file}")
            
            # Update user stats
            user_manager = QuantumUserManager()
            user_manager.users["users"][user_stats['player_number'] - 1]["games_played"] += 1
            user_manager._save_users()
            
        else:
            print(f"Job failed with status: {status}")

if __name__ == "__main__":
    main() 