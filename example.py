from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
import time
from datetime import datetime, timedelta
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set up IBM Quantum account
service = QiskitRuntimeService(channel="ibm_cloud")

def create_stability_test_circuit(num_qubits, depth):
    """Create a circuit to test qubit stability with increasing depth."""
    qc = QuantumCircuit(num_qubits)
    
    # Initial Hadamard gates to create superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Add layers of operations
    for _ in range(depth):
        # Add CNOT gates between adjacent qubits
        for i in range(0, num_qubits-1, 2):
            qc.cx(i, i+1)
        # Add single-qubit gates
        for i in range(num_qubits):
            qc.h(i)
            qc.s(i)
    
    qc.measure_all()
    return qc

def run_stability_test(backend, num_qubits, max_depth, shots_per_depth):
    """Run stability test with increasing circuit depth."""
    results = {
        'depths': [],
        'success_rates': [],
        'error_rates': [],
        'execution_times': [],
        'gate_counts': []
    }
    
    for depth in range(1, max_depth + 1):
        print(f"\nTesting circuit depth {depth}...")
        
        # Create and transpile circuit
        qc = create_stability_test_circuit(num_qubits, depth)
        qc_transpiled = transpile(qc, backend=backend)
        
        # Record gate count
        gate_count = len(qc_transpiled.data)
        results['gate_counts'].append(gate_count)
        
        # Run the circuit
        start_time = time.time()
        with Session(backend=backend) as session:
            sampler = Sampler()
            job = sampler.run([qc_transpiled], shots=shots_per_depth)
            
            # Wait for completion
            while True:
                status = job.status()
                if status in ['DONE', 'ERROR', 'CANCELLED']:
                    break
                time.sleep(30)
            
            if status == 'DONE':
                result = job.result()
                prob_dist = result[0].data.meas.get_counts()
                
                # Calculate success rate (probability of measuring all zeros)
                success_rate = prob_dist.get('0' * num_qubits, 0) / shots_per_depth
                error_rate = 1 - success_rate
                
                # Record results
                results['depths'].append(depth)
                results['success_rates'].append(success_rate)
                results['error_rates'].append(error_rate)
                results['execution_times'].append(time.time() - start_time)
                
                print(f"Depth {depth}: Success rate = {success_rate:.2%}, Error rate = {error_rate:.2%}")
            else:
                print(f"Depth {depth} failed with status: {status}")
    
    return results

def plot_results(results, backend_name):
    """Plot the stability test results."""
    plt.figure(figsize=(15, 10))
    
    # Success rate vs depth
    plt.subplot(2, 2, 1)
    plt.plot(results['depths'], results['success_rates'], 'b-o')
    plt.title('Success Rate vs Circuit Depth')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Success Rate')
    plt.grid(True)
    
    # Error rate vs depth
    plt.subplot(2, 2, 2)
    plt.plot(results['depths'], results['error_rates'], 'r-o')
    plt.title('Error Rate vs Circuit Depth')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Error Rate')
    plt.grid(True)
    
    # Execution time vs depth
    plt.subplot(2, 2, 3)
    plt.plot(results['depths'], results['execution_times'], 'g-o')
    plt.title('Execution Time vs Circuit Depth')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Execution Time (s)')
    plt.grid(True)
    
    # Gate count vs depth
    plt.subplot(2, 2, 4)
    plt.plot(results['depths'], results['gate_counts'], 'm-o')
    plt.title('Gate Count vs Circuit Depth')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Number of Gates')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'stability_test_{backend_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def main():
    # Test parameters
    num_qubits = 2  # Number of qubits to test
    max_depth = 10  # Maximum circuit depth to test
    shots_per_depth = 1000  # Number of shots per depth
    
    # Get backend
    backend = service.backend("ibm_brisbane")
    print(f"Using backend: {backend.name}")
    print(f"Number of qubits: {backend.num_qubits}")
    print(f"Pending jobs in queue: {backend.status().pending_jobs}")
    
    # Run stability test
    results = run_stability_test(backend, num_qubits, max_depth, shots_per_depth)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'stability_test_{backend.name}_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'backend': backend.name,
            'num_qubits': num_qubits,
            'max_depth': max_depth,
            'shots_per_depth': shots_per_depth,
            'results': results
        }, f, indent=2)
    
    # Plot results
    plot_results(results, backend.name)
    
    print(f"\nResults have been saved to:")
    print(f"- {results_file} (raw data in JSON format)")
    print(f"- stability_test_{backend.name}_{timestamp}.png (plots)")

if __name__ == "__main__":
    main()
