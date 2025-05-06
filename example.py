from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
import time
from datetime import datetime, timedelta
import json

# Set up IBM Quantum account
service = QiskitRuntimeService(channel="ibm_cloud")

# Create a quantum circuit
qc = QuantumCircuit(2)
qc.h(0)  # Hadamard gate
qc.cx(0, 1)  # CNOT gate
qc.measure_all()

# Submit job using the Sampler primitive
backend = service.backend("ibm_brisbane")
print(f"Using backend: {backend.name}")
print(f"Number of qubits: {backend.num_qubits}")
print(f"Pending jobs in queue: {backend.status().pending_jobs}")
print("Transpiling circuit...")
qc_transpiled = transpile(qc, backend=backend)

# Create a log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"quantum_results_{timestamp}.txt"

with open(log_file, 'w') as f:
    f.write(f"Quantum Computation Results - {timestamp}\n")
    f.write("===================================\n\n")
    f.write(f"Backend: {backend.name}\n")
    f.write(f"Number of qubits: {backend.num_qubits}\n")
    f.write(f"Pending jobs in queue: {backend.status().pending_jobs}\n\n")
    f.write("Circuit Information:\n")
    f.write(f"Number of gates: {len(qc_transpiled.data)}\n")
    f.write(f"Circuit depth: {qc_transpiled.depth()}\n\n")
    f.write("Job Status Updates:\n")

with Session(backend=backend) as session:
    sampler = Sampler()
    print("\nSubmitting job to quantum computer...")
    job = sampler.run([qc_transpiled], shots=1000)
    
    # Check job status periodically
    start_time = datetime.now()
    last_status = None
    while True:
        status = job.status()
        current_time = datetime.now()
        elapsed_time = current_time - start_time
        
        # Only print if status changed
        if status != last_status:
            status_update = f"\nJob status: {status}\nTime elapsed: {elapsed_time}"
            print(status_update)
            
            with open(log_file, 'a') as f:
                f.write(f"{status_update}\n")
            
            if status == 'QUEUED':
                # Rough estimate: each job takes about 2-3 minutes
                estimated_wait = backend.status().pending_jobs * 2.5
                eta = current_time + timedelta(minutes=estimated_wait)
                eta_update = f"Estimated completion time: {eta.strftime('%H:%M:%S')}"
                print(eta_update)
                with open(log_file, 'a') as f:
                    f.write(f"{eta_update}\n")
            elif status == 'RUNNING':
                running_update = "Job is being executed on the quantum computer..."
                print(running_update)
                with open(log_file, 'a') as f:
                    f.write(f"{running_update}\n")
        
        if status in ['DONE', 'ERROR', 'CANCELLED']:
            break
            
        last_status = status
        time.sleep(30)  # Check every 30 seconds
    
    if status == 'DONE':
        result = job.result()
        results_summary = f"""
Detailed Results:
----------------
Total shots: 1000

Measurement probabilities:
"""
        # Get the probability distribution from the result
        prob_dist = result[0].data.meas.get_counts()
        total_shots = sum(prob_dist.values())
        
        for state, count in prob_dist.items():
            probability = count / total_shots
            results_summary += f"State {state}: {probability:.2%} ({count} counts)\n"
        
        print(results_summary)
        with open(log_file, 'a') as f:
            f.write(results_summary)
        
        # Save raw results to a separate JSON file
        json_file = f"quantum_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'counts': prob_dist,
                'total_shots': total_shots,
                'circuit_info': {
                    'num_gates': len(qc_transpiled.data),
                    'depth': qc_transpiled.depth()
                }
            }, f, indent=2)
            
        print(f"\nResults have been saved to:")
        print(f"- {log_file} (human-readable format)")
        print(f"- {json_file} (raw data in JSON format)")
    else:
        error_msg = f"\nJob ended with status: {status}"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(error_msg)
