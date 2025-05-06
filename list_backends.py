from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize the service
service = QiskitRuntimeService(channel="ibm_quantum")

# List all available backends
print("Available backends:")
for backend in service.backends():
    print(f"- {backend.name}") 