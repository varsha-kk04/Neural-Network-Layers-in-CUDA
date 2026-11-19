
import matplotlib.pyplot as plt
import os

os.makedirs("graphs", exist_ok=True)

# Graph 1: Execution Time
methods = ["CPU", "GPU Kernel-only", "GPU End-to-end"]
times = [0.688750, 0.032312, 0.110540]

plt.figure(figsize=(9, 6))
bars = plt.bar(methods, times)

plt.title("CPU vs GPU Execution Time")
plt.xlabel("Implementation")
plt.ylabel("Execution Time (ms)")
plt.bar_label(bars, fmt="%.6f")
plt.tight_layout()
plt.savefig("graphs/execution_time.png", dpi=300)
plt.close()

# Graph 2: Speedup
methods = ["Kernel-only", "End-to-end"]
speedups = [21.315611, 6.230776]

plt.figure(figsize=(9, 6))
bars = plt.bar(methods, speedups)

plt.title("GPU Speedup Compared with CPU")
plt.xlabel("GPU Implementation")
plt.ylabel("Speedup (x)")
plt.bar_label(bars, fmt="%.2fx")
plt.tight_layout()
plt.savefig("graphs/speedup.png", dpi=300)
plt.close()

# Graph 3: Neural Network Dimensions
layers = ["Input", "Hidden", "Output"]
dimensions = [128, 64, 10]

plt.figure(figsize=(9, 6))
bars = plt.bar(layers, dimensions)

plt.title("Neural Network Layer Dimensions")
plt.xlabel("Neural Network Layer")
plt.ylabel("Number of Neurons")
plt.bar_label(bars)
plt.tight_layout()
plt.savefig("graphs/layer_dimensions.png", dpi=300)
plt.close()

print("All 3 graphs created successfully!")
print("Check the graphs folder.")
