\# Experiment 5: Neural Network Layers in CUDA



\*\*Course:\*\* Cloud / GPU Computing Laboratory  

\*\*Experiment No.:\*\* 5  

\*\*Experiment Title:\*\* Neural Network Layers in CUDA  

\*\*Platform:\*\* Windows 11, CUDA Toolkit 13.4, Visual Studio 2022 Build Tools



\---



\## 1. Aim



To implement and evaluate basic neural-network layers using custom CUDA kernels: matrix multiplication with bias, ReLU activation and sigmoid activation.



The same neural-network forward pass is executed on both CPU and GPU. The results are compared to verify numerical correctness, execution times are measured, and GPU speedup is calculated.



\## 2. Learning Outcomes



1\. Understand CUDA kernels, blocks, threads and grid dimensions.

2\. Implement matrix multiplication with bias using CUDA.

3\. Implement ReLU and sigmoid activation functions.

4\. Allocate GPU memory and transfer data between host and device.

5\. Compare CPU and GPU computation results.

6\. Measure execution time and calculate GPU speedup.



\### Hardware Used in This Experiment



| Component | Configuration |

|---|---|

| GPU | NVIDIA GeForce RTX 5060 Ti |

| GPU Memory | 16 GB |

| CUDA Toolkit | 13.4 |

| Compiler | Visual Studio 2022 Build Tools |

| Operating System | Windows |



\## 3. Theory



A neural network forward pass calculates outputs by applying weighted sums and activation functions.



This experiment implements a two-layer feed-forward neural network.



\### Layer 1



Matrix multiplication with bias:



`Z1 = X × W1 + b1`



ReLU activation:



`A1 = ReLU(Z1)`



\### Layer 2



Matrix multiplication with bias:



`Z2 = A1 × W2 + b2`



Sigmoid activation:



`Y = Sigmoid(Z2)`



This experiment performs inference (forward propagation) only. It does not include training, loss calculation, an optimizer or backpropagation.



\### 3.1 Matrix Multiplication with Bias



For matrices A and B:



`C\[row, col] = Σ(A\[row, k] × B\[k, col]) + bias\[col]`



Each CUDA thread calculates one output matrix element.



\### 3.2 ReLU Activation



`ReLU(x) = max(0, x)`



Negative values become zero, while positive values remain unchanged.



\### 3.3 Sigmoid Activation



`Sigmoid(x) = 1 / (1 + exp(-x))`



It converts input values into a range between 0 and 1.



\### 3.4 CPU and GPU Execution



\- \*\*CPU:\*\* Uses ordinary nested loops and standard C++ functions.

\- \*\*GPU:\*\* Transfers input data and weights to device memory, launches CUDA kernels and transfers the final output back to the CPU.



\## 4. Neural Network Configuration



| Parameter | Value |

|---|---:|

| BATCH | 256 |

| INPUT\_SIZE | 128 |

| HIDDEN\_SIZE | 64 |

| OUTPUT\_SIZE | 10 |

| REPEATS | 20 |



\### Tensor Dimensions



| Tensor | Shape |

|---|---|

| Input X | 256 × 128 |

| Weight W1 | 128 × 64 |

| Bias b1 | 64 elements |

| Hidden activation | 256 × 64 |

| Weight W2 | 64 × 10 |

| Bias b2 | 10 elements |

| Final output | 256 × 10 |



The input and weights are generated using deterministic sine and cosine functions so that the CPU and GPU receive identical data.



\## 5. Procedure: Windows Setup



\### Step 1: Check the NVIDIA GPU and Driver



Open the command window and execute:



```cmd

nvidia-smi

```



This command displays the NVIDIA GPU name, driver version, CUDA compatibility information and GPU memory.



\*\*Observed GPU:\*\* NVIDIA GeForce RTX 5060 Ti.



\### Step 2: Check the CUDA Compiler



Execute:



```cmd

nvcc --version

```



This verifies that the CUDA compiler is installed and accessible.



\### Step 3: Find the CUDA Compiler Location



```cmd

where.exe nvcc

```



This displays the location of `nvcc.exe`.



\### Step 4: Configure the Visual Studio C++ Compiler



CUDA requires Microsoft's C++ compiler (`cl.exe`).



The Visual Studio 2022 Build Tools installation must include:



\- Desktop development with C++

\- MSVC v143 x64/x86 build tools

\- Windows 10 or Windows 11 SDK



\### Step 5: Initialize the Developer Environment



Open the Visual Studio 2022 Developer Command Prompt or initialize the environment in CMD using:



```cmd

call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\VsDevCmd.bat" -arch=x64 -host\_arch=x64

```



Verify the compiler:



```cmd

where cl

```



The command should display the location of `cl.exe`.



\*\*Important:\*\* Use the initialized Developer Command Prompt environment for compiling the CUDA program.



\## 7. Procedure: Create and Compile the CUDA Program



\### Step 1: Create the Project Folder



```cmd

mkdir "%USERPROFILE%\\Documents\\Experiment5\_CUDA"

```



\### Step 2: Open the Project Folder



```cmd

cd /d "%USERPROFILE%\\Documents\\Experiment5\_CUDA"

```



\### Step 3: Create the CUDA Source File



```cmd

notepad neural\_network.cu

```



Create the file in Notepad and save it with the `.cu` extension.



\### Step 4: Implement the Program in Eight Phases



The complete CUDA program is available in:



\*\*\[`neural\_network.cu`](neural\_network.cu)\*\*



The program is developed in the following eight phases.



| Phase | Implementation |

|---|---|

| Phase 1 | Header files, CUDA error checking and network constants |

| Phase 2 | CUDA kernels for matrix multiplication, ReLU and sigmoid |

| Phase 3 | CPU implementation of the neural-network forward pass |

| Phase 4 | GPU forward-pass kernel launches and numerical comparison helper |

| Phase 5 | Individual CUDA kernel validation tests |

| Phase 6 | GPU detection, input generation and CPU execution timing |

| Phase 7 | GPU memory allocation, data transfer, warm-up and GPU timing |

| Phase 8 | CPU/GPU comparison, result printing and GPU memory cleanup |



\### Important CUDA Kernel Code



\*\*1. Matrix Multiplication with Bias\*\*



```cpp

\_\_global\_\_ void matrixMulBiasKernel(

&#x20;   const float\* A, const float\* B,

&#x20;   const float\* bias, float\* C,

&#x20;   int M, int N, int K)

{

&#x20;   int row = blockIdx.y \* blockDim.y + threadIdx.y;

&#x20;   int col = blockIdx.x \* blockDim.x + threadIdx.x;



&#x20;   if (row < M \&\& col < N) {

&#x20;       float sum = 0.0f;



&#x20;       for (int k = 0; k < K; ++k) {

&#x20;           sum += A\[row \* K + k] \* B\[k \* N + col];

&#x20;       }



&#x20;       C\[row \* N + col] = sum + bias\[col];

&#x20;   }

}

```



\*\*2. ReLU Activation Kernel\*\*



```cpp

\_\_global\_\_ void reluKernel(float\* values, int count)

{

&#x20;   int idx = blockIdx.x \* blockDim.x + threadIdx.x;



&#x20;   if (idx < count) {

&#x20;       values\[idx] = fmaxf(0.0f, values\[idx]);

&#x20;   }

}

```



\*\*3. Sigmoid Activation Kernel\*\*



```cpp

\_\_global\_\_ void sigmoidKernel(float\* values, int count)

{

&#x20;   int idx = blockIdx.x \* blockDim.x + threadIdx.x;



&#x20;   if (idx < count) {

&#x20;       values\[idx] = 1.0f / (1.0f + expf(-values\[idx]));

&#x20;   }

}

```



The complete source code, including CPU functions, GPU memory management, kernel tests, timing and output validation, is provided in `neural\_network.cu`.



\### Step 5: Compile the Complete Program



Execute:



```cmd

nvcc -O2 neural\_network.cu -o neural\_network.exe

```



The CUDA compiler generates the executable file.



\### Step 6: Verify the Executable



```cmd

dir neural\_network.exe

```



The executable should be listed in the project directory.



\## 8. Procedure: Program Execution and Validation



\### Step 1: Execute the Program



```cmd

neural\_network.exe

```



\### Step 2: Validate the Individual CUDA Kernels



The program performs three individual kernel tests.



| Test | Input | Expected Output |

|---|---|---|

| Matrix multiplication + bias | A = \[\[1,2,3],\[4,5,6]], B = \[\[1,2],\[3,4],\[5,6]], bias = \[1,-1] | \[\[23,27],\[50,63]] |

| ReLU | \[-2,0,3,-4,5] | \[0,0,3,0,5] |

| Sigmoid | \[-2,0,2] | \[0.119203, 0.5, 0.880797] |



All three tests must display `PASSED`.



\### Step 3: Check the Forward-Pass Dimensions



The program should display:



\- Input: 256 × 128

\- Hidden layer: 256 × 64

\- Output: 256 × 10



\### Step 4: Compare CPU and GPU Results



The program calculates the maximum absolute difference between CPU and GPU output values.



The outputs are considered matching when:



`Maximum absolute difference < 0.0001`



The expected validation message is:



`CPU and GPU outputs: MATCH (within tolerance)`



\### Step 5: Record the Execution Results



Record the CPU time, GPU kernel-only time, GPU end-to-end time and speedup values printed by the program.



\### Step 6: Repeat the Execution



Run the executable two more times to observe timing variations.



The experiment records the actual run presented in the observation table below.



\## 9. Timing Methodology



\### CPU Timing

The CPU forward pass is executed 20 times, and the average execution time is calculated.



\### GPU Warm-up

The GPU forward pass is executed once before timing to reduce first-launch initialization effects.



\### GPU Kernel-only Timing

CUDA events measure repeated kernel execution when the input data and weights are already stored in GPU memory.



\### GPU End-to-end Timing

The host clock measures the complete GPU operation, including:



1\. Host-to-device data transfers.

2\. CUDA kernel execution.

3\. GPU synchronization.

4\. Device-to-host output transfer.



End-to-end time is generally greater than kernel-only time because it includes data-transfer and other overheads.



\## 10. Formulae and Performance Calculations



\### Kernel-only Speedup



`Kernel-only speedup = CPU execution time / GPU kernel-only time`



\### End-to-end Speedup



`End-to-end speedup = CPU execution time / GPU end-to-end time`



\### Maximum Absolute Output Difference



`Maximum difference = max |CPU output\[i] - GPU output\[i]|`



A speedup greater than 1 indicates that the GPU execution is faster for the measured scope.



\## 11. Observation Table and Actual Results



The following values were obtained from the experiment executed on the NVIDIA GeForce RTX 5060 Ti.



| Measurement | Actual Result |

|---|---:|

| Average CPU compute time | 0.688750 ms |

| Average GPU kernel-only time | 0.032312 ms |

| Average GPU end-to-end time | 0.110540 ms |

| Kernel-only speedup | 21.315611× |

| End-to-end speedup | 6.230776× |

| Maximum absolute output difference | 0.000000 |



\### Performance Calculation



Kernel-only speedup:



`0.688750 / 0.032312 = 21.315611×`



End-to-end speedup:



`0.688750 / 0.110540 = 6.230776×`



The GPU kernel-only execution is approximately 21.32 times faster than the CPU computation for this measurement.



The complete GPU operation, including memory transfers, is approximately 6.23 times faster.



\### Graph 1: CPU vs GPU Execution Time



!\[CPU vs GPU Execution Time Graph](graphs/execution\_time.png)



\*\*Observation:\*\* The GPU kernel-only execution time is lower than the CPU execution time. The end-to-end GPU time is higher than kernel-only time because it includes memory transfers and synchronization.



\### Graph 2: GPU Speedup Comparison



!\[GPU Speedup Graph](graphs/speedup.png)



\*\*Observation:\*\* The kernel-only speedup is 21.315611×, while the end-to-end speedup is 6.230776×.



\### Graph 3: Neural Network Layer Dimensions



!\[Neural Network Layer Dimensions Graph](graphs/layer\_dimensions.png)



\*\*Observation:\*\* The network processes 128 input features, 64 hidden-layer neurons and 10 output neurons.



\## 12. Output Screenshot



The following screenshot records the successful compilation and execution of the CUDA program on the NVIDIA GeForce RTX 5060 Ti.



\### Successful Execution Output



!\[Experiment 5 Execution Output](screenshots/03\_execution\_output.png)



\### Output Verification



| Verification | Result |

|---|---|

| GPU detected | NVIDIA GeForce RTX 5060 Ti |

| Matrix multiplication + bias | PASSED |

| ReLU | PASSED |

| Sigmoid | PASSED |

| CPU and GPU dimensions | Correct |

| Maximum output difference | 0.000000 |

| CPU and GPU outputs | MATCH (within tolerance) |



\### First 10 GPU Output Values



```text

0.498196 0.503101 0.508006 0.512911 0.517814

0.497729 0.502638 0.507548 0.512457 0.517364

```



\## 13. Result



The neural-network layers were successfully implemented using custom CUDA kernels.



Matrix multiplication with bias, ReLU activation and sigmoid activation were individually tested and passed.



The two-layer neural-network forward pass was executed on both CPU and GPU. The maximum absolute output difference was 0.000000, confirming that the CPU and GPU outputs matched within the specified tolerance.



The experiment achieved a kernel-only speedup of \*\*21.315611×\*\* and an end-to-end speedup of \*\*6.230776×\*\* on the NVIDIA GeForce RTX 5060 Ti.



\## 17. Conclusion



Thus, the implementation and evaluation of neural-network layers using CUDA were successfully completed. The experiment demonstrated how custom CUDA kernels perform matrix multiplication with bias and activation functions on the GPU.



The CPU and GPU outputs were verified successfully, and execution-time measurements showed the performance advantage of GPU computation. The experiment also demonstrated the effect of memory-transfer overhead on end-to-end execution time.



\---



\*\*Project Files Used in This Experiment\*\*



```text

Experiment5\_CUDA/

│

├── neural\_network.cu

├── create\_graphs.py

├── README.md

│

├── graphs/

│   ├── execution\_time.png

│   ├── speedup.png

│   └── layer\_dimensions.png

│

└── screenshots/

&#x20;   └── 03\_execution\_output.png

```

